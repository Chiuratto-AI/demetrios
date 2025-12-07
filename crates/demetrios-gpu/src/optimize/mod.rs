//! GPU Optimization Module
//!
//! This module provides analysis and optimization tools for GPU kernels:
//!
//! - **Coalescing Analysis**: Detect and optimize memory access patterns
//! - **Bank Conflict Detection**: Identify shared memory bank conflicts
//! - **Occupancy Calculator**: Compute and optimize GPU occupancy
//! - **Register Pressure Analysis**: Detect register spilling risks
//! - **Memory Pool**: Fast memory allocation for iterative algorithms
//! - **Prefetching**: Latency hiding through prefetch and double buffering
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::optimize::{AnalysisPipeline, KernelResources, ArchLimits};
//!
//! let pipeline = AnalysisPipeline::for_a100();
//! let resources = KernelResources::new(256, 64, 8192);
//! let analysis = pipeline.analyze(&kernel, &resources);
//!
//! if !analysis.is_acceptable() {
//!     for rec in &analysis.recommendations {
//!         println!("{}: {}", rec.priority, rec.description);
//!     }
//! }
//! ```

pub mod banks;
pub mod coalesce;
pub mod occupancy;
pub mod pool;
pub mod prefetch;
pub mod registers;

// Re-exports for convenience
pub use banks::{BankConflictAnalyzer, BankConflictFix, BankConflictReport, SharedAccessPattern};
pub use coalesce::{
    AccessPattern, AccessReport, CoalesceAnalyzer, CoalesceMetrics, KernelCoalesceReport,
    OptimizationHint, Stride,
};
pub use occupancy::{
    ArchLimits, KernelResources, LimitingFactor, OccupancyCalculator, OccupancyReport,
    OccupancySuggestion,
};
pub use pool::{CachingAllocator, DeviceId, DevicePtr, MemoryPool, PoolConfig, PoolStats};
pub use prefetch::{
    AsyncCopy, DoubleBuffer, MemoryAccessPattern, Prefetch, PrefetchInserter, PrefetchLevel,
    PrefetchSchedule, SoftwarePipeline, TripleBuffer,
};
pub use registers::{
    LiveRange, PressureHotspot, RegisterPressureAnalyzer, RegisterPressureReport,
    RegisterSuggestion,
};

use crate::ir::function::GpuFunction;
use crate::ir::types::AddressSpace;

/// Optimization recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Must fix - severe performance issue
    Critical,
    /// Should fix - significant improvement possible
    High,
    /// Nice to have - moderate improvement
    Medium,
    /// Minor optimization
    Low,
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Priority::Critical => write!(f, "CRITICAL"),
            Priority::High => write!(f, "HIGH"),
            Priority::Medium => write!(f, "MEDIUM"),
            Priority::Low => write!(f, "LOW"),
        }
    }
}

/// Optimization category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Memory coalescing issues
    MemoryCoalescing,
    /// Bank conflict issues
    BankConflicts,
    /// Occupancy issues
    Occupancy,
    /// Register pressure issues
    RegisterPressure,
    /// Prefetching opportunities
    Prefetching,
}

impl std::fmt::Display for Category {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Category::MemoryCoalescing => write!(f, "Memory Coalescing"),
            Category::BankConflicts => write!(f, "Bank Conflicts"),
            Category::Occupancy => write!(f, "Occupancy"),
            Category::RegisterPressure => write!(f, "Register Pressure"),
            Category::Prefetching => write!(f, "Prefetching"),
        }
    }
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Priority level
    pub priority: Priority,
    /// Category of issue
    pub category: Category,
    /// Human-readable description
    pub description: String,
    /// Estimated speedup (if known)
    pub estimated_speedup: Option<f32>,
}

impl std::fmt::Display for Recommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}: {}",
            self.priority, self.category, self.description
        )?;
        if let Some(speedup) = self.estimated_speedup {
            write!(f, " (est. {:.1}x speedup)", speedup)?;
        }
        Ok(())
    }
}

/// Complete kernel analysis result
#[derive(Debug)]
pub struct KernelAnalysis {
    /// Kernel name
    pub name: String,
    /// Coalescing analysis
    pub coalescing: KernelCoalesceReport,
    /// Occupancy analysis
    pub occupancy: OccupancyReport,
    /// Register pressure analysis
    pub register_pressure: RegisterPressureReport,
    /// Bank conflict reports
    pub bank_conflicts: Vec<BankConflictReport>,
    /// Overall performance score (0.0 - 1.0)
    pub overall_score: f32,
    /// Optimization recommendations
    pub recommendations: Vec<Recommendation>,
}

impl KernelAnalysis {
    /// Check if kernel performance is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.overall_score > 0.5
            && !self.register_pressure.spilling
            && self
                .recommendations
                .iter()
                .all(|r| r.priority != Priority::Critical)
    }

    /// Check if kernel performance is good
    pub fn is_good(&self) -> bool {
        self.overall_score > 0.75
            && !self.register_pressure.spilling
            && self
                .recommendations
                .iter()
                .all(|r| r.priority != Priority::Critical && r.priority != Priority::High)
    }

    /// Get critical recommendations only
    pub fn critical_recommendations(&self) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority == Priority::Critical)
            .collect()
    }

    /// Get high priority recommendations
    pub fn high_priority_recommendations(&self) -> Vec<&Recommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.priority == Priority::Critical || r.priority == Priority::High)
            .collect()
    }
}

impl std::fmt::Display for KernelAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Kernel Analysis: {} ===", self.name)?;
        writeln!(f, "Overall Score: {:.0}%", self.overall_score * 100.0)?;
        writeln!(f)?;
        writeln!(f, "Occupancy: {:.1}%", self.occupancy.percentage())?;
        writeln!(
            f,
            "  Active warps: {}/{}",
            self.occupancy.active_warps, self.occupancy.theoretical_warps
        )?;
        writeln!(f, "  Limiting factor: {}", self.occupancy.limiting_factor)?;
        writeln!(f)?;
        writeln!(f, "Register Pressure:")?;
        writeln!(f, "  Max pressure: {}", self.register_pressure.max_pressure)?;
        writeln!(
            f,
            "  Spilling: {}",
            if self.register_pressure.spilling {
                "YES"
            } else {
                "no"
            }
        )?;
        writeln!(f)?;
        writeln!(f, "Memory Coalescing:")?;
        writeln!(
            f,
            "  Overall efficiency: {:.1}%",
            self.coalescing.overall_efficiency * 100.0
        )?;
        writeln!(
            f,
            "  Bandwidth utilization: {:.1}%",
            self.coalescing.estimated_bandwidth_utilization * 100.0
        )?;
        writeln!(f)?;
        if !self.recommendations.is_empty() {
            writeln!(f, "Recommendations:")?;
            for rec in &self.recommendations {
                writeln!(f, "  {}", rec)?;
            }
        }
        Ok(())
    }
}

/// Full analysis pipeline
pub struct AnalysisPipeline {
    arch: ArchLimits,
    coalesce_analyzer: CoalesceAnalyzer,
    bank_analyzer: BankConflictAnalyzer,
    occupancy_calc: OccupancyCalculator,
    register_analyzer: RegisterPressureAnalyzer,
}

impl AnalysisPipeline {
    /// Create a new analysis pipeline for the given architecture
    pub fn new(arch: ArchLimits) -> Self {
        Self {
            occupancy_calc: OccupancyCalculator::new(arch.clone()),
            register_analyzer: RegisterPressureAnalyzer::new(255),
            coalesce_analyzer: CoalesceAnalyzer::new(),
            bank_analyzer: BankConflictAnalyzer::new(),
            arch,
        }
    }

    /// Create pipeline for NVIDIA A100
    pub fn for_a100() -> Self {
        Self::new(ArchLimits::a100())
    }

    /// Create pipeline for NVIDIA H100
    pub fn for_h100() -> Self {
        Self::new(ArchLimits::h100())
    }

    /// Create pipeline for NVIDIA L4
    pub fn for_l4() -> Self {
        Self::new(ArchLimits::l4())
    }

    /// Get architecture limits
    pub fn arch(&self) -> &ArchLimits {
        &self.arch
    }

    /// Run full analysis on a kernel
    pub fn analyze(&self, kernel: &GpuFunction, resources: &KernelResources) -> KernelAnalysis {
        let occupancy = self.occupancy_calc.calculate(resources);
        let register_pressure = self.register_analyzer.analyze(kernel);

        // Analyze memory accesses for coalescing
        let coalescing = self.analyze_coalescing(kernel);
        let bank_conflicts = self.analyze_bank_conflicts(kernel);

        let mut recommendations = Vec::new();

        // Generate recommendations based on occupancy
        if occupancy.occupancy < 0.5 {
            let priority = if occupancy.occupancy < 0.25 {
                Priority::Critical
            } else {
                Priority::High
            };

            recommendations.push(Recommendation {
                priority,
                category: Category::Occupancy,
                description: format!(
                    "Low occupancy ({:.0}%) due to {}. {}",
                    occupancy.occupancy * 100.0,
                    occupancy.limiting_factor,
                    match occupancy.limiting_factor {
                        LimitingFactor::Registers =>
                            "Try reducing register usage or decreasing block size.",
                        LimitingFactor::SharedMemory => "Try reducing shared memory usage.",
                        LimitingFactor::BlockSize => "Try increasing block size.",
                        LimitingFactor::MaxBlocksPerSm => "Try decreasing block size.",
                        LimitingFactor::Optimal => "Occupancy is optimal.",
                    }
                ),
                estimated_speedup: Some(1.0 / occupancy.occupancy.max(0.1)),
            });
        }

        // Generate recommendations based on register pressure
        if register_pressure.spilling {
            recommendations.push(Recommendation {
                priority: Priority::Critical,
                category: Category::RegisterPressure,
                description: format!(
                    "Register spilling detected! {} registers needed but {} available. \
                     Spilling to local memory is catastrophic for performance.",
                    register_pressure.max_pressure, 255
                ),
                estimated_speedup: Some(5.0),
            });
        } else if register_pressure.max_pressure > 200 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: Category::RegisterPressure,
                description: format!(
                    "High register pressure ({}). Consider reducing complexity or using shared memory.",
                    register_pressure.max_pressure
                ),
                estimated_speedup: Some(1.2),
            });
        }

        // Generate recommendations based on coalescing
        if coalescing.overall_efficiency < 0.5 {
            recommendations.push(Recommendation {
                priority: Priority::High,
                category: Category::MemoryCoalescing,
                description: format!(
                    "Poor memory coalescing ({:.0}% efficiency). Consider restructuring data access patterns.",
                    coalescing.overall_efficiency * 100.0
                ),
                estimated_speedup: Some(2.0),
            });
        } else if coalescing.overall_efficiency < 0.75 {
            recommendations.push(Recommendation {
                priority: Priority::Medium,
                category: Category::MemoryCoalescing,
                description: format!(
                    "Suboptimal memory coalescing ({:.0}% efficiency).",
                    coalescing.overall_efficiency * 100.0
                ),
                estimated_speedup: Some(1.3),
            });
        }

        // Generate recommendations based on bank conflicts
        for conflict in &bank_conflicts {
            if conflict.is_severe() {
                recommendations.push(Recommendation {
                    priority: Priority::High,
                    category: Category::BankConflicts,
                    description: format!(
                        "Severe bank conflicts detected ({}-way). Add padding to shared memory arrays.",
                        conflict.conflict_degree
                    ),
                    estimated_speedup: Some(conflict.conflict_degree as f32 / 2.0),
                });
            } else if conflict.has_conflicts() {
                recommendations.push(Recommendation {
                    priority: Priority::Low,
                    category: Category::BankConflicts,
                    description: format!(
                        "Minor bank conflicts ({}-way). Consider padding if performance-critical.",
                        conflict.conflict_degree
                    ),
                    estimated_speedup: Some(1.1),
                });
            }
        }

        // Calculate overall score
        let occupancy_score = occupancy.occupancy;
        let coalescing_score = coalescing.overall_efficiency;
        let register_score = if register_pressure.spilling { 0.1 } else { 1.0 };
        let bank_score = if bank_conflicts.iter().any(|c| c.is_severe()) {
            0.5
        } else {
            1.0
        };

        let overall_score = occupancy_score * coalescing_score * register_score * bank_score;

        // Sort recommendations by priority
        recommendations.sort_by(|a, b| a.priority.cmp(&b.priority));

        KernelAnalysis {
            name: kernel.name.clone(),
            coalescing,
            occupancy,
            register_pressure,
            bank_conflicts,
            overall_score,
            recommendations,
        }
    }

    /// Analyze coalescing for a kernel
    fn analyze_coalescing(&self, kernel: &GpuFunction) -> KernelCoalesceReport {
        let mut report = KernelCoalesceReport::new(&kernel.name);

        // Analyze each memory access instruction
        for (block_idx, block) in kernel.blocks.values().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                if let Some(access) = self.analyze_memory_access(inst, block_idx * 1000 + inst_idx)
                {
                    report.add_access(access);
                }
            }
        }

        // If no accesses found, assume good efficiency
        if report.accesses.is_empty() {
            report.overall_efficiency = 1.0;
            report.estimated_bandwidth_utilization = 1.0;
        }

        report
    }

    /// Analyze a single memory access instruction
    fn analyze_memory_access(
        &self,
        inst: &crate::ir::inst::Instruction,
        inst_idx: usize,
    ) -> Option<AccessReport> {
        use crate::ir::inst::Instruction;

        match inst {
            Instruction::Load { ptr, ty, space, .. } => {
                let element_size = ty.size_bytes().unwrap_or(4);
                let pattern = AccessPattern::Linear {
                    base: *ptr,
                    stride: Stride::Constant(element_size as i64),
                };
                let metrics = self
                    .coalesce_analyzer
                    .compute_metrics(&pattern, element_size);
                let hint = self.coalesce_analyzer.suggest_optimization(&pattern);

                Some(AccessReport {
                    instruction_index: inst_idx,
                    memory_space: *space,
                    element_size,
                    pattern,
                    metrics,
                    hint,
                })
            }
            Instruction::Store { ty, space, .. } => {
                let element_size = ty.size_bytes().unwrap_or(4);
                // Assume unit stride for stores
                let pattern = AccessPattern::Linear {
                    base: crate::ir::inst::ValueId(0),
                    stride: Stride::Constant(element_size as i64),
                };
                let metrics = self
                    .coalesce_analyzer
                    .compute_metrics(&pattern, element_size);

                Some(AccessReport {
                    instruction_index: inst_idx,
                    memory_space: *space,
                    element_size,
                    pattern,
                    metrics,
                    hint: None,
                })
            }
            _ => None,
        }
    }

    /// Analyze bank conflicts for a kernel
    fn analyze_bank_conflicts(&self, kernel: &GpuFunction) -> Vec<BankConflictReport> {
        let mut reports = Vec::new();

        // Look for shared memory accesses
        for block in kernel.blocks.values() {
            for inst in &block.instructions {
                if let Some(report) = self.analyze_shared_access(inst) {
                    reports.push(report);
                }
            }
        }

        reports
    }

    /// Analyze a single shared memory access
    fn analyze_shared_access(
        &self,
        inst: &crate::ir::inst::Instruction,
    ) -> Option<BankConflictReport> {
        use crate::ir::inst::Instruction;

        match inst {
            Instruction::Load { space, .. } | Instruction::Store { space, .. }
                if *space == AddressSpace::Shared =>
            {
                // Assume linear access pattern with unit stride
                let pattern = SharedAccessPattern::Linear { base: 0, stride: 4 };
                Some(self.bank_analyzer.analyze_access(&pattern))
            }
            _ => None,
        }
    }

    /// Quick check if kernel is "good enough"
    pub fn is_acceptable(&self, analysis: &KernelAnalysis) -> bool {
        analysis.is_acceptable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::function::{BasicBlock, FunctionKind, GpuFunction};
    use crate::ir::inst::{BlockId, Constant, Instruction};
    use indexmap::IndexMap;

    fn make_test_kernel(name: &str, instructions: Vec<Instruction>) -> GpuFunction {
        let entry = BlockId(0);
        let mut block = BasicBlock::new(entry);
        block.instructions = instructions;

        let mut blocks = IndexMap::new();
        blocks.insert(entry, block);

        GpuFunction {
            name: name.to_string(),
            kind: FunctionKind::Kernel,
            params: Vec::new(),
            return_type: None,
            blocks,
            entry,
            value_types: IndexMap::new(),
            shared_mem: Vec::new(),
            dynamic_shared_mem: 0,
            max_threads: None,
            min_blocks: None,
        }
    }

    #[test]
    fn test_analysis_pipeline() {
        let pipeline = AnalysisPipeline::for_a100();

        let kernel = make_test_kernel(
            "test",
            vec![Instruction::Const {
                dst: crate::ir::inst::ValueId(0),
                value: Constant::I32(42),
            }],
        );

        let resources = KernelResources::minimal(256);
        let analysis = pipeline.analyze(&kernel, &resources);

        assert!(!analysis.name.is_empty());
        assert!(analysis.overall_score > 0.0);
    }

    #[test]
    fn test_high_register_pressure() {
        let pipeline = AnalysisPipeline::for_a100();

        // Create kernel with many live values that are all used
        let mut instructions = Vec::new();
        for i in 0..300 {
            instructions.push(Instruction::Const {
                dst: crate::ir::inst::ValueId(i),
                value: Constant::I32(i as i32),
            });
        }
        // Use all values to keep them live
        for i in 1..300u32 {
            instructions.push(Instruction::BinOp {
                dst: crate::ir::inst::ValueId(300 + i),
                op: crate::ir::inst::BinOp::Add,
                lhs: crate::ir::inst::ValueId(i - 1),
                rhs: crate::ir::inst::ValueId(i),
                ty: crate::ir::types::ScalarType::I32,
            });
        }

        let kernel = make_test_kernel("high_pressure", instructions);
        let resources = KernelResources::new(256, 128, 0);
        let analysis = pipeline.analyze(&kernel, &resources);

        // Should detect high register pressure
        assert!(analysis.register_pressure.max_pressure > 200);
    }

    #[test]
    fn test_low_occupancy() {
        let pipeline = AnalysisPipeline::for_a100();

        let kernel = make_test_kernel("low_occ", vec![]);
        let resources = KernelResources::new(32, 255, 64 * 1024); // Very constrained

        let analysis = pipeline.analyze(&kernel, &resources);

        // Should have low occupancy
        assert!(analysis.occupancy.occupancy < 0.5);
        assert!(analysis
            .recommendations
            .iter()
            .any(|r| r.category == Category::Occupancy));
    }

    #[test]
    fn test_recommendations_sorted() {
        let pipeline = AnalysisPipeline::for_a100();

        let mut instructions = Vec::new();
        for i in 0..300 {
            instructions.push(Instruction::Const {
                dst: crate::ir::inst::ValueId(i),
                value: Constant::I32(i as i32),
            });
        }

        let kernel = make_test_kernel("test", instructions);
        let resources = KernelResources::new(32, 255, 64 * 1024);
        let analysis = pipeline.analyze(&kernel, &resources);

        // Recommendations should be sorted by priority
        for window in analysis.recommendations.windows(2) {
            assert!(window[0].priority <= window[1].priority);
        }
    }

    #[test]
    fn test_is_acceptable() {
        let pipeline = AnalysisPipeline::for_a100();

        // Good kernel with reasonable resources that achieve good occupancy
        // Using 256 threads, 16 registers (low), and minimal shared memory
        let kernel = make_test_kernel("good", vec![]);
        let resources = KernelResources::new(256, 16, 0);
        let analysis = pipeline.analyze(&kernel, &resources);

        // With 16 registers, 256 threads, and no shared memory,
        // we should get good occupancy and no critical issues
        assert!(
            analysis.is_acceptable(),
            "Expected acceptable kernel, got score={}, occupancy={}, spilling={}, recommendations={:?}",
            analysis.overall_score,
            analysis.occupancy.occupancy,
            analysis.register_pressure.spilling,
            analysis.recommendations.iter().map(|r| (&r.priority, &r.description)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_display() {
        let pipeline = AnalysisPipeline::for_a100();
        let kernel = make_test_kernel("display_test", vec![]);
        let resources = KernelResources::minimal(256);
        let analysis = pipeline.analyze(&kernel, &resources);

        let display = format!("{}", analysis);
        assert!(display.contains("display_test"));
        assert!(display.contains("Occupancy"));
    }
}
