//! Memory Coalescing Analysis
//!
//! Analyzes memory access patterns to determine coalescing efficiency.
//! Coalesced memory access is critical for GPU performance - when threads
//! in a warp access consecutive memory addresses, the hardware can combine
//! these into fewer memory transactions.

use crate::ir::inst::{Instruction, ValueId};
use crate::ir::types::AddressSpace;

/// Memory access pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    /// addr = base + tid * stride
    /// Coalesced iff stride == element_size
    Linear { base: ValueId, stride: Stride },

    /// addr = base + (tid % width) * stride
    /// Coalesced if width >= 32 and stride == element_size
    Strided {
        base: ValueId,
        stride: Stride,
        width: u32,
    },

    /// addr = base + indirect[tid]
    /// Never coalesced (gather/scatter)
    Indirect {
        base: ValueId,
        index_source: ValueId,
    },

    /// addr = base (broadcast)
    /// Single transaction, but serialized
    Broadcast { base: ValueId },

    /// Cannot determine statically
    Unknown,
}

/// Stride representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stride {
    /// Constant stride in bytes
    Constant(i64),
    /// Stride depends on register
    Dynamic(ValueId),
    /// Unknown stride
    Unknown,
}

impl Stride {
    /// Check if stride equals element size (unit stride)
    pub fn is_unit(&self, element_size: usize) -> bool {
        matches!(self, Stride::Constant(s) if *s == element_size as i64)
    }

    /// Check if stride is known at compile time
    pub fn is_constant(&self) -> bool {
        matches!(self, Stride::Constant(_))
    }

    /// Get constant value if known
    pub fn as_constant(&self) -> Option<i64> {
        match self {
            Stride::Constant(s) => Some(*s),
            _ => None,
        }
    }
}

/// Coalescing efficiency metrics
#[derive(Debug, Clone, Copy)]
pub struct CoalesceMetrics {
    /// Transactions per warp (1 = perfectly coalesced, 32 = worst case)
    pub transactions_per_warp: f32,
    /// Bytes actually used / bytes transferred
    pub efficiency: f32,
    /// Is this pattern worth optimizing?
    pub needs_optimization: bool,
}

impl CoalesceMetrics {
    /// Perfect coalescing metrics
    pub fn perfect() -> Self {
        Self {
            transactions_per_warp: 1.0,
            efficiency: 1.0,
            needs_optimization: false,
        }
    }

    /// Worst case (fully uncoalesced)
    pub fn worst_case() -> Self {
        Self {
            transactions_per_warp: 32.0,
            efficiency: 0.03125, // 4 bytes used / 128 bytes fetched
            needs_optimization: true,
        }
    }

    /// Create metrics with specific values
    pub fn new(transactions: f32, efficiency: f32) -> Self {
        Self {
            transactions_per_warp: transactions,
            efficiency,
            needs_optimization: efficiency < 0.5,
        }
    }
}

/// Optimization hints for improving coalescing
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Use shared memory to reorder accesses
    SharedMemoryTranspose { stride: usize },
    /// Sort indices before gather
    SortIndices,
    /// Convert Array-of-Structures to Structure-of-Arrays
    AosToSoa,
    /// Use texture cache for irregular access
    UseTextureCache,
    /// Consider vectorized loads (float4)
    VectorizeLoads { width: u32 },
}

/// Analyze memory access coalescing
pub struct CoalesceAnalyzer {
    /// Cache line size (typically 128 bytes)
    cache_line_size: usize,
    /// Warp size (32 on NVIDIA)
    warp_size: u32,
}

impl CoalesceAnalyzer {
    /// Create a new analyzer with default settings
    pub fn new() -> Self {
        Self {
            cache_line_size: 128,
            warp_size: 32,
        }
    }

    /// Create analyzer with custom cache line size
    pub fn with_cache_line_size(mut self, size: usize) -> Self {
        self.cache_line_size = size;
        self
    }

    /// Create analyzer with custom warp size
    pub fn with_warp_size(mut self, size: u32) -> Self {
        self.warp_size = size;
        self
    }

    /// Analyze a single memory access instruction
    pub fn analyze_access(
        &self,
        addr_computation: &[Instruction],
        element_size: usize,
    ) -> AccessPattern {
        self.trace_address(addr_computation, element_size)
    }

    /// Trace address computation backwards to find pattern
    fn trace_address(&self, insts: &[Instruction], element_size: usize) -> AccessPattern {
        // Look for patterns in the instruction sequence
        for inst in insts.iter().rev() {
            match inst {
                // GetElementPtr is already in a good form
                Instruction::GetElementPtr { base, .. } => {
                    return AccessPattern::Linear {
                        base: *base,
                        stride: Stride::Constant(element_size as i64),
                    };
                }

                // Fused multiply-add pattern: addr = base + tid * stride
                Instruction::FMA { a, c, .. } => {
                    // This could be tid * stride + base
                    // Simplified analysis - assume 'a' might be thread-dependent
                    return AccessPattern::Linear {
                        base: *c,
                        stride: Stride::Dynamic(*a),
                    };
                }

                // Binary add might be address computation
                Instruction::BinOp {
                    op: crate::ir::inst::BinOp::Add,
                    lhs,
                    rhs,
                    ..
                } => {
                    // Check if one operand is from a multiply
                    return AccessPattern::Linear {
                        base: *lhs,
                        stride: Stride::Dynamic(*rhs),
                    };
                }

                // Load from a constant base could be broadcast
                Instruction::Const { dst, .. } => {
                    return AccessPattern::Broadcast { base: *dst };
                }

                _ => {}
            }
        }

        AccessPattern::Unknown
    }

    /// Compute coalescing metrics for a pattern
    pub fn compute_metrics(&self, pattern: &AccessPattern, element_size: usize) -> CoalesceMetrics {
        match pattern {
            AccessPattern::Linear { stride, .. } => {
                if stride.is_unit(element_size) {
                    // Perfect coalescing
                    let elements_per_line = self.cache_line_size / element_size;
                    let transactions =
                        (self.warp_size as usize + elements_per_line - 1) / elements_per_line;

                    CoalesceMetrics::new(
                        transactions as f32,
                        (self.warp_size as usize * element_size) as f32
                            / (transactions * self.cache_line_size) as f32,
                    )
                } else if let Stride::Constant(s) = stride {
                    self.strided_metrics(*s, element_size)
                } else {
                    CoalesceMetrics::worst_case()
                }
            }

            AccessPattern::Broadcast { .. } => {
                // Single cache line, broadcast to all threads
                CoalesceMetrics {
                    transactions_per_warp: 1.0,
                    efficiency: element_size as f32 / self.cache_line_size as f32,
                    needs_optimization: false, // This is intentional
                }
            }

            AccessPattern::Indirect { .. } => {
                // Assume worst case for indirect
                CoalesceMetrics::worst_case()
            }

            AccessPattern::Strided { stride, width, .. } => {
                if *width >= self.warp_size && stride.is_unit(element_size) {
                    CoalesceMetrics::perfect()
                } else {
                    CoalesceMetrics::worst_case()
                }
            }

            AccessPattern::Unknown => {
                // Conservative estimate
                CoalesceMetrics {
                    transactions_per_warp: 16.0, // Assume some coalescence
                    efficiency: 0.125,
                    needs_optimization: true,
                }
            }
        }
    }

    /// Metrics for constant stride access
    fn strided_metrics(&self, stride: i64, element_size: usize) -> CoalesceMetrics {
        let stride = stride.unsigned_abs() as usize;

        if stride == element_size {
            // Unit stride - perfect
            return CoalesceMetrics::perfect();
        }

        // Calculate how many cache lines a warp touches
        let span = stride * (self.warp_size as usize - 1) + element_size;
        let cache_lines = (span + self.cache_line_size - 1) / self.cache_line_size;

        CoalesceMetrics {
            transactions_per_warp: cache_lines as f32,
            efficiency: (self.warp_size as usize * element_size) as f32
                / (cache_lines * self.cache_line_size) as f32,
            needs_optimization: cache_lines > 4,
        }
    }

    /// Suggest optimization for poor coalescing
    pub fn suggest_optimization(&self, pattern: &AccessPattern) -> Option<OptimizationHint> {
        match pattern {
            AccessPattern::Linear {
                stride: Stride::Constant(s),
                ..
            } if *s > 4 && *s <= 32 => {
                // Strided access might benefit from shared memory transpose
                Some(OptimizationHint::SharedMemoryTranspose {
                    stride: *s as usize,
                })
            }

            AccessPattern::Indirect { .. } => {
                // Gather might benefit from sorting indices
                Some(OptimizationHint::SortIndices)
            }

            AccessPattern::Linear {
                stride: Stride::Constant(s),
                ..
            } if *s > 128 => {
                // Very large stride - consider AoS to SoA transform
                Some(OptimizationHint::AosToSoa)
            }

            _ => None,
        }
    }
}

impl Default for CoalesceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Report for a single memory access
#[derive(Debug)]
pub struct AccessReport {
    /// Instruction index in the kernel
    pub instruction_index: usize,
    /// Memory space being accessed
    pub memory_space: AddressSpace,
    /// Element size in bytes
    pub element_size: usize,
    /// Detected access pattern
    pub pattern: AccessPattern,
    /// Coalescing metrics
    pub metrics: CoalesceMetrics,
    /// Optimization hint if applicable
    pub hint: Option<OptimizationHint>,
}

/// Kernel-level coalescing report
#[derive(Debug)]
pub struct KernelCoalesceReport {
    /// Kernel name
    pub kernel_name: String,
    /// Individual access reports
    pub accesses: Vec<AccessReport>,
    /// Overall efficiency (average)
    pub overall_efficiency: f32,
    /// Estimated bandwidth utilization
    pub estimated_bandwidth_utilization: f32,
}

impl KernelCoalesceReport {
    /// Create a new report
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            accesses: Vec::new(),
            overall_efficiency: 0.0,
            estimated_bandwidth_utilization: 0.0,
        }
    }

    /// Add an access report
    pub fn add_access(&mut self, access: AccessReport) {
        self.accesses.push(access);
        self.recalculate_overall();
    }

    /// Recalculate overall metrics
    fn recalculate_overall(&mut self) {
        if self.accesses.is_empty() {
            self.overall_efficiency = 0.0;
            self.estimated_bandwidth_utilization = 0.0;
            return;
        }

        let total_efficiency: f32 = self.accesses.iter().map(|a| a.metrics.efficiency).sum();
        self.overall_efficiency = total_efficiency / self.accesses.len() as f32;

        // Bandwidth utilization considers global memory accesses more heavily
        let global_accesses: Vec<_> = self
            .accesses
            .iter()
            .filter(|a| a.memory_space == AddressSpace::Global)
            .collect();

        if global_accesses.is_empty() {
            self.estimated_bandwidth_utilization = 1.0;
        } else {
            let global_efficiency: f32 = global_accesses.iter().map(|a| a.metrics.efficiency).sum();
            self.estimated_bandwidth_utilization = global_efficiency / global_accesses.len() as f32;
        }
    }

    /// Check if kernel has good coalescing
    pub fn is_well_coalesced(&self) -> bool {
        self.overall_efficiency > 0.75
    }

    /// Get the worst access pattern
    pub fn worst_access(&self) -> Option<&AccessReport> {
        self.accesses
            .iter()
            .filter(|a| a.memory_space == AddressSpace::Global)
            .min_by(|a, b| {
                a.metrics
                    .efficiency
                    .partial_cmp(&b.metrics.efficiency)
                    .unwrap()
            })
    }

    /// Get all accesses that need optimization
    pub fn accesses_needing_optimization(&self) -> Vec<&AccessReport> {
        self.accesses
            .iter()
            .filter(|a| a.metrics.needs_optimization)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_stride_coalescing() {
        let analyzer = CoalesceAnalyzer::new();

        let pattern = AccessPattern::Linear {
            base: ValueId(0),
            stride: Stride::Constant(4), // float32
        };

        let metrics = analyzer.compute_metrics(&pattern, 4);

        // 32 floats = 128 bytes = 1 cache line
        assert_eq!(metrics.transactions_per_warp, 1.0);
        assert_eq!(metrics.efficiency, 1.0);
        assert!(!metrics.needs_optimization);
    }

    #[test]
    fn test_strided_coalescing() {
        let analyzer = CoalesceAnalyzer::new();

        // Stride of 8 bytes (every other float)
        let pattern = AccessPattern::Linear {
            base: ValueId(0),
            stride: Stride::Constant(8),
        };

        let metrics = analyzer.compute_metrics(&pattern, 4);

        // 32 threads × 8 byte stride = 256 byte span = 2 cache lines
        assert_eq!(metrics.transactions_per_warp, 2.0);
        assert_eq!(metrics.efficiency, 0.5); // Half utilization
    }

    #[test]
    fn test_column_major_coalescing() {
        let analyzer = CoalesceAnalyzer::new();

        // Column-major access in a 1024-wide matrix
        // Each thread accesses consecutive columns = stride of 1024*4 bytes
        let pattern = AccessPattern::Linear {
            base: ValueId(0),
            stride: Stride::Constant(4096),
        };

        let metrics = analyzer.compute_metrics(&pattern, 4);

        // Each access hits different cache line - terrible!
        assert!(metrics.transactions_per_warp >= 32.0);
        assert!(metrics.needs_optimization);
    }

    #[test]
    fn test_broadcast_pattern() {
        let analyzer = CoalesceAnalyzer::new();

        let pattern = AccessPattern::Broadcast { base: ValueId(0) };
        let metrics = analyzer.compute_metrics(&pattern, 4);

        // Single transaction, but low efficiency (only 4 bytes used of 128)
        assert_eq!(metrics.transactions_per_warp, 1.0);
        assert!(!metrics.needs_optimization); // Broadcast is intentional
    }

    #[test]
    fn test_optimization_hints() {
        let analyzer = CoalesceAnalyzer::new();

        // Small stride should suggest shared memory transpose
        let pattern = AccessPattern::Linear {
            base: ValueId(0),
            stride: Stride::Constant(16),
        };
        let hint = analyzer.suggest_optimization(&pattern);
        assert!(matches!(
            hint,
            Some(OptimizationHint::SharedMemoryTranspose { .. })
        ));

        // Indirect access should suggest sorting
        let pattern = AccessPattern::Indirect {
            base: ValueId(0),
            index_source: ValueId(1),
        };
        let hint = analyzer.suggest_optimization(&pattern);
        assert!(matches!(hint, Some(OptimizationHint::SortIndices)));

        // Very large stride should suggest AoS to SoA
        let pattern = AccessPattern::Linear {
            base: ValueId(0),
            stride: Stride::Constant(256),
        };
        let hint = analyzer.suggest_optimization(&pattern);
        assert!(matches!(hint, Some(OptimizationHint::AosToSoa)));
    }

    #[test]
    fn test_kernel_report() {
        let mut report = KernelCoalesceReport::new("test_kernel");

        report.add_access(AccessReport {
            instruction_index: 0,
            memory_space: AddressSpace::Global,
            element_size: 4,
            pattern: AccessPattern::Linear {
                base: ValueId(0),
                stride: Stride::Constant(4),
            },
            metrics: CoalesceMetrics::perfect(),
            hint: None,
        });

        report.add_access(AccessReport {
            instruction_index: 1,
            memory_space: AddressSpace::Global,
            element_size: 4,
            pattern: AccessPattern::Linear {
                base: ValueId(0),
                stride: Stride::Constant(128),
            },
            metrics: CoalesceMetrics::new(32.0, 0.03125),
            hint: Some(OptimizationHint::AosToSoa),
        });

        assert_eq!(report.accesses.len(), 2);
        assert!(!report.is_well_coalesced());
        assert!(report.worst_access().is_some());
    }
}
