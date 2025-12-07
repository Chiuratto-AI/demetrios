//! Register Pressure Analysis
//!
//! When a kernel uses more registers than available, the compiler "spills"
//! to local memory (which is actually global memory). This is catastrophic
//! for performance.
//!
//! This module analyzes register usage and live ranges to detect potential
//! spilling and suggest optimizations.

use crate::ir::function::GpuFunction;
use crate::ir::inst::{Instruction, ValueId};
use std::collections::{HashMap, HashSet};

/// Live range of a register/value
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Instruction index where value is first defined
    pub start: usize,
    /// Instruction index after last use
    pub end: usize,
    /// All instruction indices where value is used
    pub uses: Vec<usize>,
}

impl LiveRange {
    /// Create a new live range
    pub fn new(start: usize) -> Self {
        Self {
            start,
            end: start + 1,
            uses: Vec::new(),
        }
    }

    /// Length of the live range
    pub fn length(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Check if this range overlaps with another
    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if a value is live at a given instruction
    pub fn is_live_at(&self, instruction: usize) -> bool {
        instruction >= self.start && instruction < self.end
    }
}

/// Point of high register pressure
#[derive(Debug, Clone)]
pub struct PressureHotspot {
    /// Instruction index
    pub instruction_index: usize,
    /// Register pressure at this point
    pub pressure: u32,
    /// Number of live registers
    pub live_registers: usize,
}

/// Suggestion for reducing register pressure
#[derive(Debug, Clone)]
pub enum RegisterSuggestion {
    /// Reduce block size to allow more registers per thread
    ReduceBlockSize { reason: String },
    /// Split kernel into multiple phases
    SplitKernel { reason: String },
    /// Move intermediate results to shared memory
    UseSharedMemory { reason: String },
    /// Reorder instructions to reduce live ranges
    ReorderInstructions { reason: String },
    /// Use more aggressive compiler optimization
    CompilerFlags { flags: Vec<String> },
}

impl std::fmt::Display for RegisterSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterSuggestion::ReduceBlockSize { reason } => {
                write!(f, "Reduce block size: {}", reason)
            }
            RegisterSuggestion::SplitKernel { reason } => {
                write!(f, "Split kernel: {}", reason)
            }
            RegisterSuggestion::UseSharedMemory { reason } => {
                write!(f, "Use shared memory: {}", reason)
            }
            RegisterSuggestion::ReorderInstructions { reason } => {
                write!(f, "Reorder instructions: {}", reason)
            }
            RegisterSuggestion::CompilerFlags { flags } => {
                write!(f, "Use compiler flags: {}", flags.join(", "))
            }
        }
    }
}

/// Register pressure analysis report
#[derive(Debug)]
pub struct RegisterPressureReport {
    /// Maximum registers live at any point
    pub max_pressure: u32,
    /// Will this kernel likely spill?
    pub spilling: bool,
    /// Estimated number of spilled registers
    pub estimated_spills: u32,
    /// Live ranges for each value
    pub live_ranges: HashMap<ValueId, LiveRange>,
    /// High-pressure points
    pub hotspots: Vec<PressureHotspot>,
    /// Suggestions for improvement
    pub suggestions: Vec<RegisterSuggestion>,
}

impl RegisterPressureReport {
    /// Check if register usage is acceptable
    pub fn is_acceptable(&self) -> bool {
        !self.spilling
    }

    /// Get values with longest live ranges
    pub fn longest_live_ranges(&self, count: usize) -> Vec<(ValueId, &LiveRange)> {
        let mut ranges: Vec<_> = self.live_ranges.iter().collect();
        ranges.sort_by(|a, b| b.1.length().cmp(&a.1.length()));
        ranges
            .into_iter()
            .take(count)
            .map(|(id, range)| (*id, range))
            .collect()
    }
}

impl std::fmt::Display for RegisterPressureReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Max register pressure: {}", self.max_pressure)?;
        writeln!(f, "Spilling: {}", if self.spilling { "YES" } else { "no" })?;
        if self.spilling {
            writeln!(f, "Estimated spills: {}", self.estimated_spills)?;
        }
        if !self.hotspots.is_empty() {
            writeln!(f, "Hotspots:")?;
            for hotspot in &self.hotspots {
                writeln!(
                    f,
                    "  Instruction {}: {} registers",
                    hotspot.instruction_index, hotspot.pressure
                )?;
            }
        }
        if !self.suggestions.is_empty() {
            writeln!(f, "Suggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  - {}", suggestion)?;
            }
        }
        Ok(())
    }
}

/// Register pressure analyzer
pub struct RegisterPressureAnalyzer {
    /// Maximum registers available per thread
    max_registers: u32,
    /// Threshold for considering high pressure
    high_pressure_threshold: u32,
}

impl RegisterPressureAnalyzer {
    /// Create a new analyzer with default settings (255 registers, 75% threshold)
    pub fn new(max_registers: u32) -> Self {
        Self {
            max_registers,
            high_pressure_threshold: max_registers * 3 / 4,
        }
    }

    /// Set custom high pressure threshold
    pub fn with_threshold(mut self, threshold: u32) -> Self {
        self.high_pressure_threshold = threshold;
        self
    }

    /// Analyze register pressure in a function
    pub fn analyze(&self, function: &GpuFunction) -> RegisterPressureReport {
        // Collect all instructions from all blocks
        let instructions: Vec<&Instruction> = function
            .blocks
            .values()
            .flat_map(|block| block.instructions.iter())
            .collect();

        // Build live ranges
        let live_ranges = self.compute_live_ranges(&instructions);

        // Compute pressure at each instruction
        let mut max_pressure = 0u32;
        let mut hotspots = Vec::new();

        for (i, _inst) in instructions.iter().enumerate() {
            let live_at_i: HashSet<_> = live_ranges
                .iter()
                .filter(|(_, range)| range.is_live_at(i))
                .map(|(id, _)| *id)
                .collect();

            let pressure = live_at_i.len() as u32;
            max_pressure = max_pressure.max(pressure);

            if pressure >= self.high_pressure_threshold {
                hotspots.push(PressureHotspot {
                    instruction_index: i,
                    pressure,
                    live_registers: live_at_i.len(),
                });
            }
        }

        let spilling = max_pressure > self.max_registers;
        let estimated_spills = if spilling {
            max_pressure - self.max_registers
        } else {
            0
        };

        let suggestions = if spilling {
            self.generate_suggestions(max_pressure)
        } else {
            Vec::new()
        };

        // Limit hotspots to top 10
        hotspots.sort_by(|a, b| b.pressure.cmp(&a.pressure));
        hotspots.truncate(10);

        RegisterPressureReport {
            max_pressure,
            spilling,
            estimated_spills,
            live_ranges,
            hotspots,
            suggestions,
        }
    }

    /// Compute live ranges for all values
    fn compute_live_ranges(&self, instructions: &[&Instruction]) -> HashMap<ValueId, LiveRange> {
        let mut ranges: HashMap<ValueId, LiveRange> = HashMap::new();

        // First pass: find definition points
        for (i, inst) in instructions.iter().enumerate() {
            if let Some(dst) = inst.dst() {
                ranges.entry(dst).or_insert_with(|| LiveRange::new(i));
            }
        }

        // Second pass: find use points and extend ranges
        for (i, inst) in instructions.iter().enumerate() {
            for reg in self.used_values(inst) {
                if let Some(range) = ranges.get_mut(&reg) {
                    range.end = range.end.max(i + 1);
                    range.uses.push(i);
                }
            }
        }

        ranges
    }

    /// Get values used by an instruction
    fn used_values(&self, inst: &Instruction) -> Vec<ValueId> {
        match inst {
            Instruction::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::UnaryOp { src, .. } => vec![*src],

            Instruction::Cmp { lhs, rhs, .. } => vec![*lhs, *rhs],

            Instruction::Convert { src, .. } => vec![*src],

            Instruction::Bitcast { src, .. } => vec![*src],

            Instruction::Load { ptr, .. } => vec![*ptr],

            Instruction::Store { ptr, value, .. } => vec![*ptr, *value],

            Instruction::Atomic { ptr, value, .. } => vec![*ptr, *value],

            Instruction::AtomicCAS {
                ptr,
                expected,
                desired,
                ..
            } => vec![*ptr, *expected, *desired],

            Instruction::GetElementPtr { base, indices, .. } => {
                let mut vals = vec![*base];
                vals.extend(indices.iter().copied());
                vals
            }

            Instruction::CondBranch { cond, .. } => vec![*cond],

            Instruction::Return { value } => value.iter().copied().collect(),

            Instruction::Select {
                cond,
                true_val,
                false_val,
                ..
            } => vec![*cond, *true_val, *false_val],

            Instruction::Phi { incoming, .. } => incoming.iter().map(|(_, val)| *val).collect(),

            Instruction::FMA { a, b, c, .. } => vec![*a, *b, *c],

            Instruction::WarpShuffle { src, lane, .. } => vec![*src, *lane],

            Instruction::WarpVote { pred, .. } => vec![*pred],

            Instruction::WarpReduce { src, .. } => vec![*src],

            Instruction::Call { args, .. } => args.clone(),

            // Instructions that don't use values
            Instruction::Const { .. }
            | Instruction::Branch { .. }
            | Instruction::Barrier { .. }
            | Instruction::MemFence { .. }
            | Instruction::ThreadIdx { .. }
            | Instruction::BlockIdx { .. }
            | Instruction::BlockDim { .. }
            | Instruction::GridDim { .. }
            | Instruction::WarpId { .. }
            | Instruction::LaneId { .. }
            | Instruction::SharedAlloc { .. } => Vec::new(),
        }
    }

    /// Generate suggestions for reducing register pressure
    fn generate_suggestions(&self, max_pressure: u32) -> Vec<RegisterSuggestion> {
        let mut suggestions = Vec::new();

        // Always suggest reducing block size when spilling
        suggestions.push(RegisterSuggestion::ReduceBlockSize {
            reason: "Lower occupancy allows more registers per thread".into(),
        });

        // Suggest splitting if pressure is very high
        if max_pressure > self.max_registers * 2 {
            suggestions.push(RegisterSuggestion::SplitKernel {
                reason: "Kernel too complex, consider splitting into phases".into(),
            });
        }

        // Suggest using shared memory for intermediates
        suggestions.push(RegisterSuggestion::UseSharedMemory {
            reason: "Move intermediate results to shared memory".into(),
        });

        // Suggest reordering if pressure is moderate
        if max_pressure <= self.max_registers + 32 {
            suggestions.push(RegisterSuggestion::ReorderInstructions {
                reason: "Reordering may reduce live ranges".into(),
            });
        }

        // Suggest compiler flags
        suggestions.push(RegisterSuggestion::CompilerFlags {
            flags: vec![
                "--maxrregcount=128".into(),
                "-Xptxas -O3".into(),
                "-lineinfo".into(),
            ],
        });

        suggestions
    }

    /// Estimate register usage from instruction count (rough heuristic)
    pub fn estimate_from_instruction_count(&self, count: usize) -> u32 {
        // Very rough estimate: sqrt(instructions) * 4
        ((count as f64).sqrt() * 4.0) as u32
    }
}

impl Default for RegisterPressureAnalyzer {
    fn default() -> Self {
        Self::new(255)
    }
}

/// Helper to build a simple function for testing
#[cfg(test)]
mod test_helpers {
    use super::*;
    use crate::ir::function::{BasicBlock, FunctionKind, GpuFunction};
    use crate::ir::inst::{BinOp, BlockId, Constant};
    use crate::ir::types::ScalarType;
    use indexmap::IndexMap;

    pub fn make_test_function(name: &str, instructions: Vec<Instruction>) -> GpuFunction {
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

    pub fn add_inst(dst: u32, lhs: u32, rhs: u32) -> Instruction {
        Instruction::BinOp {
            dst: ValueId(dst),
            op: BinOp::Add,
            lhs: ValueId(lhs),
            rhs: ValueId(rhs),
            ty: ScalarType::I32,
        }
    }

    pub fn const_inst(dst: u32, val: i32) -> Instruction {
        Instruction::Const {
            dst: ValueId(dst),
            value: Constant::I32(val),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::*;
    use super::*;

    #[test]
    fn test_simple_pressure() {
        let func = make_test_function(
            "test",
            vec![
                const_inst(0, 1),
                const_inst(1, 2),
                add_inst(2, 0, 1),
                const_inst(3, 3),
                add_inst(4, 2, 3),
            ],
        );

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        assert!(!report.spilling);
        assert!(report.max_pressure < 10);
    }

    #[test]
    fn test_live_range_computation() {
        let func = make_test_function(
            "test",
            vec![
                const_inst(0, 1),  // Defined at 0
                const_inst(1, 2),  // Defined at 1
                add_inst(2, 0, 1), // Uses 0, 1 at 2
            ],
        );

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        // Value 0 should be live from 0 to 3 (used at 2)
        let range0 = report.live_ranges.get(&ValueId(0)).unwrap();
        assert_eq!(range0.start, 0);
        assert_eq!(range0.end, 3);

        // Value 1 should be live from 1 to 3 (used at 2)
        let range1 = report.live_ranges.get(&ValueId(1)).unwrap();
        assert_eq!(range1.start, 1);
        assert_eq!(range1.end, 3);
    }

    #[test]
    fn test_high_pressure_detection() {
        // Create a function with many simultaneously live values
        let mut instructions = Vec::new();

        // Create 300 constants (exceeds 255 limit)
        for i in 0..300 {
            instructions.push(const_inst(i, i as i32));
        }

        // Use all of them at the end
        for i in 1..300 {
            instructions.push(add_inst(300 + i, i - 1, i));
        }

        let func = make_test_function("high_pressure", instructions);

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        // Should detect spilling
        assert!(report.spilling);
        assert!(report.max_pressure > 255);
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_hotspot_detection() {
        let mut instructions = Vec::new();

        // Create 200 constants that are all used
        for i in 0..200 {
            instructions.push(const_inst(i, i as i32));
        }
        // Add uses of all values at the end
        for i in 1..200 {
            instructions.push(add_inst(200 + i, i - 1, i));
        }

        let func = make_test_function("hotspots", instructions);

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        // Should detect high pressure
        assert!(report.max_pressure > 100);
    }

    #[test]
    fn test_suggestions() {
        let mut instructions = Vec::new();

        // Create high pressure scenario - many live values
        for i in 0..300 {
            instructions.push(const_inst(i, i as i32));
        }
        // Use all of them at the end to keep them live
        for i in 1..300 {
            instructions.push(add_inst(300 + i, i - 1, i));
        }

        let func = make_test_function("suggestions", instructions);

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        // If spilling, should have suggestions
        if report.spilling {
            assert!(!report.suggestions.is_empty());
        }
    }

    #[test]
    fn test_live_range_overlap() {
        let range1 = LiveRange {
            start: 0,
            end: 5,
            uses: vec![1, 3],
        };
        let range2 = LiveRange {
            start: 3,
            end: 8,
            uses: vec![4, 6],
        };
        let range3 = LiveRange {
            start: 6,
            end: 10,
            uses: vec![7, 9],
        };

        assert!(range1.overlaps(&range2)); // 0-5 overlaps 3-8
        assert!(range2.overlaps(&range3)); // 3-8 overlaps 6-10
        assert!(!range1.overlaps(&range3)); // 0-5 doesn't overlap 6-10
    }

    #[test]
    fn test_longest_live_ranges() {
        let func = make_test_function(
            "test",
            vec![
                const_inst(0, 1),  // Long lived
                const_inst(1, 2),  // Short lived
                add_inst(2, 0, 1), // Uses 1 (ends its range)
                const_inst(3, 3),
                add_inst(4, 0, 3), // Uses 0 (ends its range)
            ],
        );

        let analyzer = RegisterPressureAnalyzer::new(255);
        let report = analyzer.analyze(&func);

        let longest = report.longest_live_ranges(1);
        assert!(!longest.is_empty());

        // Value 0 should have the longest range (0 to 5)
        let (id, _) = longest[0];
        assert_eq!(id, ValueId(0));
    }
}
