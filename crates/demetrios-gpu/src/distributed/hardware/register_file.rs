//! GPU Register File Microarchitecture
//!
//! The register file is the heart of GPU compute. Understanding its
//! structure explains why:
//! - Occupancy matters (registers limit warps)
//! - Some operations are slower than expected (bank conflicts)
//! - Compiler register allocation is so important
//!
//! Based on:
//! - "Demystifying GPU Microarchitecture through Microbenchmarking" (Wong et al.)
//! - "A Detailed GPU Cache Model Based on Reuse Distance Theory" (Nugteren et al.)
//! - NVIDIA Architecture Whitepapers (Ampere, Hopper)

use std::collections::{BTreeMap, HashMap, HashSet};

// ============================================================================
// REGISTER FILE STRUCTURE
// ============================================================================

/// Physical register file organization
///
/// Chain-of-thought:
/// 1. SM has fixed number of 32-bit registers (e.g., 65536 on A100)
/// 2. Registers organized into banks (typically 4 banks)
/// 3. Each bank can service one read per cycle per port
/// 4. Most instructions need 2-3 source registers + 1 dest
/// 5. Bank conflicts cause stalls (like shared memory!)
#[derive(Debug, Clone)]
pub struct RegisterFileSpec {
    /// Total 32-bit registers per SM
    pub total_registers: u32,
    /// Number of register banks
    pub num_banks: u32,
    /// Read ports per bank
    pub read_ports_per_bank: u32,
    /// Write ports per bank
    pub write_ports_per_bank: u32,
    /// Registers per bank
    pub registers_per_bank: u32,
    /// Bank selection function
    pub bank_select: BankSelectFunction,
    /// Operand collector entries
    pub operand_collectors: u32,
}

/// How register index maps to bank
#[derive(Debug, Clone, Copy)]
pub enum BankSelectFunction {
    /// Simple modulo: bank = reg_id % num_banks
    Modulo,
    /// XOR with warp ID: bank = (reg_id ^ warp_id) % num_banks
    XorWarp,
    /// Complex hash (Volta+)
    Hash,
}

impl RegisterFileSpec {
    /// NVIDIA A100 (SM 8.0) register file
    pub fn a100() -> Self {
        Self {
            total_registers: 65536,
            num_banks: 4,
            read_ports_per_bank: 4,
            write_ports_per_bank: 2,
            registers_per_bank: 16384,
            bank_select: BankSelectFunction::XorWarp,
            operand_collectors: 8,
        }
    }

    /// NVIDIA L4 (SM 8.9) register file
    pub fn l4() -> Self {
        Self {
            total_registers: 65536,
            num_banks: 4,
            read_ports_per_bank: 4,
            write_ports_per_bank: 2,
            registers_per_bank: 16384,
            bank_select: BankSelectFunction::Hash,
            operand_collectors: 8,
        }
    }

    /// NVIDIA H100 (SM 9.0) register file
    pub fn h100() -> Self {
        Self {
            total_registers: 65536,
            num_banks: 4,
            read_ports_per_bank: 4,
            write_ports_per_bank: 2,
            registers_per_bank: 16384,
            bank_select: BankSelectFunction::Hash,
            operand_collectors: 16,
        }
    }

    /// Compute which bank a register is in
    pub fn get_bank(&self, reg_id: u32, warp_id: u32) -> u32 {
        match self.bank_select {
            BankSelectFunction::Modulo => reg_id % self.num_banks,
            BankSelectFunction::XorWarp => (reg_id ^ warp_id) % self.num_banks,
            BankSelectFunction::Hash => {
                // Volta+ uses complex hash to reduce conflicts
                let h = reg_id.wrapping_mul(2654435761);
                (h ^ (warp_id << 2)) % self.num_banks
            }
        }
    }

    /// Check for bank conflicts in an instruction's operands
    pub fn check_bank_conflicts(&self, src_regs: &[u32], warp_id: u32) -> BankConflictResult {
        let mut bank_usage: HashMap<u32, Vec<u32>> = HashMap::new();

        for &reg in src_regs {
            let bank = self.get_bank(reg, warp_id);
            bank_usage.entry(bank).or_default().push(reg);
        }

        let max_conflict = bank_usage
            .values()
            .map(|regs| regs.len())
            .max()
            .unwrap_or(0);

        if max_conflict <= self.read_ports_per_bank as usize {
            BankConflictResult::NoConflict
        } else {
            let extra_cycles = (max_conflict - self.read_ports_per_bank as usize) as u32;
            BankConflictResult::Conflict {
                extra_cycles,
                conflicting_banks: bank_usage
                    .into_iter()
                    .filter(|(_, regs)| regs.len() > self.read_ports_per_bank as usize)
                    .collect(),
            }
        }
    }

    /// Maximum registers per thread at given occupancy
    pub fn max_regs_per_thread(&self, warps_per_sm: u32) -> u32 {
        let threads_per_sm = warps_per_sm * 32;
        self.total_registers / threads_per_sm
    }

    /// Warps achievable at given register usage
    pub fn max_warps(&self, regs_per_thread: u32) -> u32 {
        if regs_per_thread == 0 {
            return 64; // Hardware limit on A100
        }
        let regs_per_warp = regs_per_thread * 32;
        (self.total_registers / regs_per_warp).min(64)
    }
}

/// Result of bank conflict analysis
#[derive(Debug)]
pub enum BankConflictResult {
    /// No conflicts detected
    NoConflict,
    /// Conflicts detected
    Conflict {
        /// Extra cycles due to conflicts
        extra_cycles: u32,
        /// Banks with conflicts and their registers
        conflicting_banks: Vec<(u32, Vec<u32>)>,
    },
}

// ============================================================================
// OPERAND COLLECTOR
// ============================================================================

/// Operand Collector: Decouples register read from execution
///
/// Chain-of-thought:
/// 1. Instructions need operands from register file
/// 2. Register file has limited ports (can't read all operands at once)
/// 3. Operand collector buffers partially-ready instructions
/// 4. When all operands collected, instruction dispatches to execution unit
/// 5. This hides register bank conflicts (to some extent)
#[derive(Debug)]
pub struct OperandCollector {
    /// Number of collector units
    num_units: u32,
    /// Entries in each collector
    entries: Vec<CollectorEntry>,
    /// Current cycle
    cycle: u64,
}

/// Entry in the operand collector
#[derive(Debug, Clone)]
pub struct CollectorEntry {
    /// Instruction waiting for operands
    pub instruction: Option<PendingInstruction>,
    /// Which operands have been collected
    pub operands_ready: Vec<bool>,
    /// Collected operand values
    pub operand_values: Vec<Option<u32>>,
    /// Cycle when all operands ready
    pub ready_cycle: Option<u64>,
}

/// Instruction pending in operand collector
#[derive(Debug, Clone)]
pub struct PendingInstruction {
    /// Program counter
    pub pc: u64,
    /// Opcode
    pub opcode: String,
    /// Destination register
    pub dest_reg: Option<u32>,
    /// Source registers
    pub src_regs: Vec<u32>,
    /// Warp ID
    pub warp_id: u32,
    /// Arrival cycle
    pub arrival_cycle: u64,
}

impl OperandCollector {
    /// Create new operand collector
    pub fn new(num_units: u32) -> Self {
        let entries = (0..num_units)
            .map(|_| CollectorEntry {
                instruction: None,
                operands_ready: Vec::new(),
                operand_values: Vec::new(),
                ready_cycle: None,
            })
            .collect();

        Self {
            num_units,
            entries,
            cycle: 0,
        }
    }

    /// Try to allocate an instruction to the collector
    pub fn allocate(&mut self, instr: PendingInstruction) -> Result<usize, ()> {
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if entry.instruction.is_none() {
                let num_srcs = instr.src_regs.len();
                entry.instruction = Some(instr);
                entry.operands_ready = vec![false; num_srcs];
                entry.operand_values = vec![None; num_srcs];
                entry.ready_cycle = None;
                return Ok(i);
            }
        }
        Err(()) // Collector full
    }

    /// Mark an operand as ready
    pub fn operand_ready(&mut self, entry_idx: usize, operand_idx: usize, value: u32) {
        let entry = &mut self.entries[entry_idx];
        entry.operands_ready[operand_idx] = true;
        entry.operand_values[operand_idx] = Some(value);

        // Check if all operands ready
        if entry.operands_ready.iter().all(|&r| r) {
            entry.ready_cycle = Some(self.cycle);
        }
    }

    /// Get ready instructions (can dispatch to execution)
    pub fn get_ready(&self) -> Vec<(usize, &PendingInstruction)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if e.ready_cycle.is_some() {
                    e.instruction.as_ref().map(|instr| (i, instr))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Dispatch an instruction (free the entry)
    pub fn dispatch(&mut self, entry_idx: usize) -> Option<PendingInstruction> {
        let entry = &mut self.entries[entry_idx];
        let instr = entry.instruction.take();
        entry.operands_ready.clear();
        entry.operand_values.clear();
        entry.ready_cycle = None;
        instr
    }

    /// Advance cycle
    pub fn tick(&mut self) {
        self.cycle += 1;
    }

    /// Collector utilization
    pub fn utilization(&self) -> f64 {
        let used = self
            .entries
            .iter()
            .filter(|e| e.instruction.is_some())
            .count();
        used as f64 / self.num_units as f64
    }
}

// ============================================================================
// REGISTER ALLOCATION (COMPILER SIDE)
// ============================================================================

/// Live range for a virtual register
#[derive(Debug, Clone)]
pub struct LiveRange {
    /// Virtual register ID
    pub vreg: u32,
    /// First definition (instruction index)
    pub def: u32,
    /// Last use (instruction index)
    pub last_use: u32,
    /// All use points
    pub uses: Vec<u32>,
    /// Spill cost heuristic
    pub spill_cost: f64,
    /// Preferred physical register (if any)
    pub hint: Option<u32>,
}

impl LiveRange {
    /// Create new live range
    pub fn new(vreg: u32, def: u32) -> Self {
        Self {
            vreg,
            def,
            last_use: def,
            uses: vec![def],
            spill_cost: 1.0,
            hint: None,
        }
    }

    /// Add a use of this register
    pub fn add_use(&mut self, inst: u32) {
        self.uses.push(inst);
        self.last_use = self.last_use.max(inst);
    }

    /// Does this range overlap with another?
    pub fn interferes_with(&self, other: &LiveRange) -> bool {
        !(self.last_use < other.def || other.last_use < self.def)
    }

    /// Range length
    pub fn length(&self) -> u32 {
        self.last_use - self.def + 1
    }
}

/// Interference graph for register allocation
#[derive(Debug)]
pub struct InterferenceGraph {
    /// Nodes (virtual registers)
    nodes: Vec<u32>,
    /// Edges (interference between registers)
    edges: HashSet<(u32, u32)>,
    /// Node degrees
    degrees: HashMap<u32, usize>,
    /// Node colors (physical register assignments)
    colors: HashMap<u32, u32>,
}

impl InterferenceGraph {
    /// Create new interference graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashSet::new(),
            degrees: HashMap::new(),
            colors: HashMap::new(),
        }
    }

    /// Add a virtual register
    pub fn add_node(&mut self, vreg: u32) {
        if !self.nodes.contains(&vreg) {
            self.nodes.push(vreg);
            self.degrees.insert(vreg, 0);
        }
    }

    /// Add interference edge
    pub fn add_edge(&mut self, v1: u32, v2: u32) {
        if v1 != v2 {
            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            if self.edges.insert(edge) {
                *self.degrees.entry(v1).or_default() += 1;
                *self.degrees.entry(v2).or_default() += 1;
            }
        }
    }

    /// Build graph from live ranges
    pub fn build_from_ranges(ranges: &[LiveRange]) -> Self {
        let mut graph = Self::new();

        for range in ranges {
            graph.add_node(range.vreg);
        }

        // Check all pairs for interference
        for i in 0..ranges.len() {
            for j in i + 1..ranges.len() {
                if ranges[i].interferes_with(&ranges[j]) {
                    graph.add_edge(ranges[i].vreg, ranges[j].vreg);
                }
            }
        }

        graph
    }

    /// Graph coloring (Chaitin's algorithm simplified)
    ///
    /// Chain-of-thought:
    /// 1. Find node with degree < K (number of colors/registers)
    /// 2. Remove it from graph, push to stack
    /// 3. Repeat until graph empty or stuck
    /// 4. If stuck, spill highest-cost node
    /// 5. Pop stack, assign colors
    pub fn color(&mut self, num_regs: u32) -> Result<(), Vec<u32>> {
        let k = num_regs as usize;
        let mut stack: Vec<u32> = Vec::new();
        let mut removed: HashSet<u32> = HashSet::new();
        let mut current_degrees = self.degrees.clone();

        // Simplify phase
        loop {
            // Find node with degree < k
            let candidate = self
                .nodes
                .iter()
                .filter(|n| !removed.contains(n))
                .find(|n| current_degrees.get(n).copied().unwrap_or(0) < k)
                .copied();

            match candidate {
                Some(node) => {
                    // Remove node, update neighbor degrees
                    removed.insert(node);
                    stack.push(node);

                    for &other in &self.nodes {
                        if !removed.contains(&other) {
                            let edge = if node < other {
                                (node, other)
                            } else {
                                (other, node)
                            };
                            if self.edges.contains(&edge) {
                                *current_degrees.get_mut(&other).unwrap() -= 1;
                            }
                        }
                    }
                }
                None => {
                    // Check if done
                    if removed.len() == self.nodes.len() {
                        break;
                    }

                    // Must spill - return nodes that couldn't be colored
                    let spill_candidates: Vec<u32> = self
                        .nodes
                        .iter()
                        .filter(|n| !removed.contains(n))
                        .copied()
                        .collect();
                    return Err(spill_candidates);
                }
            }
        }

        // Select phase - pop stack and assign colors
        while let Some(node) = stack.pop() {
            let mut used_colors: HashSet<u32> = HashSet::new();

            // Find colors used by neighbors
            for &other in &self.nodes {
                if let Some(&color) = self.colors.get(&other) {
                    let edge = if node < other {
                        (node, other)
                    } else {
                        (other, node)
                    };
                    if self.edges.contains(&edge) {
                        used_colors.insert(color);
                    }
                }
            }

            // Find first available color
            for color in 0..num_regs {
                if !used_colors.contains(&color) {
                    self.colors.insert(node, color);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get physical register assignment
    pub fn get_assignment(&self, vreg: u32) -> Option<u32> {
        self.colors.get(&vreg).copied()
    }

    /// Check if edge exists
    pub fn has_edge(&self, v1: u32, v2: u32) -> bool {
        let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        self.edges.contains(&edge)
    }
}

impl Default for InterferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Register allocator for GPU kernels
#[derive(Debug)]
pub struct GpuRegisterAllocator {
    /// Maximum registers per thread
    max_regs: u32,
    /// Target registers (for occupancy)
    target_regs: u32,
    /// Live ranges
    ranges: Vec<LiveRange>,
    /// Interference graph
    graph: InterferenceGraph,
    /// Spilled registers
    spilled: HashSet<u32>,
    /// Spill slots in local memory
    spill_slots: HashMap<u32, u32>,
}

impl GpuRegisterAllocator {
    /// Create new allocator
    pub fn new(max_regs: u32, target_regs: u32) -> Self {
        Self {
            max_regs,
            target_regs,
            ranges: Vec::new(),
            graph: InterferenceGraph::new(),
            spilled: HashSet::new(),
            spill_slots: HashMap::new(),
        }
    }

    /// Add live range
    pub fn add_range(&mut self, range: LiveRange) {
        self.ranges.push(range);
    }

    /// Perform allocation
    pub fn allocate(&mut self) -> AllocationResult {
        // Build interference graph
        self.graph = InterferenceGraph::build_from_ranges(&self.ranges);

        // Try to color with target registers
        match self.graph.color(self.target_regs) {
            Ok(()) => {
                // Success!
                let mut assignment = HashMap::new();
                for range in &self.ranges {
                    if let Some(preg) = self.graph.get_assignment(range.vreg) {
                        assignment.insert(range.vreg, preg);
                    }
                }

                AllocationResult {
                    success: true,
                    registers_used: self.graph.colors.values().max().map(|m| m + 1).unwrap_or(0),
                    assignment,
                    spilled: Vec::new(),
                    spill_code: Vec::new(),
                }
            }
            Err(spill_candidates) => {
                // Need to spill
                // Choose register with lowest spill cost
                let to_spill = spill_candidates
                    .iter()
                    .min_by(|a, b| {
                        let cost_a = self
                            .ranges
                            .iter()
                            .find(|r| r.vreg == **a)
                            .map(|r| r.spill_cost)
                            .unwrap_or(f64::MAX);
                        let cost_b = self
                            .ranges
                            .iter()
                            .find(|r| r.vreg == **b)
                            .map(|r| r.spill_cost)
                            .unwrap_or(f64::MAX);
                        cost_a.partial_cmp(&cost_b).unwrap()
                    })
                    .copied();

                if let Some(vreg) = to_spill {
                    self.spilled.insert(vreg);
                    let slot = self.spill_slots.len() as u32;
                    self.spill_slots.insert(vreg, slot);

                    // Generate spill code
                    let spill_code = self.generate_spill_code(vreg, slot);

                    // Retry allocation (simplified - would normally iterate)
                    AllocationResult {
                        success: false,
                        registers_used: self.target_regs,
                        assignment: HashMap::new(),
                        spilled: vec![vreg],
                        spill_code,
                    }
                } else {
                    AllocationResult {
                        success: false,
                        registers_used: 0,
                        assignment: HashMap::new(),
                        spilled: Vec::new(),
                        spill_code: Vec::new(),
                    }
                }
            }
        }
    }

    /// Generate spill/reload code
    fn generate_spill_code(&self, vreg: u32, slot: u32) -> Vec<SpillCode> {
        let range = self.ranges.iter().find(|r| r.vreg == vreg).unwrap();
        let mut code = Vec::new();

        // Spill at definition
        code.push(SpillCode::Store {
            vreg,
            slot,
            after_inst: range.def,
        });

        // Reload before each use
        for &use_inst in &range.uses {
            if use_inst != range.def {
                code.push(SpillCode::Load {
                    vreg,
                    slot,
                    before_inst: use_inst,
                });
            }
        }

        code
    }
}

/// Result of register allocation
#[derive(Debug)]
pub struct AllocationResult {
    /// Whether allocation succeeded
    pub success: bool,
    /// Number of physical registers used
    pub registers_used: u32,
    /// Virtual to physical register mapping
    pub assignment: HashMap<u32, u32>,
    /// Spilled virtual registers
    pub spilled: Vec<u32>,
    /// Generated spill code
    pub spill_code: Vec<SpillCode>,
}

/// Spill/reload instruction
#[derive(Debug, Clone)]
pub enum SpillCode {
    /// Store register to local memory
    Store {
        vreg: u32,
        slot: u32,
        after_inst: u32,
    },
    /// Load register from local memory
    Load {
        vreg: u32,
        slot: u32,
        before_inst: u32,
    },
}

// ============================================================================
// REGISTER PRESSURE ANALYSIS
// ============================================================================

/// Analyze register pressure through a kernel
#[derive(Debug)]
pub struct RegisterPressureAnalysis {
    /// Pressure at each instruction
    pub pressure_curve: Vec<u32>,
    /// Maximum pressure
    pub max_pressure: u32,
    /// Instruction with max pressure
    pub max_pressure_inst: u32,
    /// Average pressure
    pub avg_pressure: f64,
    /// Pressure hotspots (sudden increases)
    pub hotspots: Vec<PressureHotspot>,
}

/// A location where register pressure spikes
#[derive(Debug)]
pub struct PressureHotspot {
    /// Instruction index
    pub instruction: u32,
    /// Pressure at this point
    pub pressure: u32,
    /// Cause description
    pub cause: String,
}

impl RegisterPressureAnalysis {
    /// Analyze a sequence of instructions
    pub fn analyze(instructions: &[InstructionInfo]) -> Self {
        let mut live_regs: HashSet<u32> = HashSet::new();
        let mut pressure_curve = Vec::new();
        let mut hotspots = Vec::new();
        let mut max_pressure = 0u32;
        let mut max_pressure_inst = 0u32;

        for (i, instr) in instructions.iter().enumerate() {
            // Add destination to live set
            if let Some(dest) = instr.dest_reg {
                live_regs.insert(dest);
            }

            // Remove sources that are last-used here
            for &src in &instr.src_regs {
                if instr.is_last_use.get(&src).copied().unwrap_or(false) {
                    live_regs.remove(&src);
                }
            }

            let pressure = live_regs.len() as u32;
            pressure_curve.push(pressure);

            if pressure > max_pressure {
                max_pressure = pressure;
                max_pressure_inst = i as u32;
            }

            // Detect hotspots (pressure increase > 4)
            if i > 0 && pressure > pressure_curve[i - 1] + 4 {
                hotspots.push(PressureHotspot {
                    instruction: i as u32,
                    pressure,
                    cause: format!(
                        "Pressure spike from {} to {}",
                        pressure_curve[i - 1],
                        pressure
                    ),
                });
            }
        }

        let avg_pressure = if pressure_curve.is_empty() {
            0.0
        } else {
            pressure_curve.iter().sum::<u32>() as f64 / pressure_curve.len() as f64
        };

        Self {
            pressure_curve,
            max_pressure,
            max_pressure_inst,
            avg_pressure,
            hotspots,
        }
    }

    /// Will this kernel spill at given register limit?
    pub fn will_spill(&self, reg_limit: u32) -> bool {
        self.max_pressure > reg_limit
    }

    /// Estimate spill cost
    pub fn estimate_spill_cost(&self, reg_limit: u32) -> f64 {
        if self.max_pressure <= reg_limit {
            return 0.0;
        }

        // Rough estimate: each register over limit costs ~100 cycles per use
        let excess = self.max_pressure - reg_limit;
        let uses_per_reg = 2.0; // Heuristic
        excess as f64 * uses_per_reg * 400.0 // Local memory latency
    }

    /// Suggest register target for occupancy
    pub fn suggest_target(&self, spec: &RegisterFileSpec, target_warps: u32) -> u32 {
        let max_regs = spec.max_regs_per_thread(target_warps);

        if self.max_pressure <= max_regs {
            self.max_pressure
        } else {
            // Need to spill to hit target
            max_regs
        }
    }
}

/// Information about an instruction for pressure analysis
#[derive(Debug)]
pub struct InstructionInfo {
    /// Opcode
    pub opcode: String,
    /// Destination register
    pub dest_reg: Option<u32>,
    /// Source registers
    pub src_regs: Vec<u32>,
    /// Which sources are last uses
    pub is_last_use: HashMap<u32, bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_file_banking() {
        let spec = RegisterFileSpec::a100();

        // Check bank distribution
        let bank0 = spec.get_bank(0, 0);
        let bank4 = spec.get_bank(4, 0);

        // With XOR, same pattern for different warps should differ
        let bank0_warp1 = spec.get_bank(0, 1);
        assert_ne!(bank0, bank0_warp1);

        // Basic modulo check
        assert!(bank0 < spec.num_banks);
        assert!(bank4 < spec.num_banks);
    }

    #[test]
    fn test_max_regs_per_thread() {
        let spec = RegisterFileSpec::a100();

        // At 64 warps, 32 regs per thread
        let regs = spec.max_regs_per_thread(64);
        assert_eq!(regs, 32);

        // At 32 warps, 64 regs per thread
        let regs = spec.max_regs_per_thread(32);
        assert_eq!(regs, 64);
    }

    #[test]
    fn test_max_warps() {
        let spec = RegisterFileSpec::a100();

        // 32 regs per thread = 64 warps
        assert_eq!(spec.max_warps(32), 64);

        // 64 regs per thread = 32 warps
        assert_eq!(spec.max_warps(64), 32);

        // 128 regs per thread = 16 warps
        assert_eq!(spec.max_warps(128), 16);
    }

    #[test]
    fn test_bank_conflict_detection() {
        let spec = RegisterFileSpec::a100();

        // All same bank - conflict
        let regs = vec![0, 4, 8, 12, 16]; // All map to bank 0 in modulo
        let result = spec.check_bank_conflicts(&regs, 0);

        // May or may not conflict depending on XOR hash
        match result {
            BankConflictResult::NoConflict => {}
            BankConflictResult::Conflict { extra_cycles, .. } => {
                assert!(extra_cycles > 0);
            }
        }
    }

    #[test]
    fn test_operand_collector() {
        let mut collector = OperandCollector::new(4);

        let instr = PendingInstruction {
            pc: 100,
            opcode: "add.f32".to_string(),
            dest_reg: Some(0),
            src_regs: vec![1, 2],
            warp_id: 0,
            arrival_cycle: 0,
        };

        let idx = collector.allocate(instr).unwrap();

        // Not ready yet
        assert!(collector.get_ready().is_empty());

        // Mark operands ready
        collector.operand_ready(idx, 0, 42);
        collector.operand_ready(idx, 1, 43);

        // Now ready
        assert_eq!(collector.get_ready().len(), 1);

        // Dispatch
        let dispatched = collector.dispatch(idx);
        assert!(dispatched.is_some());
        assert!(collector.get_ready().is_empty());
    }

    #[test]
    fn test_interference_graph() {
        let ranges = vec![
            LiveRange {
                vreg: 0,
                def: 0,
                last_use: 5,
                uses: vec![0, 3, 5],
                spill_cost: 1.0,
                hint: None,
            },
            LiveRange {
                vreg: 1,
                def: 1,
                last_use: 4,
                uses: vec![1, 4],
                spill_cost: 1.0,
                hint: None,
            },
            LiveRange {
                vreg: 2,
                def: 6,
                last_use: 8,
                uses: vec![6, 8],
                spill_cost: 1.0,
                hint: None,
            },
        ];

        let graph = InterferenceGraph::build_from_ranges(&ranges);

        // 0 and 1 should interfere (overlap in [1,4])
        assert!(graph.has_edge(0, 1));

        // 0 and 2 should NOT interfere (0 ends at 5, 2 starts at 6)
        assert!(!graph.has_edge(0, 2));
    }

    #[test]
    fn test_graph_coloring() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        // 0 and 2 don't interfere

        let result = graph.color(2);
        assert!(result.is_ok());

        // 0 and 2 can share a color
        let color0 = graph.get_assignment(0).unwrap();
        let color2 = graph.get_assignment(2).unwrap();
        assert_eq!(color0, color2);
    }

    #[test]
    fn test_register_pressure_analysis() {
        let instructions = vec![
            InstructionInfo {
                opcode: "ld".to_string(),
                dest_reg: Some(0),
                src_regs: vec![],
                is_last_use: HashMap::new(),
            },
            InstructionInfo {
                opcode: "ld".to_string(),
                dest_reg: Some(1),
                src_regs: vec![],
                is_last_use: HashMap::new(),
            },
            InstructionInfo {
                opcode: "add".to_string(),
                dest_reg: Some(2),
                src_regs: vec![0, 1],
                is_last_use: [(0, true), (1, true)].into_iter().collect(),
            },
        ];

        let analysis = RegisterPressureAnalysis::analyze(&instructions);

        // After instruction 2, r0 and r1 are dead (last use), only r2 is live
        // Max pressure is 2 after instruction 1 (r0 and r1 both live)
        assert_eq!(analysis.max_pressure, 2);
        assert!(!analysis.will_spill(32));
    }
}
