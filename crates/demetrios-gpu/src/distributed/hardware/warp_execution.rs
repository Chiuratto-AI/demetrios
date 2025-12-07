//! Warp execution model and divergence analysis
//!
//! Understanding warp execution is critical for GPU performance.
//! Key concepts:
//! - SIMT (Single Instruction, Multiple Threads)
//! - Warp = 32 threads executing in lockstep
//! - Divergence = different threads take different paths
//! - Reconvergence = threads rejoin after divergent region

use std::collections::{HashMap, VecDeque};

/// Warp size (fixed at 32 for all NVIDIA GPUs since G80)
pub const WARP_SIZE: usize = 32;

/// Thread mask (which threads are active)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadMask(pub u32);

impl ThreadMask {
    /// All threads active
    pub const ALL: Self = Self(0xFFFFFFFF);
    /// No threads active
    pub const NONE: Self = Self(0);

    /// Create new mask
    pub fn new(mask: u32) -> Self {
        Self(mask)
    }

    /// Check if lane is active
    pub fn is_active(&self, lane: usize) -> bool {
        (self.0 >> lane) & 1 == 1
    }

    /// Count active threads
    pub fn active_count(&self) -> u32 {
        self.0.count_ones()
    }

    /// Union of two masks
    pub fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Intersection of two masks
    pub fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Complement (invert all bits)
    pub fn complement(self) -> Self {
        Self(!self.0)
    }

    /// First active lane
    pub fn first_active(&self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }

    /// Last active lane
    pub fn last_active(&self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(31 - self.0.leading_zeros() as usize)
        }
    }

    /// Create mask for lanes 0..n
    pub fn first_n(n: usize) -> Self {
        if n >= 32 {
            Self::ALL
        } else if n == 0 {
            Self::NONE
        } else {
            Self((1u32 << n) - 1)
        }
    }

    /// Create mask for single lane
    pub fn single(lane: usize) -> Self {
        if lane >= 32 {
            Self::NONE
        } else {
            Self(1u32 << lane)
        }
    }
}

impl std::fmt::Binary for ThreadMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032b}", self.0)
    }
}

/// Control flow instruction types
#[derive(Debug, Clone)]
pub enum ControlFlowInst {
    /// Unconditional branch
    Branch { target: usize },
    /// Conditional branch
    BranchCond {
        predicate: String,
        target_true: usize,
        target_false: usize,
    },
    /// Function call
    Call { target: String },
    /// Return
    Return,
    /// Barrier (all threads must reach)
    Barrier { id: u32 },
    /// Exit (terminate thread)
    Exit,
}

/// Divergence result from a branch
#[derive(Debug)]
pub enum DivergenceResult {
    /// All threads take same path
    Uniform { taken: bool },
    /// Threads diverged
    Diverged {
        branch_mask: ThreadMask,
        fallthrough_mask: ThreadMask,
    },
}

/// Divergence state for a warp
#[derive(Debug, Clone)]
pub struct DivergenceStack {
    /// Stack of (reconvergence_pc, active_mask) pairs
    stack: Vec<(usize, ThreadMask)>,
    /// Current active mask
    pub active_mask: ThreadMask,
    /// Current PC
    pub pc: usize,
}

impl DivergenceStack {
    /// Create new divergence stack
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            active_mask: ThreadMask::ALL,
            pc: 0,
        }
    }

    /// Push divergence point
    pub fn push(&mut self, reconvergence_pc: usize, mask: ThreadMask) {
        self.stack.push((reconvergence_pc, mask));
    }

    /// Pop at reconvergence point
    pub fn pop(&mut self) -> Option<(usize, ThreadMask)> {
        self.stack.pop()
    }

    /// Check if at reconvergence point
    pub fn at_reconvergence(&self, pc: usize) -> bool {
        self.stack
            .last()
            .map(|(rpc, _)| *rpc == pc)
            .unwrap_or(false)
    }

    /// Handle conditional branch
    pub fn handle_branch(
        &mut self,
        condition_mask: ThreadMask,
        target_true: usize,
        target_false: usize,
        reconvergence_pc: usize,
    ) -> DivergenceResult {
        let active = self.active_mask;
        let take_branch = active.intersection(condition_mask);
        let fall_through = active.intersection(condition_mask.complement());

        if take_branch.0 != 0 && fall_through.0 != 0 {
            // DIVERGENCE!
            // Push reconvergence point with fall-through threads
            self.push(reconvergence_pc, fall_through);

            // Execute branch path first
            self.active_mask = take_branch;
            self.pc = target_true;

            DivergenceResult::Diverged {
                branch_mask: take_branch,
                fallthrough_mask: fall_through,
            }
        } else if take_branch.0 != 0 {
            // All take branch
            self.pc = target_true;
            DivergenceResult::Uniform { taken: true }
        } else {
            // All fall through
            self.pc = target_false;
            DivergenceResult::Uniform { taken: false }
        }
    }

    /// Handle reconvergence
    pub fn handle_reconvergence(&mut self) {
        while self.at_reconvergence(self.pc) {
            if let Some((_, mask)) = self.pop() {
                self.active_mask = self.active_mask.union(mask);
            }
        }
    }

    /// Get current divergence depth
    pub fn divergence_depth(&self) -> usize {
        self.stack.len()
    }
}

impl Default for DivergenceStack {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WARP SCHEDULING
// ============================================================================

/// Warp scheduler types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerType {
    /// Round-robin among ready warps
    RoundRobin,
    /// Greedy: pick warp with most ready threads
    Greedy,
    /// Two-level: group warps, schedule groups
    TwoLevel,
    /// Loose round-robin (Fermi+)
    LooseRoundRobin,
    /// GTO (Greedy Then Oldest)
    Gto,
}

/// Stall reasons for warps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StallReason {
    /// Waiting for operand (register dependency)
    Scoreboard,
    /// Waiting for memory
    MemoryDependency,
    /// Waiting for barrier
    Barrier,
    /// No instruction buffer space
    InstructionBuffer,
    /// Waiting for texture unit
    Texture,
    /// Waiting for math unit (high latency)
    MathPipeline,
    /// Other
    Other,
}

/// Warp state for scheduling
#[derive(Debug, Clone)]
pub struct WarpState {
    /// Warp ID
    pub id: u32,
    /// Active thread mask
    pub active_mask: ThreadMask,
    /// Current instruction PC
    pub pc: usize,
    /// Is warp ready to issue?
    pub ready: bool,
    /// Stall reason if not ready
    pub stall_reason: Option<StallReason>,
    /// Instructions executed
    pub instructions_executed: u64,
    /// Cycles stalled
    pub cycles_stalled: u64,
    /// Divergence stack
    pub divergence: DivergenceStack,
}

impl WarpState {
    /// Create new warp
    pub fn new(id: u32) -> Self {
        Self {
            id,
            active_mask: ThreadMask::ALL,
            pc: 0,
            ready: true,
            stall_reason: None,
            instructions_executed: 0,
            cycles_stalled: 0,
            divergence: DivergenceStack::new(),
        }
    }

    /// Mark warp as stalled
    pub fn stall(&mut self, reason: StallReason) {
        self.ready = false;
        self.stall_reason = Some(reason);
    }

    /// Mark warp as ready
    pub fn unstall(&mut self) {
        self.ready = true;
        self.stall_reason = None;
    }

    /// Execute one instruction
    pub fn execute(&mut self) {
        self.instructions_executed += 1;
    }

    /// SIMD efficiency (active threads / total)
    pub fn simd_efficiency(&self) -> f64 {
        self.active_mask.active_count() as f64 / WARP_SIZE as f64
    }
}

/// Warp scheduler simulation
pub struct WarpScheduler {
    /// Scheduler type
    pub scheduler_type: SchedulerType,
    /// All warps
    warps: Vec<WarpState>,
    /// Current cycle
    cycle: u64,
    /// Warps per scheduler
    warps_per_scheduler: u32,
    /// Number of schedulers per SM
    num_schedulers: u32,
    /// Last scheduled warp per scheduler (for round-robin)
    last_scheduled: Vec<u32>,
}

impl WarpScheduler {
    /// Create new scheduler
    pub fn new(scheduler_type: SchedulerType, num_warps: u32, num_schedulers: u32) -> Self {
        let warps = (0..num_warps).map(WarpState::new).collect();

        let warps_per_scheduler = (num_warps + num_schedulers - 1) / num_schedulers;

        Self {
            scheduler_type,
            warps,
            cycle: 0,
            warps_per_scheduler,
            num_schedulers,
            last_scheduled: vec![0; num_schedulers as usize],
        }
    }

    /// Get warp by ID
    pub fn warp(&self, id: u32) -> Option<&WarpState> {
        self.warps.get(id as usize)
    }

    /// Get mutable warp by ID
    pub fn warp_mut(&mut self, id: u32) -> Option<&mut WarpState> {
        self.warps.get_mut(id as usize)
    }

    /// Select warp to execute for a scheduler
    pub fn select_warp(&mut self, scheduler_id: u32) -> Option<u32> {
        let start = scheduler_id * self.warps_per_scheduler;
        let end = ((scheduler_id + 1) * self.warps_per_scheduler).min(self.warps.len() as u32);

        match self.scheduler_type {
            SchedulerType::RoundRobin => self.select_round_robin(scheduler_id, start, end),
            SchedulerType::Greedy => self.select_greedy(start, end),
            SchedulerType::Gto => self.select_gto(scheduler_id, start, end),
            _ => self.select_round_robin(scheduler_id, start, end),
        }
    }

    fn select_round_robin(&mut self, scheduler_id: u32, start: u32, end: u32) -> Option<u32> {
        let last = self.last_scheduled[scheduler_id as usize];

        // Try from last+1 to end
        for i in (last + 1)..end {
            if self.warps[i as usize].ready {
                self.last_scheduled[scheduler_id as usize] = i;
                return Some(i);
            }
        }

        // Try from start to last
        for i in start..=last {
            if i < self.warps.len() as u32 && self.warps[i as usize].ready {
                self.last_scheduled[scheduler_id as usize] = i;
                return Some(i);
            }
        }

        None
    }

    fn select_greedy(&self, start: u32, end: u32) -> Option<u32> {
        let mut best: Option<(u32, u32)> = None;

        for i in start..end {
            if i < self.warps.len() as u32 && self.warps[i as usize].ready {
                let count = self.warps[i as usize].active_mask.active_count();
                if best.map(|(_, c)| count > c).unwrap_or(true) {
                    best = Some((i, count));
                }
            }
        }

        best.map(|(id, _)| id)
    }

    fn select_gto(&mut self, scheduler_id: u32, start: u32, end: u32) -> Option<u32> {
        // GTO: continue with same warp if possible, else round-robin
        let last = self.last_scheduled[scheduler_id as usize];

        if last >= start
            && last < end
            && self
                .warps
                .get(last as usize)
                .map(|w| w.ready)
                .unwrap_or(false)
        {
            return Some(last);
        }

        self.select_round_robin(scheduler_id, start, end)
    }

    /// Advance one cycle
    pub fn step(&mut self) {
        self.cycle += 1;

        // Count stalled warps
        for warp in &mut self.warps {
            if !warp.ready {
                warp.cycles_stalled += 1;
            }
        }
    }

    /// Get current cycle
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// Get occupancy (active warps / max warps)
    pub fn occupancy(&self) -> f64 {
        let ready_count = self.warps.iter().filter(|w| w.ready).count();
        ready_count as f64 / self.warps.len() as f64
    }

    /// Get average active threads per warp
    pub fn average_active_threads(&self) -> f64 {
        let total: u32 = self
            .warps
            .iter()
            .map(|w| w.active_mask.active_count())
            .sum();
        total as f64 / self.warps.len() as f64
    }

    /// SIMD efficiency (active threads / total threads)
    pub fn simd_efficiency(&self) -> f64 {
        self.average_active_threads() / WARP_SIZE as f64
    }

    /// Get stall breakdown
    pub fn stall_breakdown(&self) -> HashMap<StallReason, usize> {
        let mut breakdown = HashMap::new();

        for warp in &self.warps {
            if let Some(reason) = warp.stall_reason {
                *breakdown.entry(reason).or_insert(0) += 1;
            }
        }

        breakdown
    }
}

// ============================================================================
// DIVERGENCE ANALYSIS
// ============================================================================

/// Basic block in control flow graph
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID
    pub id: usize,
    /// Instructions (simplified as strings)
    pub instructions: Vec<String>,
    /// Terminator instruction
    pub terminator: Option<ControlFlowInst>,
}

/// Divergence analyzer for control flow graphs
pub struct DivergenceAnalyzer {
    /// Basic blocks
    blocks: Vec<BasicBlock>,
    /// Edges between blocks
    edges: Vec<(usize, usize)>,
    /// Post-dominator tree
    post_dominators: HashMap<usize, usize>,
    /// Reverse edges for analysis
    reverse_edges: HashMap<usize, Vec<usize>>,
}

impl DivergenceAnalyzer {
    /// Create new analyzer
    pub fn new(blocks: Vec<BasicBlock>, edges: Vec<(usize, usize)>) -> Self {
        let mut analyzer = Self {
            blocks,
            edges: edges.clone(),
            post_dominators: HashMap::new(),
            reverse_edges: HashMap::new(),
        };

        // Build reverse edge map
        for &(from, to) in &edges {
            analyzer.reverse_edges.entry(to).or_default().push(from);
        }

        analyzer.compute_post_dominators();
        analyzer
    }

    /// Compute post-dominator tree
    fn compute_post_dominators(&mut self) {
        let n = self.blocks.len();
        if n == 0 {
            return;
        }

        // Assume last block is exit
        let exit = n - 1;

        // BFS from exit to find post-dominators
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        queue.push_back(exit);
        visited[exit] = true;

        while let Some(block) = queue.pop_front() {
            if let Some(preds) = self.reverse_edges.get(&block) {
                for &pred in preds {
                    if !visited[pred] {
                        visited[pred] = true;
                        self.post_dominators.insert(pred, block);
                        queue.push_back(pred);
                    }
                }
            }
        }
    }

    /// Find reconvergence point for a branch
    pub fn reconvergence_point(&self, branch_block: usize) -> Option<usize> {
        self.post_dominators.get(&branch_block).copied()
    }

    /// Analyze potential divergence
    pub fn analyze(&self) -> DivergenceReport {
        let mut divergent_branches = Vec::new();
        let mut max_divergence_depth: usize = 0;
        let mut current_depth: usize = 0;

        for block in &self.blocks {
            if let Some(ControlFlowInst::BranchCond {
                target_true,
                target_false,
                ..
            }) = &block.terminator
            {
                let reconverge = self.reconvergence_point(block.id);

                divergent_branches.push(DivergentBranch {
                    block_id: block.id,
                    target_true: *target_true,
                    target_false: *target_false,
                    reconvergence: reconverge,
                });

                current_depth += 1;
                max_divergence_depth = max_divergence_depth.max(current_depth);
            }

            // Check for reconvergence
            if self.post_dominators.values().any(|&v| v == block.id) {
                current_depth = current_depth.saturating_sub(1);
            }
        }

        DivergenceReport {
            divergent_branches,
            max_divergence_depth,
            estimated_simd_efficiency: self.estimate_simd_efficiency(),
        }
    }

    /// Estimate SIMD efficiency based on control flow
    fn estimate_simd_efficiency(&self) -> f64 {
        let num_branches = self
            .blocks
            .iter()
            .filter(|b| matches!(&b.terminator, Some(ControlFlowInst::BranchCond { .. })))
            .count();

        // Heuristic: each branch potentially halves efficiency
        1.0 / (1.0 + 0.1 * num_branches as f64)
    }

    /// Get all blocks
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }
}

/// Divergence analysis report
#[derive(Debug)]
pub struct DivergenceReport {
    /// Potentially divergent branches
    pub divergent_branches: Vec<DivergentBranch>,
    /// Maximum nesting depth of divergence
    pub max_divergence_depth: usize,
    /// Estimated SIMD efficiency
    pub estimated_simd_efficiency: f64,
}

/// Information about a divergent branch
#[derive(Debug)]
pub struct DivergentBranch {
    /// Block containing the branch
    pub block_id: usize,
    /// True branch target
    pub target_true: usize,
    /// False branch target
    pub target_false: usize,
    /// Reconvergence point (post-dominator)
    pub reconvergence: Option<usize>,
}

// ============================================================================
// WARP EXECUTION TRACE
// ============================================================================

/// Execution trace for analysis
#[derive(Debug, Clone)]
pub struct WarpTrace {
    /// Events in the trace
    events: Vec<TraceEvent>,
}

/// Trace event types
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// Instruction executed
    Execute {
        pc: usize,
        active_mask: ThreadMask,
        instruction: String,
    },
    /// Branch taken
    Branch {
        pc: usize,
        taken_mask: ThreadMask,
        target: usize,
    },
    /// Divergence occurred
    Diverge {
        pc: usize,
        branch_mask: ThreadMask,
        fallthrough_mask: ThreadMask,
    },
    /// Reconvergence occurred
    Reconverge { pc: usize, mask: ThreadMask },
    /// Barrier synchronization
    Barrier { id: u32, mask: ThreadMask },
    /// Memory access
    MemoryAccess {
        pc: usize,
        mask: ThreadMask,
        is_write: bool,
        coalesced: bool,
    },
}

impl WarpTrace {
    /// Create new trace
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add event
    pub fn add(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Get all events
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Count divergence events
    pub fn divergence_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, TraceEvent::Diverge { .. }))
            .count()
    }

    /// Calculate average SIMD efficiency from trace
    pub fn average_simd_efficiency(&self) -> f64 {
        let execute_events: Vec<_> = self
            .events
            .iter()
            .filter_map(|e| match e {
                TraceEvent::Execute { active_mask, .. } => Some(active_mask.active_count()),
                _ => None,
            })
            .collect();

        if execute_events.is_empty() {
            1.0
        } else {
            let total: u32 = execute_events.iter().sum();
            total as f64 / (execute_events.len() as f64 * WARP_SIZE as f64)
        }
    }

    /// Count coalesced memory accesses
    pub fn coalesced_access_ratio(&self) -> f64 {
        let mem_events: Vec<_> = self
            .events
            .iter()
            .filter_map(|e| match e {
                TraceEvent::MemoryAccess { coalesced, .. } => Some(*coalesced),
                _ => None,
            })
            .collect();

        if mem_events.is_empty() {
            1.0
        } else {
            let coalesced = mem_events.iter().filter(|&&c| c).count();
            coalesced as f64 / mem_events.len() as f64
        }
    }
}

impl Default for WarpTrace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_mask_basics() {
        let mask = ThreadMask::ALL;
        assert_eq!(mask.active_count(), 32);

        let mask = ThreadMask::new(0b1010);
        assert!(mask.is_active(1));
        assert!(mask.is_active(3));
        assert!(!mask.is_active(0));
        assert_eq!(mask.active_count(), 2);
    }

    #[test]
    fn test_thread_mask_operations() {
        let a = ThreadMask::new(0b1100);
        let b = ThreadMask::new(0b1010);

        assert_eq!(a.union(b).0, 0b1110);
        assert_eq!(a.intersection(b).0, 0b1000);
        assert_eq!(ThreadMask::new(0b1111).complement().0, !0b1111u32);
    }

    #[test]
    fn test_thread_mask_first_n() {
        assert_eq!(ThreadMask::first_n(4).0, 0b1111);
        assert_eq!(ThreadMask::first_n(8).0, 0xFF);
        assert_eq!(ThreadMask::first_n(0), ThreadMask::NONE);
        assert_eq!(ThreadMask::first_n(32), ThreadMask::ALL);
    }

    #[test]
    fn test_thread_mask_single() {
        assert_eq!(ThreadMask::single(0).0, 1);
        assert_eq!(ThreadMask::single(5).0, 32);
        assert_eq!(ThreadMask::single(31).0, 1 << 31);
    }

    #[test]
    fn test_divergence_stack() {
        let mut stack = DivergenceStack::new();

        // Simulate: if (tid < 16) { ... } else { ... }
        let condition = ThreadMask::new(0x0000FFFF); // Lanes 0-15 take branch

        let result = stack.handle_branch(
            condition, 100, // true target
            200, // false target
            300, // reconvergence
        );

        match result {
            DivergenceResult::Diverged {
                branch_mask,
                fallthrough_mask,
            } => {
                assert_eq!(branch_mask.active_count(), 16);
                assert_eq!(fallthrough_mask.active_count(), 16);
            }
            _ => panic!("Expected divergence"),
        }

        assert_eq!(stack.divergence_depth(), 1);
    }

    #[test]
    fn test_divergence_uniform() {
        let mut stack = DivergenceStack::new();
        stack.active_mask = ThreadMask::first_n(16); // Only first 16 active

        // Condition that all active threads satisfy
        let condition = ThreadMask::first_n(16);

        let result = stack.handle_branch(condition, 100, 200, 300);

        assert!(matches!(result, DivergenceResult::Uniform { taken: true }));
        assert_eq!(stack.divergence_depth(), 0);
    }

    #[test]
    fn test_warp_state() {
        let mut warp = WarpState::new(0);

        assert!(warp.ready);
        assert_eq!(warp.simd_efficiency(), 1.0);

        warp.stall(StallReason::MemoryDependency);
        assert!(!warp.ready);
        assert_eq!(warp.stall_reason, Some(StallReason::MemoryDependency));

        warp.unstall();
        assert!(warp.ready);
        assert!(warp.stall_reason.is_none());
    }

    #[test]
    fn test_warp_scheduler() {
        let mut scheduler = WarpScheduler::new(SchedulerType::RoundRobin, 16, 4);

        assert_eq!(scheduler.occupancy(), 1.0);
        assert_eq!(scheduler.simd_efficiency(), 1.0);

        let selected = scheduler.select_warp(0);
        assert!(selected.is_some());
    }

    #[test]
    fn test_warp_scheduler_stalls() {
        let mut scheduler = WarpScheduler::new(SchedulerType::RoundRobin, 4, 1);

        // Stall all warps
        for i in 0..4 {
            scheduler
                .warp_mut(i)
                .unwrap()
                .stall(StallReason::Scoreboard);
        }

        assert_eq!(scheduler.occupancy(), 0.0);
        assert!(scheduler.select_warp(0).is_none());

        let breakdown = scheduler.stall_breakdown();
        assert_eq!(breakdown.get(&StallReason::Scoreboard), Some(&4));
    }

    #[test]
    fn test_divergence_analyzer() {
        let blocks = vec![
            BasicBlock {
                id: 0,
                instructions: vec!["setp.lt %p, %tid, 16".into()],
                terminator: Some(ControlFlowInst::BranchCond {
                    predicate: "%p".into(),
                    target_true: 1,
                    target_false: 2,
                }),
            },
            BasicBlock {
                id: 1,
                instructions: vec!["// true path".into()],
                terminator: Some(ControlFlowInst::Branch { target: 3 }),
            },
            BasicBlock {
                id: 2,
                instructions: vec!["// false path".into()],
                terminator: Some(ControlFlowInst::Branch { target: 3 }),
            },
            BasicBlock {
                id: 3,
                instructions: vec!["// reconvergence".into()],
                terminator: Some(ControlFlowInst::Return),
            },
        ];

        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        let analyzer = DivergenceAnalyzer::new(blocks, edges);
        let report = analyzer.analyze();

        assert_eq!(report.divergent_branches.len(), 1);
        assert!(report.estimated_simd_efficiency < 1.0);
    }

    #[test]
    fn test_warp_trace() {
        let mut trace = WarpTrace::new();

        trace.add(TraceEvent::Execute {
            pc: 0,
            active_mask: ThreadMask::ALL,
            instruction: "add.f32".into(),
        });

        trace.add(TraceEvent::Diverge {
            pc: 1,
            branch_mask: ThreadMask::first_n(16),
            fallthrough_mask: ThreadMask::new(0xFFFF0000),
        });

        trace.add(TraceEvent::Execute {
            pc: 2,
            active_mask: ThreadMask::first_n(16),
            instruction: "mul.f32".into(),
        });

        assert_eq!(trace.divergence_count(), 1);
        assert!(trace.average_simd_efficiency() < 1.0);
    }

    #[test]
    fn test_memory_coalescing_trace() {
        let mut trace = WarpTrace::new();

        trace.add(TraceEvent::MemoryAccess {
            pc: 0,
            mask: ThreadMask::ALL,
            is_write: false,
            coalesced: true,
        });

        trace.add(TraceEvent::MemoryAccess {
            pc: 1,
            mask: ThreadMask::ALL,
            is_write: true,
            coalesced: false,
        });

        assert_eq!(trace.coalesced_access_ratio(), 0.5);
    }
}
