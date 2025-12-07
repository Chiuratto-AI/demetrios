//! Formal specification of CUDA/PTX memory model
//!
//! Based on:
//! - "A Formal Analysis of the NVIDIA PTX Memory Consistency Model" (Alglave et al.)
//! - NVIDIA PTX ISA documentation
//! - NVIDIA CUDA C++ Programming Guide
//!
//! Key concepts:
//! - Scoped synchronization (CTA, GPU, SYS)
//! - Acquire-release semantics
//! - Causality and visibility

use std::collections::{HashMap, HashSet};

/// Memory location (abstract address)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Location(pub u64);

/// Thread identifier (scope-qualified)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadId {
    /// System-level ID
    pub system: u32,
    /// GPU ID within system
    pub gpu: u32,
    /// CTA (block) ID within GPU
    pub cta: u32,
    /// Thread ID within CTA
    pub thread: u32,
}

impl ThreadId {
    /// Create a new thread ID
    pub fn new(system: u32, gpu: u32, cta: u32, thread: u32) -> Self {
        Self {
            system,
            gpu,
            cta,
            thread,
        }
    }

    /// Check if two threads are in same scope
    pub fn same_scope(&self, other: &Self, scope: Scope) -> bool {
        match scope {
            Scope::Cta => {
                self.system == other.system && self.gpu == other.gpu && self.cta == other.cta
            }
            Scope::Gpu => self.system == other.system && self.gpu == other.gpu,
            Scope::Sys => true,
        }
    }
}

/// Synchronization scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// Cooperative Thread Array (block)
    Cta,
    /// GPU (device)
    Gpu,
    /// System (all devices + CPU)
    Sys,
}

impl Scope {
    /// Check if self is at least as wide as other
    pub fn includes(&self, other: &Scope) -> bool {
        *self >= *other
    }
}

/// Memory operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Load
    Read,
    /// Store
    Write,
    /// Read-Modify-Write (atomic)
    Rmw,
    /// Fence
    Fence,
}

/// Memory ordering tag
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderingTag {
    /// No ordering
    Relaxed,
    /// Acquire
    Acquire,
    /// Release
    Release,
    /// Acquire + Release
    AcqRel,
    /// Sequential consistency
    SeqCst,
}

/// A memory event in the execution
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Unique event ID
    pub id: u64,
    /// Thread that issued this event
    pub thread: ThreadId,
    /// Type of operation
    pub op_type: OpType,
    /// Memory location
    pub location: Location,
    /// Value read or written
    pub value: u64,
    /// Ordering
    pub ordering: OrderingTag,
    /// Scope
    pub scope: Scope,
    /// Program order (within thread)
    pub po_index: u64,
}

/// Relations between events
#[derive(Debug, Clone, Default)]
pub struct ExecutionRelations {
    /// Program order: (e1, e2) means e1 precedes e2 in same thread
    pub po: HashSet<(u64, u64)>,
    /// Reads-from: (w, r) means read r reads from write w
    pub rf: HashMap<u64, u64>, // read -> write
    /// Coherence order: total order on writes to same location
    pub co: HashMap<Location, Vec<u64>>,
    /// From-reads: (r, w) means w is co-after the write r reads from
    pub fr: HashSet<(u64, u64)>,
    /// Synchronizes-with (release-acquire pairs)
    pub sw: HashSet<(u64, u64)>,
    /// Happens-before
    pub hb: HashSet<(u64, u64)>,
}

impl ExecutionRelations {
    /// Create new empty relations
    pub fn new() -> Self {
        Self::default()
    }

    /// Add program order edge
    pub fn add_po(&mut self, e1: u64, e2: u64) {
        self.po.insert((e1, e2));
    }

    /// Add reads-from edge
    pub fn add_rf(&mut self, write: u64, read: u64) {
        self.rf.insert(read, write);
    }

    /// Add to coherence order
    pub fn add_co(&mut self, loc: Location, write: u64) {
        self.co.entry(loc).or_default().push(write);
    }
}

/// Memory model checker
pub struct MemoryModelChecker {
    events: HashMap<u64, MemoryEvent>,
    relations: ExecutionRelations,
}

impl MemoryModelChecker {
    /// Create new checker
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
            relations: ExecutionRelations::new(),
        }
    }

    /// Add event to execution
    pub fn add_event(&mut self, event: MemoryEvent) {
        self.events.insert(event.id, event);
    }

    /// Get relations (mutable)
    pub fn relations_mut(&mut self) -> &mut ExecutionRelations {
        &mut self.relations
    }

    /// Compute synchronizes-with relation
    ///
    /// sw(w, r) iff:
    /// - w is a release write (or stronger)
    /// - r is an acquire read (or stronger)
    /// - rf(w, r) (r reads from w)
    /// - w.scope includes r.scope (or vice versa)
    pub fn compute_sw(&mut self) {
        for (&read_id, &write_id) in &self.relations.rf {
            let read = self.events.get(&read_id).unwrap();
            let write = self.events.get(&write_id).unwrap();

            // Check ordering requirements
            let write_releases = write.ordering >= OrderingTag::Release;
            let read_acquires = read.ordering >= OrderingTag::Acquire;

            // Check scope compatibility
            let scopes_compatible =
                write.scope.includes(&read.scope) || read.scope.includes(&write.scope);

            // Check threads are in the relevant scope
            let min_scope = write.scope.min(read.scope);
            let in_scope = write.thread.same_scope(&read.thread, min_scope);

            if write_releases && read_acquires && scopes_compatible && in_scope {
                self.relations.sw.insert((write_id, read_id));
            }
        }
    }

    /// Compute happens-before relation
    ///
    /// hb = (po ∪ sw)⁺ (transitive closure)
    pub fn compute_hb(&mut self) {
        // Start with po and sw
        let mut hb: HashSet<(u64, u64)> = self.relations.po.clone();
        hb.extend(&self.relations.sw);

        // Transitive closure via fixed-point iteration
        loop {
            let mut added = false;
            let current: Vec<_> = hb.iter().copied().collect();

            for &(a, b) in &current {
                for &(c, d) in &current {
                    if b == c && !hb.contains(&(a, d)) {
                        hb.insert((a, d));
                        added = true;
                    }
                }
            }

            if !added {
                break;
            }
        }

        self.relations.hb = hb;
    }

    /// Compute from-reads relation
    ///
    /// fr(r, w) iff:
    /// - rf(w', r) for some w'
    /// - co(w', w) (w is coherence-after w')
    pub fn compute_fr(&mut self) {
        for (&read_id, &rf_write_id) in &self.relations.rf {
            let read = self.events.get(&read_id).unwrap();

            if let Some(co_order) = self.relations.co.get(&read.location) {
                let rf_pos = co_order.iter().position(|&w| w == rf_write_id);

                if let Some(pos) = rf_pos {
                    // All writes after rf_write in co order
                    for &later_write in &co_order[pos + 1..] {
                        self.relations.fr.insert((read_id, later_write));
                    }
                }
            }
        }
    }

    /// Check coherence axiom: co ∪ rf ∪ fr ∪ po-loc is acyclic
    pub fn check_coherence(&self) -> bool {
        // Build combined relation
        let mut edges: HashSet<(u64, u64)> = HashSet::new();

        // Add co edges
        for writes in self.relations.co.values() {
            for i in 0..writes.len() {
                for j in i + 1..writes.len() {
                    edges.insert((writes[i], writes[j]));
                }
            }
        }

        // Add rf edges
        for (&read, &write) in &self.relations.rf {
            edges.insert((write, read));
        }

        // Add fr edges
        edges.extend(&self.relations.fr);

        // Add po-loc edges (po restricted to same location)
        for &(e1, e2) in &self.relations.po {
            let ev1 = self.events.get(&e1).unwrap();
            let ev2 = self.events.get(&e2).unwrap();
            if ev1.location == ev2.location {
                edges.insert((e1, e2));
            }
        }

        // Check acyclicity via DFS
        self.is_acyclic(&edges)
    }

    /// Check happens-before consistency
    ///
    /// For every read r reading from write w:
    /// - NOT hb(r, w) (can't read from future)
    /// - For every other write w' to same location:
    ///   NOT (hb(w, w') AND hb(w', r)) (can't skip intermediate write)
    pub fn check_hb_consistency(&self) -> bool {
        for (&read_id, &write_id) in &self.relations.rf {
            // Check r doesn't happen-before w
            if self.relations.hb.contains(&(read_id, write_id)) {
                return false; // Read from future!
            }

            let read = self.events.get(&read_id).unwrap();

            // Check no intervening write
            if let Some(writes) = self.relations.co.get(&read.location) {
                for &other_write in writes {
                    if other_write != write_id {
                        let w_before_w2 = self.relations.hb.contains(&(write_id, other_write));
                        let w2_before_r = self.relations.hb.contains(&(other_write, read_id));

                        if w_before_w2 && w2_before_r {
                            return false; // Skipped intermediate write!
                        }
                    }
                }
            }
        }

        true
    }

    /// Check if relation is acyclic
    fn is_acyclic(&self, edges: &HashSet<(u64, u64)>) -> bool {
        // Build adjacency list
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nodes: HashSet<u64> = HashSet::new();

        for &(from, to) in edges {
            adj.entry(from).or_default().push(to);
            nodes.insert(from);
            nodes.insert(to);
        }

        // DFS for cycle detection
        let mut visited: HashSet<u64> = HashSet::new();
        let mut rec_stack: HashSet<u64> = HashSet::new();

        fn dfs(
            node: u64,
            adj: &HashMap<u64, Vec<u64>>,
            visited: &mut HashSet<u64>,
            rec_stack: &mut HashSet<u64>,
        ) -> bool {
            visited.insert(node);
            rec_stack.insert(node);

            if let Some(neighbors) = adj.get(&node) {
                for &next in neighbors {
                    if !visited.contains(&next) {
                        if dfs(next, adj, visited, rec_stack) {
                            return true; // Cycle found
                        }
                    } else if rec_stack.contains(&next) {
                        return true; // Back edge = cycle
                    }
                }
            }

            rec_stack.remove(&node);
            false
        }

        for &node in &nodes {
            if !visited.contains(&node) && dfs(node, &adj, &mut visited, &mut rec_stack) {
                return false; // Has cycle
            }
        }

        true
    }

    /// Validate entire execution
    pub fn validate(&mut self) -> ValidationResult {
        self.compute_sw();
        self.compute_hb();
        self.compute_fr();

        let coherence_ok = self.check_coherence();
        let hb_ok = self.check_hb_consistency();

        ValidationResult {
            valid: coherence_ok && hb_ok,
            coherence_satisfied: coherence_ok,
            hb_consistent: hb_ok,
            data_races: self.find_data_races(),
        }
    }

    /// Find data races
    ///
    /// A data race exists between events e1, e2 if:
    /// - At least one is a write
    /// - They access the same location
    /// - They are not ordered by hb
    /// - They are from different threads
    pub fn find_data_races(&self) -> Vec<DataRace> {
        let mut races = Vec::new();

        let events: Vec<_> = self.events.values().collect();

        for i in 0..events.len() {
            for j in i + 1..events.len() {
                let e1 = events[i];
                let e2 = events[j];

                // Same location?
                if e1.location != e2.location {
                    continue;
                }

                // Different threads?
                if e1.thread == e2.thread {
                    continue;
                }

                // At least one write?
                if e1.op_type == OpType::Read && e2.op_type == OpType::Read {
                    continue;
                }

                // Not ordered by hb?
                if self.relations.hb.contains(&(e1.id, e2.id))
                    || self.relations.hb.contains(&(e2.id, e1.id))
                {
                    continue;
                }

                // Data race!
                races.push(DataRace {
                    event1: e1.id,
                    event2: e2.id,
                    location: e1.location,
                });
            }
        }

        races
    }
}

impl Default for MemoryModelChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Overall validity
    pub valid: bool,
    /// Coherence axiom satisfied
    pub coherence_satisfied: bool,
    /// Happens-before consistency
    pub hb_consistent: bool,
    /// Data races found
    pub data_races: Vec<DataRace>,
}

/// Data race information
#[derive(Debug)]
pub struct DataRace {
    /// First event ID
    pub event1: u64,
    /// Second event ID
    pub event2: u64,
    /// Memory location
    pub location: Location,
}

// ============================================================================
// LITMUS TESTS
// ============================================================================

/// Litmus test for memory model verification
pub struct LitmusTest {
    /// Test name
    pub name: String,
    /// Threads in the test
    pub threads: Vec<LitmusThread>,
    /// Initial state
    pub initial_state: HashMap<Location, u64>,
    /// Expected outcome
    pub allowed_outcomes: Vec<HashMap<String, u64>>,
    /// Forbidden outcomes
    pub forbidden_outcomes: Vec<HashMap<String, u64>>,
}

/// Thread in a litmus test
pub struct LitmusThread {
    /// Thread ID
    pub id: ThreadId,
    /// Instructions
    pub instructions: Vec<LitmusInstruction>,
}

/// Instruction in a litmus test
#[derive(Debug, Clone)]
pub enum LitmusInstruction {
    /// Store value to location
    Store {
        location: Location,
        value: u64,
        ordering: OrderingTag,
        scope: Scope,
    },
    /// Load from location into register
    Load {
        register: String,
        location: Location,
        ordering: OrderingTag,
        scope: Scope,
    },
    /// Fence
    Fence { ordering: OrderingTag, scope: Scope },
}

impl LitmusTest {
    /// Classic Message Passing test
    ///
    /// Thread 0: x = 1; fence; y = 1
    /// Thread 1: while (y == 0); r1 = x
    ///
    /// With proper fences, r1 must be 1
    pub fn message_passing() -> Self {
        let x = Location(0);
        let y = Location(8);

        let t0 = ThreadId::new(0, 0, 0, 0);
        let t1 = ThreadId::new(0, 0, 0, 1);

        Self {
            name: "MP".to_string(),
            threads: vec![
                LitmusThread {
                    id: t0,
                    instructions: vec![
                        LitmusInstruction::Store {
                            location: x,
                            value: 1,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Fence {
                            ordering: OrderingTag::Release,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Store {
                            location: y,
                            value: 1,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
                LitmusThread {
                    id: t1,
                    instructions: vec![
                        LitmusInstruction::Load {
                            register: "r0".to_string(),
                            location: y,
                            ordering: OrderingTag::Acquire,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Load {
                            register: "r1".to_string(),
                            location: x,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
            ],
            initial_state: [(x, 0), (y, 0)].into_iter().collect(),
            allowed_outcomes: vec![
                // r0=0, r1=0 (didn't see y yet)
                [("r0".to_string(), 0), ("r1".to_string(), 0)]
                    .into_iter()
                    .collect(),
                // r0=0, r1=1 (saw x but not y - shouldn't happen with proper ordering)
                // r0=1, r1=1 (saw both)
                [("r0".to_string(), 1), ("r1".to_string(), 1)]
                    .into_iter()
                    .collect(),
            ],
            forbidden_outcomes: vec![
                // r0=1, r1=0 is FORBIDDEN (if y==1, x must be 1)
                [("r0".to_string(), 1), ("r1".to_string(), 0)]
                    .into_iter()
                    .collect(),
            ],
        }
    }

    /// Store Buffering test (can produce r1 = 0, r2 = 0 on relaxed)
    pub fn store_buffering() -> Self {
        let x = Location(0);
        let y = Location(8);

        let t0 = ThreadId::new(0, 0, 0, 0);
        let t1 = ThreadId::new(0, 0, 0, 1);

        Self {
            name: "SB".to_string(),
            threads: vec![
                LitmusThread {
                    id: t0,
                    instructions: vec![
                        LitmusInstruction::Store {
                            location: x,
                            value: 1,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Load {
                            register: "r0".to_string(),
                            location: y,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
                LitmusThread {
                    id: t1,
                    instructions: vec![
                        LitmusInstruction::Store {
                            location: y,
                            value: 1,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Load {
                            register: "r1".to_string(),
                            location: x,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
            ],
            initial_state: [(x, 0), (y, 0)].into_iter().collect(),
            allowed_outcomes: vec![
                // All outcomes allowed on relaxed
                [("r0".to_string(), 0), ("r1".to_string(), 0)]
                    .into_iter()
                    .collect(),
                [("r0".to_string(), 0), ("r1".to_string(), 1)]
                    .into_iter()
                    .collect(),
                [("r0".to_string(), 1), ("r1".to_string(), 0)]
                    .into_iter()
                    .collect(),
                [("r0".to_string(), 1), ("r1".to_string(), 1)]
                    .into_iter()
                    .collect(),
            ],
            forbidden_outcomes: vec![],
        }
    }

    /// Independent Reads of Independent Writes (IRIW)
    pub fn iriw() -> Self {
        let x = Location(0);
        let y = Location(8);

        let t0 = ThreadId::new(0, 0, 0, 0);
        let t1 = ThreadId::new(0, 0, 0, 1);
        let t2 = ThreadId::new(0, 0, 0, 2);
        let t3 = ThreadId::new(0, 0, 0, 3);

        Self {
            name: "IRIW".to_string(),
            threads: vec![
                LitmusThread {
                    id: t0,
                    instructions: vec![LitmusInstruction::Store {
                        location: x,
                        value: 1,
                        ordering: OrderingTag::Relaxed,
                        scope: Scope::Gpu,
                    }],
                },
                LitmusThread {
                    id: t1,
                    instructions: vec![LitmusInstruction::Store {
                        location: y,
                        value: 1,
                        ordering: OrderingTag::Relaxed,
                        scope: Scope::Gpu,
                    }],
                },
                LitmusThread {
                    id: t2,
                    instructions: vec![
                        LitmusInstruction::Load {
                            register: "r0".to_string(),
                            location: x,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Fence {
                            ordering: OrderingTag::AcqRel,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Load {
                            register: "r1".to_string(),
                            location: y,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
                LitmusThread {
                    id: t3,
                    instructions: vec![
                        LitmusInstruction::Load {
                            register: "r2".to_string(),
                            location: y,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Fence {
                            ordering: OrderingTag::AcqRel,
                            scope: Scope::Gpu,
                        },
                        LitmusInstruction::Load {
                            register: "r3".to_string(),
                            location: x,
                            ordering: OrderingTag::Relaxed,
                            scope: Scope::Gpu,
                        },
                    ],
                },
            ],
            initial_state: [(x, 0), (y, 0)].into_iter().collect(),
            allowed_outcomes: vec![], // Complex - depends on memory model
            forbidden_outcomes: vec![
                // r0=1, r1=0, r2=1, r3=0 is typically forbidden
                // (different threads see writes in different orders)
            ],
        }
    }

    /// Check if an outcome is allowed
    pub fn is_outcome_allowed(&self, outcome: &HashMap<String, u64>) -> bool {
        // Check not forbidden
        for forbidden in &self.forbidden_outcomes {
            if forbidden == outcome {
                return false;
            }
        }

        // Check in allowed (if allowed list is non-empty)
        if self.allowed_outcomes.is_empty() {
            true
        } else {
            self.allowed_outcomes.contains(outcome)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_ordering() {
        assert!(Scope::Sys.includes(&Scope::Gpu));
        assert!(Scope::Gpu.includes(&Scope::Cta));
        assert!(!Scope::Cta.includes(&Scope::Gpu));
    }

    #[test]
    fn test_thread_same_scope() {
        let t1 = ThreadId::new(0, 0, 0, 0);
        let t2 = ThreadId::new(0, 0, 0, 1);
        let t3 = ThreadId::new(0, 0, 1, 0);
        let t4 = ThreadId::new(0, 1, 0, 0);

        assert!(t1.same_scope(&t2, Scope::Cta));
        assert!(!t1.same_scope(&t3, Scope::Cta));
        assert!(t1.same_scope(&t3, Scope::Gpu));
        assert!(!t1.same_scope(&t4, Scope::Gpu));
        assert!(t1.same_scope(&t4, Scope::Sys));
    }

    #[test]
    fn test_basic_execution_valid() {
        let mut checker = MemoryModelChecker::new();

        let t0 = ThreadId::new(0, 0, 0, 0);
        let loc = Location(0);

        // Simple sequential execution: write then read
        checker.add_event(MemoryEvent {
            id: 0,
            thread: t0,
            op_type: OpType::Write,
            location: loc,
            value: 42,
            ordering: OrderingTag::Relaxed,
            scope: Scope::Gpu,
            po_index: 0,
        });

        checker.add_event(MemoryEvent {
            id: 1,
            thread: t0,
            op_type: OpType::Read,
            location: loc,
            value: 42,
            ordering: OrderingTag::Relaxed,
            scope: Scope::Gpu,
            po_index: 1,
        });

        checker.relations_mut().add_po(0, 1);
        checker.relations_mut().add_rf(0, 1);
        checker.relations_mut().add_co(loc, 0);

        let result = checker.validate();
        assert!(result.valid);
        assert!(result.data_races.is_empty());
    }

    #[test]
    fn test_message_passing_litmus() {
        let mp = LitmusTest::message_passing();
        assert_eq!(mp.name, "MP");
        assert_eq!(mp.threads.len(), 2);

        // Forbidden outcome
        let forbidden: HashMap<String, u64> = [("r0".to_string(), 1), ("r1".to_string(), 0)].into();
        assert!(!mp.is_outcome_allowed(&forbidden));

        // Allowed outcome
        let allowed: HashMap<String, u64> = [("r0".to_string(), 1), ("r1".to_string(), 1)].into();
        assert!(mp.is_outcome_allowed(&allowed));
    }

    #[test]
    fn test_store_buffering_litmus() {
        let sb = LitmusTest::store_buffering();
        assert_eq!(sb.name, "SB");

        // All outcomes allowed on relaxed
        let outcome: HashMap<String, u64> = [("r0".to_string(), 0), ("r1".to_string(), 0)].into();
        assert!(sb.is_outcome_allowed(&outcome));
    }

    #[test]
    fn test_data_race_detection() {
        let mut checker = MemoryModelChecker::new();

        let t0 = ThreadId::new(0, 0, 0, 0);
        let t1 = ThreadId::new(0, 0, 0, 1);
        let loc = Location(0);

        // Two concurrent writes without synchronization
        checker.add_event(MemoryEvent {
            id: 0,
            thread: t0,
            op_type: OpType::Write,
            location: loc,
            value: 1,
            ordering: OrderingTag::Relaxed,
            scope: Scope::Gpu,
            po_index: 0,
        });

        checker.add_event(MemoryEvent {
            id: 1,
            thread: t1,
            op_type: OpType::Write,
            location: loc,
            value: 2,
            ordering: OrderingTag::Relaxed,
            scope: Scope::Gpu,
            po_index: 0,
        });

        let result = checker.validate();
        assert!(!result.data_races.is_empty());
    }
}
