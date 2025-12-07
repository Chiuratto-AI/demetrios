//! Formal Verification of the Complete GPU Stack
//!
//! This module implements formal verification techniques including:
//! - Temporal logic properties (LTL/CTL)
//! - Model checking with explicit state enumeration
//! - SMT-LIB formula generation for solver integration
//! - Runtime verification monitors

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Temporal logic formula (LTL - Linear Temporal Logic)
#[derive(Debug, Clone, PartialEq)]
pub enum LtlFormula {
    /// Atomic proposition
    Atom(String),
    /// Negation
    Not(Box<LtlFormula>),
    /// Conjunction
    And(Box<LtlFormula>, Box<LtlFormula>),
    /// Disjunction
    Or(Box<LtlFormula>, Box<LtlFormula>),
    /// Implication
    Implies(Box<LtlFormula>, Box<LtlFormula>),
    /// Next state (X)
    Next(Box<LtlFormula>),
    /// Globally (G) - always true
    Globally(Box<LtlFormula>),
    /// Finally (F) - eventually true
    Finally(Box<LtlFormula>),
    /// Until (U)
    Until(Box<LtlFormula>, Box<LtlFormula>),
    /// Release (R)
    Release(Box<LtlFormula>, Box<LtlFormula>),
}

impl LtlFormula {
    /// Create atomic proposition
    pub fn atom(name: &str) -> Self {
        LtlFormula::Atom(name.to_string())
    }

    /// Negation
    pub fn not(self) -> Self {
        LtlFormula::Not(Box::new(self))
    }

    /// Conjunction
    pub fn and(self, other: Self) -> Self {
        LtlFormula::And(Box::new(self), Box::new(other))
    }

    /// Disjunction
    pub fn or(self, other: Self) -> Self {
        LtlFormula::Or(Box::new(self), Box::new(other))
    }

    /// Implication
    pub fn implies(self, other: Self) -> Self {
        LtlFormula::Implies(Box::new(self), Box::new(other))
    }

    /// Next state
    pub fn next(self) -> Self {
        LtlFormula::Next(Box::new(self))
    }

    /// Globally (always)
    pub fn globally(self) -> Self {
        LtlFormula::Globally(Box::new(self))
    }

    /// Finally (eventually)
    pub fn finally(self) -> Self {
        LtlFormula::Finally(Box::new(self))
    }

    /// Until
    pub fn until(self, other: Self) -> Self {
        LtlFormula::Until(Box::new(self), Box::new(other))
    }

    /// Get all atomic propositions in formula
    pub fn atoms(&self) -> HashSet<String> {
        let mut atoms = HashSet::new();
        self.collect_atoms(&mut atoms);
        atoms
    }

    fn collect_atoms(&self, atoms: &mut HashSet<String>) {
        match self {
            LtlFormula::Atom(name) => {
                atoms.insert(name.clone());
            }
            LtlFormula::Not(f)
            | LtlFormula::Next(f)
            | LtlFormula::Globally(f)
            | LtlFormula::Finally(f) => {
                f.collect_atoms(atoms);
            }
            LtlFormula::And(l, r)
            | LtlFormula::Or(l, r)
            | LtlFormula::Implies(l, r)
            | LtlFormula::Until(l, r)
            | LtlFormula::Release(l, r) => {
                l.collect_atoms(atoms);
                r.collect_atoms(atoms);
            }
        }
    }

    /// Convert to Negation Normal Form (NNF)
    pub fn to_nnf(&self) -> Self {
        match self {
            LtlFormula::Atom(a) => LtlFormula::Atom(a.clone()),
            LtlFormula::Not(inner) => match inner.as_ref() {
                LtlFormula::Atom(a) => LtlFormula::Not(Box::new(LtlFormula::Atom(a.clone()))),
                LtlFormula::Not(f) => f.to_nnf(),
                LtlFormula::And(l, r) => {
                    let nl = LtlFormula::Not(l.clone()).to_nnf();
                    let nr = LtlFormula::Not(r.clone()).to_nnf();
                    LtlFormula::Or(Box::new(nl), Box::new(nr))
                }
                LtlFormula::Or(l, r) => {
                    let nl = LtlFormula::Not(l.clone()).to_nnf();
                    let nr = LtlFormula::Not(r.clone()).to_nnf();
                    LtlFormula::And(Box::new(nl), Box::new(nr))
                }
                LtlFormula::Implies(l, r) => {
                    // ¬(l → r) = l ∧ ¬r
                    let nl = l.to_nnf();
                    let nr = LtlFormula::Not(r.clone()).to_nnf();
                    LtlFormula::And(Box::new(nl), Box::new(nr))
                }
                LtlFormula::Next(f) => {
                    LtlFormula::Next(Box::new(LtlFormula::Not(f.clone()).to_nnf()))
                }
                LtlFormula::Globally(f) => {
                    // ¬G(f) = F(¬f)
                    LtlFormula::Finally(Box::new(LtlFormula::Not(f.clone()).to_nnf()))
                }
                LtlFormula::Finally(f) => {
                    // ¬F(f) = G(¬f)
                    LtlFormula::Globally(Box::new(LtlFormula::Not(f.clone()).to_nnf()))
                }
                LtlFormula::Until(l, r) => {
                    // ¬(l U r) = (¬r R ¬l)
                    let nl = LtlFormula::Not(l.clone()).to_nnf();
                    let nr = LtlFormula::Not(r.clone()).to_nnf();
                    LtlFormula::Release(Box::new(nr), Box::new(nl))
                }
                LtlFormula::Release(l, r) => {
                    // ¬(l R r) = (¬l U ¬r)
                    let nl = LtlFormula::Not(l.clone()).to_nnf();
                    let nr = LtlFormula::Not(r.clone()).to_nnf();
                    LtlFormula::Until(Box::new(nl), Box::new(nr))
                }
            },
            LtlFormula::And(l, r) => LtlFormula::And(Box::new(l.to_nnf()), Box::new(r.to_nnf())),
            LtlFormula::Or(l, r) => LtlFormula::Or(Box::new(l.to_nnf()), Box::new(r.to_nnf())),
            LtlFormula::Implies(l, r) => {
                // l → r = ¬l ∨ r
                let nl = LtlFormula::Not(l.clone()).to_nnf();
                let nr = r.to_nnf();
                LtlFormula::Or(Box::new(nl), Box::new(nr))
            }
            LtlFormula::Next(f) => LtlFormula::Next(Box::new(f.to_nnf())),
            LtlFormula::Globally(f) => LtlFormula::Globally(Box::new(f.to_nnf())),
            LtlFormula::Finally(f) => LtlFormula::Finally(Box::new(f.to_nnf())),
            LtlFormula::Until(l, r) => {
                LtlFormula::Until(Box::new(l.to_nnf()), Box::new(r.to_nnf()))
            }
            LtlFormula::Release(l, r) => {
                LtlFormula::Release(Box::new(l.to_nnf()), Box::new(r.to_nnf()))
            }
        }
    }
}

impl fmt::Display for LtlFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LtlFormula::Atom(a) => write!(f, "{}", a),
            LtlFormula::Not(inner) => write!(f, "¬({})", inner),
            LtlFormula::And(l, r) => write!(f, "({} ∧ {})", l, r),
            LtlFormula::Or(l, r) => write!(f, "({} ∨ {})", l, r),
            LtlFormula::Implies(l, r) => write!(f, "({} → {})", l, r),
            LtlFormula::Next(inner) => write!(f, "X({})", inner),
            LtlFormula::Globally(inner) => write!(f, "G({})", inner),
            LtlFormula::Finally(inner) => write!(f, "F({})", inner),
            LtlFormula::Until(l, r) => write!(f, "({} U {})", l, r),
            LtlFormula::Release(l, r) => write!(f, "({} R {})", l, r),
        }
    }
}

/// CTL formula (Computation Tree Logic)
#[derive(Debug, Clone, PartialEq)]
pub enum CtlFormula {
    /// Atomic proposition
    Atom(String),
    /// Negation
    Not(Box<CtlFormula>),
    /// Conjunction
    And(Box<CtlFormula>, Box<CtlFormula>),
    /// Disjunction
    Or(Box<CtlFormula>, Box<CtlFormula>),
    /// EX - exists next
    ExistsNext(Box<CtlFormula>),
    /// AX - all next
    AllNext(Box<CtlFormula>),
    /// EF - exists finally
    ExistsFinally(Box<CtlFormula>),
    /// AF - all finally
    AllFinally(Box<CtlFormula>),
    /// EG - exists globally
    ExistsGlobally(Box<CtlFormula>),
    /// AG - all globally
    AllGlobally(Box<CtlFormula>),
    /// EU - exists until
    ExistsUntil(Box<CtlFormula>, Box<CtlFormula>),
    /// AU - all until
    AllUntil(Box<CtlFormula>, Box<CtlFormula>),
}

impl CtlFormula {
    /// Create atomic proposition
    pub fn atom(name: &str) -> Self {
        CtlFormula::Atom(name.to_string())
    }

    /// AG (always globally) - most common safety property
    pub fn ag(f: Self) -> Self {
        CtlFormula::AllGlobally(Box::new(f))
    }

    /// EF (exists finally) - reachability
    pub fn ef(f: Self) -> Self {
        CtlFormula::ExistsFinally(Box::new(f))
    }

    /// AF (all finally) - inevitability
    pub fn af(f: Self) -> Self {
        CtlFormula::AllFinally(Box::new(f))
    }
}

impl fmt::Display for CtlFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CtlFormula::Atom(a) => write!(f, "{}", a),
            CtlFormula::Not(inner) => write!(f, "¬({})", inner),
            CtlFormula::And(l, r) => write!(f, "({} ∧ {})", l, r),
            CtlFormula::Or(l, r) => write!(f, "({} ∨ {})", l, r),
            CtlFormula::ExistsNext(inner) => write!(f, "EX({})", inner),
            CtlFormula::AllNext(inner) => write!(f, "AX({})", inner),
            CtlFormula::ExistsFinally(inner) => write!(f, "EF({})", inner),
            CtlFormula::AllFinally(inner) => write!(f, "AF({})", inner),
            CtlFormula::ExistsGlobally(inner) => write!(f, "EG({})", inner),
            CtlFormula::AllGlobally(inner) => write!(f, "AG({})", inner),
            CtlFormula::ExistsUntil(l, r) => write!(f, "E({} U {})", l, r),
            CtlFormula::AllUntil(l, r) => write!(f, "A({} U {})", l, r),
        }
    }
}

/// State in a finite state machine
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// State identifier
    pub id: u32,
    /// Atomic propositions true in this state
    pub labels: HashSet<String>,
}

impl std::hash::Hash for State {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        // Hash labels in a deterministic order
        let mut sorted_labels: Vec<_> = self.labels.iter().collect();
        sorted_labels.sort();
        for label in sorted_labels {
            label.hash(state);
        }
    }
}

impl State {
    /// Create new state
    pub fn new(id: u32) -> Self {
        Self {
            id,
            labels: HashSet::new(),
        }
    }

    /// Add label
    pub fn with_label(mut self, label: &str) -> Self {
        self.labels.insert(label.to_string());
        self
    }

    /// Check if proposition holds
    pub fn satisfies(&self, prop: &str) -> bool {
        self.labels.contains(prop)
    }
}

/// Transition in a finite state machine
#[derive(Debug, Clone)]
pub struct Transition {
    /// Source state
    pub from: u32,
    /// Destination state
    pub to: u32,
    /// Transition label (action name)
    pub action: Option<String>,
}

/// Kripke structure for model checking
#[derive(Debug)]
pub struct KripkeStructure {
    /// States
    states: HashMap<u32, State>,
    /// Transitions (from -> list of (to, action))
    transitions: HashMap<u32, Vec<(u32, Option<String>)>>,
    /// Initial states
    initial_states: HashSet<u32>,
    /// All atomic propositions
    propositions: HashSet<String>,
}

impl KripkeStructure {
    /// Create new empty structure
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            transitions: HashMap::new(),
            initial_states: HashSet::new(),
            propositions: HashSet::new(),
        }
    }

    /// Add a state
    pub fn add_state(&mut self, state: State) {
        for label in &state.labels {
            self.propositions.insert(label.clone());
        }
        self.states.insert(state.id, state);
    }

    /// Mark state as initial
    pub fn set_initial(&mut self, state_id: u32) {
        self.initial_states.insert(state_id);
    }

    /// Add transition
    pub fn add_transition(&mut self, from: u32, to: u32, action: Option<String>) {
        self.transitions.entry(from).or_default().push((to, action));
    }

    /// Get successors of a state
    pub fn successors(&self, state_id: u32) -> Vec<u32> {
        self.transitions
            .get(&state_id)
            .map(|ts| ts.iter().map(|(to, _)| *to).collect())
            .unwrap_or_default()
    }

    /// Get state by ID
    pub fn get_state(&self, id: u32) -> Option<&State> {
        self.states.get(&id)
    }

    /// Number of states
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions
    pub fn num_transitions(&self) -> usize {
        self.transitions.values().map(|v| v.len()).sum()
    }
}

impl Default for KripkeStructure {
    fn default() -> Self {
        Self::new()
    }
}

/// Model checker for CTL formulas
#[derive(Debug)]
pub struct CtlModelChecker<'a> {
    /// The Kripke structure to check
    structure: &'a KripkeStructure,
}

impl<'a> CtlModelChecker<'a> {
    /// Create model checker for structure
    pub fn new(structure: &'a KripkeStructure) -> Self {
        Self { structure }
    }

    /// Check if formula holds in initial states
    pub fn check(&self, formula: &CtlFormula) -> bool {
        let sat_states = self.sat(formula);
        self.structure
            .initial_states
            .iter()
            .all(|s| sat_states.contains(s))
    }

    /// Compute set of states satisfying formula
    pub fn sat(&self, formula: &CtlFormula) -> HashSet<u32> {
        match formula {
            CtlFormula::Atom(prop) => self
                .structure
                .states
                .iter()
                .filter(|(_, s)| s.satisfies(prop))
                .map(|(id, _)| *id)
                .collect(),
            CtlFormula::Not(f) => {
                let sat_f = self.sat(f);
                self.structure
                    .states
                    .keys()
                    .filter(|id| !sat_f.contains(id))
                    .copied()
                    .collect()
            }
            CtlFormula::And(f, g) => {
                let sat_f = self.sat(f);
                let sat_g = self.sat(g);
                sat_f.intersection(&sat_g).copied().collect()
            }
            CtlFormula::Or(f, g) => {
                let sat_f = self.sat(f);
                let sat_g = self.sat(g);
                sat_f.union(&sat_g).copied().collect()
            }
            CtlFormula::ExistsNext(f) => self.sat_ex(f),
            CtlFormula::AllNext(f) => self.sat_ax(f),
            CtlFormula::ExistsFinally(f) => self.sat_ef(f),
            CtlFormula::AllFinally(f) => self.sat_af(f),
            CtlFormula::ExistsGlobally(f) => self.sat_eg(f),
            CtlFormula::AllGlobally(f) => self.sat_ag(f),
            CtlFormula::ExistsUntil(f, g) => self.sat_eu(f, g),
            CtlFormula::AllUntil(f, g) => self.sat_au(f, g),
        }
    }

    /// SAT(EX f) - states with some successor satisfying f
    fn sat_ex(&self, f: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        self.structure
            .states
            .keys()
            .filter(|id| {
                self.structure
                    .successors(**id)
                    .iter()
                    .any(|s| sat_f.contains(s))
            })
            .copied()
            .collect()
    }

    /// SAT(AX f) - states with all successors satisfying f
    fn sat_ax(&self, f: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        self.structure
            .states
            .keys()
            .filter(|id| {
                let succs = self.structure.successors(**id);
                !succs.is_empty() && succs.iter().all(|s| sat_f.contains(s))
            })
            .copied()
            .collect()
    }

    /// SAT(EF f) - reachable to f
    fn sat_ef(&self, f: &CtlFormula) -> HashSet<u32> {
        // EF f = true U f
        let sat_f = self.sat(f);
        let mut result = sat_f.clone();

        // Fixed-point iteration
        loop {
            let prev_size = result.len();

            for (id, _) in &self.structure.states {
                if !result.contains(id) {
                    if self
                        .structure
                        .successors(*id)
                        .iter()
                        .any(|s| result.contains(s))
                    {
                        result.insert(*id);
                    }
                }
            }

            if result.len() == prev_size {
                break;
            }
        }

        result
    }

    /// SAT(AF f) - inevitably f
    fn sat_af(&self, f: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        let mut result = sat_f.clone();

        // Fixed-point iteration
        loop {
            let prev_size = result.len();

            for (id, _) in &self.structure.states {
                if !result.contains(id) {
                    let succs = self.structure.successors(*id);
                    if !succs.is_empty() && succs.iter().all(|s| result.contains(s)) {
                        result.insert(*id);
                    }
                }
            }

            if result.len() == prev_size {
                break;
            }
        }

        result
    }

    /// SAT(EG f) - exists path where f always holds
    fn sat_eg(&self, f: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        let mut result = sat_f.clone();

        // Fixed-point iteration (greatest fixed point)
        loop {
            let prev_size = result.len();

            let to_remove: Vec<u32> = result
                .iter()
                .filter(|id| {
                    let succs = self.structure.successors(**id);
                    succs.is_empty() || !succs.iter().any(|s| result.contains(s))
                })
                .copied()
                .collect();

            for id in to_remove {
                result.remove(&id);
            }

            if result.len() == prev_size {
                break;
            }
        }

        result
    }

    /// SAT(AG f) - f holds on all paths
    fn sat_ag(&self, f: &CtlFormula) -> HashSet<u32> {
        // AG f = ¬EF(¬f)
        let not_f = CtlFormula::Not(Box::new(f.clone()));
        let ef_not_f = self.sat_ef(&not_f);

        self.structure
            .states
            .keys()
            .filter(|id| !ef_not_f.contains(id))
            .copied()
            .collect()
    }

    /// SAT(E(f U g))
    fn sat_eu(&self, f: &CtlFormula, g: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        let sat_g = self.sat(g);
        let mut result = sat_g.clone();

        loop {
            let prev_size = result.len();

            for (id, _) in &self.structure.states {
                if !result.contains(id) && sat_f.contains(id) {
                    if self
                        .structure
                        .successors(*id)
                        .iter()
                        .any(|s| result.contains(s))
                    {
                        result.insert(*id);
                    }
                }
            }

            if result.len() == prev_size {
                break;
            }
        }

        result
    }

    /// SAT(A(f U g))
    fn sat_au(&self, f: &CtlFormula, g: &CtlFormula) -> HashSet<u32> {
        let sat_f = self.sat(f);
        let sat_g = self.sat(g);
        let mut result = sat_g.clone();

        loop {
            let prev_size = result.len();

            for (id, _) in &self.structure.states {
                if !result.contains(id) && sat_f.contains(id) {
                    let succs = self.structure.successors(*id);
                    if !succs.is_empty() && succs.iter().all(|s| result.contains(s)) {
                        result.insert(*id);
                    }
                }
            }

            if result.len() == prev_size {
                break;
            }
        }

        result
    }

    /// Find counterexample path for AG formula
    pub fn counterexample_ag(&self, f: &CtlFormula) -> Option<Vec<u32>> {
        let not_f = CtlFormula::Not(Box::new(f.clone()));
        let sat_not_f = self.sat(&not_f);

        // BFS from initial states to find path to ¬f
        for &init in &self.structure.initial_states {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            let mut parent: HashMap<u32, u32> = HashMap::new();

            queue.push_back(init);
            visited.insert(init);

            while let Some(current) = queue.pop_front() {
                if sat_not_f.contains(&current) {
                    // Found counterexample, reconstruct path
                    let mut path = vec![current];
                    let mut node = current;
                    while let Some(&p) = parent.get(&node) {
                        path.push(p);
                        node = p;
                    }
                    path.reverse();
                    return Some(path);
                }

                for succ in self.structure.successors(current) {
                    if !visited.contains(&succ) {
                        visited.insert(succ);
                        parent.insert(succ, current);
                        queue.push_back(succ);
                    }
                }
            }
        }

        None
    }
}

/// SMT-LIB formula generation
#[derive(Debug)]
pub struct SmtLibGenerator {
    /// Variable declarations
    declarations: Vec<String>,
    /// Assertions
    assertions: Vec<String>,
    /// Variable counter
    var_counter: u32,
}

impl SmtLibGenerator {
    /// Create new generator
    pub fn new() -> Self {
        Self {
            declarations: Vec::new(),
            assertions: Vec::new(),
            var_counter: 0,
        }
    }

    /// Declare boolean variable
    pub fn declare_bool(&mut self, name: &str) {
        self.declarations
            .push(format!("(declare-const {} Bool)", name));
    }

    /// Declare integer variable
    pub fn declare_int(&mut self, name: &str) {
        self.declarations
            .push(format!("(declare-const {} Int)", name));
    }

    /// Declare bitvector variable
    pub fn declare_bv(&mut self, name: &str, bits: u32) {
        self.declarations
            .push(format!("(declare-const {} (_ BitVec {}))", name, bits));
    }

    /// Add assertion
    pub fn assert(&mut self, formula: &str) {
        self.assertions.push(format!("(assert {})", formula));
    }

    /// Fresh variable name
    pub fn fresh_var(&mut self, prefix: &str) -> String {
        self.var_counter += 1;
        format!("{}_{}", prefix, self.var_counter)
    }

    /// Generate SMT-LIB output
    pub fn generate(&self) -> String {
        let mut output = String::new();

        output.push_str("(set-logic QF_BV)\n");
        output.push_str("(set-option :produce-models true)\n\n");

        for decl in &self.declarations {
            output.push_str(decl);
            output.push('\n');
        }

        output.push('\n');

        for assertion in &self.assertions {
            output.push_str(assertion);
            output.push('\n');
        }

        output.push_str("\n(check-sat)\n");
        output.push_str("(get-model)\n");

        output
    }

    /// Generate memory safety constraint
    pub fn memory_bounds_constraint(&mut self, ptr: &str, size: &str, max_addr: u64) {
        self.declare_bv(ptr, 64);
        self.declare_bv(size, 64);

        // ptr + size <= max_addr and no overflow
        let constraint = format!(
            "(and (bvule (bvadd {} {}) #x{:016x}) (bvuge (bvadd {} {}) {}))",
            ptr, size, max_addr, ptr, size, ptr
        );
        self.assert(&constraint);
    }

    /// Generate race condition constraint
    pub fn race_freedom_constraint(&mut self, addr1: &str, addr2: &str, size1: &str, size2: &str) {
        // Either ranges don't overlap, or accesses are synchronized
        let no_overlap = format!(
            "(or (bvuge {} (bvadd {} {})) (bvuge {} (bvadd {} {})))",
            addr1, addr2, size2, addr2, addr1, size1
        );
        self.assert(&no_overlap);
    }
}

impl Default for SmtLibGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime verification monitor
#[derive(Debug)]
pub struct RuntimeMonitor {
    /// Monitor name
    pub name: String,
    /// Current state
    current_state: u32,
    /// State transitions (current_state, event) -> next_state
    transitions: HashMap<(u32, String), u32>,
    /// Accepting states
    accepting_states: HashSet<u32>,
    /// Error states
    error_states: HashSet<u32>,
    /// Event history
    event_history: VecDeque<String>,
    /// Maximum history size
    max_history: usize,
    /// Violation count
    pub violations: u64,
}

impl RuntimeMonitor {
    /// Create new monitor
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            current_state: 0,
            transitions: HashMap::new(),
            accepting_states: HashSet::new(),
            error_states: HashSet::new(),
            event_history: VecDeque::new(),
            max_history: 100,
            violations: 0,
        }
    }

    /// Add transition
    pub fn add_transition(&mut self, from: u32, event: &str, to: u32) {
        self.transitions.insert((from, event.to_string()), to);
    }

    /// Set accepting states
    pub fn set_accepting(&mut self, states: &[u32]) {
        self.accepting_states = states.iter().copied().collect();
    }

    /// Set error states
    pub fn set_error(&mut self, states: &[u32]) {
        self.error_states = states.iter().copied().collect();
    }

    /// Process event
    pub fn process(&mut self, event: &str) -> MonitorResult {
        self.event_history.push_back(event.to_string());
        if self.event_history.len() > self.max_history {
            self.event_history.pop_front();
        }

        if let Some(&next_state) = self
            .transitions
            .get(&(self.current_state, event.to_string()))
        {
            self.current_state = next_state;

            if self.error_states.contains(&self.current_state) {
                self.violations += 1;
                MonitorResult::Violated
            } else if self.accepting_states.contains(&self.current_state) {
                MonitorResult::Accepted
            } else {
                MonitorResult::Ongoing
            }
        } else {
            // Undefined transition - stay in current state
            MonitorResult::Ongoing
        }
    }

    /// Reset monitor
    pub fn reset(&mut self) {
        self.current_state = 0;
        self.event_history.clear();
    }

    /// Check if in accepting state
    pub fn is_accepting(&self) -> bool {
        self.accepting_states.contains(&self.current_state)
    }

    /// Check if in error state
    pub fn is_error(&self) -> bool {
        self.error_states.contains(&self.current_state)
    }

    /// Create monitor for mutex protocol
    pub fn mutex_protocol() -> Self {
        let mut monitor = Self::new("mutex_protocol");

        // States: 0=unlocked, 1=locked, 2=error
        monitor.add_transition(0, "lock", 1);
        monitor.add_transition(1, "unlock", 0);
        monitor.add_transition(0, "unlock", 2); // Error: unlock without lock
        monitor.add_transition(1, "lock", 2); // Error: double lock

        monitor.set_accepting(&[0]);
        monitor.set_error(&[2]);

        monitor
    }

    /// Create monitor for request-response protocol
    pub fn request_response() -> Self {
        let mut monitor = Self::new("request_response");

        // States: 0=idle, 1=pending, 2=error
        monitor.add_transition(0, "request", 1);
        monitor.add_transition(1, "response", 0);
        monitor.add_transition(0, "response", 2); // Error: response without request
        monitor.add_transition(1, "request", 2); // Error: double request

        monitor.set_accepting(&[0]);
        monitor.set_error(&[2]);

        monitor
    }
}

/// Result of monitor evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorResult {
    /// Property is currently being satisfied
    Ongoing,
    /// Property has been satisfied (accepting state)
    Accepted,
    /// Property has been violated (error state)
    Violated,
}

/// GPU property specifications
pub mod gpu_properties {
    use super::*;

    /// No data race between warps
    pub fn warp_race_freedom() -> CtlFormula {
        // AG(¬(warp_conflict))
        CtlFormula::ag(CtlFormula::Not(Box::new(CtlFormula::atom("warp_conflict"))))
    }

    /// Memory coalescing is always possible
    pub fn memory_coalescing() -> CtlFormula {
        // AG(memory_request → AF(coalesced))
        let request = CtlFormula::atom("memory_request");
        let coalesced = CtlFormula::af(CtlFormula::atom("coalesced"));
        let implication = CtlFormula::Or(
            Box::new(CtlFormula::Not(Box::new(request))),
            Box::new(coalesced),
        );
        CtlFormula::ag(implication)
    }

    /// Kernel eventually completes
    pub fn kernel_termination() -> CtlFormula {
        // AF(kernel_complete)
        CtlFormula::af(CtlFormula::atom("kernel_complete"))
    }

    /// No deadlock in synchronization
    pub fn deadlock_freedom() -> CtlFormula {
        // AG(EF(can_progress))
        CtlFormula::ag(CtlFormula::ef(CtlFormula::atom("can_progress")))
    }

    /// Cache coherence
    pub fn cache_coherence() -> CtlFormula {
        // AG(¬(stale_data ∧ valid_line))
        let stale = CtlFormula::atom("stale_data");
        let valid = CtlFormula::atom("valid_line");
        let conflict = CtlFormula::And(Box::new(stale), Box::new(valid));
        CtlFormula::ag(CtlFormula::Not(Box::new(conflict)))
    }

    /// NVLink ordering preserved
    pub fn nvlink_ordering() -> LtlFormula {
        // G(send → F(receive))
        let send = LtlFormula::atom("send");
        let receive = LtlFormula::atom("receive").finally();
        send.implies(receive).globally()
    }
}

/// Verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Property name
    pub property: String,
    /// Whether property holds
    pub holds: bool,
    /// Counterexample if property violated
    pub counterexample: Option<Vec<u32>>,
    /// Number of states explored
    pub states_explored: usize,
    /// Verification time in milliseconds
    pub time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltl_formula_construction() {
        let p = LtlFormula::atom("p");
        let q = LtlFormula::atom("q");

        let formula = p.clone().and(q.clone());
        assert!(matches!(formula, LtlFormula::And(_, _)));

        let g_p = p.clone().globally();
        assert!(matches!(g_p, LtlFormula::Globally(_)));

        let p_until_q = p.until(q);
        assert!(matches!(p_until_q, LtlFormula::Until(_, _)));
    }

    #[test]
    fn test_ltl_atoms() {
        let formula = LtlFormula::atom("p")
            .and(LtlFormula::atom("q"))
            .implies(LtlFormula::atom("r").globally());

        let atoms = formula.atoms();
        assert!(atoms.contains("p"));
        assert!(atoms.contains("q"));
        assert!(atoms.contains("r"));
        assert_eq!(atoms.len(), 3);
    }

    #[test]
    fn test_ltl_nnf() {
        // ¬G(p) should become F(¬p)
        let formula = LtlFormula::atom("p").globally().not();
        let nnf = formula.to_nnf();

        assert!(matches!(nnf, LtlFormula::Finally(_)));
    }

    #[test]
    fn test_ltl_display() {
        let formula = LtlFormula::atom("p").globally();
        assert_eq!(format!("{}", formula), "G(p)");

        let complex = LtlFormula::atom("p").until(LtlFormula::atom("q"));
        assert_eq!(format!("{}", complex), "(p U q)");
    }

    #[test]
    fn test_ctl_construction() {
        let safe = CtlFormula::ag(CtlFormula::atom("safe"));
        assert!(matches!(safe, CtlFormula::AllGlobally(_)));

        let reach = CtlFormula::ef(CtlFormula::atom("goal"));
        assert!(matches!(reach, CtlFormula::ExistsFinally(_)));
    }

    #[test]
    fn test_kripke_structure() {
        let mut ks = KripkeStructure::new();

        ks.add_state(State::new(0).with_label("init"));
        ks.add_state(State::new(1).with_label("running"));
        ks.add_state(State::new(2).with_label("done"));

        ks.set_initial(0);
        ks.add_transition(0, 1, Some("start".to_string()));
        ks.add_transition(1, 2, Some("finish".to_string()));

        assert_eq!(ks.num_states(), 3);
        assert_eq!(ks.num_transitions(), 2);
        assert_eq!(ks.successors(0), vec![1]);
    }

    #[test]
    fn test_ctl_model_checking() {
        let mut ks = KripkeStructure::new();

        // Simple two-state system
        ks.add_state(State::new(0).with_label("safe"));
        ks.add_state(State::new(1).with_label("safe"));
        ks.set_initial(0);
        ks.add_transition(0, 1, None);
        ks.add_transition(1, 0, None);

        let checker = CtlModelChecker::new(&ks);

        // AG(safe) should hold
        let safe = CtlFormula::ag(CtlFormula::atom("safe"));
        assert!(checker.check(&safe));

        // EF(¬safe) should not hold
        let unsafe_reach = CtlFormula::ef(CtlFormula::Not(Box::new(CtlFormula::atom("safe"))));
        assert!(!checker.check(&unsafe_reach));
    }

    #[test]
    fn test_ctl_counterexample() {
        let mut ks = KripkeStructure::new();

        // System that can reach unsafe state
        ks.add_state(State::new(0).with_label("safe"));
        ks.add_state(State::new(1).with_label("safe"));
        ks.add_state(State::new(2)); // Not safe!
        ks.set_initial(0);
        ks.add_transition(0, 1, None);
        ks.add_transition(1, 2, None);

        let checker = CtlModelChecker::new(&ks);

        // AG(safe) should not hold
        let safe_formula = CtlFormula::atom("safe");
        let counterex = checker.counterexample_ag(&safe_formula);

        assert!(counterex.is_some());
        let path = counterex.unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_smt_generator() {
        let mut gen = SmtLibGenerator::new();

        gen.declare_bv("ptr", 64);
        gen.declare_bv("size", 64);
        gen.assert("(bvult ptr #xffffffffffffffff)");

        let output = gen.generate();
        assert!(output.contains("declare-const ptr"));
        assert!(output.contains("check-sat"));
    }

    #[test]
    fn test_smt_memory_bounds() {
        let mut gen = SmtLibGenerator::new();
        gen.memory_bounds_constraint("ptr", "size", 0x1000);

        let output = gen.generate();
        assert!(output.contains("bvadd"));
        assert!(output.contains("bvule"));
    }

    #[test]
    fn test_runtime_monitor_mutex() {
        let mut monitor = RuntimeMonitor::mutex_protocol();

        // Valid sequence
        assert_eq!(monitor.process("lock"), MonitorResult::Ongoing);
        assert_eq!(monitor.process("unlock"), MonitorResult::Accepted);

        // Invalid: double lock
        monitor.reset();
        assert_eq!(monitor.process("lock"), MonitorResult::Ongoing);
        assert_eq!(monitor.process("lock"), MonitorResult::Violated);
    }

    #[test]
    fn test_runtime_monitor_request_response() {
        let mut monitor = RuntimeMonitor::request_response();

        // Valid sequence
        assert_eq!(monitor.process("request"), MonitorResult::Ongoing);
        assert_eq!(monitor.process("response"), MonitorResult::Accepted);

        // Invalid: response without request
        monitor.reset();
        assert_eq!(monitor.process("response"), MonitorResult::Violated);
    }

    #[test]
    fn test_gpu_properties() {
        let race_free = gpu_properties::warp_race_freedom();
        assert!(matches!(race_free, CtlFormula::AllGlobally(_)));

        let term = gpu_properties::kernel_termination();
        assert!(matches!(term, CtlFormula::AllFinally(_)));

        let ordering = gpu_properties::nvlink_ordering();
        assert!(matches!(ordering, LtlFormula::Globally(_)));
    }

    #[test]
    fn test_state_labels() {
        let state = State::new(0).with_label("safe").with_label("ready");

        assert!(state.satisfies("safe"));
        assert!(state.satisfies("ready"));
        assert!(!state.satisfies("error"));
    }

    #[test]
    fn test_ctl_sat_operations() {
        let mut ks = KripkeStructure::new();

        // Chain: 0 -> 1 -> 2
        ks.add_state(State::new(0).with_label("start"));
        ks.add_state(State::new(1).with_label("middle"));
        ks.add_state(State::new(2).with_label("end"));
        ks.set_initial(0);
        ks.add_transition(0, 1, None);
        ks.add_transition(1, 2, None);
        ks.add_transition(2, 2, None); // Self-loop

        let checker = CtlModelChecker::new(&ks);

        // EF(end) should include all states
        let ef_end = CtlFormula::ef(CtlFormula::atom("end"));
        let sat = checker.sat(&ef_end);
        assert_eq!(sat.len(), 3);

        // EX(middle) should be {0}
        let ex_middle = CtlFormula::ExistsNext(Box::new(CtlFormula::atom("middle")));
        let sat_ex = checker.sat(&ex_middle);
        assert!(sat_ex.contains(&0));
        assert_eq!(sat_ex.len(), 1);
    }
}
