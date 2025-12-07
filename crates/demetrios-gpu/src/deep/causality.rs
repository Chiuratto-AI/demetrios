//! Causal Structure for Computation
//!
//! This module implements causal reasoning primitives based on:
//! - Causal sets (discrete spacetime)
//! - Pearl's do-calculus
//! - Light cone constraints
//! - Interventions and counterfactuals
//!
//! # Key Insight
//!
//! In physics, causality is fundamental. Effects cannot precede causes.
//! Light cones constrain what can influence what.
//!
//! In computation, we model this as:
//! - Data dependencies form a causal graph
//! - Parallel execution respects causal structure
//! - Interventions allow "what-if" reasoning

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;

// ============================================================================
// CAUSAL SET (Discrete Spacetime)
// ============================================================================

/// A causal set: a discrete model of spacetime
///
/// Elements are events, partial order is causality.
/// x ≤ y means x can causally influence y.
#[derive(Debug, Clone)]
pub struct CausalSet<E: Clone + Eq + Hash> {
    /// All events
    events: HashSet<E>,
    /// Causal relations: (cause, effect)
    relations: HashSet<(E, E)>,
    /// Cached transitive closure
    closure: Option<HashSet<(E, E)>>,
}

impl<E: Clone + Eq + Hash> CausalSet<E> {
    /// Create an empty causal set
    pub fn new() -> Self {
        Self {
            events: HashSet::new(),
            relations: HashSet::new(),
            closure: None,
        }
    }

    /// Add an event
    pub fn add_event(&mut self, event: E) {
        self.events.insert(event);
        self.closure = None;
    }

    /// Add a causal relation: cause → effect
    pub fn add_causation(&mut self, cause: E, effect: E) {
        self.events.insert(cause.clone());
        self.events.insert(effect.clone());
        self.relations.insert((cause, effect));
        self.closure = None;
    }

    /// Check if a causes b (directly or transitively)
    pub fn causes(&mut self, a: &E, b: &E) -> bool {
        self.ensure_closure();
        self.closure
            .as_ref()
            .unwrap()
            .contains(&(a.clone(), b.clone()))
    }

    /// Check if a and b are causally related
    pub fn are_related(&mut self, a: &E, b: &E) -> bool {
        self.causes(a, b) || self.causes(b, a)
    }

    /// Check if a and b are spacelike separated (can be parallel)
    pub fn are_spacelike(&mut self, a: &E, b: &E) -> bool {
        !self.are_related(a, b) && a != b
    }

    /// Get all events that can be parallel to this event
    pub fn parallel_to(&mut self, event: &E) -> Vec<E> {
        let events: Vec<E> = self.events.iter().cloned().collect();
        events
            .into_iter()
            .filter(|e| self.are_spacelike(e, event))
            .collect()
    }

    /// Get the past light cone of an event
    pub fn past_cone(&mut self, event: &E) -> HashSet<E> {
        self.ensure_closure();
        let closure = self.closure.as_ref().unwrap();
        self.events
            .iter()
            .filter(|e| closure.contains(&((*e).clone(), event.clone())))
            .cloned()
            .collect()
    }

    /// Get the future light cone of an event
    pub fn future_cone(&mut self, event: &E) -> HashSet<E> {
        self.ensure_closure();
        let closure = self.closure.as_ref().unwrap();
        self.events
            .iter()
            .filter(|e| closure.contains(&(event.clone(), (*e).clone())))
            .cloned()
            .collect()
    }

    /// Compute transitive closure (Floyd-Warshall style)
    fn ensure_closure(&mut self) {
        if self.closure.is_some() {
            return;
        }

        let mut closure = self.relations.clone();

        // Warshall's algorithm
        let events: Vec<E> = self.events.iter().cloned().collect();
        for k in &events {
            for i in &events {
                for j in &events {
                    if closure.contains(&(i.clone(), k.clone()))
                        && closure.contains(&(k.clone(), j.clone()))
                    {
                        closure.insert((i.clone(), j.clone()));
                    }
                }
            }
        }

        self.closure = Some(closure);
    }

    /// Get a valid causal ordering (topological sort)
    pub fn causal_order(&self) -> Option<Vec<E>> {
        let mut in_degree: HashMap<E, usize> = HashMap::new();
        for e in &self.events {
            in_degree.insert(e.clone(), 0);
        }

        for (_, effect) in &self.relations {
            *in_degree.get_mut(effect).unwrap() += 1;
        }

        let mut queue: VecDeque<E> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(e, _)| e.clone())
            .collect();

        let mut order = Vec::new();

        while let Some(event) = queue.pop_front() {
            order.push(event.clone());

            for (cause, effect) in &self.relations {
                if cause == &event {
                    let d = in_degree.get_mut(effect).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(effect.clone());
                    }
                }
            }
        }

        if order.len() == self.events.len() {
            Some(order)
        } else {
            None // Cycle detected
        }
    }
}

impl<E: Clone + Eq + Hash> Default for CausalSet<E> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CAUSAL RELATION
// ============================================================================

/// Types of causal relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalRelation {
    /// A directly causes B
    Direct,
    /// A causes B through intermediaries
    Indirect,
    /// A and B have a common cause
    Confounded,
    /// A and B are independent
    Independent,
    /// Causal direction is unknown
    Unknown,
}

// ============================================================================
// LIGHT CONE
// ============================================================================

/// A light cone in discrete spacetime
///
/// Represents the causal structure around an event:
/// - Past cone: all events that could have influenced this one
/// - Future cone: all events this one could influence
/// - Spacelike: events that cannot be causally connected
#[derive(Debug, Clone)]
pub struct LightCone<E: Clone + Eq + Hash> {
    /// The central event
    pub event: E,
    /// Past light cone
    pub past: HashSet<E>,
    /// Future light cone
    pub future: HashSet<E>,
    /// Spacelike separated events
    pub spacelike: HashSet<E>,
}

impl<E: Clone + Eq + Hash> LightCone<E> {
    /// Create a light cone from a causal set
    pub fn from_causal_set(event: E, causal_set: &mut CausalSet<E>) -> Self {
        let past = causal_set.past_cone(&event);
        let future = causal_set.future_cone(&event);
        let spacelike: HashSet<E> = causal_set
            .events
            .iter()
            .filter(|e| *e != &event && !past.contains(*e) && !future.contains(*e))
            .cloned()
            .collect();

        Self {
            event,
            past,
            future,
            spacelike,
        }
    }

    /// Check if an event is in the past cone
    pub fn in_past(&self, event: &E) -> bool {
        self.past.contains(event)
    }

    /// Check if an event is in the future cone
    pub fn in_future(&self, event: &E) -> bool {
        self.future.contains(event)
    }

    /// Check if an event is spacelike separated
    pub fn is_spacelike(&self, event: &E) -> bool {
        self.spacelike.contains(event)
    }
}

// ============================================================================
// SPACETIME EVENT
// ============================================================================

/// An event in discrete spacetime
#[derive(Debug, Clone, PartialEq)]
pub struct SpacetimeEvent {
    /// Unique identifier
    pub id: usize,
    /// Discrete time coordinate
    pub time: i64,
    /// Spatial coordinates
    pub space: Vec<i64>,
    /// Associated data
    pub data: Option<String>,
}

impl Eq for SpacetimeEvent {}

impl Hash for SpacetimeEvent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl SpacetimeEvent {
    /// Create a new event
    pub fn new(id: usize, time: i64, space: Vec<i64>) -> Self {
        Self {
            id,
            time,
            space,
            data: None,
        }
    }

    /// Create with data
    pub fn with_data(mut self, data: String) -> Self {
        self.data = Some(data);
        self
    }

    /// Check if this event can causally influence another
    /// (respecting light cone: Δt >= |Δx| in natural units)
    pub fn can_influence(&self, other: &SpacetimeEvent) -> bool {
        let dt = other.time - self.time;
        if dt < 0 {
            return false; // Future cannot influence past
        }

        // Spatial distance
        let dx_squared: i64 = self
            .space
            .iter()
            .zip(other.space.iter())
            .map(|(a, b)| (b - a).pow(2))
            .sum();

        // Light cone: dt² >= dx² (in natural units where c=1)
        (dt * dt) >= dx_squared
    }
}

// ============================================================================
// CAUSAL GRAPH (Pearl's Framework)
// ============================================================================

/// A causal graph (DAG) for causal inference
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Variable names
    variables: Vec<String>,
    /// Variable indices
    var_index: HashMap<String, usize>,
    /// Edges: parent → child
    edges: HashSet<(usize, usize)>,
    /// Observed variables
    observed: HashSet<usize>,
}

impl CausalGraph {
    /// Create an empty causal graph
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            var_index: HashMap::new(),
            edges: HashSet::new(),
            observed: HashSet::new(),
        }
    }

    /// Add a variable
    pub fn add_variable(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.var_index.get(name) {
            return idx;
        }
        let idx = self.variables.len();
        self.variables.push(name.to_string());
        self.var_index.insert(name.to_string(), idx);
        idx
    }

    /// Add a causal edge: cause → effect
    pub fn add_edge(&mut self, cause: &str, effect: &str) {
        let c = self.add_variable(cause);
        let e = self.add_variable(effect);
        self.edges.insert((c, e));
    }

    /// Mark a variable as observed
    pub fn observe(&mut self, var: &str) {
        if let Some(&idx) = self.var_index.get(var) {
            self.observed.insert(idx);
        }
    }

    /// Get parents of a variable
    pub fn parents(&self, var: &str) -> Vec<String> {
        if let Some(&idx) = self.var_index.get(var) {
            self.edges
                .iter()
                .filter(|(_, e)| *e == idx)
                .map(|(p, _)| self.variables[*p].clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get children of a variable
    pub fn children(&self, var: &str) -> Vec<String> {
        if let Some(&idx) = self.var_index.get(var) {
            self.edges
                .iter()
                .filter(|(p, _)| *p == idx)
                .map(|(_, c)| self.variables[*c].clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check if there's a directed path from a to b
    pub fn has_path(&self, from: &str, to: &str) -> bool {
        let Some(&from_idx) = self.var_index.get(from) else {
            return false;
        };
        let Some(&to_idx) = self.var_index.get(to) else {
            return false;
        };

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from_idx);

        while let Some(current) = queue.pop_front() {
            if current == to_idx {
                return true;
            }
            if visited.insert(current) {
                for &(p, c) in &self.edges {
                    if p == current && !visited.contains(&c) {
                        queue.push_back(c);
                    }
                }
            }
        }

        false
    }

    /// Check d-separation
    pub fn d_separated(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        // Simplified d-separation check
        // Full implementation would use Bayes-Ball algorithm

        let z_idx: HashSet<usize> = z
            .iter()
            .filter_map(|v| self.var_index.get(v))
            .copied()
            .collect();

        // If any conditioning variable blocks all paths, they're d-separated
        !self.has_unblocked_path(x, y, &z_idx)
    }

    fn has_unblocked_path(&self, from: &str, to: &str, blocked: &HashSet<usize>) -> bool {
        let Some(&from_idx) = self.var_index.get(from) else {
            return false;
        };
        let Some(&to_idx) = self.var_index.get(to) else {
            return false;
        };

        // Simple BFS ignoring blocked nodes
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from_idx);

        while let Some(current) = queue.pop_front() {
            if current == to_idx {
                return true;
            }
            if blocked.contains(&current) {
                continue;
            }
            if visited.insert(current) {
                // Check both directions (undirected for d-separation)
                for &(p, c) in &self.edges {
                    if p == current && !visited.contains(&c) {
                        queue.push_back(c);
                    }
                    if c == current && !visited.contains(&p) {
                        queue.push_back(p);
                    }
                }
            }
        }

        false
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// INTERVENTION (do-operator)
// ============================================================================

/// An intervention: setting a variable to a specific value
/// This is Pearl's do(X = x) operator
#[derive(Debug, Clone)]
pub struct Intervention {
    /// Variable being intervened on
    pub variable: String,
    /// Value being set
    pub value: f64,
}

impl Intervention {
    /// Create a new intervention
    pub fn new(variable: &str, value: f64) -> Self {
        Self {
            variable: variable.to_string(),
            value,
        }
    }
}

impl fmt::Display for Intervention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "do({} = {})", self.variable, self.value)
    }
}

// ============================================================================
// COUNTERFACTUAL
// ============================================================================

/// A counterfactual query: "What would Y be if X had been x?"
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// The outcome variable
    pub outcome: String,
    /// The intervention
    pub intervention: Intervention,
    /// Evidence (observations)
    pub evidence: HashMap<String, f64>,
}

impl Counterfactual {
    /// Create a counterfactual query
    pub fn new(outcome: &str, intervention: Intervention) -> Self {
        Self {
            outcome: outcome.to_string(),
            intervention,
            evidence: HashMap::new(),
        }
    }

    /// Add evidence
    pub fn with_evidence(mut self, var: &str, value: f64) -> Self {
        self.evidence.insert(var.to_string(), value);
        self
    }
}

impl fmt::Display for Counterfactual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.outcome, self.intervention)
    }
}

// ============================================================================
// DO-CALCULUS
// ============================================================================

/// Pearl's do-calculus rules for causal inference
#[derive(Debug, Clone)]
pub struct DoCalculus {
    graph: CausalGraph,
}

impl DoCalculus {
    /// Create from a causal graph
    pub fn new(graph: CausalGraph) -> Self {
        Self { graph }
    }

    /// Rule 1: Insertion/deletion of observations
    /// P(y|do(x), z, w) = P(y|do(x), w) if (Y ⊥ Z | X, W)_G̅_X
    pub fn rule1_applicable(&self, y: &str, x: &str, z: &str, w: &HashSet<String>) -> bool {
        // Check d-separation in mutilated graph
        let mut cond = w.clone();
        cond.insert(x.to_string());
        self.graph.d_separated(y, z, &cond)
    }

    /// Rule 2: Action/observation exchange
    /// P(y|do(x), do(z), w) = P(y|do(x), z, w) if (Y ⊥ Z | X, W)_G̅_X_Z̲
    pub fn rule2_applicable(&self, y: &str, x: &str, z: &str, w: &HashSet<String>) -> bool {
        // Check d-separation in modified graph
        let mut cond = w.clone();
        cond.insert(x.to_string());
        self.graph.d_separated(y, z, &cond)
    }

    /// Rule 3: Insertion/deletion of actions
    /// P(y|do(x), do(z), w) = P(y|do(x), w) if (Y ⊥ Z | X, W)_G̅_X_Z̅(W)
    pub fn rule3_applicable(&self, y: &str, x: &str, z: &str, w: &HashSet<String>) -> bool {
        // Check d-separation with Z removed
        let mut cond = w.clone();
        cond.insert(x.to_string());
        self.graph.d_separated(y, z, &cond) && !self.graph.has_path(z, y)
    }

    /// Check if causal effect is identifiable
    pub fn is_identifiable(&self, y: &str, x: &str) -> bool {
        // Simplified: check if there's a backdoor path that's not blocked
        // Full implementation would use ID algorithm

        let parents = self.graph.parents(x);
        if parents.is_empty() {
            return true; // No confounding
        }

        // Check if conditioning on parents blocks all backdoor paths
        let parent_set: HashSet<String> = parents.into_iter().collect();
        self.graph.d_separated(x, y, &parent_set)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_set_basic() {
        let mut cs: CausalSet<&str> = CausalSet::new();
        cs.add_causation("A", "B");
        cs.add_causation("B", "C");

        assert!(cs.causes(&"A", &"B"));
        assert!(cs.causes(&"A", &"C")); // Transitive
        assert!(!cs.causes(&"C", &"A"));
    }

    #[test]
    fn test_spacelike_separation() {
        let mut cs: CausalSet<&str> = CausalSet::new();
        cs.add_causation("A", "B");
        cs.add_event("C");

        assert!(cs.are_spacelike(&"B", &"C"));
        assert!(!cs.are_spacelike(&"A", &"B"));
    }

    #[test]
    fn test_causal_order() {
        let mut cs: CausalSet<&str> = CausalSet::new();
        cs.add_causation("A", "B");
        cs.add_causation("B", "C");
        cs.add_causation("A", "C");

        let order = cs.causal_order().unwrap();
        let a_pos = order.iter().position(|x| *x == "A").unwrap();
        let b_pos = order.iter().position(|x| *x == "B").unwrap();
        let c_pos = order.iter().position(|x| *x == "C").unwrap();

        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_light_cone() {
        let mut cs: CausalSet<&str> = CausalSet::new();
        cs.add_causation("Past1", "Present");
        cs.add_causation("Past2", "Present");
        cs.add_causation("Present", "Future1");
        cs.add_causation("Present", "Future2");
        cs.add_event("Spacelike");

        let cone = LightCone::from_causal_set("Present", &mut cs);

        assert!(cone.in_past(&"Past1"));
        assert!(cone.in_past(&"Past2"));
        assert!(cone.in_future(&"Future1"));
        assert!(cone.in_future(&"Future2"));
        assert!(cone.is_spacelike(&"Spacelike"));
    }

    #[test]
    fn test_spacetime_causality() {
        let e1 = SpacetimeEvent::new(1, 0, vec![0, 0, 0]);
        let e2 = SpacetimeEvent::new(2, 1, vec![0, 0, 0]); // Same place, later
        let e3 = SpacetimeEvent::new(3, 1, vec![2, 0, 0]); // Too far

        assert!(e1.can_influence(&e2));
        assert!(!e1.can_influence(&e3)); // Outside light cone
    }

    #[test]
    fn test_causal_graph() {
        let mut g = CausalGraph::new();
        g.add_edge("X", "Y");
        g.add_edge("Z", "X");
        g.add_edge("Z", "Y");

        assert!(g.has_path("X", "Y"));
        assert!(g.has_path("Z", "Y"));
        assert!(!g.has_path("Y", "X"));

        assert_eq!(g.parents("Y").len(), 2);
    }

    #[test]
    fn test_intervention() {
        let i = Intervention::new("X", 1.0);
        assert_eq!(format!("{}", i), "do(X = 1)");
    }

    #[test]
    fn test_counterfactual() {
        let i = Intervention::new("X", 1.0);
        let cf = Counterfactual::new("Y", i).with_evidence("Z", 0.5);

        assert_eq!(cf.outcome, "Y");
        assert_eq!(cf.evidence.get("Z"), Some(&0.5));
    }
}
