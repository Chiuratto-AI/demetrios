//! Geometry Rule Engine
//!
//! Forward-chaining deduction rules for geometry.
//! Each rule has premises, conclusion, and confidence decay.

use std::collections::HashMap;

use super::predicates::{Predicate, PredicateId, PredicateKind, PredicatePattern};
use super::proof_state::ProofState;

/// A geometry deduction rule
#[derive(Debug, Clone)]
pub struct GeometryRule {
    /// Rule name (for tracing)
    pub name: String,
    /// Premise patterns
    pub premises: Vec<PredicatePattern>,
    /// How to construct the conclusion from bindings
    pub conclusion: ConclusionTemplate,
    /// Confidence decay factor
    pub decay: f64,
    /// Rule priority (higher = try first)
    pub priority: i32,
}

/// Template for constructing conclusions
#[derive(Debug, Clone)]
pub enum ConclusionTemplate {
    /// Collinear from three bound variables
    Collinear { p1: String, p2: String, p3: String },
    /// Parallel from four bound variables (two lines)
    Parallel {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    /// Perpendicular from four bound variables
    Perpendicular {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    /// Equal length from four bound variables
    EqualLength {
        s1_p1: String,
        s1_p2: String,
        s2_p1: String,
        s2_p2: String,
    },
    /// Concyclic from four bound variables
    Concyclic {
        p1: String,
        p2: String,
        p3: String,
        p4: String,
    },
    /// On circle
    OnCircle {
        point: String,
        center: String,
        on_circle: String,
    },
    /// Midpoint
    Midpoint { mid: String, p1: String, p2: String },
    /// Right angle
    RightAngle {
        p1: String,
        vertex: String,
        p2: String,
    },
}

impl ConclusionTemplate {
    /// Instantiate the template with variable bindings
    pub fn instantiate(&self, bindings: &HashMap<String, String>) -> Option<Predicate> {
        match self {
            ConclusionTemplate::Collinear { p1, p2, p3 } => {
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                let v3 = bindings.get(p3)?;
                Some(Predicate::collinear(v1, v2, v3))
            }
            ConclusionTemplate::Parallel {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => {
                let v1 = bindings.get(l1_p1)?;
                let v2 = bindings.get(l1_p2)?;
                let v3 = bindings.get(l2_p1)?;
                let v4 = bindings.get(l2_p2)?;
                Some(Predicate::parallel(v1, v2, v3, v4))
            }
            ConclusionTemplate::Perpendicular {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => {
                let v1 = bindings.get(l1_p1)?;
                let v2 = bindings.get(l1_p2)?;
                let v3 = bindings.get(l2_p1)?;
                let v4 = bindings.get(l2_p2)?;
                Some(Predicate::perpendicular(v1, v2, v3, v4))
            }
            ConclusionTemplate::EqualLength {
                s1_p1,
                s1_p2,
                s2_p1,
                s2_p2,
            } => {
                let v1 = bindings.get(s1_p1)?;
                let v2 = bindings.get(s1_p2)?;
                let v3 = bindings.get(s2_p1)?;
                let v4 = bindings.get(s2_p2)?;
                Some(Predicate::equal_length(v1, v2, v3, v4))
            }
            ConclusionTemplate::Concyclic { p1, p2, p3, p4 } => {
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                let v3 = bindings.get(p3)?;
                let v4 = bindings.get(p4)?;
                Some(Predicate::concyclic(v1, v2, v3, v4))
            }
            ConclusionTemplate::OnCircle {
                point,
                center,
                on_circle,
            } => {
                let v1 = bindings.get(point)?;
                let v2 = bindings.get(center)?;
                let v3 = bindings.get(on_circle)?;
                Some(Predicate::on_circle(v1, v2, v3))
            }
            ConclusionTemplate::Midpoint { mid, p1, p2 } => {
                let vm = bindings.get(mid)?;
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                Some(Predicate::midpoint(vm, v1, v2))
            }
            ConclusionTemplate::RightAngle { p1, vertex, p2 } => {
                let v1 = bindings.get(p1)?;
                let vv = bindings.get(vertex)?;
                let v2 = bindings.get(p2)?;
                Some(Predicate::right_angle(v1, vv, v2))
            }
        }
    }
}

/// Result of matching a rule
#[derive(Debug, Clone)]
pub struct RuleMatch {
    /// The rule that matched
    pub rule_name: String,
    /// Variable bindings
    pub bindings: HashMap<String, String>,
    /// Matched premise predicate IDs
    pub premise_ids: Vec<PredicateId>,
    /// Instantiated conclusion
    pub conclusion: Predicate,
}

impl GeometryRule {
    /// Try to match this rule against the proof state
    /// Returns all possible matches
    pub fn match_state(&self, state: &ProofState) -> Vec<RuleMatch> {
        let mut matches = Vec::new();

        // Get all predicates matching first premise
        if self.premises.is_empty() {
            return matches;
        }

        // Recursive helper to find all valid bindings
        fn find_bindings(
            premises: &[PredicatePattern],
            state: &ProofState,
            current_bindings: HashMap<String, String>,
            premise_ids: Vec<PredicateId>,
        ) -> Vec<(HashMap<String, String>, Vec<PredicateId>)> {
            if premises.is_empty() {
                return vec![(current_bindings, premise_ids)];
            }

            let pattern = &premises[0];
            let remaining = &premises[1..];
            let mut results = Vec::new();

            for pred in state.predicates_by_kind(pattern.kind.clone()) {
                if let Some(new_bindings) = pattern.match_predicate(pred) {
                    // Check compatibility with current bindings
                    let mut compatible = true;
                    let mut merged = current_bindings.clone();

                    for (var, val) in new_bindings {
                        if let Some(existing) = merged.get(&var) {
                            if existing != &val {
                                compatible = false;
                                break;
                            }
                        } else {
                            merged.insert(var, val);
                        }
                    }

                    if compatible {
                        let mut new_ids = premise_ids.clone();
                        new_ids.push(pred.id);
                        results.extend(find_bindings(remaining, state, merged, new_ids));
                    }
                }
            }

            results
        }

        let bindings_list = find_bindings(&self.premises, state, HashMap::new(), Vec::new());

        for (bindings, premise_ids) in bindings_list {
            if let Some(conclusion) = self.conclusion.instantiate(&bindings) {
                // Check that conclusion doesn't already exist
                if !state.has_predicate(&conclusion.key()) {
                    matches.push(RuleMatch {
                        rule_name: self.name.clone(),
                        bindings,
                        premise_ids,
                        conclusion,
                    });
                }
            }
        }

        matches
    }
}

/// Database of geometry rules
pub struct RuleDatabase {
    rules: Vec<GeometryRule>,
}

impl RuleDatabase {
    /// Create empty database
    pub fn new() -> Self {
        RuleDatabase { rules: Vec::new() }
    }

    /// Create database with standard rules
    pub fn standard() -> Self {
        let mut db = RuleDatabase::new();

        // Collinearity transitivity
        // If collinear(A,B,C) and collinear(A,B,D) then collinear(B,C,D)
        db.add_rule(GeometryRule {
            name: "collinear_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Collinear, vec!["A", "B", "C"]),
                PredicatePattern::new(PredicateKind::Collinear, vec!["A", "B", "D"]),
            ],
            conclusion: ConclusionTemplate::Collinear {
                p1: "B".to_string(),
                p2: "C".to_string(),
                p3: "D".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Parallel transitivity
        // If parallel(L1, L2) and parallel(L2, L3) then parallel(L1, L3)
        db.add_rule(GeometryRule {
            name: "parallel_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Parallel, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::Parallel, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::Parallel {
                l1_p1: "A".to_string(),
                l1_p2: "B".to_string(),
                l2_p1: "E".to_string(),
                l2_p2: "F".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Perpendicular to parallel implies perpendicular
        // If perp(L1, L2) and parallel(L2, L3) then perp(L1, L3)
        db.add_rule(GeometryRule {
            name: "perp_para_perp".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Perpendicular, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::Parallel, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::Perpendicular {
                l1_p1: "A".to_string(),
                l1_p2: "B".to_string(),
                l2_p1: "E".to_string(),
                l2_p2: "F".to_string(),
            },
            decay: 0.98,
            priority: 9,
        });

        // Midpoint theorem: line through midpoints is parallel to third side
        // If midpoint(M, A, B) and midpoint(N, A, C) then parallel(MN, BC)
        db.add_rule(GeometryRule {
            name: "midpoint_parallel".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Midpoint, vec!["M", "A", "B"]),
                PredicatePattern::new(PredicateKind::Midpoint, vec!["N", "A", "C"]),
            ],
            conclusion: ConclusionTemplate::Parallel {
                l1_p1: "M".to_string(),
                l1_p2: "N".to_string(),
                l2_p1: "B".to_string(),
                l2_p2: "C".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Inscribed angle theorem (same arc)
        // If on_circle(A,O,R) and on_circle(B,O,R) and on_circle(P,O,R) and on_circle(Q,O,R)
        // This is complex - simplified version
        // If concyclic(A,B,P,Q) then angles subtended by AB from P and Q are equal

        // Cyclic quadrilateral: four concyclic points
        // If on_circle(A,O,R) and on_circle(B,O,R) and on_circle(C,O,R) and on_circle(D,O,R)
        // then concyclic(A,B,C,D)

        // Equal length transitivity
        // If equal_length(AB, CD) and equal_length(CD, EF) then equal_length(AB, EF)
        db.add_rule(GeometryRule {
            name: "equal_length_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::EqualLength, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::EqualLength, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::EqualLength {
                s1_p1: "A".to_string(),
                s1_p2: "B".to_string(),
                s2_p1: "E".to_string(),
                s2_p2: "F".to_string(),
            },
            decay: 0.99,
            priority: 9,
        });

        // Midpoint implies equal lengths
        // If midpoint(M, A, B) then equal_length(AM, MB)
        db.add_rule(GeometryRule {
            name: "midpoint_equal".to_string(),
            premises: vec![PredicatePattern::new(
                PredicateKind::Midpoint,
                vec!["M", "A", "B"],
            )],
            conclusion: ConclusionTemplate::EqualLength {
                s1_p1: "A".to_string(),
                s1_p2: "M".to_string(),
                s2_p1: "M".to_string(),
                s2_p2: "B".to_string(),
            },
            decay: 1.0, // No decay - definitional
            priority: 10,
        });

        // On same circle implies concyclic (4 points)
        // This requires collecting 4 on_circle predicates with same circle

        // Perpendicular bisector: perpendicular at midpoint
        // If midpoint(M, A, B) and perpendicular(line_through_M, AB)
        // then equidistant from A and B

        db
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: GeometryRule) {
        self.rules.push(rule);
        // Sort by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get all rules
    pub fn rules(&self) -> &[GeometryRule] {
        &self.rules
    }

    /// Find all applicable rules for current state
    pub fn find_matches(&self, state: &ProofState) -> Vec<RuleMatch> {
        let mut all_matches = Vec::new();

        for rule in &self.rules {
            all_matches.extend(rule.match_state(state));
        }

        all_matches
    }
}

impl Default for RuleDatabase {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collinear_trans_rule() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        let db = RuleDatabase::standard();
        let matches = db.find_matches(&state);

        // Should find collinear transitivity
        let trans_match = matches.iter().find(|m| m.rule_name == "collinear_trans");
        assert!(trans_match.is_some());
    }

    #[test]
    fn test_midpoint_rules() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "M", "N"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        let db = RuleDatabase::standard();
        let matches = db.find_matches(&state);

        // Should find midpoint parallel theorem
        let para_match = matches.iter().find(|m| m.rule_name == "midpoint_parallel");
        assert!(para_match.is_some());

        // Should find midpoint equal lengths (twice, for M and N)
        let eq_matches: Vec<_> = matches
            .iter()
            .filter(|m| m.rule_name == "midpoint_equal")
            .collect();
        assert_eq!(eq_matches.len(), 2);
    }
}
