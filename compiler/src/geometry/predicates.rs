//! Geometric Predicates
//!
//! First-order predicates for symbolic geometry reasoning.
//! Each predicate carries epistemic metadata (confidence, provenance).

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::epistemic::{Confidence, Revisability, Source};

use super::primitives::{Angle, Circle, Line, Segment};

/// Kind of geometric predicate
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum PredicateKind {
    // Point relations
    /// Three points are collinear
    Collinear,
    /// Four points are concyclic (lie on a circle)
    Concyclic,
    /// Point is on a line
    OnLine,
    /// Point is on a circle
    OnCircle,
    /// Two points are equal
    EqualPoints,

    // Line relations
    /// Two lines are parallel
    Parallel,
    /// Two lines are perpendicular
    Perpendicular,

    // Length relations
    /// Two segments have equal length
    EqualLength,
    /// Ratio of two lengths equals a value
    LengthRatio,

    // Angle relations
    /// Two angles are equal
    EqualAngle,
    /// Angle is right (90 degrees)
    RightAngle,
    /// Sum of angles equals a value
    AngleSum,

    // Triangle relations
    /// Two triangles are similar
    Similar,
    /// Two triangles are congruent
    Congruent,

    // Special points
    /// Point is midpoint of segment
    Midpoint,
    /// Line is angle bisector
    AngleBisector,
    /// Line is perpendicular bisector
    PerpBisector,
    /// Point is circumcenter
    Circumcenter,
    /// Point is incenter
    Incenter,
    /// Point is centroid
    Centroid,
    /// Point is orthocenter
    Orthocenter,

    // Circle relations
    /// Line is tangent to circle at point
    Tangent,
    /// Point is center of circle
    CircleCenter,

    // Algebraic equality (for AR)
    /// Two expressions are equal
    AlgebraicEqual,
}

impl PredicateKind {
    /// Get the arity (number of arguments) for this predicate kind
    pub fn arity(&self) -> usize {
        match self {
            PredicateKind::Collinear => 3,
            PredicateKind::Concyclic => 4,
            PredicateKind::OnLine => 2,   // point, line (2 points)
            PredicateKind::OnCircle => 2, // point, circle
            PredicateKind::EqualPoints => 2,
            PredicateKind::Parallel => 4, // line1 (2 pts), line2 (2 pts)
            PredicateKind::Perpendicular => 4,
            PredicateKind::EqualLength => 4, // seg1 (2 pts), seg2 (2 pts)
            PredicateKind::LengthRatio => 5, // seg1, seg2, ratio
            PredicateKind::EqualAngle => 6,  // angle1 (3 pts), angle2 (3 pts)
            PredicateKind::RightAngle => 3,
            PredicateKind::AngleSum => 7, // angle1, angle2, sum
            PredicateKind::Similar => 6,  // tri1, tri2
            PredicateKind::Congruent => 6,
            PredicateKind::Midpoint => 3,      // mid, p1, p2
            PredicateKind::AngleBisector => 4, // line, angle
            PredicateKind::PerpBisector => 4,  // line, segment
            PredicateKind::Circumcenter => 4,  // center, p1, p2, p3
            PredicateKind::Incenter => 4,
            PredicateKind::Centroid => 4,
            PredicateKind::Orthocenter => 4,
            PredicateKind::Tangent => 4, // line, circle, point
            PredicateKind::CircleCenter => 2,
            PredicateKind::AlgebraicEqual => 2, // expr1, expr2
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            PredicateKind::Collinear => "collinear",
            PredicateKind::Concyclic => "concyclic",
            PredicateKind::OnLine => "on_line",
            PredicateKind::OnCircle => "on_circle",
            PredicateKind::EqualPoints => "equal_points",
            PredicateKind::Parallel => "parallel",
            PredicateKind::Perpendicular => "perpendicular",
            PredicateKind::EqualLength => "equal_length",
            PredicateKind::LengthRatio => "length_ratio",
            PredicateKind::EqualAngle => "equal_angle",
            PredicateKind::RightAngle => "right_angle",
            PredicateKind::AngleSum => "angle_sum",
            PredicateKind::Similar => "similar",
            PredicateKind::Congruent => "congruent",
            PredicateKind::Midpoint => "midpoint",
            PredicateKind::AngleBisector => "angle_bisector",
            PredicateKind::PerpBisector => "perp_bisector",
            PredicateKind::Circumcenter => "circumcenter",
            PredicateKind::Incenter => "incenter",
            PredicateKind::Centroid => "centroid",
            PredicateKind::Orthocenter => "orthocenter",
            PredicateKind::Tangent => "tangent",
            PredicateKind::CircleCenter => "circle_center",
            PredicateKind::AlgebraicEqual => "algebraic_equal",
        }
    }
}

/// Epistemic status of a predicate
#[derive(Debug, Clone)]
pub struct PredicateEpistemic {
    /// Confidence in the predicate (0.0 to 1.0)
    pub confidence: Confidence,
    /// How this predicate was derived
    pub source: Source,
    /// Whether this can be revised
    pub revisability: Revisability,
    /// Depth in proof tree (0 = axiom)
    pub depth: usize,
    /// IDs of predicates this was derived from
    pub derived_from: Vec<PredicateId>,
}

impl Default for PredicateEpistemic {
    fn default() -> Self {
        PredicateEpistemic {
            confidence: Confidence::new(1.0),
            source: Source::Unknown,
            revisability: Revisability::Revisable {
                conditions: vec!["new_evidence".to_string()],
            },
            depth: 0,
            derived_from: vec![],
        }
    }
}

impl PredicateEpistemic {
    /// Create axiom epistemic status (from problem statement)
    pub fn axiom() -> Self {
        PredicateEpistemic {
            confidence: Confidence::new(1.0),
            source: Source::Axiom,
            revisability: Revisability::NonRevisable,
            depth: 0,
            derived_from: vec![],
        }
    }

    /// Create derived epistemic status
    pub fn derived(parents: &[&Predicate], rule_name: &str, decay: f64) -> Self {
        // Combine parent confidences with decay
        let combined_conf = parents
            .iter()
            .map(|p| p.epistemic.confidence.value())
            .fold(1.0, |acc, c| acc * c)
            * decay;

        let parent_ids: Vec<PredicateId> = parents.iter().map(|p| p.id).collect();
        let max_depth = parents.iter().map(|p| p.epistemic.depth).max().unwrap_or(0);

        PredicateEpistemic {
            confidence: Confidence::new(combined_conf),
            source: Source::Derivation(rule_name.to_string()),
            revisability: Revisability::Revisable {
                conditions: vec!["parent_revision".to_string()],
            },
            depth: max_depth + 1,
            derived_from: parent_ids,
        }
    }

    /// Create from neural prediction
    pub fn from_neural(model: &str, confidence: f64) -> Self {
        PredicateEpistemic {
            confidence: Confidence::new(confidence),
            source: Source::ModelPrediction {
                model: model.to_string(),
                version: None,
            },
            revisability: Revisability::Revisable {
                conditions: vec!["verification".to_string()],
            },
            depth: 0,
            derived_from: vec![],
        }
    }

    /// Decay confidence
    pub fn decay(&self, factor: f64) -> Self {
        PredicateEpistemic {
            confidence: Confidence::new(self.confidence.value() * factor),
            ..self.clone()
        }
    }
}

/// Unique identifier for predicates
pub type PredicateId = u64;

/// A geometric predicate with epistemic metadata
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Unique identifier
    pub id: PredicateId,
    /// Kind of predicate
    pub kind: PredicateKind,
    /// Arguments (point labels, in canonical order)
    pub args: Vec<String>,
    /// Epistemic metadata
    pub epistemic: PredicateEpistemic,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(kind: PredicateKind, args: Vec<String>) -> Self {
        use std::collections::hash_map::DefaultHasher;

        // Generate ID from kind + args
        let mut hasher = DefaultHasher::new();
        kind.hash(&mut hasher);
        for arg in &args {
            arg.hash(&mut hasher);
        }
        let id = hasher.finish();

        Predicate {
            id,
            kind,
            args,
            epistemic: PredicateEpistemic::default(),
        }
    }

    /// Create with epistemic status
    pub fn with_epistemic(mut self, epistemic: PredicateEpistemic) -> Self {
        self.epistemic = epistemic;
        self
    }

    /// Create collinear predicate
    pub fn collinear(p1: &str, p2: &str, p3: &str) -> Self {
        let mut args = vec![p1.to_string(), p2.to_string(), p3.to_string()];
        args.sort(); // Canonical order
        Predicate::new(PredicateKind::Collinear, args)
    }

    /// Create concyclic predicate
    pub fn concyclic(p1: &str, p2: &str, p3: &str, p4: &str) -> Self {
        let mut args = vec![
            p1.to_string(),
            p2.to_string(),
            p3.to_string(),
            p4.to_string(),
        ];
        args.sort();
        Predicate::new(PredicateKind::Concyclic, args)
    }

    /// Create parallel predicate
    pub fn parallel(l1_p1: &str, l1_p2: &str, l2_p1: &str, l2_p2: &str) -> Self {
        // Canonical: sort within each line, then sort lines
        let mut l1 = vec![l1_p1.to_string(), l1_p2.to_string()];
        let mut l2 = vec![l2_p1.to_string(), l2_p2.to_string()];
        l1.sort();
        l2.sort();

        let (first, second) = if l1 <= l2 { (l1, l2) } else { (l2, l1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::Parallel, args)
    }

    /// Create perpendicular predicate
    pub fn perpendicular(l1_p1: &str, l1_p2: &str, l2_p1: &str, l2_p2: &str) -> Self {
        let mut l1 = vec![l1_p1.to_string(), l1_p2.to_string()];
        let mut l2 = vec![l2_p1.to_string(), l2_p2.to_string()];
        l1.sort();
        l2.sort();

        let (first, second) = if l1 <= l2 { (l1, l2) } else { (l2, l1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::Perpendicular, args)
    }

    /// Create equal length predicate
    pub fn equal_length(s1_p1: &str, s1_p2: &str, s2_p1: &str, s2_p2: &str) -> Self {
        let mut s1 = vec![s1_p1.to_string(), s1_p2.to_string()];
        let mut s2 = vec![s2_p1.to_string(), s2_p2.to_string()];
        s1.sort();
        s2.sort();

        let (first, second) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };
        let args = vec![
            first[0].clone(),
            first[1].clone(),
            second[0].clone(),
            second[1].clone(),
        ];

        Predicate::new(PredicateKind::EqualLength, args)
    }

    /// Create midpoint predicate
    pub fn midpoint(mid: &str, p1: &str, p2: &str) -> Self {
        let mut endpoints = vec![p1.to_string(), p2.to_string()];
        endpoints.sort();
        let args = vec![mid.to_string(), endpoints[0].clone(), endpoints[1].clone()];
        Predicate::new(PredicateKind::Midpoint, args)
    }

    /// Create right angle predicate
    pub fn right_angle(p1: &str, vertex: &str, p2: &str) -> Self {
        let mut rays = vec![p1.to_string(), p2.to_string()];
        rays.sort();
        let args = vec![rays[0].clone(), vertex.to_string(), rays[1].clone()];
        Predicate::new(PredicateKind::RightAngle, args)
    }

    /// Create on_circle predicate
    pub fn on_circle(point: &str, center: &str, on_circle: &str) -> Self {
        Predicate::new(
            PredicateKind::OnCircle,
            vec![point.to_string(), center.to_string(), on_circle.to_string()],
        )
    }

    /// Get canonical key for deduplication
    pub fn key(&self) -> String {
        format!("{}:{}", self.kind.name(), self.args.join(","))
    }

    /// Check if this predicate is high confidence
    pub fn is_high_confidence(&self, threshold: f64) -> bool {
        self.epistemic.confidence.value() >= threshold
    }

    /// Get all referenced point labels
    pub fn referenced_points(&self) -> &[String] {
        &self.args
    }
}

impl PartialEq for Predicate {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.args == other.args
    }
}

impl Eq for Predicate {}

impl Hash for Predicate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.args.hash(state);
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.kind.name(), self.args.join(", "))
    }
}

/// Pattern for matching predicates in rules
#[derive(Debug, Clone)]
pub struct PredicatePattern {
    /// Kind to match
    pub kind: PredicateKind,
    /// Variable names for arguments (for binding)
    pub vars: Vec<String>,
}

impl PredicatePattern {
    pub fn new(kind: PredicateKind, vars: Vec<&str>) -> Self {
        PredicatePattern {
            kind,
            vars: vars.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Try to match a predicate, returning variable bindings if successful
    pub fn match_predicate(&self, pred: &Predicate) -> Option<HashMap<String, String>> {
        if self.kind != pred.kind || self.vars.len() != pred.args.len() {
            return None;
        }

        let mut bindings = HashMap::new();
        for (var, arg) in self.vars.iter().zip(pred.args.iter()) {
            if let Some(existing) = bindings.get(var) {
                if existing != arg {
                    return None; // Conflict
                }
            } else {
                bindings.insert(var.clone(), arg.clone());
            }
        }

        Some(bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_canonical() {
        let p1 = Predicate::collinear("A", "B", "C");
        let p2 = Predicate::collinear("C", "A", "B");
        assert_eq!(p1.key(), p2.key());
    }

    #[test]
    fn test_parallel_canonical() {
        let p1 = Predicate::parallel("A", "B", "C", "D");
        let p2 = Predicate::parallel("C", "D", "B", "A");
        assert_eq!(p1.key(), p2.key());
    }

    #[test]
    fn test_pattern_match() {
        let pattern = PredicatePattern::new(PredicateKind::Collinear, vec!["X", "Y", "Z"]);
        let pred = Predicate::collinear("A", "B", "C");

        let bindings = pattern.match_predicate(&pred).unwrap();
        assert_eq!(bindings.len(), 3);
    }

    #[test]
    fn test_epistemic_decay() {
        let epi = PredicateEpistemic::axiom();
        assert_eq!(epi.confidence.value(), 1.0);

        let decayed = epi.decay(0.95);
        assert!((decayed.confidence.value() - 0.95).abs() < 0.001);
    }
}
