//! Epistemic Type Integration
//!
//! This module extends the type checker with epistemic awareness, enabling
//! dependent ontological types with confidence tracking.
//!
//! # Knowledge[τ, ε, δ, Φ] Type System
//!
//! - τ (tau): Temporal index - when the knowledge is valid
//! - ε (epsilon): Epistemic status - confidence and source
//! - δ (delta): Domain constraint - ontological binding
//! - Φ (phi): Provenance functor - derivation trace
//!
//! # Dependent Ontological Types
//!
//! ```text
//! type Aspirin = Knowledge[
//!     ChEBI:15365,
//!     ε ≥ 0.95,
//!     δ ⊆ SmallMolecule,
//!     Φ: ChEBI → verified
//! ]
//! ```

use std::collections::HashMap;

use crate::epistemic::{
    Confidence, EpistemicStatus, Evidence, OntologyBinding, Revisability, Source,
};
use crate::ontology::{FoundationOntologies, OntologyResolver, ParsedTermRef, SubsumptionResult};

/// Temporal index placeholder (until full temporal support)
#[derive(Debug, Clone)]
pub struct TemporalIndex {
    pub timestamp: Option<String>,
}

/// A dependent ontological type with epistemic constraints
#[derive(Debug, Clone)]
pub struct OntologicalType {
    /// The ontology term this type represents
    pub binding: OntologyBinding,
    /// Minimum required confidence
    pub min_confidence: Confidence,
    /// Required evidence types
    pub required_evidence: Vec<EvidenceRequirement>,
    /// Temporal validity constraints
    pub temporal_constraint: Option<TemporalConstraint>,
    /// Provenance requirements
    pub provenance_constraint: Option<ProvenanceConstraint>,
}

/// Evidence requirement for a type constraint
#[derive(Debug, Clone)]
pub enum EvidenceRequirement {
    /// Require any evidence
    Any,
    /// Require publication evidence
    Publication,
    /// Require experimental evidence
    Experimental,
    /// Require computational evidence
    Computational,
    /// Require a specific minimum strength
    MinStrength(Confidence),
}

/// Temporal constraint for type validity
#[derive(Debug, Clone)]
pub enum TemporalConstraint {
    /// Valid at a specific point in time
    AtTime(TemporalIndex),
    /// Valid during an interval
    During {
        start: TemporalIndex,
        end: TemporalIndex,
    },
    /// Must be current (no more than N days old)
    Current { max_age_days: u32 },
}

/// Provenance constraint for derivation tracking
#[derive(Debug, Clone)]
pub enum ProvenanceConstraint {
    /// Must originate from a specific source
    FromSource(Source),
    /// Must pass through a verification step
    Verified,
    /// Maximum derivation depth
    MaxDepth(u32),
    /// Must have human review
    HumanReviewed,
}

/// Result of checking an epistemic constraint
#[derive(Debug, Clone)]
pub enum ConstraintResult {
    /// Constraint satisfied
    Satisfied,
    /// Constraint violated with explanation
    Violated(String),
    /// Constraint cannot be checked (missing information)
    Indeterminate(String),
}

/// Epistemic type checker integration
pub struct EpistemicChecker {
    /// Foundation ontologies for quick lookups
    foundations: FoundationOntologies,
    /// Ontology resolver for full resolution
    resolver: OntologyResolver,
    /// Type bindings in scope
    bindings: HashMap<String, OntologicalType>,
}

impl EpistemicChecker {
    /// Create a new epistemic checker
    pub fn new() -> Self {
        let resolver =
            OntologyResolver::default_resolver().expect("Failed to create ontology resolver");
        Self {
            foundations: FoundationOntologies::bootstrap(),
            resolver,
            bindings: HashMap::new(),
        }
    }

    /// Check if a value's epistemic status satisfies a type's requirements
    pub fn check_constraint(
        &self,
        value_status: &EpistemicStatus,
        type_constraint: &OntologicalType,
    ) -> ConstraintResult {
        // Check confidence requirement
        if value_status.confidence.value() < type_constraint.min_confidence.value() {
            return ConstraintResult::Violated(format!(
                "Confidence {} is below required minimum {}",
                value_status.confidence.value(),
                type_constraint.min_confidence.value()
            ));
        }

        // Check evidence requirements
        for requirement in &type_constraint.required_evidence {
            if !self.check_evidence_requirement(&value_status.evidence, requirement) {
                return ConstraintResult::Violated(format!(
                    "Evidence requirement not satisfied: {:?}",
                    requirement
                ));
            }
        }

        // Check temporal constraint
        if let Some(ref temporal) = type_constraint.temporal_constraint {
            match self.check_temporal_constraint(value_status, temporal) {
                ConstraintResult::Violated(msg) => return ConstraintResult::Violated(msg),
                ConstraintResult::Indeterminate(msg) => {
                    return ConstraintResult::Indeterminate(msg);
                }
                _ => {}
            }
        }

        // Check provenance constraint
        if let Some(ref provenance) = type_constraint.provenance_constraint {
            match self.check_provenance_constraint(value_status, provenance) {
                ConstraintResult::Violated(msg) => return ConstraintResult::Violated(msg),
                ConstraintResult::Indeterminate(msg) => {
                    return ConstraintResult::Indeterminate(msg);
                }
                _ => {}
            }
        }

        ConstraintResult::Satisfied
    }

    /// Check evidence requirement
    fn check_evidence_requirement(
        &self,
        evidence: &[Evidence],
        requirement: &EvidenceRequirement,
    ) -> bool {
        match requirement {
            EvidenceRequirement::Any => !evidence.is_empty(),
            EvidenceRequirement::Publication => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Publication { .. })),
            EvidenceRequirement::Experimental => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Experiment { .. })),
            EvidenceRequirement::Computational => evidence
                .iter()
                .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Computation { .. })),
            EvidenceRequirement::MinStrength(min) => {
                evidence.iter().any(|e| e.strength.value() >= min.value())
            }
        }
    }

    /// Check temporal constraint
    fn check_temporal_constraint(
        &self,
        _status: &EpistemicStatus,
        _constraint: &TemporalConstraint,
    ) -> ConstraintResult {
        // Temporal checking would require timestamp information in the status
        // For now, accept if constraint is present
        ConstraintResult::Indeterminate("Temporal constraints not yet fully implemented".into())
    }

    /// Check provenance constraint
    fn check_provenance_constraint(
        &self,
        status: &EpistemicStatus,
        constraint: &ProvenanceConstraint,
    ) -> ConstraintResult {
        match constraint {
            ProvenanceConstraint::FromSource(required_source) => {
                if self.sources_compatible(&status.source, required_source) {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated(format!(
                        "Source {:?} does not match required {:?}",
                        status.source, required_source
                    ))
                }
            }
            ProvenanceConstraint::Verified => {
                // Check if evidence includes verification
                if status
                    .evidence
                    .iter()
                    .any(|e| matches!(e.kind, crate::epistemic::EvidenceKind::Verified { .. }))
                {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated("No verification evidence found".into())
                }
            }
            ProvenanceConstraint::MaxDepth(_) => {
                ConstraintResult::Indeterminate("Depth tracking not yet implemented".into())
            }
            ProvenanceConstraint::HumanReviewed => {
                // Check for human review evidence
                if status.evidence.iter().any(|e| {
                    matches!(
                        e.kind,
                        crate::epistemic::EvidenceKind::HumanAssertion { .. }
                    )
                }) {
                    ConstraintResult::Satisfied
                } else {
                    ConstraintResult::Violated("No human review evidence found".into())
                }
            }
        }
    }

    /// Check if two sources are compatible
    fn sources_compatible(&self, actual: &Source, required: &Source) -> bool {
        match (actual, required) {
            (
                Source::OntologyAssertion {
                    ontology: o1,
                    term: t1,
                },
                Source::OntologyAssertion {
                    ontology: o2,
                    term: t2,
                },
            ) => o1 == o2 && t1 == t2,
            _ => false,
        }
    }

    /// Check ontological subsumption
    pub fn check_subsumption(&mut self, child: &str, parent: &str) -> SubsumptionResult {
        self.resolver
            .is_subclass_of(child, parent)
            .unwrap_or(SubsumptionResult::Unknown)
    }

    /// Bind a variable to an ontological type
    pub fn bind(&mut self, name: String, ty: OntologicalType) {
        self.bindings.insert(name, ty);
    }

    /// Look up a binding
    pub fn lookup(&self, name: &str) -> Option<&OntologicalType> {
        self.bindings.get(name)
    }

    /// Create an ontological type from a CURIE with default constraints
    pub fn type_from_curie(&self, curie: &str) -> Result<OntologicalType, String> {
        let parsed = ParsedTermRef::parse(curie).map_err(|e| e.to_string())?;

        Ok(OntologicalType {
            binding: parsed.to_binding(),
            min_confidence: Confidence::new(0.0), // No minimum by default
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        })
    }

    /// Create an ontological type with confidence requirement
    pub fn type_with_confidence(
        &self,
        curie: &str,
        min_confidence: f64,
    ) -> Result<OntologicalType, String> {
        let mut ty = self.type_from_curie(curie)?;
        ty.min_confidence = Confidence::new(min_confidence);
        Ok(ty)
    }
}

impl Default for EpistemicChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute epistemic heterogeneity between two types
pub fn epistemic_heterogeneity(a: &EpistemicStatus, b: &EpistemicStatus) -> f64 {
    // Difference in confidence levels
    let confidence_diff = (a.confidence.value() - b.confidence.value()).abs();

    // Source compatibility (0 if same, 1 if different)
    let source_diff = if std::mem::discriminant(&a.source) == std::mem::discriminant(&b.source) {
        0.0
    } else {
        0.5
    };

    // Combine heterogeneity factors
    (confidence_diff + source_diff) / 2.0
}

/// Combine epistemic statuses using Bayesian methods
pub fn combine_epistemic_bayesian(statuses: &[EpistemicStatus]) -> EpistemicStatus {
    if statuses.is_empty() {
        return EpistemicStatus::default();
    }

    if statuses.len() == 1 {
        return statuses[0].clone();
    }

    // Bayesian combination of confidence values
    // Using log-odds combination
    let combined_confidence = {
        let mut log_odds_sum = 0.0;

        for status in statuses {
            let p = status.confidence.value().clamp(0.001, 0.999);
            let log_odds = (p / (1.0 - p)).ln();
            log_odds_sum += log_odds;
        }

        let avg_log_odds = log_odds_sum / statuses.len() as f64;
        let combined_p = 1.0 / (1.0 + (-avg_log_odds).exp());
        combined_p.clamp(0.0, 1.0)
    };

    // Combine evidence from all sources
    let combined_evidence: Vec<Evidence> =
        statuses.iter().flat_map(|s| s.evidence.clone()).collect();

    // Use most restrictive revisability
    let combined_revisability = Revisability::Revisable {
        conditions: statuses
            .iter()
            .filter_map(|s| {
                if let Revisability::Revisable { conditions } = &s.revisability {
                    Some(conditions.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect(),
    };

    EpistemicStatus {
        confidence: Confidence::new(combined_confidence),
        revisability: combined_revisability,
        source: Source::Derivation("bayesian_combination".into()),
        evidence: combined_evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_checker_new() {
        let checker = EpistemicChecker::new();
        assert!(checker.bindings.is_empty());
    }

    #[test]
    fn test_type_from_curie() {
        let checker = EpistemicChecker::new();
        let ty = checker.type_from_curie("CHEBI:15365").unwrap();
        assert!(ty.min_confidence.value() == 0.0);
    }

    #[test]
    fn test_type_with_confidence() {
        let checker = EpistemicChecker::new();
        let ty = checker.type_with_confidence("GO:0008150", 0.95).unwrap();
        assert!(ty.min_confidence.value() >= 0.95);
    }

    #[test]
    fn test_constraint_check_confidence() {
        let checker = EpistemicChecker::new();

        let value_status = EpistemicStatus {
            confidence: Confidence::new(0.9),
            ..Default::default()
        };

        let high_requirement = OntologicalType {
            binding: ParsedTermRef::parse("TEST:001").unwrap().to_binding(),
            min_confidence: Confidence::new(0.95),
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        };

        let result = checker.check_constraint(&value_status, &high_requirement);
        assert!(matches!(result, ConstraintResult::Violated(_)));

        let low_requirement = OntologicalType {
            binding: ParsedTermRef::parse("TEST:001").unwrap().to_binding(),
            min_confidence: Confidence::new(0.8),
            required_evidence: vec![],
            temporal_constraint: None,
            provenance_constraint: None,
        };

        let result = checker.check_constraint(&value_status, &low_requirement);
        assert!(matches!(result, ConstraintResult::Satisfied));
    }

    #[test]
    fn test_epistemic_heterogeneity() {
        let a = EpistemicStatus {
            confidence: Confidence::new(0.9),
            ..Default::default()
        };
        let b = EpistemicStatus {
            confidence: Confidence::new(0.8),
            ..Default::default()
        };

        let het = epistemic_heterogeneity(&a, &b);
        assert!(het >= 0.0 && het <= 1.0);
        assert!(het > 0.0); // Different confidence should produce non-zero heterogeneity
    }

    #[test]
    fn test_bayesian_combination() {
        let statuses = vec![
            EpistemicStatus {
                confidence: Confidence::new(0.9),
                ..Default::default()
            },
            EpistemicStatus {
                confidence: Confidence::new(0.8),
                ..Default::default()
            },
        ];

        let combined = combine_epistemic_bayesian(&statuses);
        // Combined confidence should be between the two
        assert!(combined.confidence.value() > 0.8 && combined.confidence.value() < 0.9);
    }

    #[test]
    fn test_bind_and_lookup() {
        let mut checker = EpistemicChecker::new();
        let ty = checker.type_from_curie("PATO:0000001").unwrap();
        checker.bind("quality".into(), ty);

        let looked_up = checker.lookup("quality");
        assert!(looked_up.is_some());
    }
}
