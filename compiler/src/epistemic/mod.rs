//! Epistemic type system for Demetrios
//!
//! This module implements Knowledge as a first-class type with:
//! - Temporal indexing (τ) for context-dependent typing
//! - Epistemic status (ε) for confidence and revisability tracking
//! - Domain binding (δ) for ontology-validated types
//! - Functor trace (Φ) for complete provenance
//!
//! # The Paradigm Shift
//!
//! Traditional languages: Types are syntactic constraints
//! Demetrios: Types are ontological assertions about reality
//!
//! Every value in Demetrios carries its epistemic history.
//!
//! # Example
//!
//! ```demetrios
//! let result: Knowledge[
//!     content = f64,
//!     τ = (2024, Lab, Experiment),
//!     ε = (confidence: 0.95, source: Measurement),
//!     δ = PATO:mass,
//!     Φ = [sensor → calibration → conversion]
//! ] = measure_mass(sample);
//! ```
//!
//! # Knowledge Type Structure
//!
//! ```text
//! Knowledge[τ, ε, δ, Φ]
//! │        │  │  │  └── Φ: Functor trace (transformation provenance)
//! │        │  │  └───── δ: Domain ontology (which ontology validates this)
//! │        │  └──────── ε: Epistemic status (confidence, revisability, source)
//! │        └─────────── τ: Context-time (temporal indexing for type evolution)
//! └──────────────────── Knowledge: First-class epistemic primitive
//! ```

pub mod agents;
pub mod composition;
pub mod confidence;
pub mod evolution;
pub mod heterogeneity;
pub mod knowledge;
pub mod operations;
pub mod provenance;
pub mod temporal;

pub use confidence::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};
pub use heterogeneity::{
    HeterogeneityConfig, HeterogeneityResolver, ResolutionResult, ResolutionStrategy,
};
pub use knowledge::{
    CompatibilityResult, DomainOntology, FederatedRef, FoundationOntology, IncompatibilityReason,
    Knowledge, KnowledgeType, KnownIndices, OntologyBinding, OntologyConstraint, OntologyRef,
    PrimitiveOntology, QuantifiedIndices, TermId, TranslationPath, TranslationStep,
};
pub use operations::{
    EpistemicConstraint, InspectField, InspectOp, KnowledgeOp, MergeOp, MergeStrategy, QueryOp,
    RelationalConstraint, ReviseOp, RevisionStrategy, TranslateOp, TranslateOptions,
    assert_knowledge, query_knowledge, revise_knowledge, translate_knowledge,
};
pub use provenance::{
    FunctorTrace, Origin, Provenance, Transformation, TransformationKind, TransformationMetadata,
};
pub use temporal::{ContextIndex, ContextTime, TemporalIndex, TemporalOffset, ValidityBounds};
