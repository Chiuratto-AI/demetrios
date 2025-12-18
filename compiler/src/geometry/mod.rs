//! Geometry Symbolic Engine for Demetrios
//!
//! Native neuro-symbolic geometry reasoning inspired by AlphaGeometry.
//! This module provides:
//!
//! - Geometric primitives (Point, Line, Circle) with epistemic semantics
//! - Predicate graph (proof state with confidence propagation)
//! - Forward-chaining deduction (DD) with refinement checking
//! - Algebraic reasoning (AR) with unit validation
//! - Integration with effects system for NeSy loop
//! - Epistemic pruning with uncertainty-based branch control
//! - GeometryReasoning effect for algebraic effect integration
//!
//! # Key Innovation
//!
//! Every predicate is `Knowledge<Predicate>` - deductions automatically
//! propagate confidence and provenance. Rules are refinement-checked
//! (Z3 proves application). Low-confidence branches trigger neural
//! suggestions via effects.

pub mod algebraic;
pub mod engine;
pub mod predicates;
pub mod primitives;
pub mod proof_state;
pub mod reasoning_effect;
pub mod rules;

pub use algebraic::{AlgebraicReasoner, Expression};
pub use engine::{DeductionResult, EngineConfig, EpistemicPruner, PruningDecision, SymbolicEngine};
pub use predicates::{Predicate, PredicateKind, PredicatePattern};
pub use primitives::{Angle, Circle, GeometryPrimitive, Line, Point, Segment};
pub use proof_state::{ProofState, ProofStep, ProvenanceNode};
pub use reasoning_effect::{
    GeometryReasoningHandler, MerkleProof, NeSyHandler, NeuralSuggester, PureSymbolicHandler,
    geometry_reasoning_effect,
};
pub use rules::{GeometryRule, RuleDatabase, RuleMatch};
