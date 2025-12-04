//! Type system for the Demetrios language
//!
//! This module implements D's advanced type system including:
//! - Core types (primitives, references, functions)
//! - Ownership and borrowing (linear, affine types)
//! - Algebraic effects
//! - Refinement types
//! - Units of measure with inference

pub mod core;
pub mod effects;
pub mod ownership;
pub mod refinement;
pub mod unit_infer;
pub mod units;

pub use self::core::*;
pub use effects::*;
pub use ownership::*;

// Don't use glob re-export for these to avoid ambiguous `medical` module conflict
pub use refinement::{
    ArithOp, CompareOp, Predicate, RefinedType, RefinementChecker, RefinementResult,
};
pub use unit_infer::{UnitExpr, UnitInference, UnitInferenceError, UnitVar};
pub use units::{Unit, UnitChecker, UnitError, UnitOp};
