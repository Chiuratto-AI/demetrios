//! Scientific domain-specific macros
//!
//! Provides compile-time code generation for:
//! - Dimensional analysis (units of measure)
//! - Automatic differentiation
//! - Linear algebra DSL
//! - Statistical modeling

pub mod units;
pub mod autodiff;

pub use units::*;
pub use autodiff::*;
