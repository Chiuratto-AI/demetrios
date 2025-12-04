//! Tree-walking interpreter for HIR
//!
//! Executes HIR directly for rapid semantic testing.

pub mod env;
pub mod eval;
pub mod value;

pub use env::Environment;
pub use eval::Interpreter;
pub use value::Value;
