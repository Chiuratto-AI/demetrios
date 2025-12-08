// tests/e2e/mod.rs — Main end-to-end test module
//
// This module organizes all end-to-end integration tests for the
// Demetrios compiler. Tests are organized by category:
//
// - common: Test harness and utilities
// - pharmacology: Real pharmacology scenario tests
// - cross_ontology: Cross-ontology alignment tests
// - diagnostics: Error message quality tests
// - edge_cases: Boundary condition tests
// - performance: Scalability and timing tests
// - golden: Snapshot comparison tests

pub mod common;

mod cross_ontology;
mod diagnostics;
mod edge_cases;
mod golden;
mod performance;
mod pharmacology;

// Re-export test harness for use in tests
pub use common::{CompileResult, TestHarness};
