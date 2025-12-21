/// epistemic — Demetrios Standard Library for Epistemic Computing
///
/// The world's first programming language with built-in epistemic honesty.
/// Every value knows what it knows and what it doesn't know.
///
/// # Modules
///
/// - `stats` - Revolutionary statistics with Beta posteriors, variance propagation,
///             active inference, and ML integration
/// - `causal` - Causal inference with Pearl's do-calculus and epistemic uncertainty
///
/// # Philosophy
///
/// "It is wrong always, everywhere, and for anyone, to believe anything
///  upon insufficient evidence." — W.K. Clifford
///
/// Demetrios makes this computational: insufficient evidence is tracked
/// as high variance, and computations propagate this uncertainty honestly.
///
/// # Quick Start
///
/// ## Statistics
///
/// ```demetrios
/// use std::epistemic::stats::{Beta, beta_update, beta_summary}
///
/// // Start with uniform prior (complete ignorance)
/// let prior = beta_uniform()
///
/// // Update with evidence: 7 successes, 3 failures
/// let posterior = beta_update(prior, 7.0, 3.0)
///
/// // Get full epistemic summary
/// let summary = beta_summary(posterior)
/// // summary.mean ≈ 0.667
/// // summary.variance ≈ 0.017 (residual ignorance)
/// ```
///
/// ## Causal Inference
///
/// ```demetrios
/// use std::epistemic::causal::{CausalDAG, NodeType, do_intervention}
///
/// // Build causal DAG
/// var dag = dag_new()
/// dag = dag_add_node(dag, "Treatment", NodeType::Treatment)
/// dag = dag_add_node(dag, "Outcome", NodeType::Outcome)
/// dag = dag_add_edge(dag, "Treatment", "Outcome", beta_new(7.0, 3.0), 0.5, 0.08)
///
/// // Estimate causal effect
/// let effect = average_treatment_effect(dag, "Treatment", "Outcome")
/// // effect carries full epistemic uncertainty
/// ```

// Core modules
pub use stats::*
pub use causal::*

// New epistemic computing modules
pub use knowledge::*
pub use propagate::*
pub use meta::*
pub use active::*
pub use merkle::*
pub use linalg::*
pub use ode::*
pub use mcmc::*
pub use pde::*
pub use uplift::*
pub use optimization::*
pub use timeseries::*

// Submodule declarations
pub mod stats
pub mod causal
pub mod knowledge
pub mod propagate
pub mod meta
pub mod active
pub mod merkle
pub mod linalg
pub mod ode
pub mod mcmc
pub mod pde
pub mod uplift
pub mod optimization
pub mod timeseries
