//! Causal Primitives: Do-Calculus as L0 Operator
//!
//! Day 34 of the Demetrios compiler implements Pearl's causal hierarchy:
//!
//! # Ladder of Causation
//!
//! ```text
//! Level 1: ASSOCIATION (Seeing)
//!   Query: P(Y | X)
//!   Type: Knowledge[τ, ε, δ, Φ, t]
//!
//! Level 2: INTERVENTION (Doing)
//!   Query: P(Y | do(X))
//!   Type: CausalKnowledge[τ, ε, δ, Φ, t, G]
//!
//! Level 3: COUNTERFACTUAL (Imagining)
//!   Query: P(Y_x | X=x', Y=y)
//!   Type: StructuralKnowledge[τ, ε, δ, Φ, t, M]
//! ```
//!
//! # Key Components
//!
//! - [`CausalGraph`]: Directed acyclic graph with d-separation
//! - [`CausalKnowledge`]: Knowledge with causal structure (Level 2)
//! - [`StructuralKnowledge`]: Full structural causal model (Level 3)
//! - [`Intervention`]: The do() operator
//! - [`CounterfactualResult`]: Results of counterfactual queries
//!
//! # Example
//!
//! ```ignore
//! use demetrios::causal::*;
//!
//! // Build causal graph
//! let mut graph = CausalGraph::new();
//! graph.add_node(CausalNode::treatment("Dose"));
//! graph.add_node(CausalNode::outcome("Effect"));
//! graph.add_edge("Dose", "Effect", EdgeType::Direct)?;
//!
//! // Create causal knowledge
//! let causal = CausalKnowledge::new(knowledge, graph, "Effect", vec!["Dose"]);
//!
//! // Compute causal effect
//! let result = causal.do_intervention(Intervention::atomic("Dose", 100.0))?;
//! ```

pub mod composition;
pub mod counterfactual;
pub mod graph;
pub mod identification;
pub mod intervention;
pub mod knowledge;
pub mod structural;

// Re-exports
pub use composition::*;
pub use counterfactual::*;
pub use graph::*;
pub use identification::*;
pub use intervention::*;
pub use knowledge::*;
pub use structural::*;
