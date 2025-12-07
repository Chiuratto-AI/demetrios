//! Deep Computational Foundations
//!
//! This module implements the deepest layers of computational theory:
//!
//! # The Fundamental Insight
//!
//! **Computation is physics. Physics is information. Information is reality.**
//!
//! Demetrios treats these three domains as a unified whole, not separate concerns.
//!
//! # Layers
//!
//! ## 1. Information-Theoretic Layer (`information`)
//! - Kolmogorov complexity as intrinsic type property
//! - Shannon entropy tracking through computation
//! - Mutual information for dependency analysis
//! - Algorithmic Information Dynamics for causal discovery
//!
//! ## 2. Causal Structure Layer (`causality`)
//! - Causal sets as discrete spacetime
//! - Light cone constraints on parallelism
//! - Causal consistency for distributed systems
//! - Interventions and counterfactuals
//!
//! ## 3. Reversible Computation Layer (`reversible`)
//! - Landauer's principle: irreversibility has thermodynamic cost
//! - Bijective functions as zero-energy primitives
//! - Uncomputation for garbage-free memory
//! - Bennett's reversible Turing machines
//!
//! ## 4. Category-Theoretic Layer (`category`)
//! - Topoi as computational universes
//! - Functors as structure-preserving transformations
//! - Natural transformations as polymorphism
//! - Limits/colimits as universal constructions
//!
//! ## 5. Emergence Layer (`emergence`)
//! - Self-organized criticality
//! - Renormalization group for hierarchical abstraction
//! - Computational irreducibility detection
//! - Phase transitions in algorithms
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DEEP COMPUTATIONAL LAYER                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │ Information  │  │   Causal     │  │    Reversible        │  │
//! │  │   Theory     │  │  Structure   │  │    Computing         │  │
//! │  │              │  │              │  │                      │  │
//! │  │ Entropy      │  │ CausalSet    │  │ Bijection           │  │
//! │  │ Complexity   │  │ LightCone    │  │ Uncompute           │  │
//! │  │ MutualInfo   │  │ Intervention │  │ ReversibleOp        │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │              Category-Theoretic Foundation                │  │
//! │  │                                                           │  │
//! │  │  Topos → Functor → NatTrans → Limit → Universal          │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │                  Emergence Layer                          │  │
//! │  │                                                           │  │
//! │  │  SelfOrganization │ Criticality │ RenormalizationGroup   │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod category;
pub mod causality;
pub mod emergence;
pub mod information;
pub mod reversible;

// Re-export core types
pub use information::{
    AlgorithmicRandomness, CompressionBound, Entropy, InformationContent, InformationFlow,
    KolmogorovComplexity, MutualInformation,
};

pub use causality::{
    CausalGraph, CausalRelation, CausalSet, Counterfactual, DoCalculus, Intervention, LightCone,
    SpacetimeEvent,
};

pub use reversible::{
    AdiabaticComputation, Bijection, LandauerBound, Reversible, ReversibleOp, ThermodynamicCost,
    Uncomputation,
};

pub use category::{
    Adjunction, Category, Colimit, Comonad, Functor, Limit, Monad, NaturalTransformation, Topos,
};

pub use emergence::{
    Criticality, Irreducibility, OrderParameter, PhaseTransition, RenormalizationGroup, Scale,
    SelfOrganization,
};
