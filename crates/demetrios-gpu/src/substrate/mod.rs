//! Substrate-Aware Epistemic Computing
//!
//! This module implements a novel GPU computing paradigm where the language
//! understands the **physical substrate** of computation, not just abstract types.
//!
//! # The Core Insight
//!
//! Scientific computing is fundamentally about physical reality:
//! - Chemistry: Electrons minimize energy on potential surfaces
//! - Biology: Systems reach thermodynamic equilibrium
//! - Physics: Particles follow least-action paths
//! - Materials: Crystals relax to ground states
//!
//! Demetrios encodes this insight at the language level.
//!
//! # Three Pillars
//!
//! ## 1. Semantic Substrate Types
//! Types that encode what physical quantity they represent, not just their
//! computational representation. A `ChemicalPotential` isn't just `f64` —
//! it carries thermodynamic semantics the compiler understands.
//!
//! ## 2. Epistemic Execution Model
//! Computation guided by what we know, not just what we compute.
//! Uncertainty guides execution: high uncertainty → more samples,
//! low confidence → reduced precision.
//!
//! ## 3. Physical Memory Topology
//! Memory layout mirrors physical reality. Elements near in physical
//! space are near in memory. Neighbor queries become cache hits.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 Substrate-Aware Epistemic Computing              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │  Substrate   │  │  Epistemic   │  │  Physical Memory     │  │
//! │  │    Types     │  │  Execution   │  │     Topology         │  │
//! │  │              │  │              │  │                      │  │
//! │  │ PhysicalQty  │  │ Uncertainty  │  │ SpaceFillingCurve    │  │
//! │  │ Conservation │  │ Confidence   │  │ NeighborIterator     │  │
//! │  │ Symmetry     │  │ Provenance   │  │ SpatialPartition     │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │              Variational Framework                        │  │
//! │  │                                                           │  │
//! │  │  Action Principles → Unified Optimization → GPU Kernels  │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │              Verification Layer                           │  │
//! │  │                                                           │  │
//! │  │  Conservation Laws │ Thermodynamics │ Gauge Invariance   │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod conservation;
pub mod epistemic;
pub mod physical_quantity;
pub mod symmetry;
pub mod thermodynamic;
pub mod topology;
pub mod variational;

// Re-export core types
pub use physical_quantity::{
    DimensionalAnalysis, Dimensions, PhysicalConstraint, PhysicalField, PhysicalQuantity,
    QuantityKind, SubstrateType, TensorField,
};

pub use conservation::{
    AngularMomentumConservation, ChargeConservation, ConservationCheckable, ConservationChecker,
    ConservationLaw, ConservationLawExt, EnergyConservation, MassConservation,
    MomentumConservation,
};

pub use symmetry::{
    Covariant, DiscreteGroup, Equivariant, GaugeTransformation, Invariant, LieGroup,
    SymmetryChecker, SE3, SO3, SU2, SU3, U1,
};

pub use epistemic::{
    AdaptivePrecision, Confidence, Epistemic, EpistemicExecution, Provenance, TemporalValidity,
    UncertaintyGuidedSampling, UncertaintyPropagation,
};

// Rename Epistemic to EpistemicValue for clarity in external API
pub type EpistemicValue<T> = Epistemic<T>;

pub use topology::{
    CellList, HilbertCurve, MortonCurve, NeighborIterator3D, PhysicalArray, PhysicalSpace, Space3D,
    SpaceFillingCurve, SpatialPartition,
};

pub use variational::{
    Action, EulerLagrange, GibbsMinimization, HamiltonJacobi, HamiltonPrinciple, Hamiltonian,
    Lagrangian, OptimizationMethod, RayleighRitz, StationaryResult, VariationalPrinciple,
    VariationalSolver,
};

pub use thermodynamic::{
    Ensemble, EquilibriumFinder, FreeEnergy, SecondLawChecker, ThermodynamicProcess,
    ThermodynamicState,
};
