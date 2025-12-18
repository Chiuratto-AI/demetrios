//! Quantum Machine Learning Module for Demetrios
//!
//! Native integration of quantum computing with epistemic semantics:
//! - Epistemic quantum states (Knowledge<QubitState> with noise-aware variance)
//! - Differentiable quantum circuits with gradient tracking
//! - VQE/QAOA with full posterior energy estimation
//! - GPU kernels for parallel quantum trials
//! - Refinement types for unitarity and no-cloning
//!
//! # Key Innovation
//!
//! Every quantum measurement is `Knowledge<T>` - noise and decoherence
//! automatically propagate as epistemic variance. This enables:
//! - "How confident am I in this quantum advantage?"
//! - Variance penalty in VQE to encourage stable circuits
//! - Provenance tracking for quantum chemistry audits
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Epistemic Quantum ML                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
//! │  │ Qubit State │───►│  Circuit    │───►│ Measurement │         │
//! │  │ Knowledge   │    │ Execution   │    │ Knowledge   │         │
//! │  │ (amp+noise) │    │ (gates)     │    │ (value+var) │         │
//! │  └─────────────┘    └─────────────┘    └─────────────┘         │
//! │         │                  │                  │                 │
//! │         ▼                  ▼                  ▼                 │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │           Epistemic Variance Propagation            │       │
//! │  │   (noise model + gate errors + measurement shots)   │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! │                            │                                   │
//! │                            ▼                                   │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              VQE / QAOA Optimization                │       │
//! │  │   Loss = Energy + λ * Variance (epistemic penalty)  │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod circuit;
pub mod gates;
pub mod noise;
pub mod states;
pub mod vqe;

pub use circuit::{CircuitBuilder, CircuitStats, QuantumCircuit};
pub use gates::{Gate, GateType, ParametricGate};
pub use noise::{AmplitudeDamping, DepolarizingNoise, NoiseModel, NoiseType};
pub use states::{DensityMatrix, EpistemicQubit, QubitState, StateVector};
pub use vqe::{Hamiltonian, PauliTerm, VQEConfig, VQEResult, VQESolver};
