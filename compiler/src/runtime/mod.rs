//! Runtime Support for Demetrios
//!
//! This module provides runtime representations and support code for
//! Demetrios programs. Key features:
//!
//! - Epistemic type runtime representations (Full/Compact/Erased modes)
//! - GPU memory layouts for vectorized epistemic operations
//! - Runtime support for confidence tracking
//! - Provenance chain management

pub mod epistemic;
pub mod gpu_epistemic;

pub use epistemic::{
    CompactKnowledge, EpistemicMode, EpistemicRuntime, ErasedKnowledge, FullKnowledge,
    RuntimeConfidence, RuntimeProvenance,
};
pub use gpu_epistemic::{AoSKnowledge, GpuEpistemicArray, GpuMemoryLayout, SoAKnowledge};
