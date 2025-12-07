//! Formal Verification of Collective Algorithms
//!
//! This module provides formal verification capabilities for distributed GPU
//! collective operations, ensuring correctness properties:
//!
//! - **Safety**: No deadlocks, no data races
//! - **Liveness**: All processes eventually complete
//! - **Correctness**: Results match sequential specification
//!
//! # Verification Approach
//!
//! We model collective algorithms as state machines and explore the state space
//! to verify properties. For ring all-reduce:
//!
//! ```text
//! State = (phase[0..n-1], in_flight_messages, local_data[0..n-1])
//!
//! Invariants:
//! 1. At any time, exactly one message in flight per ring direction
//! 2. All processes in same phase ± 1
//! 3. No process waits for message that will never be sent
//! ```
//!
//! # Theorems
//!
//! The module provides formal proofs for key algorithms:
//!
//! - **Ring All-Reduce**: Correct in 2(n-1) steps
//! - **Recursive Halving-Doubling**: Correct in 2⌈log₂n⌉ steps
//! - **Dissemination Barrier**: All processes sync in ⌈log₂n⌉ rounds
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::verify::{
//!     CollectiveVerifier, ReduceOp, VerificationResult,
//! };
//!
//! let verifier = CollectiveVerifier::new(4); // 4 processes
//!
//! // Verify ring all-reduce
//! let result = verifier.verify_ring_allreduce(ReduceOp::Sum);
//!
//! assert!(result.is_correct);
//! assert!(result.deadlock_free);
//! assert!(result.deterministic);
//! ```

pub mod correctness;

// Re-export verification types
pub use correctness::{
    AbstractValue, CollectiveVerifier, InFlightMessage, Phase, ProcessState, ReduceOp, SystemState,
    VerificationResult, WaitCondition,
};

/// Properties that can be verified
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiableProperty {
    /// No deadlock can occur
    DeadlockFreedom,
    /// All processes eventually terminate
    Termination,
    /// Result is deterministic regardless of message ordering
    Determinism,
    /// Result matches sequential specification
    Correctness,
    /// No data races occur
    DataRaceFreedom,
    /// Progress is guaranteed under fair scheduling
    ProgressUnderFairness,
}

/// Collective algorithm that can be verified
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiableAlgorithm {
    /// Ring-based all-reduce
    RingAllReduce,
    /// Recursive halving-doubling all-reduce
    RecursiveHalvingDoubling,
    /// Tree-based broadcast
    TreeBroadcast,
    /// Ring all-gather
    RingAllGather,
    /// Reduce-scatter
    ReduceScatter,
    /// Dissemination barrier
    DisseminationBarrier,
    /// Pairwise exchange
    PairwiseExchange,
}

impl VerifiableAlgorithm {
    /// Get the optimal step count for this algorithm
    pub fn optimal_steps(&self, n: usize) -> usize {
        match self {
            Self::RingAllReduce => 2 * (n - 1),
            Self::RecursiveHalvingDoubling => 2 * (n as f64).log2().ceil() as usize,
            Self::TreeBroadcast => (n as f64).log2().ceil() as usize,
            Self::RingAllGather => n - 1,
            Self::ReduceScatter => n - 1,
            Self::DisseminationBarrier => (n as f64).log2().ceil() as usize,
            Self::PairwiseExchange => 1,
        }
    }

    /// Get the lower bound on communication volume
    pub fn volume_lower_bound(&self, n: usize, data_size: usize) -> usize {
        match self {
            // All-reduce: 2(n-1)/n * data ≈ 2 * data for large n
            Self::RingAllReduce | Self::RecursiveHalvingDoubling => 2 * (n - 1) * data_size / n,
            // Broadcast: data must reach all n-1 other processes
            Self::TreeBroadcast => data_size,
            // All-gather: each process contributes data/n, result is data
            Self::RingAllGather => (n - 1) * data_size / n,
            // Reduce-scatter: similar to all-gather
            Self::ReduceScatter => (n - 1) * data_size / n,
            // Barrier: only control messages
            Self::DisseminationBarrier => 0,
            // Exchange: bidirectional data
            Self::PairwiseExchange => data_size,
        }
    }
}

/// Summary of verification results across multiple properties
#[derive(Debug)]
pub struct VerificationSummary {
    /// Algorithm verified
    pub algorithm: VerifiableAlgorithm,
    /// Number of processes
    pub num_processes: usize,
    /// Properties verified and their results
    pub properties: Vec<(VerifiableProperty, bool)>,
    /// Total states explored
    pub states_explored: usize,
    /// Maximum state space depth
    pub max_depth: usize,
}

impl VerificationSummary {
    /// Check if all properties passed
    pub fn all_passed(&self) -> bool {
        self.properties.iter().all(|(_, passed)| *passed)
    }

    /// Get failed properties
    pub fn failed_properties(&self) -> Vec<VerifiableProperty> {
        self.properties
            .iter()
            .filter_map(|(prop, passed)| if !passed { Some(*prop) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_steps() {
        // Ring all-reduce with 4 processes: 2(4-1) = 6 steps
        assert_eq!(VerifiableAlgorithm::RingAllReduce.optimal_steps(4), 6);

        // Recursive halving-doubling with 8 processes: 2*log2(8) = 6 steps
        assert_eq!(
            VerifiableAlgorithm::RecursiveHalvingDoubling.optimal_steps(8),
            6
        );

        // Tree broadcast with 8 processes: log2(8) = 3 steps
        assert_eq!(VerifiableAlgorithm::TreeBroadcast.optimal_steps(8), 3);
    }

    #[test]
    fn test_volume_lower_bound() {
        let n = 4;
        let data_size = 1024;

        // Ring all-reduce: 2(n-1)/n * data = 2*3/4*1024 = 1536
        let volume = VerifiableAlgorithm::RingAllReduce.volume_lower_bound(n, data_size);
        assert_eq!(volume, 1536);
    }

    #[test]
    fn test_verification_summary() {
        let summary = VerificationSummary {
            algorithm: VerifiableAlgorithm::RingAllReduce,
            num_processes: 4,
            properties: vec![
                (VerifiableProperty::DeadlockFreedom, true),
                (VerifiableProperty::Correctness, true),
                (VerifiableProperty::Determinism, false),
            ],
            states_explored: 100,
            max_depth: 10,
        };

        assert!(!summary.all_passed());
        assert_eq!(summary.failed_properties().len(), 1);
        assert_eq!(
            summary.failed_properties()[0],
            VerifiableProperty::Determinism
        );
    }

    #[test]
    fn test_verifiable_properties() {
        let props = [
            VerifiableProperty::DeadlockFreedom,
            VerifiableProperty::Termination,
            VerifiableProperty::Determinism,
            VerifiableProperty::Correctness,
            VerifiableProperty::DataRaceFreedom,
            VerifiableProperty::ProgressUnderFairness,
        ];

        // All properties should be distinct
        for i in 0..props.len() {
            for j in (i + 1)..props.len() {
                assert_ne!(props[i], props[j]);
            }
        }
    }
}
