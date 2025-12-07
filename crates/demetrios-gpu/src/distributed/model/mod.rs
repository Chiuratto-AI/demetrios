//! Performance Modeling for Distributed GPU Computing
//!
//! This module provides analytical performance models for predicting and
//! optimizing distributed GPU collective operations.
//!
//! # Models Implemented
//!
//! ## α-β Model (Hockney)
//!
//! The fundamental model for point-to-point communication:
//!
//! ```text
//! T(n) = α + β·n
//!
//! where:
//!   α = latency (seconds)
//!   β = 1/bandwidth (seconds/byte)
//!   n = message size (bytes)
//! ```
//!
//! ## LogP Model
//!
//! Extended model capturing network characteristics:
//!
//! ```text
//! L = latency (time for small message)
//! o = overhead (CPU time per message)
//! g = gap (minimum time between messages)
//! P = number of processors
//! ```
//!
//! ## BSP Model (Bulk Synchronous Parallel)
//!
//! Superstep-based execution model:
//!
//! ```text
//! T = Σᵢ (wᵢ + hᵢ·g + l)
//!
//! where:
//!   wᵢ = computation in superstep i
//!   hᵢ = maximum messages sent/received
//!   g = cost per message
//!   l = barrier synchronization cost
//! ```
//!
//! # Roofline Model
//!
//! Identifies performance bottlenecks:
//!
//! ```text
//! Achievable FLOPS = min(Peak FLOPS, Arithmetic Intensity × Bandwidth)
//!
//! Bottleneck identification:
//! - Compute-bound: AI > Ridge Point
//! - Memory-bound: AI < Ridge Point
//! - Network-bound: Communication > Computation
//! ```
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::model::{
//!     NetworkParams, CollectiveModel, AllReduceAlgorithm,
//!     DistributedRoofline,
//! };
//!
//! // Create network parameters for NVLink
//! let params = NetworkParams::nvlink4();
//!
//! // Model collective performance
//! let model = CollectiveModel::new(params, 8); // 8 GPUs
//!
//! // Predict all-reduce time
//! let time = model.allreduce_time(1024 * 1024, AllReduceAlgorithm::Ring);
//!
//! // Find optimal algorithm
//! let best = model.best_allreduce_algorithm(1024 * 1024);
//!
//! // Roofline analysis
//! let roofline = DistributedRoofline::h100();
//! let bottleneck = roofline.identify_bottleneck(100.0, 1e12, 1e11);
//! ```

pub mod performance;

// Re-export performance modeling types
pub use performance::{
    AllReduceAlgorithm, Bottleneck, BroadcastAlgorithm, CollectiveModel, DistributedRoofline,
    NetworkParams, ReduceAlgorithm, StrongScalingAnalysis, WeakScalingAnalysis,
};

/// Lower bounds for collective operations (information-theoretic)
pub mod lower_bounds {
    /// Bandwidth lower bound for all-reduce
    ///
    /// All-reduce must move at least 2(n-1)/n × data bytes
    /// This approaches 2×data as n→∞
    pub fn allreduce_bandwidth(num_gpus: usize, data_bytes: usize) -> usize {
        2 * (num_gpus - 1) * data_bytes / num_gpus
    }

    /// Latency lower bound for all-reduce
    ///
    /// Minimum latency is 2⌈log₂n⌉ messages (recursive halving-doubling)
    pub fn allreduce_latency(num_gpus: usize) -> usize {
        2 * (num_gpus as f64).log2().ceil() as usize
    }

    /// Bandwidth lower bound for broadcast
    ///
    /// Broadcast must deliver data to n-1 recipients
    /// Optimal: each recipient receives exactly once
    pub fn broadcast_bandwidth(data_bytes: usize) -> usize {
        data_bytes
    }

    /// Latency lower bound for broadcast
    ///
    /// Tree broadcast achieves ⌈log₂n⌉ steps
    pub fn broadcast_latency(num_gpus: usize) -> usize {
        (num_gpus as f64).log2().ceil() as usize
    }

    /// Bandwidth lower bound for all-gather
    ///
    /// Each GPU contributes data/n, result is full data on all GPUs
    pub fn allgather_bandwidth(num_gpus: usize, data_bytes: usize) -> usize {
        (num_gpus - 1) * data_bytes / num_gpus
    }

    /// Bandwidth lower bound for reduce-scatter
    ///
    /// Similar to all-gather in reverse
    pub fn reduce_scatter_bandwidth(num_gpus: usize, data_bytes: usize) -> usize {
        (num_gpus - 1) * data_bytes / num_gpus
    }
}

/// Scaling law constants and formulas
pub mod scaling {
    /// Amdahl's Law: Maximum speedup with parallel fraction p and n processors
    ///
    /// S(n) = 1 / ((1-p) + p/n)
    pub fn amdahl_speedup(parallel_fraction: f64, num_processors: usize) -> f64 {
        let p = parallel_fraction;
        let n = num_processors as f64;
        1.0 / ((1.0 - p) + p / n)
    }

    /// Maximum speedup achievable (n→∞)
    pub fn amdahl_limit(parallel_fraction: f64) -> f64 {
        1.0 / (1.0 - parallel_fraction)
    }

    /// Gustafson's Law: Scaled speedup
    ///
    /// S(n) = n - (1-p)(n-1)
    pub fn gustafson_speedup(parallel_fraction: f64, num_processors: usize) -> f64 {
        let p = parallel_fraction;
        let n = num_processors as f64;
        n - (1.0 - p) * (n - 1.0)
    }

    /// Parallel efficiency
    ///
    /// E(n) = S(n) / n
    pub fn parallel_efficiency(speedup: f64, num_processors: usize) -> f64 {
        speedup / num_processors as f64
    }

    /// Iso-efficiency function: work needed to maintain efficiency
    ///
    /// For overhead function f(n,W), iso-efficiency requires:
    /// W ≥ f(n,W) / (E_target - E_achieved)
    pub fn iso_efficiency_work(
        overhead: f64,
        target_efficiency: f64,
        achieved_efficiency: f64,
    ) -> f64 {
        if target_efficiency <= achieved_efficiency {
            0.0
        } else {
            overhead / (target_efficiency - achieved_efficiency)
        }
    }
}

/// Communication patterns and their costs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationPattern {
    /// One-to-all broadcast
    Broadcast,
    /// All-to-one reduction
    Reduce,
    /// All-to-all with reduction
    AllReduce,
    /// All-to-all gather
    AllGather,
    /// Scatter from root
    Scatter,
    /// Reduce and scatter result
    ReduceScatter,
    /// All-to-all personalized exchange
    AllToAll,
    /// Barrier synchronization
    Barrier,
}

impl CommunicationPattern {
    /// Get the optimal algorithm complexity in terms of α (latency) and β (bandwidth)
    ///
    /// Returns (latency_factor, bandwidth_factor) where:
    /// - Total time = latency_factor × α + bandwidth_factor × β × n
    pub fn optimal_complexity(&self, num_gpus: usize) -> (f64, f64) {
        let p = num_gpus as f64;
        let log_p = p.log2().ceil();

        match self {
            // Broadcast: log(p) latency, 1× bandwidth
            Self::Broadcast => (log_p, 1.0),
            // Reduce: log(p) latency, 1× bandwidth
            Self::Reduce => (log_p, 1.0),
            // All-reduce: 2log(p) latency, 2(p-1)/p bandwidth
            Self::AllReduce => (2.0 * log_p, 2.0 * (p - 1.0) / p),
            // All-gather: (p-1) latency for ring, (p-1)/p bandwidth
            Self::AllGather => (p - 1.0, (p - 1.0) / p),
            // Scatter: log(p) latency, (p-1)/p bandwidth
            Self::Scatter => (log_p, (p - 1.0) / p),
            // Reduce-scatter: log(p) latency, (p-1)/p bandwidth
            Self::ReduceScatter => (log_p, (p - 1.0) / p),
            // All-to-all: (p-1) latency, (p-1) bandwidth (each sends to all others)
            Self::AllToAll => (p - 1.0, p - 1.0),
            // Barrier: log(p) latency, 0 bandwidth
            Self::Barrier => (log_p, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allreduce_lower_bound() {
        // 8 GPUs, 1024 bytes
        let bound = lower_bounds::allreduce_bandwidth(8, 1024);
        // 2 * 7 * 1024 / 8 = 1792
        assert_eq!(bound, 1792);
    }

    #[test]
    fn test_allreduce_latency_bound() {
        // 8 GPUs: 2 * ceil(log2(8)) = 2 * 3 = 6
        assert_eq!(lower_bounds::allreduce_latency(8), 6);
        // 16 GPUs: 2 * ceil(log2(16)) = 2 * 4 = 8
        assert_eq!(lower_bounds::allreduce_latency(16), 8);
    }

    #[test]
    fn test_amdahl_speedup() {
        // 90% parallel, 4 processors
        let speedup = scaling::amdahl_speedup(0.9, 4);
        // S = 1 / (0.1 + 0.9/4) = 1 / (0.1 + 0.225) = 1 / 0.325 ≈ 3.08
        assert!((speedup - 3.08).abs() < 0.1);

        // Maximum speedup
        let limit = scaling::amdahl_limit(0.9);
        // 1 / 0.1 = 10
        assert!((limit - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_gustafson_speedup() {
        // 90% parallel, 4 processors
        let speedup = scaling::gustafson_speedup(0.9, 4);
        // S = 4 - 0.1 * 3 = 4 - 0.3 = 3.7
        assert!((speedup - 3.7).abs() < 0.001);
    }

    #[test]
    fn test_parallel_efficiency() {
        let eff = scaling::parallel_efficiency(3.2, 4);
        // 3.2 / 4 = 0.8 = 80%
        assert!((eff - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_communication_complexity() {
        // All-reduce with 8 GPUs
        let (lat, bw) = CommunicationPattern::AllReduce.optimal_complexity(8);
        // Latency: 2 * log2(8) = 6
        assert!((lat - 6.0).abs() < 0.001);
        // Bandwidth: 2 * 7/8 = 1.75
        assert!((bw - 1.75).abs() < 0.001);
    }

    #[test]
    fn test_barrier_complexity() {
        let (lat, bw) = CommunicationPattern::Barrier.optimal_complexity(8);
        // Barrier has log(p) latency, 0 bandwidth
        assert!((lat - 3.0).abs() < 0.001);
        assert!((bw - 0.0).abs() < 0.001);
    }
}
