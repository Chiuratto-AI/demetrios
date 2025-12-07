//! Performance Modeling for Collective Operations
//!
//! Predict execution time based on:
//! - Message size
//! - Network topology
//! - Algorithm choice
//! - System characteristics

/// Network parameters for alpha-beta model
///
/// T(n) = alpha + beta * n
/// - alpha = latency (startup cost)
/// - beta = inverse bandwidth (time per byte)
#[derive(Debug, Clone, Copy)]
pub struct NetworkParams {
    /// Startup latency (seconds)
    pub alpha: f64,
    /// Inverse bandwidth (seconds/byte)
    pub beta: f64,
    /// CPU overhead for send (seconds)
    pub overhead_send: f64,
    /// CPU overhead for receive (seconds)
    pub overhead_recv: f64,
}

impl NetworkParams {
    /// Create custom network parameters
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            overhead_send: alpha * 0.5,
            overhead_recv: alpha * 0.5,
        }
    }

    /// NVLink 3.0 (A100) - ~300 GB/s bidirectional
    pub fn nvlink3() -> Self {
        Self {
            alpha: 1.0e-6,       // 1 us
            beta: 1.0 / 300.0e9, // 300 GB/s
            overhead_send: 0.5e-6,
            overhead_recv: 0.5e-6,
        }
    }

    /// NVLink 4.0 (H100) - ~450 GB/s bidirectional
    pub fn nvlink4() -> Self {
        Self {
            alpha: 0.8e-6,
            beta: 1.0 / 450.0e9,
            overhead_send: 0.4e-6,
            overhead_recv: 0.4e-6,
        }
    }

    /// PCIe 4.0 x16 - ~32 GB/s
    pub fn pcie4() -> Self {
        Self {
            alpha: 5.0e-6,      // 5 us
            beta: 1.0 / 32.0e9, // 32 GB/s
            overhead_send: 2.0e-6,
            overhead_recv: 2.0e-6,
        }
    }

    /// PCIe 5.0 x16 - ~64 GB/s
    pub fn pcie5() -> Self {
        Self {
            alpha: 4.0e-6,
            beta: 1.0 / 64.0e9,
            overhead_send: 1.5e-6,
            overhead_recv: 1.5e-6,
        }
    }

    /// InfiniBand HDR - ~25 GB/s per direction
    pub fn ib_hdr() -> Self {
        Self {
            alpha: 1.0e-6,
            beta: 1.0 / 25.0e9,
            overhead_send: 0.5e-6,
            overhead_recv: 0.5e-6,
        }
    }

    /// InfiniBand NDR - ~50 GB/s per direction
    pub fn ib_ndr() -> Self {
        Self {
            alpha: 0.8e-6,
            beta: 1.0 / 50.0e9,
            overhead_send: 0.4e-6,
            overhead_recv: 0.4e-6,
        }
    }

    /// Ethernet 100G
    pub fn ethernet_100g() -> Self {
        Self {
            alpha: 10.0e-6,
            beta: 1.0 / 12.5e9, // ~12.5 GB/s
            overhead_send: 5.0e-6,
            overhead_recv: 5.0e-6,
        }
    }

    /// Time to send message of given size
    pub fn send_time(&self, bytes: usize) -> f64 {
        self.overhead_send + self.alpha + self.beta * bytes as f64
    }

    /// Effective bandwidth for message size
    pub fn effective_bandwidth(&self, bytes: usize) -> f64 {
        bytes as f64 / self.send_time(bytes)
    }

    /// Crossover point where latency = bandwidth cost
    pub fn crossover_bytes(&self) -> usize {
        (self.alpha / self.beta) as usize
    }

    /// Peak bandwidth (bytes/sec)
    pub fn peak_bandwidth(&self) -> f64 {
        1.0 / self.beta
    }
}

impl Default for NetworkParams {
    fn default() -> Self {
        Self::nvlink3()
    }
}

/// Broadcast algorithm choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastAlgorithm {
    /// Root sends to all directly (n-1 messages)
    Flat,
    /// Binary tree (log n steps)
    BinomialTree,
    /// Pipelined tree (overlapped)
    PipelinedTree,
    /// Scatter then allgather
    ScatterAllgather,
}

/// Reduce algorithm choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceAlgorithm {
    /// All send to root
    Flat,
    /// Binary tree reduction
    BinomialTree,
    /// Reduce-scatter + gather
    Rabenseifner,
}

/// All-reduce algorithm choices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllReduceAlgorithm {
    /// Reduce to root + broadcast
    ReduceBroadcast,
    /// Ring (bandwidth optimal)
    Ring,
    /// Recursive halving-doubling (latency optimal)
    RecursiveHalvingDoubling,
    /// Rabenseifner (reduce-scatter + allgather)
    Rabenseifner,
    /// Double binary tree
    DoubleBinaryTree,
}

impl AllReduceAlgorithm {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ReduceBroadcast => "Reduce to root then broadcast",
            Self::Ring => "Ring all-reduce (bandwidth optimal for large messages)",
            Self::RecursiveHalvingDoubling => "Recursive halving-doubling (latency optimal)",
            Self::Rabenseifner => "Rabenseifner (reduce-scatter + allgather)",
            Self::DoubleBinaryTree => "Double binary tree",
        }
    }
}

/// Performance model for collective operations
pub struct CollectiveModel {
    /// Network parameters
    network: NetworkParams,
    /// Number of processes
    num_procs: usize,
}

impl CollectiveModel {
    pub fn new(network: NetworkParams, num_procs: usize) -> Self {
        Self { network, num_procs }
    }

    /// Get network parameters
    pub fn network(&self) -> &NetworkParams {
        &self.network
    }

    /// Get number of processes
    pub fn num_procs(&self) -> usize {
        self.num_procs
    }

    /// Model time for broadcast
    pub fn broadcast_time(&self, bytes: usize, algorithm: BroadcastAlgorithm) -> f64 {
        let n = self.num_procs;
        let alpha = self.network.alpha;
        let beta = self.network.beta;
        let m = bytes as f64;

        match algorithm {
            BroadcastAlgorithm::Flat => {
                // Root sends to all: (n-1) * (alpha + beta*m)
                (n - 1) as f64 * (alpha + beta * m)
            }
            BroadcastAlgorithm::BinomialTree => {
                // log(n) steps, each sends full message
                let log_n = (n as f64).log2().ceil();
                log_n * (alpha + beta * m)
            }
            BroadcastAlgorithm::PipelinedTree => {
                // Pipelined: log(n) * alpha + (n-1)/n * beta * m
                let log_n = (n as f64).log2().ceil();
                log_n * alpha + (n - 1) as f64 / n as f64 * beta * m
            }
            BroadcastAlgorithm::ScatterAllgather => {
                // Scatter: alpha + beta*m/n, Allgather: log(n)*alpha + beta*m*(n-1)/n
                let scatter = alpha + beta * m / n as f64;
                let allgather =
                    (n as f64).log2().ceil() * alpha + beta * m * (n - 1) as f64 / n as f64;
                scatter + allgather
            }
        }
    }

    /// Model time for reduce
    pub fn reduce_time(&self, bytes: usize, algorithm: ReduceAlgorithm) -> f64 {
        // Reduce is symmetric to broadcast
        self.broadcast_time(
            bytes,
            match algorithm {
                ReduceAlgorithm::Flat => BroadcastAlgorithm::Flat,
                ReduceAlgorithm::BinomialTree => BroadcastAlgorithm::BinomialTree,
                ReduceAlgorithm::Rabenseifner => BroadcastAlgorithm::ScatterAllgather,
            },
        )
    }

    /// Model time for all-reduce
    pub fn allreduce_time(&self, bytes: usize, algorithm: AllReduceAlgorithm) -> f64 {
        let n = self.num_procs;
        let alpha = self.network.alpha;
        let beta = self.network.beta;
        let m = bytes as f64;

        match algorithm {
            AllReduceAlgorithm::ReduceBroadcast => {
                // Reduce to root + broadcast
                2.0 * (n as f64).log2().ceil() * (alpha + beta * m)
            }
            AllReduceAlgorithm::Ring => {
                // 2(n-1) steps, each sends m/n bytes
                2.0 * (n - 1) as f64 * (alpha + beta * m / n as f64)
            }
            AllReduceAlgorithm::RecursiveHalvingDoubling => {
                // 2*log(n) steps, total 2*m bytes transferred
                2.0 * (n as f64).log2().ceil() * alpha + 2.0 * beta * m
            }
            AllReduceAlgorithm::Rabenseifner => {
                // Same as recursive halving-doubling for bandwidth
                2.0 * (n as f64).log2().ceil() * alpha + 2.0 * beta * m * (n - 1) as f64 / n as f64
            }
            AllReduceAlgorithm::DoubleBinaryTree => {
                // Two trees: reduce + broadcast
                2.0 * (n as f64).log2().ceil() * (alpha + beta * m)
            }
        }
    }

    /// Model time for all-to-all
    pub fn alltoall_time(&self, bytes_per_proc: usize) -> f64 {
        let n = self.num_procs;
        let alpha = self.network.alpha;
        let beta = self.network.beta;
        let m = bytes_per_proc as f64;

        // Each process sends (n-1) messages of size m
        (n - 1) as f64 * (alpha + beta * m)
    }

    /// Model time for all-gather
    pub fn allgather_time(&self, bytes_per_proc: usize) -> f64 {
        let n = self.num_procs;
        let alpha = self.network.alpha;
        let beta = self.network.beta;
        let m = bytes_per_proc as f64;

        // Ring allgather: (n-1) steps, each sends m bytes
        (n - 1) as f64 * (alpha + beta * m)
    }

    /// Model time for reduce-scatter
    pub fn reduce_scatter_time(&self, total_bytes: usize) -> f64 {
        let n = self.num_procs;
        let alpha = self.network.alpha;
        let beta = self.network.beta;
        let m = total_bytes as f64;

        // Ring reduce-scatter: (n-1) steps, each sends m/n bytes
        (n - 1) as f64 * (alpha + beta * m / n as f64)
    }

    /// Select best algorithm for given message size
    pub fn best_allreduce_algorithm(&self, bytes: usize) -> AllReduceAlgorithm {
        let crossover = self.network.crossover_bytes();

        if self.num_procs <= 2 {
            return AllReduceAlgorithm::Ring;
        }

        if bytes < crossover / 4 {
            // Small message: minimize latency
            AllReduceAlgorithm::RecursiveHalvingDoubling
        } else if bytes < crossover * 4 {
            // Medium message: balanced
            AllReduceAlgorithm::Rabenseifner
        } else {
            // Large message: maximize bandwidth
            AllReduceAlgorithm::Ring
        }
    }

    /// Estimate speedup from parallelization
    pub fn parallel_speedup(&self, compute_time: f64, bytes: usize) -> f64 {
        let comm_time = self.allreduce_time(bytes, AllReduceAlgorithm::Ring);
        let sequential_time = compute_time * self.num_procs as f64;
        let parallel_time = compute_time + comm_time;

        sequential_time / parallel_time
    }

    /// Compute efficiency
    pub fn efficiency(&self, compute_time: f64, bytes: usize) -> f64 {
        self.parallel_speedup(compute_time, bytes) / self.num_procs as f64
    }

    /// Compare all algorithms
    pub fn compare_algorithms(&self, bytes: usize) -> Vec<(AllReduceAlgorithm, f64)> {
        let algorithms = [
            AllReduceAlgorithm::ReduceBroadcast,
            AllReduceAlgorithm::Ring,
            AllReduceAlgorithm::RecursiveHalvingDoubling,
            AllReduceAlgorithm::Rabenseifner,
            AllReduceAlgorithm::DoubleBinaryTree,
        ];

        let mut results: Vec<_> = algorithms
            .iter()
            .map(|&alg| (alg, self.allreduce_time(bytes, alg)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }
}

/// Strong scaling analysis
#[derive(Debug)]
pub struct StrongScalingAnalysis {
    pub num_procs: usize,
    pub speedup: f64,
    pub efficiency: f64,
    pub serial_fraction: f64,
    pub amdahl_limit: f64,
    pub parallel_time: f64,
    pub communication_time: f64,
    pub computation_time: f64,
}

impl StrongScalingAnalysis {
    /// Compute strong scaling for given parameters
    pub fn compute(model: &CollectiveModel, total_work: f64, bytes: usize) -> Self {
        let n = model.num_procs;
        let work_per_proc = total_work / n as f64;
        let comm_time = model.allreduce_time(bytes, AllReduceAlgorithm::Ring);

        let sequential_time = total_work;
        let parallel_time = work_per_proc + comm_time;

        let speedup = sequential_time / parallel_time;
        let efficiency = speedup / n as f64;

        // Amdahl's law: speedup limited by serial fraction
        let serial_fraction = comm_time / parallel_time;
        let amdahl_limit = if serial_fraction > 0.0 {
            1.0 / serial_fraction
        } else {
            f64::INFINITY
        };

        Self {
            num_procs: n,
            speedup,
            efficiency,
            serial_fraction,
            amdahl_limit,
            parallel_time,
            communication_time: comm_time,
            computation_time: work_per_proc,
        }
    }

    /// Check if scaling is reasonable (efficiency > 50%)
    pub fn is_reasonable(&self) -> bool {
        self.efficiency > 0.5
    }
}

/// Weak scaling analysis
#[derive(Debug)]
pub struct WeakScalingAnalysis {
    pub num_procs: usize,
    pub efficiency: f64,
    pub parallel_time: f64,
    pub communication_time: f64,
    pub computation_time: f64,
    pub communication_overhead: f64,
}

impl WeakScalingAnalysis {
    /// Compute weak scaling for given parameters
    pub fn compute(model: &CollectiveModel, work_per_proc: f64, bytes_per_proc: usize) -> Self {
        let n = model.num_procs;

        // Total bytes scales with num_procs
        let total_bytes = bytes_per_proc * n;
        let comm_time = model.allreduce_time(total_bytes, AllReduceAlgorithm::Ring);

        let single_proc_time = work_per_proc;
        let parallel_time = work_per_proc + comm_time;

        let efficiency = single_proc_time / parallel_time;

        Self {
            num_procs: n,
            efficiency,
            parallel_time,
            communication_time: comm_time,
            computation_time: work_per_proc,
            communication_overhead: comm_time / work_per_proc,
        }
    }

    /// Check if weak scaling is maintained (efficiency > 80%)
    pub fn is_maintained(&self) -> bool {
        self.efficiency > 0.8
    }
}

/// Roofline model for distributed computing
pub struct DistributedRoofline {
    /// Peak compute (FLOPS) per device
    peak_compute: f64,
    /// Peak memory bandwidth (bytes/sec) per device
    peak_bandwidth: f64,
    /// Network bandwidth (bytes/sec)
    network_bandwidth: f64,
    /// Number of devices
    num_devices: usize,
}

impl DistributedRoofline {
    pub fn new(
        peak_compute: f64,
        peak_bandwidth: f64,
        network_bandwidth: f64,
        num_devices: usize,
    ) -> Self {
        Self {
            peak_compute,
            peak_bandwidth,
            network_bandwidth,
            num_devices,
        }
    }

    /// A100 configuration
    pub fn a100(num_devices: usize) -> Self {
        Self {
            peak_compute: 312e12,     // 312 TFLOPS FP16
            peak_bandwidth: 2.0e12,   // 2 TB/s HBM
            network_bandwidth: 600e9, // 600 GB/s NVLink
            num_devices,
        }
    }

    /// H100 configuration
    pub fn h100(num_devices: usize) -> Self {
        Self {
            peak_compute: 989e12,     // 989 TFLOPS FP16
            peak_bandwidth: 3.35e12,  // 3.35 TB/s HBM3
            network_bandwidth: 900e9, // 900 GB/s NVLink
            num_devices,
        }
    }

    /// Compute arithmetic intensity where we become network-bound
    pub fn network_crossover_intensity(&self) -> f64 {
        self.peak_compute / self.network_bandwidth
    }

    /// Memory ridge point (arithmetic intensity where memory-bound meets compute-bound)
    pub fn memory_ridge_point(&self) -> f64 {
        self.peak_compute / self.peak_bandwidth
    }

    /// Achievable performance given arithmetic intensity
    pub fn achievable_performance(&self, arithmetic_intensity: f64) -> f64 {
        let compute_bound = self.peak_compute * self.num_devices as f64;
        let memory_bound = self.peak_bandwidth * arithmetic_intensity * self.num_devices as f64;
        let network_bound = self.network_bandwidth * arithmetic_intensity;

        compute_bound.min(memory_bound).min(network_bound)
    }

    /// Scaling efficiency at given intensity
    pub fn scaling_efficiency(&self, arithmetic_intensity: f64) -> f64 {
        let single_device_perf =
            (self.peak_compute).min(self.peak_bandwidth * arithmetic_intensity);

        let multi_device_perf = self.achievable_performance(arithmetic_intensity);

        multi_device_perf / (single_device_perf * self.num_devices as f64)
    }

    /// Determine which resource is the bottleneck
    pub fn bottleneck(&self, arithmetic_intensity: f64) -> Bottleneck {
        let compute = self.peak_compute * self.num_devices as f64;
        let memory = self.peak_bandwidth * arithmetic_intensity * self.num_devices as f64;
        let network = self.network_bandwidth * arithmetic_intensity;

        let min = compute.min(memory).min(network);

        if (min - network).abs() < 1e-6 {
            Bottleneck::Network
        } else if (min - memory).abs() < 1e-6 {
            Bottleneck::Memory
        } else {
            Bottleneck::Compute
        }
    }
}

/// Resource bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Network,
}

impl Bottleneck {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Compute => "Compute-bound",
            Self::Memory => "Memory-bound",
            Self::Network => "Network-bound",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_params() {
        let nvlink = NetworkParams::nvlink3();

        // Peak bandwidth should be ~300 GB/s
        let bw = nvlink.peak_bandwidth();
        assert!(bw > 200e9 && bw < 400e9);

        // Crossover should be hundreds of KB
        let crossover = nvlink.crossover_bytes();
        assert!(crossover > 100_000 && crossover < 1_000_000);
    }

    #[test]
    fn test_ring_allreduce_model() {
        let network = NetworkParams::nvlink3();
        let model = CollectiveModel::new(network, 8);

        // 1 GB message
        let time = model.allreduce_time(1024 * 1024 * 1024, AllReduceAlgorithm::Ring);

        // Should be roughly 2 * (8-1) * (alpha + beta * 1GB/8)
        // With NVLink3: ~14 * (1us + 128MB / 300GB/s) = ~14 * 0.43ms = ~6ms
        assert!(
            time > 0.001 && time < 0.020,
            "Ring all-reduce time: {}",
            time
        );
    }

    #[test]
    fn test_algorithm_selection() {
        let network = NetworkParams::pcie4();
        let model = CollectiveModel::new(network, 8);

        // Small message should use recursive halving-doubling
        let small = model.best_allreduce_algorithm(1024);
        assert!(matches!(
            small,
            AllReduceAlgorithm::RecursiveHalvingDoubling
        ));

        // Large message should use ring
        let large = model.best_allreduce_algorithm(100 * 1024 * 1024);
        assert!(matches!(large, AllReduceAlgorithm::Ring));
    }

    #[test]
    fn test_strong_scaling() {
        let network = NetworkParams::nvlink3();
        let model = CollectiveModel::new(network, 8);

        // 1 second of compute, 100 MB of communication
        let analysis = StrongScalingAnalysis::compute(&model, 1.0, 100 * 1024 * 1024);

        // Should have reasonable efficiency
        assert!(
            analysis.efficiency > 0.5,
            "Efficiency too low: {}",
            analysis.efficiency
        );
        assert!(
            analysis.speedup > 4.0,
            "Speedup too low: {}",
            analysis.speedup
        );
    }

    #[test]
    fn test_weak_scaling() {
        let network = NetworkParams::nvlink3();
        let model = CollectiveModel::new(network, 8);

        let analysis = WeakScalingAnalysis::compute(&model, 1.0, 10 * 1024 * 1024);

        assert!(analysis.efficiency > 0.0);
        assert!(analysis.communication_overhead >= 0.0);
    }

    #[test]
    fn test_roofline() {
        let roofline = DistributedRoofline::a100(8);

        // Low arithmetic intensity should be memory/network bound
        let bottleneck_low = roofline.bottleneck(0.1);
        assert!(matches!(
            bottleneck_low,
            Bottleneck::Memory | Bottleneck::Network
        ));

        // For multi-GPU, network can become the bottleneck even at high intensity
        // The network crossover point is peak_compute / network_bandwidth
        // For A100: 312e12 / 600e9 = 520 FLOP/byte
        // With 8 GPUs, we need even higher intensity to be compute-bound
        let crossover = roofline.network_crossover_intensity();
        assert!(crossover > 0.0);

        // At very high intensity relative to crossover, should be compute-bound
        // For single GPU test (network doesn't scale with devices)
        let single_gpu = DistributedRoofline::a100(1);
        let bottleneck_high = single_gpu.bottleneck(1000.0);
        assert!(matches!(bottleneck_high, Bottleneck::Compute));
    }

    #[test]
    fn test_compare_algorithms() {
        let network = NetworkParams::nvlink3();
        let model = CollectiveModel::new(network, 8);

        let comparison = model.compare_algorithms(1024 * 1024); // 1 MB

        // Should return sorted list
        assert!(!comparison.is_empty());
        for i in 1..comparison.len() {
            assert!(comparison[i].1 >= comparison[i - 1].1);
        }
    }

    #[test]
    fn test_broadcast_algorithms() {
        let network = NetworkParams::pcie4();
        let model = CollectiveModel::new(network, 16);

        let flat = model.broadcast_time(1024, BroadcastAlgorithm::Flat);
        let tree = model.broadcast_time(1024, BroadcastAlgorithm::BinomialTree);

        // For small messages, tree should be better than flat for many processes
        assert!(tree < flat);
    }
}
