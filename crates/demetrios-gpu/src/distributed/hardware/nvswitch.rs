//! NVSwitch and Multi-GPU Fabric Topology
//!
//! This module implements GPU interconnect fabric modeling including:
//! - NVSwitch crossbar fabric simulation
//! - DGX A100/H100 topology definitions
//! - Topology-aware collective algorithms
//! - Multi-node fabric with NVLink and InfiniBand

use std::collections::{HashMap, HashSet, VecDeque};

/// NVSwitch chip specification
#[derive(Debug, Clone)]
pub struct NvSwitchSpec {
    /// Switch generation (e.g., "NVSwitch3" for H100)
    pub generation: String,
    /// Number of NVLink ports
    pub num_ports: u32,
    /// Per-port bandwidth in GB/s
    pub port_bandwidth_gbps: f64,
    /// Total switch bandwidth in GB/s
    pub aggregate_bandwidth_gbps: f64,
    /// Switching latency in nanoseconds
    pub latency_ns: u64,
    /// Number of virtual channels
    pub virtual_channels: u32,
}

impl NvSwitchSpec {
    /// NVSwitch for A100 (3rd generation)
    pub fn nvswitch_a100() -> Self {
        Self {
            generation: "NVSwitch3".to_string(),
            num_ports: 18,
            port_bandwidth_gbps: 50.0,       // 50 GB/s per NVLink
            aggregate_bandwidth_gbps: 900.0, // 18 * 50
            latency_ns: 120,
            virtual_channels: 4,
        }
    }

    /// NVSwitch for H100 (4th generation)
    pub fn nvswitch_h100() -> Self {
        Self {
            generation: "NVSwitch4".to_string(),
            num_ports: 64,
            port_bandwidth_gbps: 112.5, // NVLink 4.0: 450 GB/s / 4 links
            aggregate_bandwidth_gbps: 7200.0, // 64 * 112.5
            latency_ns: 100,
            virtual_channels: 8,
        }
    }
}

/// NVLink connection between two endpoints
#[derive(Debug, Clone)]
pub struct NvLinkConnection {
    /// Source GPU ID
    pub src_gpu: u32,
    /// Destination GPU ID
    pub dst_gpu: u32,
    /// NVLink generation (3 for A100, 4 for H100)
    pub generation: u32,
    /// Number of NVLink lanes
    pub num_links: u32,
    /// Per-link bandwidth in GB/s
    pub link_bandwidth_gbps: f64,
    /// Current utilization (0.0 to 1.0)
    pub utilization: f64,
}

impl NvLinkConnection {
    /// Create NVLink 3.0 connection (A100)
    pub fn nvlink3(src: u32, dst: u32, links: u32) -> Self {
        Self {
            src_gpu: src,
            dst_gpu: dst,
            generation: 3,
            num_links: links,
            link_bandwidth_gbps: 50.0,
            utilization: 0.0,
        }
    }

    /// Create NVLink 4.0 connection (H100)
    pub fn nvlink4(src: u32, dst: u32, links: u32) -> Self {
        Self {
            src_gpu: src,
            dst_gpu: dst,
            generation: 4,
            num_links: links,
            link_bandwidth_gbps: 112.5,
            utilization: 0.0,
        }
    }

    /// Total bandwidth in GB/s
    pub fn total_bandwidth(&self) -> f64 {
        self.num_links as f64 * self.link_bandwidth_gbps
    }

    /// Available bandwidth in GB/s
    pub fn available_bandwidth(&self) -> f64 {
        self.total_bandwidth() * (1.0 - self.utilization)
    }
}

/// GPU interconnect topology
#[derive(Debug, Clone)]
pub struct FabricTopology {
    /// Number of GPUs
    pub num_gpus: u32,
    /// Number of NVSwitches
    pub num_switches: u32,
    /// Adjacency matrix (GPU to GPU bandwidth in GB/s)
    bandwidth_matrix: Vec<Vec<f64>>,
    /// Hop count matrix
    hop_matrix: Vec<Vec<u32>>,
    /// NVLink connections
    connections: Vec<NvLinkConnection>,
    /// Topology type
    pub topology_type: TopologyType,
}

/// Types of GPU topologies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyType {
    /// Point-to-point (PCIe only)
    PointToPoint,
    /// Ring topology
    Ring,
    /// Full mesh (all-to-all direct)
    FullMesh,
    /// NVSwitch-based (DGX)
    NvSwitchFabric,
    /// Hybrid (NVSwitch + NVLink)
    Hybrid,
}

impl FabricTopology {
    /// Create DGX A100 topology (8 GPUs, 6 NVSwitches)
    pub fn dgx_a100() -> Self {
        let num_gpus = 8;
        let num_switches = 6;

        // Full bisection bandwidth: each GPU has 12 NVLinks, 600 GB/s total
        // With 6 NVSwitches, each GPU connects to each switch with 2 NVLinks
        let mut bandwidth_matrix = vec![vec![0.0; num_gpus as usize]; num_gpus as usize];
        let mut hop_matrix = vec![vec![0; num_gpus as usize]; num_gpus as usize];
        let mut connections = Vec::new();

        // In DGX A100, any GPU can reach any other GPU through NVSwitch
        // with full bandwidth (600 GB/s)
        for i in 0..num_gpus {
            for j in 0..num_gpus {
                if i != j {
                    bandwidth_matrix[i as usize][j as usize] = 600.0;
                    hop_matrix[i as usize][j as usize] = 1; // Single hop through switch

                    if i < j {
                        connections.push(NvLinkConnection::nvlink3(i, j, 12));
                    }
                }
            }
        }

        Self {
            num_gpus,
            num_switches,
            bandwidth_matrix,
            hop_matrix,
            connections,
            topology_type: TopologyType::NvSwitchFabric,
        }
    }

    /// Create DGX H100 topology (8 GPUs, 4 NVSwitches)
    pub fn dgx_h100() -> Self {
        let num_gpus = 8;
        let num_switches = 4;

        // H100 has NVLink 4.0 with 900 GB/s per GPU
        let mut bandwidth_matrix = vec![vec![0.0; num_gpus as usize]; num_gpus as usize];
        let mut hop_matrix = vec![vec![0; num_gpus as usize]; num_gpus as usize];
        let mut connections = Vec::new();

        for i in 0..num_gpus {
            for j in 0..num_gpus {
                if i != j {
                    bandwidth_matrix[i as usize][j as usize] = 900.0;
                    hop_matrix[i as usize][j as usize] = 1;

                    if i < j {
                        connections.push(NvLinkConnection::nvlink4(i, j, 8));
                    }
                }
            }
        }

        Self {
            num_gpus,
            num_switches,
            bandwidth_matrix,
            hop_matrix,
            connections,
            topology_type: TopologyType::NvSwitchFabric,
        }
    }

    /// Create simple ring topology
    pub fn ring(num_gpus: u32) -> Self {
        let mut bandwidth_matrix = vec![vec![0.0; num_gpus as usize]; num_gpus as usize];
        let mut hop_matrix = vec![vec![0; num_gpus as usize]; num_gpus as usize];
        let mut connections = Vec::new();

        for i in 0..num_gpus {
            let next = (i + 1) % num_gpus;
            let prev = (i + num_gpus - 1) % num_gpus;

            bandwidth_matrix[i as usize][next as usize] = 50.0;
            bandwidth_matrix[i as usize][prev as usize] = 50.0;
            hop_matrix[i as usize][next as usize] = 1;
            hop_matrix[i as usize][prev as usize] = 1;

            if i < next {
                connections.push(NvLinkConnection::nvlink3(i, next, 1));
            }
        }

        // Calculate multi-hop distances
        for i in 0..num_gpus {
            for j in 0..num_gpus {
                if i != j && hop_matrix[i as usize][j as usize] == 0 {
                    // Find shortest path in ring
                    let clockwise = ((j as i32 - i as i32).rem_euclid(num_gpus as i32)) as u32;
                    let counter = num_gpus - clockwise;
                    hop_matrix[i as usize][j as usize] = clockwise.min(counter);
                }
            }
        }

        Self {
            num_gpus,
            num_switches: 0,
            bandwidth_matrix,
            hop_matrix,
            connections,
            topology_type: TopologyType::Ring,
        }
    }

    /// Get bandwidth between two GPUs
    pub fn bandwidth(&self, src: u32, dst: u32) -> f64 {
        if src < self.num_gpus && dst < self.num_gpus {
            self.bandwidth_matrix[src as usize][dst as usize]
        } else {
            0.0
        }
    }

    /// Get hop count between two GPUs
    pub fn hops(&self, src: u32, dst: u32) -> u32 {
        if src < self.num_gpus && dst < self.num_gpus {
            self.hop_matrix[src as usize][dst as usize]
        } else {
            u32::MAX
        }
    }

    /// Get total bisection bandwidth
    pub fn bisection_bandwidth(&self) -> f64 {
        // For NVSwitch topology, it's the sum of all GPU bandwidths / 2
        match self.topology_type {
            TopologyType::NvSwitchFabric => {
                let per_gpu = self.bandwidth_matrix[0][1];
                per_gpu * self.num_gpus as f64 / 2.0
            }
            TopologyType::Ring => {
                // Ring bisection is limited
                2.0 * self.bandwidth_matrix[0][1]
            }
            _ => {
                // Sum all outgoing bandwidth / 2
                let total: f64 = self
                    .bandwidth_matrix
                    .iter()
                    .flat_map(|row| row.iter())
                    .sum();
                total / 4.0 // Divide by 2 for bidirectional, 2 for bisection
            }
        }
    }

    /// Check if topology provides full connectivity
    pub fn is_fully_connected(&self) -> bool {
        for i in 0..self.num_gpus {
            for j in 0..self.num_gpus {
                if i != j && self.hops(i, j) == u32::MAX {
                    return false;
                }
            }
        }
        true
    }
}

/// NVSwitch crossbar switch
#[derive(Debug)]
pub struct NvSwitchCrossbar {
    /// Switch specification
    pub spec: NvSwitchSpec,
    /// Switch ID
    pub switch_id: u32,
    /// Connected GPU ports
    connected_gpus: HashSet<u32>,
    /// Port utilization (port_id -> utilization)
    port_utilization: HashMap<u32, f64>,
    /// Pending transfers
    pending_transfers: VecDeque<CrossbarTransfer>,
    /// Statistics
    pub stats: CrossbarStats,
}

/// A transfer through the crossbar
#[derive(Debug, Clone)]
pub struct CrossbarTransfer {
    /// Source GPU
    pub src_gpu: u32,
    /// Destination GPU
    pub dst_gpu: u32,
    /// Size in bytes
    pub size_bytes: u64,
    /// Remaining bytes
    pub remaining_bytes: u64,
    /// Priority (0 = highest)
    pub priority: u32,
}

/// Crossbar statistics
#[derive(Debug, Clone, Default)]
pub struct CrossbarStats {
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Number of transfers completed
    pub transfers_completed: u64,
    /// Total cycles active
    pub active_cycles: u64,
    /// Port conflicts encountered
    pub port_conflicts: u64,
}

impl NvSwitchCrossbar {
    /// Create new NVSwitch crossbar
    pub fn new(switch_id: u32, spec: NvSwitchSpec) -> Self {
        Self {
            spec,
            switch_id,
            connected_gpus: HashSet::new(),
            port_utilization: HashMap::new(),
            pending_transfers: VecDeque::new(),
            stats: CrossbarStats::default(),
        }
    }

    /// Connect a GPU to the switch
    pub fn connect_gpu(&mut self, gpu_id: u32) {
        self.connected_gpus.insert(gpu_id);
        self.port_utilization.insert(gpu_id, 0.0);
    }

    /// Submit a transfer request
    pub fn submit_transfer(&mut self, src: u32, dst: u32, size_bytes: u64, priority: u32) {
        let transfer = CrossbarTransfer {
            src_gpu: src,
            dst_gpu: dst,
            size_bytes,
            remaining_bytes: size_bytes,
            priority,
        };
        self.pending_transfers.push_back(transfer);
    }

    /// Simulate one cycle of crossbar operation
    pub fn tick(&mut self, cycle_time_ns: u64) -> Vec<CrossbarTransfer> {
        let mut completed = Vec::new();

        if self.pending_transfers.is_empty() {
            return completed;
        }

        self.stats.active_cycles += 1;

        // Calculate bytes that can be transferred this cycle
        let bytes_per_cycle =
            (self.spec.port_bandwidth_gbps * 1e9 * cycle_time_ns as f64 / 1e9) as u64;

        // Track port usage this cycle
        let mut src_ports_used: HashSet<u32> = HashSet::new();
        let mut dst_ports_used: HashSet<u32> = HashSet::new();

        // Process transfers (simple round-robin)
        let mut remaining = VecDeque::new();

        while let Some(mut transfer) = self.pending_transfers.pop_front() {
            // Check for port conflicts
            if src_ports_used.contains(&transfer.src_gpu)
                || dst_ports_used.contains(&transfer.dst_gpu)
            {
                self.stats.port_conflicts += 1;
                remaining.push_back(transfer);
                continue;
            }

            // Reserve ports
            src_ports_used.insert(transfer.src_gpu);
            dst_ports_used.insert(transfer.dst_gpu);

            // Transfer data
            let transferred = transfer.remaining_bytes.min(bytes_per_cycle);
            transfer.remaining_bytes -= transferred;
            self.stats.bytes_transferred += transferred;

            if transfer.remaining_bytes == 0 {
                self.stats.transfers_completed += 1;
                completed.push(transfer);
            } else {
                remaining.push_back(transfer);
            }
        }

        self.pending_transfers = remaining;
        completed
    }

    /// Get current port utilization
    pub fn get_utilization(&self) -> f64 {
        if self.port_utilization.is_empty() {
            return 0.0;
        }
        let total: f64 = self.port_utilization.values().sum();
        total / self.port_utilization.len() as f64
    }
}

/// Topology-aware collective operation scheduler
#[derive(Debug)]
pub struct CollectiveScheduler {
    /// Fabric topology
    topology: FabricTopology,
    /// Algorithm selection strategy
    algorithm: CollectiveAlgorithm,
}

/// Collective algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveAlgorithm {
    /// Ring-based allreduce
    Ring,
    /// Tree-based reduction
    Tree,
    /// Direct (all-to-all through switch)
    Direct,
    /// Recursive halving-doubling
    RecursiveHalving,
    /// Bucket (for large messages)
    Bucket,
    /// Auto-select based on topology
    Auto,
}

/// A collective operation step
#[derive(Debug, Clone)]
pub struct CollectiveStep {
    /// Step index
    pub step: u32,
    /// List of (src, dst, size) transfers
    pub transfers: Vec<(u32, u32, u64)>,
    /// Description
    pub description: String,
}

impl CollectiveScheduler {
    /// Create scheduler for topology
    pub fn new(topology: FabricTopology) -> Self {
        Self {
            topology,
            algorithm: CollectiveAlgorithm::Auto,
        }
    }

    /// Set algorithm
    pub fn set_algorithm(&mut self, algo: CollectiveAlgorithm) {
        self.algorithm = algo;
    }

    /// Schedule an allreduce operation
    pub fn schedule_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        let algo = if self.algorithm == CollectiveAlgorithm::Auto {
            self.select_algorithm(data_size)
        } else {
            self.algorithm
        };

        match algo {
            CollectiveAlgorithm::Ring => self.ring_allreduce(data_size),
            CollectiveAlgorithm::Tree => self.tree_allreduce(data_size),
            CollectiveAlgorithm::Direct => self.direct_allreduce(data_size),
            CollectiveAlgorithm::RecursiveHalving => self.recursive_halving_allreduce(data_size),
            CollectiveAlgorithm::Bucket => self.bucket_allreduce(data_size),
            CollectiveAlgorithm::Auto => unreachable!(),
        }
    }

    /// Auto-select best algorithm
    fn select_algorithm(&self, data_size: u64) -> CollectiveAlgorithm {
        match self.topology.topology_type {
            TopologyType::NvSwitchFabric => {
                // NVSwitch can do direct for small messages
                if data_size < 1024 * 1024 {
                    CollectiveAlgorithm::Direct
                } else {
                    CollectiveAlgorithm::Ring
                }
            }
            TopologyType::Ring => CollectiveAlgorithm::Ring,
            TopologyType::FullMesh => {
                if data_size < 256 * 1024 {
                    CollectiveAlgorithm::Direct
                } else {
                    CollectiveAlgorithm::Bucket
                }
            }
            _ => CollectiveAlgorithm::Ring,
        }
    }

    /// Ring allreduce schedule
    fn ring_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        let n = self.topology.num_gpus;
        let chunk_size = data_size / n as u64;
        let mut steps = Vec::new();

        // Reduce-scatter phase: n-1 steps
        for step in 0..(n - 1) {
            let mut transfers = Vec::new();
            for gpu in 0..n {
                let next = (gpu + 1) % n;
                let _chunk = (gpu as i32 - step as i32).rem_euclid(n as i32) as u32;
                transfers.push((gpu, next, chunk_size));
            }
            steps.push(CollectiveStep {
                step,
                transfers,
                description: format!("Reduce-scatter step {}", step),
            });
        }

        // Allgather phase: n-1 steps
        for step in 0..(n - 1) {
            let mut transfers = Vec::new();
            for gpu in 0..n {
                let next = (gpu + 1) % n;
                transfers.push((gpu, next, chunk_size));
            }
            steps.push(CollectiveStep {
                step: n - 1 + step,
                transfers,
                description: format!("Allgather step {}", step),
            });
        }

        steps
    }

    /// Tree allreduce schedule
    fn tree_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        let n = self.topology.num_gpus;
        let mut steps = Vec::new();

        // Reduce phase (gather to root)
        let levels = (n as f64).log2().ceil() as u32;
        for level in 0..levels {
            let stride = 1 << level;
            let mut transfers = Vec::new();
            for i in (0..n).step_by(stride as usize * 2) {
                if i + stride < n {
                    transfers.push((i + stride, i, data_size));
                }
            }
            if !transfers.is_empty() {
                steps.push(CollectiveStep {
                    step: level,
                    transfers,
                    description: format!("Tree reduce level {}", level),
                });
            }
        }

        // Broadcast phase (scatter from root)
        for level in (0..levels).rev() {
            let stride = 1 << level;
            let mut transfers = Vec::new();
            for i in (0..n).step_by(stride as usize * 2) {
                if i + stride < n {
                    transfers.push((i, i + stride, data_size));
                }
            }
            if !transfers.is_empty() {
                steps.push(CollectiveStep {
                    step: levels + (levels - 1 - level),
                    transfers,
                    description: format!("Tree broadcast level {}", level),
                });
            }
        }

        steps
    }

    /// Direct allreduce (all-to-all)
    fn direct_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        let n = self.topology.num_gpus;
        let chunk_size = data_size / n as u64;
        let mut steps = Vec::new();

        // Single step: all GPUs exchange with all others
        let mut transfers = Vec::new();
        for src in 0..n {
            for dst in 0..n {
                if src != dst {
                    transfers.push((src, dst, chunk_size));
                }
            }
        }

        steps.push(CollectiveStep {
            step: 0,
            transfers,
            description: "Direct all-to-all exchange".to_string(),
        });

        steps
    }

    /// Recursive halving-doubling allreduce
    fn recursive_halving_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        let n = self.topology.num_gpus;
        let mut steps = Vec::new();
        let levels = (n as f64).log2().ceil() as u32;

        // Reduce-scatter using recursive halving
        for level in 0..levels {
            let distance = 1 << (levels - 1 - level);
            let chunk_size = data_size >> (level + 1);
            let mut transfers = Vec::new();

            for gpu in 0..n {
                let partner = gpu ^ distance;
                if partner < n && gpu < partner {
                    transfers.push((gpu, partner, chunk_size));
                    transfers.push((partner, gpu, chunk_size));
                }
            }

            steps.push(CollectiveStep {
                step: level,
                transfers,
                description: format!("Recursive halving step {}", level),
            });
        }

        // Allgather using recursive doubling
        for level in 0..levels {
            let distance = 1 << level;
            let chunk_size = data_size >> (levels - level);
            let mut transfers = Vec::new();

            for gpu in 0..n {
                let partner = gpu ^ distance;
                if partner < n && gpu < partner {
                    transfers.push((gpu, partner, chunk_size));
                    transfers.push((partner, gpu, chunk_size));
                }
            }

            steps.push(CollectiveStep {
                step: levels + level,
                transfers,
                description: format!("Recursive doubling step {}", level),
            });
        }

        steps
    }

    /// Bucket allreduce
    fn bucket_allreduce(&self, data_size: u64) -> Vec<CollectiveStep> {
        // For large messages, use bucketing to maximize bandwidth
        let n = self.topology.num_gpus;
        let bucket_size = data_size / n as u64;
        let mut steps = Vec::new();

        // Each GPU responsible for one bucket
        // Step 1: Reduce each bucket to its owner
        for bucket in 0..n {
            let mut transfers = Vec::new();
            for src in 0..n {
                if src != bucket {
                    transfers.push((src, bucket, bucket_size));
                }
            }
            steps.push(CollectiveStep {
                step: bucket,
                transfers,
                description: format!("Bucket {} reduce", bucket),
            });
        }

        // Step 2: Broadcast each bucket from owner
        for bucket in 0..n {
            let mut transfers = Vec::new();
            for dst in 0..n {
                if dst != bucket {
                    transfers.push((bucket, dst, bucket_size));
                }
            }
            steps.push(CollectiveStep {
                step: n + bucket,
                transfers,
                description: format!("Bucket {} broadcast", bucket),
            });
        }

        steps
    }

    /// Estimate time for collective in microseconds
    pub fn estimate_time(&self, data_size: u64) -> f64 {
        let steps = self.schedule_allreduce(data_size);
        let mut total_time: f64 = 0.0;

        for step in &steps {
            // Find the bottleneck transfer in this step
            let mut max_time: f64 = 0.0;
            for (src, dst, size) in &step.transfers {
                let bw = self.topology.bandwidth(*src, *dst);
                if bw > 0.0 {
                    let time = *size as f64 / (bw * 1e9) * 1e6; // Convert to microseconds
                    max_time = max_time.max(time);
                }
            }
            total_time += max_time;
        }

        total_time
    }
}

/// Multi-node fabric configuration
#[derive(Debug)]
pub struct MultiNodeFabric {
    /// Number of nodes
    pub num_nodes: u32,
    /// GPUs per node
    pub gpus_per_node: u32,
    /// Intra-node topology
    intra_node: FabricTopology,
    /// Inter-node bandwidth per link in GB/s
    pub inter_node_bandwidth_gbps: f64,
    /// Inter-node connections (node pairs)
    inter_node_links: Vec<(u32, u32)>,
    /// Inter-node connection type
    pub inter_node_type: InterNodeType,
}

/// Types of inter-node connections
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterNodeType {
    /// InfiniBand HDR (200 Gbps)
    InfiniBandHDR,
    /// InfiniBand NDR (400 Gbps)
    InfiniBandNDR,
    /// NVLink (DGX SuperPOD)
    NvLinkBridge,
    /// Ethernet (100GbE, 400GbE)
    Ethernet,
}

impl InterNodeType {
    /// Bandwidth in GB/s per link
    pub fn bandwidth_gbps(&self) -> f64 {
        match self {
            Self::InfiniBandHDR => 25.0, // 200 Gbps = 25 GB/s
            Self::InfiniBandNDR => 50.0, // 400 Gbps = 50 GB/s
            Self::NvLinkBridge => 450.0, // NVLink 4.0
            Self::Ethernet => 50.0,      // 400GbE
        }
    }
}

impl MultiNodeFabric {
    /// Create DGX SuperPOD configuration
    pub fn dgx_superpod(num_nodes: u32) -> Self {
        let intra_node = FabricTopology::dgx_a100();
        let gpus_per_node = intra_node.num_gpus;

        // Full fat-tree interconnect between nodes
        let mut inter_node_links = Vec::new();
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                inter_node_links.push((i, j));
            }
        }

        Self {
            num_nodes,
            gpus_per_node,
            intra_node,
            inter_node_bandwidth_gbps: 200.0, // 8x 25 GB/s HDR links
            inter_node_links,
            inter_node_type: InterNodeType::InfiniBandHDR,
        }
    }

    /// Create custom multi-node fabric
    pub fn new(num_nodes: u32, intra_node: FabricTopology, inter_type: InterNodeType) -> Self {
        let gpus_per_node = intra_node.num_gpus;

        let mut inter_node_links = Vec::new();
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                inter_node_links.push((i, j));
            }
        }

        Self {
            num_nodes,
            gpus_per_node,
            intra_node,
            inter_node_bandwidth_gbps: inter_type.bandwidth_gbps(),
            inter_node_links,
            inter_node_type: inter_type,
        }
    }

    /// Get total number of GPUs
    pub fn total_gpus(&self) -> u32 {
        self.num_nodes * self.gpus_per_node
    }

    /// Convert global GPU ID to (node, local_gpu)
    pub fn global_to_local(&self, global_gpu: u32) -> (u32, u32) {
        let node = global_gpu / self.gpus_per_node;
        let local = global_gpu % self.gpus_per_node;
        (node, local)
    }

    /// Convert (node, local_gpu) to global GPU ID
    pub fn local_to_global(&self, node: u32, local_gpu: u32) -> u32 {
        node * self.gpus_per_node + local_gpu
    }

    /// Get bandwidth between two global GPU IDs
    pub fn bandwidth(&self, src: u32, dst: u32) -> f64 {
        let (src_node, src_local) = self.global_to_local(src);
        let (dst_node, dst_local) = self.global_to_local(dst);

        if src_node == dst_node {
            // Intra-node
            self.intra_node.bandwidth(src_local, dst_local)
        } else {
            // Inter-node: limited by inter-node links
            self.inter_node_bandwidth_gbps
        }
    }

    /// Check if two GPUs are on the same node
    pub fn same_node(&self, gpu1: u32, gpu2: u32) -> bool {
        let (node1, _) = self.global_to_local(gpu1);
        let (node2, _) = self.global_to_local(gpu2);
        node1 == node2
    }

    /// Get aggregate inter-node bandwidth
    pub fn aggregate_inter_node_bandwidth(&self) -> f64 {
        self.inter_node_links.len() as f64 * self.inter_node_bandwidth_gbps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvswitch_spec() {
        let a100 = NvSwitchSpec::nvswitch_a100();
        assert_eq!(a100.num_ports, 18);
        assert_eq!(a100.aggregate_bandwidth_gbps, 900.0);

        let h100 = NvSwitchSpec::nvswitch_h100();
        assert!(h100.aggregate_bandwidth_gbps > a100.aggregate_bandwidth_gbps);
    }

    #[test]
    fn test_nvlink_connection() {
        let link = NvLinkConnection::nvlink3(0, 1, 12);
        assert_eq!(link.total_bandwidth(), 600.0);

        let link4 = NvLinkConnection::nvlink4(0, 1, 8);
        assert_eq!(link4.total_bandwidth(), 900.0);
    }

    #[test]
    fn test_dgx_a100_topology() {
        let topo = FabricTopology::dgx_a100();

        assert_eq!(topo.num_gpus, 8);
        assert_eq!(topo.num_switches, 6);
        assert!(topo.is_fully_connected());

        // All pairs should have same bandwidth
        assert_eq!(topo.bandwidth(0, 1), 600.0);
        assert_eq!(topo.bandwidth(0, 7), 600.0);

        // Single hop through switch
        assert_eq!(topo.hops(0, 7), 1);
    }

    #[test]
    fn test_dgx_h100_topology() {
        let topo = FabricTopology::dgx_h100();

        assert_eq!(topo.num_gpus, 8);
        assert_eq!(topo.bandwidth(0, 1), 900.0);
    }

    #[test]
    fn test_ring_topology() {
        let topo = FabricTopology::ring(4);

        assert_eq!(topo.num_gpus, 4);
        assert_eq!(topo.hops(0, 1), 1);
        assert_eq!(topo.hops(0, 2), 2); // Across the ring
        assert!(topo.is_fully_connected());
    }

    #[test]
    fn test_nvswitch_crossbar() {
        let mut crossbar = NvSwitchCrossbar::new(0, NvSwitchSpec::nvswitch_a100());

        for i in 0..8 {
            crossbar.connect_gpu(i);
        }

        // Submit transfer
        crossbar.submit_transfer(0, 1, 1024 * 1024, 0);

        // Run cycles
        let mut completed = 0;
        for _ in 0..100 {
            let done = crossbar.tick(100);
            completed += done.len();
        }

        assert!(completed >= 1 || crossbar.stats.bytes_transferred > 0);
    }

    #[test]
    fn test_collective_scheduler_ring() {
        let topo = FabricTopology::ring(4);
        let mut scheduler = CollectiveScheduler::new(topo);
        scheduler.set_algorithm(CollectiveAlgorithm::Ring);

        let steps = scheduler.schedule_allreduce(1024 * 1024);

        // Ring allreduce: 2*(n-1) steps
        assert_eq!(steps.len(), 6);
    }

    #[test]
    fn test_collective_scheduler_tree() {
        let topo = FabricTopology::dgx_a100();
        let mut scheduler = CollectiveScheduler::new(topo);
        scheduler.set_algorithm(CollectiveAlgorithm::Tree);

        let steps = scheduler.schedule_allreduce(1024 * 1024);

        // Tree: log2(n) reduce + log2(n) broadcast
        assert!(steps.len() >= 4);
    }

    #[test]
    fn test_collective_auto_select() {
        let topo = FabricTopology::dgx_a100();
        let scheduler = CollectiveScheduler::new(topo);

        // Small message should use direct
        let small_steps = scheduler.schedule_allreduce(1024);

        // Large message should use ring
        let large_steps = scheduler.schedule_allreduce(100 * 1024 * 1024);

        // Different algorithms produce different step counts
        assert_ne!(small_steps.len(), large_steps.len());
    }

    #[test]
    fn test_collective_time_estimate() {
        let topo = FabricTopology::dgx_a100();
        let scheduler = CollectiveScheduler::new(topo);

        let time = scheduler.estimate_time(1024 * 1024 * 1024); // 1 GB

        // Should be a reasonable time in microseconds
        assert!(time > 0.0);
        assert!(time < 1_000_000.0); // Less than 1 second
    }

    #[test]
    fn test_multi_node_fabric() {
        let fabric = MultiNodeFabric::dgx_superpod(4);

        assert_eq!(fabric.total_gpus(), 32);

        // Same node
        assert!(fabric.same_node(0, 7));
        assert!(!fabric.same_node(0, 8));

        // Intra-node bandwidth
        assert_eq!(fabric.bandwidth(0, 1), 600.0);

        // Inter-node bandwidth
        assert_eq!(fabric.bandwidth(0, 8), 200.0);
    }

    #[test]
    fn test_multi_node_global_local() {
        let fabric = MultiNodeFabric::dgx_superpod(4);

        assert_eq!(fabric.global_to_local(0), (0, 0));
        assert_eq!(fabric.global_to_local(7), (0, 7));
        assert_eq!(fabric.global_to_local(8), (1, 0));
        assert_eq!(fabric.global_to_local(15), (1, 7));

        assert_eq!(fabric.local_to_global(0, 0), 0);
        assert_eq!(fabric.local_to_global(1, 3), 11);
    }

    #[test]
    fn test_inter_node_types() {
        assert_eq!(InterNodeType::InfiniBandHDR.bandwidth_gbps(), 25.0);
        assert_eq!(InterNodeType::InfiniBandNDR.bandwidth_gbps(), 50.0);
        assert_eq!(InterNodeType::NvLinkBridge.bandwidth_gbps(), 450.0);
    }

    #[test]
    fn test_bisection_bandwidth() {
        let dgx = FabricTopology::dgx_a100();
        let bisection = dgx.bisection_bandwidth();

        // DGX A100: 600 GB/s * 8 / 2 = 2400 GB/s
        assert_eq!(bisection, 2400.0);

        let ring = FabricTopology::ring(4);
        let ring_bisection = ring.bisection_bandwidth();

        // Ring has limited bisection
        assert!(ring_bisection < bisection);
    }
}
