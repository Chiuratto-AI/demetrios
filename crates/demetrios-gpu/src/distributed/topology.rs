//! GPU Topology Discovery and Modeling
//!
//! This module provides topology discovery and modeling for multi-GPU systems:
//!
//! - Interconnect type detection (NVLink, PCIe, NVSwitch)
//! - Bandwidth and latency estimation
//! - Optimal routing computation
//! - All-reduce tree construction

use crate::optimize::pool::DeviceId;
use crate::runtime::{BufferError, Device};
use std::collections::{HashMap, HashSet};

/// Get a Device instance for a DeviceId
///
/// In the current simulation mode, this always returns a CPU device.
/// In real GPU mode, this would map to the actual GPU device.
pub fn get_device(_id: DeviceId) -> Device {
    Device::cpu()
}

/// GPU interconnect type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterconnectType {
    /// Same GPU (no communication needed)
    SameDevice,
    /// NVLink (900 GB/s on A100)
    NVLink { version: u8, lanes: u8 },
    /// NVSwitch (full bisection bandwidth)
    NVSwitch,
    /// PCIe (32 GB/s for Gen4 x16)
    PCIe { gen: u8, lanes: u8 },
    /// Network (InfiniBand, RoCE)
    Network { bandwidth_gbps: f64 },
    /// Across nodes
    InterNode {
        bandwidth_gbps: f64,
        latency_us: f64,
    },
}

impl InterconnectType {
    /// Bandwidth in bytes per second
    pub fn bandwidth_bytes_per_sec(&self) -> f64 {
        match self {
            Self::SameDevice => f64::INFINITY,
            Self::NVLink { version, lanes } => {
                let per_lane = match version {
                    1 => 20.0e9, // NVLink 1.0: 20 GB/s per lane
                    2 => 25.0e9, // NVLink 2.0: 25 GB/s
                    3 => 25.0e9, // NVLink 3.0: 25 GB/s (more lanes)
                    4 => 25.0e9, // NVLink 4.0: 25 GB/s
                    _ => 25.0e9,
                };
                per_lane * (*lanes as f64)
            }
            Self::NVSwitch => 900.0e9, // Full bisection
            Self::PCIe { gen, lanes } => {
                let per_lane = match gen {
                    3 => 1.0e9, // 1 GB/s per lane
                    4 => 2.0e9, // 2 GB/s per lane
                    5 => 4.0e9, // 4 GB/s per lane
                    _ => 2.0e9,
                };
                per_lane * (*lanes as f64)
            }
            Self::Network { bandwidth_gbps } => bandwidth_gbps * 1e9 / 8.0,
            Self::InterNode { bandwidth_gbps, .. } => bandwidth_gbps * 1e9 / 8.0,
        }
    }

    /// Latency in microseconds
    pub fn latency_us(&self) -> f64 {
        match self {
            Self::SameDevice => 0.0,
            Self::NVLink { .. } => 1.0,   // ~1 us
            Self::NVSwitch => 2.0,        // ~2 us
            Self::PCIe { .. } => 5.0,     // ~5 us
            Self::Network { .. } => 10.0, // ~10 us
            Self::InterNode { latency_us, .. } => *latency_us,
        }
    }

    /// Cost metric for routing decisions (lower is better)
    pub fn cost(&self, bytes: usize) -> f64 {
        let transfer_time = bytes as f64 / self.bandwidth_bytes_per_sec();
        let latency = self.latency_us() * 1e-6;
        transfer_time + latency
    }

    /// Check if this is a high-bandwidth interconnect
    pub fn is_high_bandwidth(&self) -> bool {
        matches!(
            self,
            Self::NVLink { .. } | Self::NVSwitch | Self::SameDevice
        )
    }
}

/// Node identifier in a cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub id: DeviceId,
    pub name: String,
    pub compute_capability: (u8, u8),
    pub memory_bytes: usize,
    pub memory_bandwidth: f64, // bytes/sec
    pub compute_tflops: f64,
    pub node_id: NodeId,
}

impl GpuDevice {
    /// Check if device supports peer access to another device
    pub fn can_peer_access(&self, _other: &GpuDevice) -> bool {
        // In real implementation, would check CUDA peer access
        true
    }

    /// Compute capability as a single number
    pub fn compute_capability_int(&self) -> u32 {
        self.compute_capability.0 as u32 * 10 + self.compute_capability.1 as u32
    }
}

/// Node information
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub hostname: String,
    pub devices: Vec<DeviceId>,
    pub cpu_cores: u32,
    pub memory_bytes: usize,
}

/// Device group for hierarchical collectives
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    pub devices: Vec<DeviceId>,
    pub interconnect: InterconnectType,
    pub name: String,
}

/// Multi-GPU topology
#[derive(Debug)]
pub struct GpuTopology {
    /// All devices in the topology
    devices: HashMap<DeviceId, GpuDevice>,
    /// Interconnect between device pairs
    interconnects: HashMap<(DeviceId, DeviceId), InterconnectType>,
    /// Nodes in the cluster
    nodes: HashMap<NodeId, NodeInfo>,
    /// Device groupings (for hierarchical collectives)
    device_groups: Vec<DeviceGroup>,
}

impl GpuTopology {
    /// Discover topology from system
    pub fn discover() -> Result<Self, BufferError> {
        let mut topology = Self {
            devices: HashMap::new(),
            interconnects: HashMap::new(),
            nodes: HashMap::new(),
            device_groups: Vec::new(),
        };

        // In real implementation, would use CUDA/NVML APIs
        // cudaGetDeviceCount, cudaDeviceCanAccessPeer, nvmlDeviceGetNvLinkState

        // Mock discovery for development
        topology.mock_discover()?;

        Ok(topology)
    }

    /// Create topology with specific configuration
    pub fn with_config(
        devices: Vec<GpuDevice>,
        interconnects: Vec<((DeviceId, DeviceId), InterconnectType)>,
    ) -> Self {
        let mut topology = Self {
            devices: HashMap::new(),
            interconnects: HashMap::new(),
            nodes: HashMap::new(),
            device_groups: Vec::new(),
        };

        for device in devices {
            topology.devices.insert(device.id, device);
        }

        for ((from, to), interconnect) in interconnects {
            topology.interconnects.insert((from, to), interconnect);
        }

        topology
    }

    /// Mock topology for testing
    fn mock_discover(&mut self) -> Result<(), BufferError> {
        // Simulate a node with 2 GPUs connected via PCIe
        let node = NodeId(0);

        self.nodes.insert(
            node,
            NodeInfo {
                id: node,
                hostname: "localhost".into(),
                devices: vec![DeviceId(0), DeviceId(1)],
                cpu_cores: 32,
                memory_bytes: 128 * 1024 * 1024 * 1024,
            },
        );

        // Add devices
        for i in 0..2 {
            self.devices.insert(
                DeviceId(i),
                GpuDevice {
                    id: DeviceId(i),
                    name: "NVIDIA L4".into(),
                    compute_capability: (8, 9),
                    memory_bytes: 24 * 1024 * 1024 * 1024,
                    memory_bandwidth: 300.0e9,
                    compute_tflops: 30.0,
                    node_id: node,
                },
            );
        }

        // Add interconnects
        self.interconnects
            .insert((DeviceId(0), DeviceId(0)), InterconnectType::SameDevice);
        self.interconnects
            .insert((DeviceId(1), DeviceId(1)), InterconnectType::SameDevice);
        self.interconnects.insert(
            (DeviceId(0), DeviceId(1)),
            InterconnectType::PCIe { gen: 4, lanes: 16 },
        );
        self.interconnects.insert(
            (DeviceId(1), DeviceId(0)),
            InterconnectType::PCIe { gen: 4, lanes: 16 },
        );

        // Create device group
        self.device_groups.push(DeviceGroup {
            devices: vec![DeviceId(0), DeviceId(1)],
            interconnect: InterconnectType::PCIe { gen: 4, lanes: 16 },
            name: "node0_gpus".into(),
        });

        Ok(())
    }

    /// Get interconnect between two devices
    pub fn get_interconnect(&self, from: DeviceId, to: DeviceId) -> Option<InterconnectType> {
        self.interconnects.get(&(from, to)).copied()
    }

    /// Get all devices
    pub fn devices(&self) -> impl Iterator<Item = &GpuDevice> {
        self.devices.values()
    }

    /// Get device by ID
    pub fn get_device(&self, id: DeviceId) -> Option<&GpuDevice> {
        self.devices.get(&id)
    }

    /// Get devices on a node
    pub fn devices_on_node(&self, node: NodeId) -> Vec<DeviceId> {
        self.devices
            .values()
            .filter(|d| d.node_id == node)
            .map(|d| d.id)
            .collect()
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &NodeInfo> {
        self.nodes.values()
    }

    /// Total memory across all devices
    pub fn total_memory(&self) -> usize {
        self.devices.values().map(|d| d.memory_bytes).sum()
    }

    /// Total compute in TFLOPS
    pub fn total_compute(&self) -> f64 {
        self.devices.values().map(|d| d.compute_tflops).sum()
    }

    /// Number of devices
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Find optimal route between devices using Dijkstra's algorithm
    pub fn optimal_route(&self, from: DeviceId, to: DeviceId) -> Vec<DeviceId> {
        if from == to {
            return vec![from];
        }

        let mut dist: HashMap<DeviceId, f64> = HashMap::new();
        let mut prev: HashMap<DeviceId, DeviceId> = HashMap::new();
        let mut visited: HashSet<DeviceId> = HashSet::new();

        for &id in self.devices.keys() {
            dist.insert(id, f64::INFINITY);
        }
        dist.insert(from, 0.0);

        while visited.len() < self.devices.len() {
            // Find unvisited node with minimum distance
            let current = dist
                .iter()
                .filter(|(id, _)| !visited.contains(id))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(&id, _)| id);

            let current = match current {
                Some(c) => c,
                None => break,
            };

            visited.insert(current);

            if current == to {
                break;
            }

            // Update distances to neighbors
            for (&(a, b), interconnect) in &self.interconnects {
                if a == current && !visited.contains(&b) {
                    let new_dist = dist[&current] + interconnect.cost(1024 * 1024);
                    if new_dist < dist[&b] {
                        dist.insert(b, new_dist);
                        prev.insert(b, current);
                    }
                }
            }
        }

        // Reconstruct path
        let mut path = vec![to];
        let mut current = to;
        while let Some(&p) = prev.get(&current) {
            path.push(p);
            current = p;
            if current == from {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Compute optimal all-reduce tree
    pub fn allreduce_tree(&self, devices: &[DeviceId]) -> AllReduceTree {
        if devices.len() <= 1 {
            return AllReduceTree::Ring(devices.to_vec());
        }

        if devices.len() == 2 {
            return AllReduceTree::Ring(devices.to_vec());
        }

        // Check if all devices have NVLink
        let all_nvlink = devices.windows(2).all(|pair| {
            matches!(
                self.get_interconnect(pair[0], pair[1]),
                Some(InterconnectType::NVLink { .. } | InterconnectType::NVSwitch)
            )
        });

        if all_nvlink {
            AllReduceTree::NVLinkRing(devices.to_vec())
        } else {
            // Hierarchical: intra-node then inter-node
            let nodes: Vec<NodeId> = devices
                .iter()
                .filter_map(|&d| self.devices.get(&d).map(|dev| dev.node_id))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            if nodes.len() > 1 {
                AllReduceTree::Hierarchical {
                    local_trees: nodes
                        .iter()
                        .map(|&n| {
                            let local_devices: Vec<_> = devices
                                .iter()
                                .filter(|&&d| {
                                    self.devices
                                        .get(&d)
                                        .map(|dev| dev.node_id == n)
                                        .unwrap_or(false)
                                })
                                .copied()
                                .collect();
                            AllReduceTree::Ring(local_devices)
                        })
                        .collect(),
                    global_tree: Box::new(AllReduceTree::Ring(
                        nodes
                            .iter()
                            .filter_map(|n| self.devices_on_node(*n).first().copied())
                            .collect(),
                    )),
                }
            } else {
                AllReduceTree::Ring(devices.to_vec())
            }
        }
    }

    /// Get device groups
    pub fn device_groups(&self) -> &[DeviceGroup] {
        &self.device_groups
    }

    /// Check if all devices are on same node
    pub fn is_single_node(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Get minimum interconnect bandwidth between any pair of devices
    pub fn min_bandwidth(&self, devices: &[DeviceId]) -> f64 {
        let mut min_bw = f64::INFINITY;

        for i in 0..devices.len() {
            for j in (i + 1)..devices.len() {
                if let Some(ic) = self.get_interconnect(devices[i], devices[j]) {
                    min_bw = min_bw.min(ic.bandwidth_bytes_per_sec());
                }
            }
        }

        min_bw
    }
}

/// All-reduce algorithm tree structure
#[derive(Debug, Clone)]
pub enum AllReduceTree {
    /// Simple ring algorithm
    Ring(Vec<DeviceId>),
    /// NVLink-optimized ring
    NVLinkRing(Vec<DeviceId>),
    /// Recursive halving-doubling
    RecursiveHalvingDoubling(Vec<DeviceId>),
    /// Hierarchical (intra-node + inter-node)
    Hierarchical {
        local_trees: Vec<AllReduceTree>,
        global_tree: Box<AllReduceTree>,
    },
    /// Tree reduction (for non-uniform topologies)
    Tree {
        root: DeviceId,
        children: Vec<AllReduceTree>,
    },
}

impl AllReduceTree {
    /// Get all devices in the tree
    pub fn devices(&self) -> Vec<DeviceId> {
        match self {
            Self::Ring(devices)
            | Self::NVLinkRing(devices)
            | Self::RecursiveHalvingDoubling(devices) => devices.clone(),
            Self::Hierarchical { local_trees, .. } => {
                local_trees.iter().flat_map(|t| t.devices()).collect()
            }
            Self::Tree { root, children } => {
                let mut devs = vec![*root];
                for child in children {
                    devs.extend(child.devices());
                }
                devs
            }
        }
    }

    /// Estimate time for all-reduce of given size
    pub fn estimate_time(&self, bytes: usize, topology: &GpuTopology) -> f64 {
        match self {
            Self::Ring(devices) | Self::NVLinkRing(devices) => {
                self.estimate_ring_time(devices, bytes, topology)
            }

            Self::RecursiveHalvingDoubling(devices) => {
                let n = devices.len();
                if n <= 1 {
                    return 0.0;
                }
                let log_n = (n as f64).log2().ceil() as usize;

                // log(n) steps, decreasing/increasing chunk sizes
                let mut total_time = 0.0;
                for step in 0..log_n {
                    let chunk = bytes >> step; // Simplified
                    if let Some(interconnect) = topology.get_interconnect(devices[0], devices[1]) {
                        total_time += interconnect.cost(chunk);
                    }
                }
                total_time * 2.0 // Reduce-scatter + allgather
            }

            Self::Hierarchical {
                local_trees,
                global_tree,
            } => {
                let mut local_time: f64 = 0.0;
                for tree in local_trees {
                    local_time = local_time.max(tree.estimate_time(bytes, topology));
                }
                let global_time = global_tree.estimate_time(bytes, topology);
                local_time + global_time + local_time // Local reduce + global + local broadcast
            }

            Self::Tree { children, .. } => {
                let mut child_time: f64 = 0.0;
                for child in children {
                    child_time = child_time.max(child.estimate_time(bytes, topology));
                }
                child_time + 1e-6 // Simplified root aggregation
            }
        }
    }

    fn estimate_ring_time(
        &self,
        devices: &[DeviceId],
        bytes: usize,
        topology: &GpuTopology,
    ) -> f64 {
        let n = devices.len();
        if n <= 1 {
            return 0.0;
        }

        let chunk_size = bytes / n;
        let steps = 2 * (n - 1);

        // Find minimum bandwidth in ring
        let min_bw = devices
            .iter()
            .enumerate()
            .filter_map(|(i, &from)| {
                let to = devices[(i + 1) % n];
                topology
                    .get_interconnect(from, to)
                    .map(|ic| ic.bandwidth_bytes_per_sec())
            })
            .fold(f64::INFINITY, f64::min);

        if min_bw == f64::INFINITY {
            return f64::INFINITY;
        }

        (steps as f64) * (chunk_size as f64) / min_bw
    }

    /// Get algorithm name
    pub fn algorithm_name(&self) -> &'static str {
        match self {
            Self::Ring(_) => "Ring",
            Self::NVLinkRing(_) => "NVLink Ring",
            Self::RecursiveHalvingDoubling(_) => "Recursive Halving-Doubling",
            Self::Hierarchical { .. } => "Hierarchical",
            Self::Tree { .. } => "Tree",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_discovery() {
        let topology = GpuTopology::discover().unwrap();

        assert!(topology.devices().count() >= 1);
        assert!(topology.total_memory() > 0);
    }

    #[test]
    fn test_interconnect_cost() {
        let nvlink = InterconnectType::NVLink {
            version: 3,
            lanes: 4,
        };
        let pcie = InterconnectType::PCIe { gen: 4, lanes: 16 };

        let bytes = 1024 * 1024; // 1 MB

        // NVLink should be faster
        assert!(nvlink.cost(bytes) < pcie.cost(bytes));
    }

    #[test]
    fn test_interconnect_bandwidth() {
        let pcie_gen4 = InterconnectType::PCIe { gen: 4, lanes: 16 };
        assert!((pcie_gen4.bandwidth_bytes_per_sec() - 32.0e9).abs() < 1e9);

        let nvlink = InterconnectType::NVLink {
            version: 3,
            lanes: 4,
        };
        assert!(nvlink.bandwidth_bytes_per_sec() > 50.0e9);
    }

    #[test]
    fn test_allreduce_tree() {
        let topology = GpuTopology::discover().unwrap();
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        if devices.len() >= 2 {
            let tree = topology.allreduce_tree(&devices);
            let time = tree.estimate_time(100 * 1024 * 1024, &topology);

            assert!(time > 0.0);
        }
    }

    #[test]
    fn test_optimal_route() {
        let topology = GpuTopology::discover().unwrap();
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        if devices.len() >= 2 {
            let route = topology.optimal_route(devices[0], devices[1]);
            assert!(!route.is_empty());
            assert_eq!(route[0], devices[0]);
            assert_eq!(*route.last().unwrap(), devices[1]);
        }
    }

    #[test]
    fn test_single_device_route() {
        let topology = GpuTopology::discover().unwrap();
        let device = topology.devices().next().unwrap().id;

        let route = topology.optimal_route(device, device);
        assert_eq!(route.len(), 1);
        assert_eq!(route[0], device);
    }
}
