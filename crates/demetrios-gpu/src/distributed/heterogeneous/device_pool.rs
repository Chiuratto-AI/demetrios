//! Device Pool for Heterogeneous GPU Systems
//!
//! This module provides a pool of GPU devices grouped by architecture,
//! with support for capability-based device selection and P2P queries.

use super::architecture::{ArchCapabilities, ArchFeature, ArchFeatures, GpuArchitecture};
use super::discovery::{DeviceDiscovery, DiscoveredDevice, DiscoveryError, P2PCapability};
use crate::optimize::pool::DeviceId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur with device pools
#[derive(Debug, Error)]
pub enum DevicePoolError {
    #[error("Discovery failed: {0}")]
    DiscoveryFailed(#[from] DiscoveryError),

    #[error("No devices found")]
    NoDevices,

    #[error("Device {0:?} not found in pool")]
    DeviceNotFound(DeviceId),

    #[error("No devices with architecture {0:?}")]
    NoDevicesForArchitecture(GpuArchitecture),

    #[error("No devices support feature {0:?}")]
    NoDevicesWithFeature(ArchFeature),
}

/// A pool of heterogeneous GPU devices
#[derive(Debug)]
pub struct DevicePool {
    /// All devices in the pool
    devices: Vec<HeterogeneousDevice>,
    /// Devices grouped by architecture
    by_architecture: HashMap<GpuArchitecture, Vec<DeviceId>>,
    /// P2P capability matrix
    p2p_matrix: HashMap<(DeviceId, DeviceId), P2PCapability>,
    /// Device capabilities cache
    capabilities: HashMap<DeviceId, ArchCapabilities>,
}

impl DevicePool {
    /// Create an empty device pool
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            by_architecture: HashMap::new(),
            p2p_matrix: HashMap::new(),
            capabilities: HashMap::new(),
        }
    }

    /// Discover all available devices and create a pool
    pub fn discover() -> Result<Self, DevicePoolError> {
        let discovered = DeviceDiscovery::enumerate()?;

        if discovered.is_empty() {
            return Err(DevicePoolError::NoDevices);
        }

        let mut pool = Self::new();

        // Add devices
        for disc in &discovered {
            pool.add_device(disc.clone());
        }

        // Query P2P matrix
        let device_ids: Vec<u32> = discovered.iter().map(|d| d.id.0).collect();
        pool.p2p_matrix = DeviceDiscovery::query_p2p_matrix(&device_ids)?;

        Ok(pool)
    }

    /// Create a pool with specific devices
    pub fn with_devices(device_ids: &[u32]) -> Result<Self, DevicePoolError> {
        let all_discovered = DeviceDiscovery::enumerate()?;
        let mut pool = Self::new();

        for disc in all_discovered {
            if device_ids.contains(&disc.id.0) {
                pool.add_device(disc);
            }
        }

        if pool.devices.is_empty() {
            return Err(DevicePoolError::NoDevices);
        }

        // Query P2P for selected devices
        pool.p2p_matrix = DeviceDiscovery::query_p2p_matrix(device_ids)?;

        Ok(pool)
    }

    /// Add a device to the pool
    fn add_device(&mut self, discovered: DiscoveredDevice) {
        let device = HeterogeneousDevice::from_discovered(discovered.clone());

        // Add to architecture group
        self.by_architecture
            .entry(device.architecture)
            .or_default()
            .push(device.id);

        // Cache capabilities
        self.capabilities
            .insert(device.id, discovered.capabilities);

        self.devices.push(device);
    }

    /// Get all devices in the pool
    pub fn devices(&self) -> &[HeterogeneousDevice] {
        &self.devices
    }

    /// Get a device by ID
    pub fn get(&self, id: DeviceId) -> Option<&HeterogeneousDevice> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Get devices by architecture
    pub fn by_architecture(&self, arch: GpuArchitecture) -> &[DeviceId] {
        self.by_architecture.get(&arch).map_or(&[], |v| v.as_slice())
    }

    /// Get all unique architectures in the pool
    pub fn architectures(&self) -> Vec<GpuArchitecture> {
        let mut archs: Vec<_> = self.by_architecture.keys().copied().collect();
        archs.sort();
        archs
    }

    /// Get devices supporting a specific feature
    pub fn with_feature(&self, feature: ArchFeature) -> Vec<DeviceId> {
        self.devices
            .iter()
            .filter(|d| d.supports(feature))
            .map(|d| d.id)
            .collect()
    }

    /// Get the lowest common architecture (for fallback compilation)
    pub fn lowest_common_architecture(&self) -> GpuArchitecture {
        self.architectures()
            .into_iter()
            .min()
            .unwrap_or(GpuArchitecture::Volta)
    }

    /// Get the fastest device (highest performance score)
    pub fn fastest_device(&self) -> Option<&HeterogeneousDevice> {
        self.devices
            .iter()
            .max_by(|a, b| a.performance_score.partial_cmp(&b.performance_score).unwrap())
    }

    /// Get devices with the highest memory
    pub fn highest_memory_device(&self) -> Option<&HeterogeneousDevice> {
        self.devices.iter().max_by_key(|d| d.memory_bytes)
    }

    /// Check if P2P is available between two devices
    pub fn can_p2p(&self, src: DeviceId, dst: DeviceId) -> bool {
        self.p2p_matrix
            .get(&(src, dst))
            .map_or(false, |p| p.can_access)
    }

    /// Get P2P capability between two devices
    pub fn p2p_capability(&self, src: DeviceId, dst: DeviceId) -> Option<&P2PCapability> {
        self.p2p_matrix.get(&(src, dst))
    }

    /// Get all device pairs connected via NVLink
    pub fn nvlink_pairs(&self) -> Vec<(DeviceId, DeviceId)> {
        self.p2p_matrix
            .iter()
            .filter(|(_, p)| p.has_nvlink())
            .map(|((src, dst), _)| (*src, *dst))
            .collect()
    }

    /// Partition devices into NVLink-connected groups
    pub fn nvlink_groups(&self) -> Vec<Vec<DeviceId>> {
        // Simple union-find to group NVLink-connected devices
        let mut groups: Vec<Vec<DeviceId>> = Vec::new();
        let mut device_to_group: HashMap<DeviceId, usize> = HashMap::new();

        for device in &self.devices {
            if !device_to_group.contains_key(&device.id) {
                let group_idx = groups.len();
                let mut group = vec![device.id];
                device_to_group.insert(device.id, group_idx);

                // Find all NVLink-connected devices
                for other in &self.devices {
                    if other.id != device.id
                        && !device_to_group.contains_key(&other.id)
                        && self
                            .p2p_capability(device.id, other.id)
                            .map_or(false, |p| p.has_nvlink())
                    {
                        group.push(other.id);
                        device_to_group.insert(other.id, group_idx);
                    }
                }

                groups.push(group);
            }
        }

        groups
    }

    /// Get total compute in TFLOPS
    pub fn total_compute_tflops(&self) -> f64 {
        self.devices.iter().map(|d| d.capabilities.fp32_tflops).sum()
    }

    /// Get total memory in bytes
    pub fn total_memory(&self) -> usize {
        self.devices.iter().map(|d| d.memory_bytes).sum()
    }

    /// Get number of devices
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Get capabilities for a device
    pub fn capabilities(&self, id: DeviceId) -> Option<&ArchCapabilities> {
        self.capabilities.get(&id)
    }

    /// Compute performance weights for load balancing
    pub fn compute_weights(&self) -> HashMap<DeviceId, f64> {
        let total_score: f64 = self.devices.iter().map(|d| d.performance_score).sum();

        self.devices
            .iter()
            .map(|d| (d.id, d.performance_score / total_score))
            .collect()
    }

    /// Compute memory-based weights for load balancing
    pub fn memory_weights(&self) -> HashMap<DeviceId, f64> {
        let total_mem: f64 = self.devices.iter().map(|d| d.memory_bytes as f64).sum();

        self.devices
            .iter()
            .map(|d| (d.id, d.memory_bytes as f64 / total_mem))
            .collect()
    }
}

impl Default for DevicePool {
    fn default() -> Self {
        Self::new()
    }
}

/// A heterogeneous GPU device with runtime state
#[derive(Debug)]
pub struct HeterogeneousDevice {
    /// Device ID
    pub id: DeviceId,
    /// Device name
    pub name: String,
    /// Architecture generation
    pub architecture: GpuArchitecture,
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Total memory (bytes)
    pub memory_bytes: usize,
    /// Full capabilities
    pub capabilities: ArchCapabilities,
    /// Normalized performance score (fastest = 1.0)
    pub performance_score: f64,
    /// Current utilization (0.0 - 1.0)
    utilization: Arc<AtomicU64>,
}

impl HeterogeneousDevice {
    /// Create from discovered device
    pub fn from_discovered(disc: DiscoveredDevice) -> Self {
        let performance_score = disc.capabilities.performance_score();

        Self {
            id: disc.id,
            name: disc.name,
            architecture: disc.architecture,
            compute_capability: disc.compute_capability,
            memory_bytes: disc.memory_bytes,
            capabilities: disc.capabilities,
            performance_score,
            utilization: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get architecture features
    pub fn features(&self) -> &ArchFeatures {
        &self.capabilities.features
    }

    /// Check if device supports a feature
    pub fn supports(&self, feature: ArchFeature) -> bool {
        self.capabilities.features.supports(feature)
    }

    /// Get current utilization
    pub fn utilization(&self) -> f64 {
        f64::from_bits(self.utilization.load(Ordering::Relaxed))
    }

    /// Set utilization
    pub fn set_utilization(&self, util: f64) {
        self.utilization
            .store(util.to_bits(), Ordering::Relaxed);
    }

    /// Check if device is idle (utilization < threshold)
    pub fn is_idle(&self, threshold: f64) -> bool {
        self.utilization() < threshold
    }

    /// Get SM string for PTX compilation
    pub fn sm_string(&self) -> &'static str {
        self.architecture.sm_string()
    }

    /// Get PTX version
    pub fn ptx_version(&self) -> (u32, u32) {
        self.architecture.ptx_version()
    }
}

impl Clone for HeterogeneousDevice {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            architecture: self.architecture,
            compute_capability: self.compute_capability,
            memory_bytes: self.memory_bytes,
            capabilities: self.capabilities.clone(),
            performance_score: self.performance_score,
            utilization: Arc::new(AtomicU64::new(self.utilization.load(Ordering::Relaxed))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_pool() {
        let pool = DevicePool::discover().unwrap();
        assert!(!pool.is_empty());
        assert!(pool.len() >= 1);
    }

    #[test]
    fn test_by_architecture() {
        let pool = DevicePool::discover().unwrap();
        let archs = pool.architectures();
        assert!(!archs.is_empty());

        for arch in archs {
            let devices = pool.by_architecture(arch);
            assert!(!devices.is_empty());
        }
    }

    #[test]
    fn test_lowest_common_architecture() {
        let pool = DevicePool::discover().unwrap();
        let lowest = pool.lowest_common_architecture();

        // All devices should be >= lowest
        for device in pool.devices() {
            assert!(device.architecture >= lowest);
        }
    }

    #[test]
    fn test_fastest_device() {
        let pool = DevicePool::discover().unwrap();
        let fastest = pool.fastest_device().unwrap();

        for device in pool.devices() {
            assert!(device.performance_score <= fastest.performance_score);
        }
    }

    #[test]
    fn test_with_feature() {
        let pool = DevicePool::discover().unwrap();

        // BF16 should be supported by Ampere+
        let bf16_devices = pool.with_feature(ArchFeature::BF16);
        for id in bf16_devices {
            let device = pool.get(id).unwrap();
            assert!(device.architecture >= GpuArchitecture::Ampere);
        }
    }

    #[test]
    fn test_p2p_capability() {
        let pool = DevicePool::discover().unwrap();

        if pool.len() >= 2 {
            let d0 = pool.devices()[0].id;
            let d1 = pool.devices()[1].id;

            let p2p = pool.p2p_capability(d0, d1);
            assert!(p2p.is_some());
        }
    }

    #[test]
    fn test_compute_weights() {
        let pool = DevicePool::discover().unwrap();
        let weights = pool.compute_weights();

        // Weights should sum to ~1.0
        let sum: f64 = weights.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_total_compute() {
        let pool = DevicePool::discover().unwrap();
        let total = pool.total_compute_tflops();
        assert!(total > 0.0);
    }

    #[test]
    fn test_device_utilization() {
        let pool = DevicePool::discover().unwrap();
        let device = &pool.devices()[0];

        assert_eq!(device.utilization(), 0.0);

        device.set_utilization(0.5);
        assert_eq!(device.utilization(), 0.5);

        assert!(!device.is_idle(0.3));
        assert!(device.is_idle(0.6));
    }

    #[test]
    fn test_nvlink_groups() {
        let pool = DevicePool::discover().unwrap();
        let groups = pool.nvlink_groups();

        // All devices should be in exactly one group
        let mut all_devices: Vec<_> = groups.iter().flatten().copied().collect();
        all_devices.sort_by_key(|d| d.0);

        let mut pool_devices: Vec<_> = pool.devices().iter().map(|d| d.id).collect();
        pool_devices.sort_by_key(|d| d.0);

        assert_eq!(all_devices, pool_devices);
    }
}
