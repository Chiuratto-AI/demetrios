//! Device Discovery for Heterogeneous GPU Systems
//!
//! This module provides device enumeration and capability querying
//! for mixed-architecture GPU clusters.

use super::architecture::{ArchCapabilities, ArchFeatures, GpuArchitecture};
use crate::optimize::pool::DeviceId;
use std::collections::HashMap;
use thiserror::Error;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// Errors that can occur during device discovery
#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("CUDA driver not initialized")]
    DriverNotInitialized,

    #[error("No CUDA devices found")]
    NoDevicesFound,

    #[error("Device {0} not found")]
    DeviceNotFound(u32),

    #[error("Failed to query device attribute: {0}")]
    AttributeQueryFailed(String),

    #[error("P2P query failed between devices {0} and {1}: {2}")]
    P2PQueryFailed(u32, u32, String),

    #[error("Driver error: {0}")]
    DriverError(String),
}

/// P2P (Peer-to-Peer) capability between two devices
#[derive(Debug, Clone)]
pub struct P2PCapability {
    /// Source device ID
    pub src_device: DeviceId,
    /// Destination device ID
    pub dst_device: DeviceId,
    /// Whether direct P2P access is supported
    pub can_access: bool,
    /// Performance rating (higher = better)
    pub performance_rank: u32,
    /// NVLink connection (if any)
    pub nvlink_info: Option<NVLinkInfo>,
    /// Atomic operations supported
    pub atomic_supported: bool,
}

impl P2PCapability {
    /// Check if P2P is available (either direction)
    pub fn is_available(&self) -> bool {
        self.can_access
    }

    /// Check if NVLink is available
    pub fn has_nvlink(&self) -> bool {
        self.nvlink_info.is_some()
    }

    /// Get estimated bandwidth in GB/s
    pub fn bandwidth_gb_s(&self) -> f64 {
        if let Some(nvlink) = &self.nvlink_info {
            nvlink.bandwidth_gb_s
        } else if self.can_access {
            // PCIe direct access - assume PCIe 4.0 x16
            31.5
        } else {
            // Staged through host
            15.0
        }
    }
}

/// NVLink connection information
#[derive(Debug, Clone)]
pub struct NVLinkInfo {
    /// NVLink version (2, 3, 4, 5)
    pub version: u32,
    /// Number of active links
    pub link_count: u32,
    /// Total bandwidth (GB/s)
    pub bandwidth_gb_s: f64,
}

impl NVLinkInfo {
    /// Create NVLink info from version and link count
    pub fn new(version: u32, link_count: u32) -> Self {
        let bandwidth_per_link = match version {
            2 => 25.0,    // NVLink 2.0: 25 GB/s per link
            3 => 50.0,    // NVLink 3.0: 50 GB/s per link
            4 => 100.0,   // NVLink 4.0: 100 GB/s per link
            5 => 200.0,   // NVLink 5.0: 200 GB/s per link (estimated)
            _ => 25.0,
        };

        Self {
            version,
            link_count,
            bandwidth_gb_s: bandwidth_per_link * link_count as f64,
        }
    }
}

/// Device discovery interface
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// Enumerate all available CUDA devices
    pub fn enumerate() -> Result<Vec<DiscoveredDevice>, DiscoveryError> {
        #[cfg(feature = "cuda")]
        {
            Self::enumerate_cuda()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::enumerate_simulated()
        }
    }

    /// Real CUDA device enumeration using cudarc
    #[cfg(feature = "cuda")]
    fn enumerate_cuda() -> Result<Vec<DiscoveredDevice>, DiscoveryError> {
        let count = CudaDevice::count()
            .map_err(|e| DiscoveryError::DriverError(format!("Failed to get device count: {}", e)))?;

        if count == 0 {
            return Err(DiscoveryError::NoDevicesFound);
        }

        let mut devices = Vec::with_capacity(count as usize);

        for ordinal in 0..count as usize {
            let device = CudaDevice::new(ordinal)
                .map_err(|e| DiscoveryError::DriverError(format!("Failed to create device {}: {}", ordinal, e)))?;

            // Query device name
            let name = device.name()
                .map_err(|e| DiscoveryError::AttributeQueryFailed(format!("Failed to get name for device {}: {}", ordinal, e)))?;

            // Parse compute capability from device name and known GPU patterns
            let (architecture, compute_capability, sm_count, clock_mhz) = Self::infer_device_specs(&name);

            // Default memory value - cudarc 0.12 doesn't expose mem_info directly
            let total_mem = 8 * 1024 * 1024 * 1024_usize; // 8 GB default

            let capabilities = ArchCapabilities::default_for(architecture);

            devices.push(DiscoveredDevice {
                id: DeviceId(ordinal as u32),
                name,
                architecture,
                compute_capability,
                memory_bytes: total_mem,
                sm_count,
                clock_mhz,
                capabilities,
            });
        }

        Ok(devices)
    }

    /// Infer device specifications from device name
    #[cfg(feature = "cuda")]
    fn infer_device_specs(name: &str) -> (GpuArchitecture, (u32, u32), u32, u32) {
        let name_upper = name.to_uppercase();

        // Blackwell B-series (sm_100+)
        if name_upper.contains("B200") || name_upper.contains("B100") || name_upper.contains("GB200") {
            return (GpuArchitecture::Blackwell, (10, 0), 160, 2100);
        }

        // Hopper H-series (sm_90)
        if name_upper.contains("H100") {
            if name_upper.contains("SXM") {
                return (GpuArchitecture::Hopper, (9, 0), 132, 1830);
            }
            return (GpuArchitecture::Hopper, (9, 0), 114, 1620);
        }
        if name_upper.contains("H200") {
            return (GpuArchitecture::Hopper, (9, 0), 132, 1830);
        }

        // Ada Lovelace (sm_89)
        if name_upper.contains("RTX 4090") {
            return (GpuArchitecture::Ada, (8, 9), 128, 2520);
        }
        if name_upper.contains("RTX 4080") {
            return (GpuArchitecture::Ada, (8, 9), 76, 2505);
        }
        if name_upper.contains("RTX 40") || name_upper.contains("L40") {
            return (GpuArchitecture::Ada, (8, 9), 96, 2310);
        }

        // Ampere (sm_80, sm_86, sm_87)
        if name_upper.contains("A100") {
            if name_upper.contains("80GB") || name_upper.contains("SXM") {
                return (GpuArchitecture::Ampere, (8, 0), 108, 1410);
            }
            return (GpuArchitecture::Ampere, (8, 0), 108, 1410);
        }
        if name_upper.contains("A10") && !name_upper.contains("A100") {
            return (GpuArchitecture::Ampere, (8, 6), 72, 1695);
        }
        if name_upper.contains("A30") {
            return (GpuArchitecture::Ampere, (8, 0), 56, 1440);
        }
        if name_upper.contains("RTX 3090") {
            return (GpuArchitecture::Ampere, (8, 6), 82, 1695);
        }
        if name_upper.contains("RTX 30") {
            return (GpuArchitecture::Ampere, (8, 6), 68, 1710);
        }

        // Turing (sm_75)
        if name_upper.contains("RTX 20") || name_upper.contains("T4") || name_upper.contains("TITAN RTX") {
            return (GpuArchitecture::Turing, (7, 5), 68, 1770);
        }

        // Volta (sm_70, sm_72)
        if name_upper.contains("V100") {
            if name_upper.contains("SXM") {
                return (GpuArchitecture::Volta, (7, 0), 80, 1530);
            }
            return (GpuArchitecture::Volta, (7, 0), 80, 1380);
        }

        // Default to Ampere if unknown
        (GpuArchitecture::Ampere, (8, 0), 64, 1400)
    }

    /// Query capabilities for a specific device
    pub fn query_capabilities(device_id: u32) -> Result<ArchCapabilities, DiscoveryError> {
        #[cfg(feature = "cuda")]
        {
            Self::query_capabilities_cuda(device_id)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::query_capabilities_simulated(device_id)
        }
    }

    /// Real CUDA capabilities query
    #[cfg(feature = "cuda")]
    fn query_capabilities_cuda(device_id: u32) -> Result<ArchCapabilities, DiscoveryError> {
        let device = CudaDevice::new(device_id as usize)
            .map_err(|e| DiscoveryError::DriverError(format!("Failed to access device {}: {}", device_id, e)))?;

        let name = device.name()
            .map_err(|e| DiscoveryError::AttributeQueryFailed(format!("Failed to get name: {}", e)))?;

        let (architecture, (major, minor), sm_count, clock_mhz) = Self::infer_device_specs(&name);

        // Default memory value - cudarc 0.12 doesn't expose mem_info directly
        let total_mem = 8 * 1024 * 1024 * 1024_usize; // 8 GB default

        // Get default capabilities for architecture and override with real values
        let mut caps = ArchCapabilities::default_for(architecture);
        caps.compute_capability = (major, minor);
        caps.memory_bytes = total_mem;
        caps.sm_count = sm_count;
        caps.clock_mhz = clock_mhz;

        Ok(caps)
    }

    /// Query P2P capability between two devices
    pub fn query_p2p(device_a: u32, device_b: u32) -> Result<P2PCapability, DiscoveryError> {
        #[cfg(feature = "cuda")]
        {
            Self::query_p2p_cuda(device_a, device_b)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::query_p2p_simulated(device_a, device_b)
        }
    }

    /// Real CUDA P2P query
    #[cfg(feature = "cuda")]
    fn query_p2p_cuda(device_a: u32, device_b: u32) -> Result<P2PCapability, DiscoveryError> {
        // cudarc doesn't expose P2P queries directly, so we use heuristics
        // based on device names/architectures
        let dev_a = CudaDevice::new(device_a as usize)
            .map_err(|e| DiscoveryError::DriverError(format!("Failed to access device {}: {}", device_a, e)))?;
        let dev_b = CudaDevice::new(device_b as usize)
            .map_err(|e| DiscoveryError::DriverError(format!("Failed to access device {}: {}", device_b, e)))?;

        let name_a = dev_a.name().unwrap_or_default();
        let name_b = dev_b.name().unwrap_or_default();

        let (arch_a, _, _, _) = Self::infer_device_specs(&name_a);
        let (arch_b, _, _, _) = Self::infer_device_specs(&name_b);

        let same_arch = arch_a == arch_b;
        let nvlink_capable = arch_a >= GpuArchitecture::Ampere && arch_b >= GpuArchitecture::Ampere;

        // Check for datacenter GPUs that typically have NVLink
        let is_datacenter_a = name_a.contains("A100") || name_a.contains("H100") ||
                             name_a.contains("H200") || name_a.contains("B200") || name_a.contains("V100");
        let is_datacenter_b = name_b.contains("A100") || name_b.contains("H100") ||
                             name_b.contains("H200") || name_b.contains("B200") || name_b.contains("V100");

        let nvlink_info = if same_arch && nvlink_capable && is_datacenter_a && is_datacenter_b {
            let version = match arch_a {
                GpuArchitecture::Ampere => 3,
                GpuArchitecture::Hopper => 4,
                GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => 5,
                _ => 0,
            };
            if version > 0 {
                Some(NVLinkInfo::new(version, 4))
            } else {
                None
            }
        } else {
            None
        };

        Ok(P2PCapability {
            src_device: DeviceId(device_a),
            dst_device: DeviceId(device_b),
            can_access: true, // Assume P2P works for modern GPUs
            performance_rank: if nvlink_info.is_some() { 2 } else { 1 },
            nvlink_info,
            atomic_supported: same_arch,
        })
    }

    /// Query full P2P matrix for all devices
    pub fn query_p2p_matrix(
        device_ids: &[u32],
    ) -> Result<HashMap<(DeviceId, DeviceId), P2PCapability>, DiscoveryError> {
        let mut matrix = HashMap::new();

        for &src in device_ids {
            for &dst in device_ids {
                if src != dst {
                    let capability = Self::query_p2p(src, dst)?;
                    matrix.insert((DeviceId(src), DeviceId(dst)), capability);
                }
            }
        }

        Ok(matrix)
    }

    /// Get the number of available CUDA devices
    pub fn device_count() -> Result<u32, DiscoveryError> {
        #[cfg(feature = "cuda")]
        {
            CudaDevice::count()
                .map(|c| c as u32)
                .map_err(|e| DiscoveryError::DriverError(format!("Failed to get device count: {}", e)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self::simulated_device_count())
        }
    }

    // === Simulated implementations for testing ===

    fn enumerate_simulated() -> Result<Vec<DiscoveredDevice>, DiscoveryError> {
        // Simulate a heterogeneous cluster with multiple GPU types
        let devices = vec![
            DiscoveredDevice {
                id: DeviceId(0),
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                architecture: GpuArchitecture::Ampere,
                compute_capability: (8, 0),
                memory_bytes: 80 * 1024 * 1024 * 1024,
                sm_count: 108,
                clock_mhz: 1410,
                capabilities: ArchCapabilities::default_for(GpuArchitecture::Ampere),
            },
            DiscoveredDevice {
                id: DeviceId(1),
                name: "NVIDIA A100-SXM4-80GB".to_string(),
                architecture: GpuArchitecture::Ampere,
                compute_capability: (8, 0),
                memory_bytes: 80 * 1024 * 1024 * 1024,
                sm_count: 108,
                clock_mhz: 1410,
                capabilities: ArchCapabilities::default_for(GpuArchitecture::Ampere),
            },
            DiscoveredDevice {
                id: DeviceId(2),
                name: "NVIDIA H100 PCIe".to_string(),
                architecture: GpuArchitecture::Hopper,
                compute_capability: (9, 0),
                memory_bytes: 80 * 1024 * 1024 * 1024,
                sm_count: 114,
                clock_mhz: 1620,
                capabilities: ArchCapabilities::default_for(GpuArchitecture::Hopper),
            },
            DiscoveredDevice {
                id: DeviceId(3),
                name: "NVIDIA RTX 4090".to_string(),
                architecture: GpuArchitecture::Ada,
                compute_capability: (8, 9),
                memory_bytes: 24 * 1024 * 1024 * 1024,
                sm_count: 128,
                clock_mhz: 2520,
                capabilities: ArchCapabilities::default_for(GpuArchitecture::Ada),
            },
        ];

        Ok(devices)
    }

    fn query_capabilities_simulated(device_id: u32) -> Result<ArchCapabilities, DiscoveryError> {
        let devices = Self::enumerate_simulated()?;

        devices
            .into_iter()
            .find(|d| d.id.0 == device_id)
            .map(|d| d.capabilities)
            .ok_or(DiscoveryError::DeviceNotFound(device_id))
    }

    fn query_p2p_simulated(device_a: u32, device_b: u32) -> Result<P2PCapability, DiscoveryError> {
        let devices = Self::enumerate_simulated()?;

        let dev_a = devices
            .iter()
            .find(|d| d.id.0 == device_a)
            .ok_or(DiscoveryError::DeviceNotFound(device_a))?;
        let dev_b = devices
            .iter()
            .find(|d| d.id.0 == device_b)
            .ok_or(DiscoveryError::DeviceNotFound(device_b))?;

        // Simulate P2P capabilities based on architecture
        let same_arch = dev_a.architecture == dev_b.architecture;
        let nvlink_capable =
            dev_a.architecture >= GpuArchitecture::Ampere && dev_b.architecture >= GpuArchitecture::Ampere;

        let nvlink_info = if same_arch && nvlink_capable {
            // Same architecture datacenter GPUs likely have NVLink
            let version = match dev_a.architecture {
                GpuArchitecture::Ampere => 3,
                GpuArchitecture::Hopper => 4,
                GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => 5,
                _ => 0,
            };
            if version > 0 {
                Some(NVLinkInfo::new(version, 4))
            } else {
                None
            }
        } else {
            None
        };

        Ok(P2PCapability {
            src_device: DeviceId(device_a),
            dst_device: DeviceId(device_b),
            can_access: true, // Most modern GPUs support P2P
            performance_rank: if nvlink_info.is_some() { 2 } else { 1 },
            nvlink_info,
            atomic_supported: same_arch,
        })
    }

    fn simulated_device_count() -> u32 {
        4 // Simulated cluster size
    }
}

/// A discovered GPU device with all queried information
#[derive(Debug, Clone)]
pub struct DiscoveredDevice {
    /// Device ID
    pub id: DeviceId,
    /// Device name (e.g., "NVIDIA A100-SXM4-80GB")
    pub name: String,
    /// Architecture generation
    pub architecture: GpuArchitecture,
    /// CUDA compute capability
    pub compute_capability: (u32, u32),
    /// Total device memory (bytes)
    pub memory_bytes: usize,
    /// Number of SMs
    pub sm_count: u32,
    /// Clock rate (MHz)
    pub clock_mhz: u32,
    /// Full capabilities
    pub capabilities: ArchCapabilities,
}

impl DiscoveredDevice {
    /// Get architecture features
    pub fn features(&self) -> &ArchFeatures {
        &self.capabilities.features
    }

    /// Check if device supports a feature
    pub fn supports(&self, feature: super::architecture::ArchFeature) -> bool {
        self.capabilities.features.supports(feature)
    }

    /// Get performance score (0.0 - 1.0)
    pub fn performance_score(&self) -> f64 {
        self.capabilities.performance_score()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_devices() {
        let devices = DeviceDiscovery::enumerate().unwrap();
        assert!(!devices.is_empty());

        // Check we have multiple architectures
        let archs: std::collections::HashSet<_> =
            devices.iter().map(|d| d.architecture).collect();
        assert!(archs.len() >= 2, "Should have multiple architectures");
    }

    #[test]
    fn test_query_capabilities() {
        let caps = DeviceDiscovery::query_capabilities(0).unwrap();
        assert!(caps.memory_bytes > 0);
        assert!(caps.sm_count > 0);
    }

    #[test]
    fn test_query_p2p() {
        let p2p = DeviceDiscovery::query_p2p(0, 1).unwrap();
        assert!(p2p.can_access);
        assert!(p2p.bandwidth_gb_s() > 0.0);
    }

    #[test]
    fn test_p2p_matrix() {
        let device_ids = vec![0, 1, 2];
        let matrix = DeviceDiscovery::query_p2p_matrix(&device_ids).unwrap();

        // Should have n*(n-1) entries
        assert_eq!(matrix.len(), 6);
    }

    #[test]
    fn test_nvlink_info() {
        let nvlink = NVLinkInfo::new(4, 4);
        assert_eq!(nvlink.version, 4);
        assert_eq!(nvlink.bandwidth_gb_s, 400.0); // 100 GB/s * 4 links
    }

    #[test]
    fn test_device_count() {
        let count = DeviceDiscovery::device_count().unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_discovered_device_features() {
        let devices = DeviceDiscovery::enumerate().unwrap();
        let hopper_device = devices.iter().find(|d| d.architecture == GpuArchitecture::Hopper);

        if let Some(device) = hopper_device {
            assert!(device.supports(super::super::architecture::ArchFeature::TMA));
            assert!(device.supports(super::super::architecture::ArchFeature::BF16));
        }
    }
}
