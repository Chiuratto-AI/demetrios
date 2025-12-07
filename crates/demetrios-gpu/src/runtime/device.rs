//! GPU Device Abstraction
//!
//! Provides a unified interface for GPU device discovery and management.

use std::fmt;
use thiserror::Error;

/// Device-related errors
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("No GPU devices available")]
    NoDevicesAvailable,

    #[error("Device not found: {0}")]
    DeviceNotFound(usize),

    #[error("Device initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Feature not supported: {0}")]
    FeatureNotSupported(String),

    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Invalid device state: {0}")]
    InvalidState(String),
}

/// GPU compute capability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub fn new(major: u32, minor: u32) -> Self {
        ComputeCapability { major, minor }
    }

    /// Check if this capability supports a minimum version
    pub fn supports(&self, min_major: u32, min_minor: u32) -> bool {
        self.major > min_major || (self.major == min_major && self.minor >= min_minor)
    }

    /// Get the SM version number (e.g., 86 for SM 8.6)
    pub fn sm_version(&self) -> u32 {
        self.major * 10 + self.minor
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM {}.{}", self.major, self.minor)
    }
}

/// Device memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total global memory in bytes
    pub total_global: usize,
    /// Free global memory in bytes
    pub free_global: usize,
    /// Total shared memory per block in bytes
    pub shared_per_block: usize,
    /// Total constant memory in bytes
    pub total_constant: usize,
    /// Memory bus width in bits
    pub bus_width: u32,
    /// Memory clock rate in kHz
    pub clock_rate: u32,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
}

impl Default for MemoryInfo {
    fn default() -> Self {
        MemoryInfo {
            total_global: 0,
            free_global: 0,
            shared_per_block: 49152, // 48KB typical
            total_constant: 65536,   // 64KB typical
            bus_width: 256,
            clock_rate: 0,
            l2_cache_size: 0,
        }
    }
}

/// Device compute limits
#[derive(Debug, Clone)]
pub struct ComputeLimits {
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum block dimensions (x, y, z)
    pub max_block_dim: [u32; 3],
    /// Maximum grid dimensions (x, y, z)
    pub max_grid_dim: [u32; 3],
    /// Warp size
    pub warp_size: u32,
    /// Number of multiprocessors (SMs)
    pub multiprocessor_count: u32,
    /// Maximum registers per block
    pub max_registers_per_block: u32,
    /// Maximum shared memory per multiprocessor
    pub max_shared_per_sm: usize,
    /// Maximum blocks per multiprocessor
    pub max_blocks_per_sm: u32,
}

impl Default for ComputeLimits {
    fn default() -> Self {
        ComputeLimits {
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [2147483647, 65535, 65535],
            warp_size: 32,
            multiprocessor_count: 1,
            max_registers_per_block: 65536,
            max_shared_per_sm: 49152,
            max_blocks_per_sm: 16,
        }
    }
}

/// GPU device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// NVIDIA CUDA GPU
    Cuda,
    /// AMD ROCm GPU
    Rocm,
    /// Intel GPU
    Intel,
    /// Apple Metal GPU
    Metal,
    /// Vulkan compute
    Vulkan,
    /// CPU simulation (for testing)
    Cpu,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cuda => write!(f, "CUDA"),
            DeviceType::Rocm => write!(f, "ROCm"),
            DeviceType::Intel => write!(f, "Intel"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::Vulkan => write!(f, "Vulkan"),
            DeviceType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Properties of a GPU device
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device index
    pub index: usize,
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: DeviceType,
    /// Compute capability
    pub compute_capability: ComputeCapability,
    /// Memory information
    pub memory: MemoryInfo,
    /// Compute limits
    pub limits: ComputeLimits,
    /// Driver version
    pub driver_version: String,
    /// Supports unified memory
    pub unified_memory: bool,
    /// Supports managed memory
    pub managed_memory: bool,
    /// Supports concurrent kernels
    pub concurrent_kernels: bool,
    /// Supports cooperative groups
    pub cooperative_groups: bool,
}

impl DeviceProperties {
    /// Create default CPU device properties (for simulation)
    pub fn cpu_default() -> Self {
        DeviceProperties {
            index: 0,
            name: "CPU Simulator".to_string(),
            device_type: DeviceType::Cpu,
            compute_capability: ComputeCapability::new(8, 0),
            memory: MemoryInfo {
                total_global: 16 * 1024 * 1024 * 1024, // 16 GB
                free_global: 16 * 1024 * 1024 * 1024,
                ..Default::default()
            },
            limits: ComputeLimits::default(),
            driver_version: "1.0.0".to_string(),
            unified_memory: true,
            managed_memory: true,
            concurrent_kernels: true,
            cooperative_groups: false,
        }
    }
}

/// GPU device handle
#[derive(Debug)]
pub struct Device {
    /// Device properties
    props: DeviceProperties,
    /// Whether the device is the current context
    is_current: bool,
}

impl Device {
    /// Create a new CPU simulation device
    pub fn cpu() -> Self {
        Device {
            props: DeviceProperties::cpu_default(),
            is_current: true,
        }
    }

    /// Create a device from properties
    pub fn from_properties(props: DeviceProperties) -> Self {
        Device {
            props,
            is_current: false,
        }
    }

    /// Get device properties
    pub fn properties(&self) -> &DeviceProperties {
        &self.props
    }

    /// Get device index
    pub fn index(&self) -> usize {
        self.props.index
    }

    /// Get device name
    pub fn name(&self) -> &str {
        &self.props.name
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        self.props.device_type
    }

    /// Get compute capability
    pub fn compute_capability(&self) -> ComputeCapability {
        self.props.compute_capability
    }

    /// Get total global memory
    pub fn total_memory(&self) -> usize {
        self.props.memory.total_global
    }

    /// Get free global memory
    pub fn free_memory(&self) -> usize {
        self.props.memory.free_global
    }

    /// Get maximum threads per block
    pub fn max_threads_per_block(&self) -> u32 {
        self.props.limits.max_threads_per_block
    }

    /// Get warp size
    pub fn warp_size(&self) -> u32 {
        self.props.limits.warp_size
    }

    /// Get number of multiprocessors
    pub fn multiprocessor_count(&self) -> u32 {
        self.props.limits.multiprocessor_count
    }

    /// Check if unified memory is supported
    pub fn supports_unified_memory(&self) -> bool {
        self.props.unified_memory
    }

    /// Make this device the current context
    pub fn set_current(&mut self) -> Result<(), DeviceError> {
        self.is_current = true;
        Ok(())
    }

    /// Check if this is the current device
    pub fn is_current(&self) -> bool {
        self.is_current
    }

    /// Synchronize all operations on this device
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        // CPU device is always synchronized
        Ok(())
    }

    /// Calculate optimal block size for a kernel
    pub fn optimal_block_size(&self, shared_mem_per_thread: usize) -> u32 {
        let max_threads = self.max_threads_per_block();
        let shared_per_block = self.props.memory.shared_per_block;

        if shared_mem_per_thread == 0 {
            // No shared memory constraint, use max
            max_threads.min(256) // 256 is often optimal
        } else {
            // Limited by shared memory
            let max_by_shared = shared_per_block / shared_mem_per_thread;
            (max_by_shared as u32).min(max_threads).next_power_of_two() / 2
        }
    }

    /// Calculate grid size for a given number of elements
    pub fn grid_size(&self, elements: usize, block_size: u32) -> u32 {
        ((elements + block_size as usize - 1) / block_size as usize) as u32
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {}, {} MB)",
            self.props.name,
            self.props.device_type,
            self.props.compute_capability,
            self.props.memory.total_global / (1024 * 1024)
        )
    }
}

/// Device manager for discovering and managing GPU devices
pub struct DeviceManager {
    devices: Vec<Device>,
    current_device: usize,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        DeviceManager {
            devices: vec![Device::cpu()], // Always have CPU fallback
            current_device: 0,
        }
    }

    /// Discover available devices
    pub fn discover(&mut self) -> Result<usize, DeviceError> {
        // In a real implementation, this would query CUDA/ROCm/etc.
        // For now, we only have the CPU simulator
        Ok(self.devices.len())
    }

    /// Get the number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get a device by index
    pub fn get_device(&self, index: usize) -> Result<&Device, DeviceError> {
        self.devices
            .get(index)
            .ok_or(DeviceError::DeviceNotFound(index))
    }

    /// Get a mutable device by index
    pub fn get_device_mut(&mut self, index: usize) -> Result<&mut Device, DeviceError> {
        self.devices
            .get_mut(index)
            .ok_or(DeviceError::DeviceNotFound(index))
    }

    /// Get the current device
    pub fn current_device(&self) -> Result<&Device, DeviceError> {
        self.get_device(self.current_device)
    }

    /// Set the current device
    pub fn set_current_device(&mut self, index: usize) -> Result<(), DeviceError> {
        if index >= self.devices.len() {
            return Err(DeviceError::DeviceNotFound(index));
        }
        self.current_device = index;
        Ok(())
    }

    /// Get all devices
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Add a device (for testing)
    pub fn add_device(&mut self, device: Device) {
        self.devices.push(device);
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_capability() {
        let cc = ComputeCapability::new(8, 6);
        assert!(cc.supports(8, 0));
        assert!(cc.supports(8, 6));
        assert!(!cc.supports(9, 0));
        assert_eq!(cc.sm_version(), 86);
    }

    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();
        assert_eq!(manager.device_count(), 1);

        let device = manager.current_device().unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_optimal_block_size() {
        let device = Device::cpu();
        let block_size = device.optimal_block_size(0);
        assert!(block_size > 0);
        assert!(block_size <= device.max_threads_per_block());
    }

    #[test]
    fn test_grid_size() {
        let device = Device::cpu();
        assert_eq!(device.grid_size(1000, 256), 4);
        assert_eq!(device.grid_size(256, 256), 1);
        assert_eq!(device.grid_size(257, 256), 2);
    }
}
