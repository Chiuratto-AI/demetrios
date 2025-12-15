//! Cross-Device Memory Management for Heterogeneous GPU Systems
//!
//! This module provides memory allocation and transfer operations across
//! mixed-architecture GPU clusters, optimizing for P2P where available.

use super::device_pool::DevicePool;
use crate::optimize::pool::DeviceId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

/// Errors during memory operations
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Allocation failed on device {device:?}: {message}")]
    AllocationFailed { device: DeviceId, message: String },

    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    #[error("Device {0:?} not found")]
    DeviceNotFound(DeviceId),

    #[error("Copy failed from {src:?} to {dst:?}: {message}")]
    CopyFailed {
        src: DeviceId,
        dst: DeviceId,
        message: String,
    },

    #[error("P2P not enabled between {src:?} and {dst:?}")]
    P2PNotEnabled { src: DeviceId, dst: DeviceId },

    #[error("Unified memory allocation failed: {0}")]
    UnifiedAllocFailed(String),

    #[error("Invalid buffer state")]
    InvalidState,
}

/// Transfer path between devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPath {
    /// Direct P2P via NVLink
    NVLink,
    /// Direct P2P via PCIe
    PCIeDirect,
    /// Staged through host memory
    StagedHost,
    /// Same device (no transfer needed)
    SameDevice,
}

impl TransferPath {
    /// Estimated bandwidth in GB/s
    pub fn bandwidth_gb_s(&self) -> f64 {
        match self {
            TransferPath::NVLink => 300.0,     // NVLink 4.0 avg
            TransferPath::PCIeDirect => 31.5,  // PCIe 4.0 x16
            TransferPath::StagedHost => 15.0,  // Through host
            TransferPath::SameDevice => f64::INFINITY,
        }
    }
}

/// Device memory buffer
pub struct DeviceBuffer {
    /// Device that owns this buffer
    pub device: DeviceId,
    /// Buffer size in bytes
    pub size: usize,
    /// Device pointer (simulated or real)
    ptr: u64,
    /// Is buffer valid?
    valid: bool,
    /// Real CUDA slice (when cuda feature enabled)
    #[cfg(feature = "cuda")]
    cuda_slice: Option<CudaSlice<u8>>,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("device", &self.device)
            .field("size", &self.size)
            .field("ptr", &self.ptr)
            .field("valid", &self.valid)
            .finish()
    }
}

impl DeviceBuffer {
    /// Create a new device buffer (simulated)
    fn new_simulated(device: DeviceId, size: usize) -> Self {
        Self {
            device,
            size,
            ptr: (device.0 as u64 * 0x1000_0000_0000) + (size as u64 & 0xFFFF_FFFF_FFFF),
            valid: true,
            #[cfg(feature = "cuda")]
            cuda_slice: None,
        }
    }

    /// Create a new device buffer with real CUDA allocation
    #[cfg(feature = "cuda")]
    fn new_cuda(device: DeviceId, size: usize, cuda_device: &Arc<CudaDevice>) -> Result<Self, MemoryError> {
        let slice: CudaSlice<u8> = cuda_device.alloc_zeros(size)
            .map_err(|e| MemoryError::AllocationFailed {
                device,
                message: e.to_string(),
            })?;

        let ptr = *slice.device_ptr() as u64;

        Ok(Self {
            device,
            size,
            ptr,
            valid: true,
            cuda_slice: Some(slice),
        })
    }

    /// Check if this buffer has a real CUDA allocation
    #[cfg(feature = "cuda")]
    pub fn has_cuda_slice(&self) -> bool {
        self.cuda_slice.is_some()
    }

    /// Get device pointer
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Check if buffer is valid
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Invalidate buffer (marks for deallocation)
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

/// Unified memory buffer accessible from all devices
#[derive(Debug)]
pub struct UnifiedBuffer {
    /// Size in bytes
    pub size: usize,
    /// Host pointer
    ptr: u64,
    /// Coherence mode
    pub coherence: CoherenceMode,
    /// Devices with active views
    active_views: RwLock<Vec<DeviceId>>,
}

impl UnifiedBuffer {
    /// Create a new unified buffer (simulated)
    fn new(size: usize, coherence: CoherenceMode) -> Self {
        Self {
            size,
            ptr: 0x7FFF_0000_0000 + (size as u64 & 0xFFFF_FFFF),
            coherence,
            active_views: RwLock::new(Vec::new()),
        }
    }

    /// Get unified pointer
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Add device to active views
    pub fn add_view(&self, device: DeviceId) {
        let mut views = self.active_views.write().unwrap();
        if !views.contains(&device) {
            views.push(device);
        }
    }

    /// Get active device views
    pub fn views(&self) -> Vec<DeviceId> {
        self.active_views.read().unwrap().clone()
    }
}

/// Memory coherence mode for unified memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoherenceMode {
    /// System-managed coherence (CUDA managed memory)
    #[default]
    Managed,
    /// Manual coherence with explicit prefetch
    Manual,
    /// Read-only (no coherence needed)
    ReadOnly,
}

/// Memory pool for a single device
#[derive(Debug, Default)]
struct DeviceMemoryPool {
    /// Total allocated bytes
    allocated: usize,
    /// Peak allocation
    peak: usize,
    /// Active buffers
    buffers: Vec<u64>,
}

impl DeviceMemoryPool {
    fn alloc(&mut self, size: usize) -> u64 {
        self.allocated += size;
        self.peak = self.peak.max(self.allocated);
        let ptr = self.buffers.len() as u64 * 0x1000 + size as u64;
        self.buffers.push(ptr);
        ptr
    }

    fn free(&mut self, _ptr: u64, size: usize) {
        self.allocated = self.allocated.saturating_sub(size);
    }
}

/// Cross-device memory manager
pub struct HeterogeneousMemoryManager {
    /// Device pool reference
    pool: Arc<DevicePool>,
    /// Per-device allocators
    allocators: RwLock<HashMap<DeviceId, DeviceMemoryPool>>,
    /// P2P enabled pairs
    p2p_enabled: RwLock<HashMap<(DeviceId, DeviceId), bool>>,
    /// Total unified memory allocated
    unified_allocated: RwLock<usize>,
    /// CUDA device handles (when cuda feature enabled)
    #[cfg(feature = "cuda")]
    cuda_devices: HashMap<DeviceId, Arc<CudaDevice>>,
}

impl HeterogeneousMemoryManager {
    /// Create a new memory manager
    pub fn new(pool: Arc<DevicePool>) -> Self {
        let mut allocators = HashMap::new();
        for device in pool.devices() {
            allocators.insert(device.id, DeviceMemoryPool::default());
        }

        // Enable P2P for all capable pairs
        let mut p2p_enabled = HashMap::new();
        for src in pool.devices() {
            for dst in pool.devices() {
                if src.id != dst.id && pool.can_p2p(src.id, dst.id) {
                    p2p_enabled.insert((src.id, dst.id), true);
                }
            }
        }

        // Initialize CUDA devices when feature enabled
        #[cfg(feature = "cuda")]
        let cuda_devices = {
            let mut devices = HashMap::new();
            for device in pool.devices() {
                if let Ok(cuda_dev) = CudaDevice::new(device.id.0 as usize) {
                    devices.insert(device.id, cuda_dev);
                }
            }
            devices
        };

        Self {
            pool,
            allocators: RwLock::new(allocators),
            p2p_enabled: RwLock::new(p2p_enabled),
            unified_allocated: RwLock::new(0),
            #[cfg(feature = "cuda")]
            cuda_devices,
        }
    }

    /// Allocate buffer on a specific device
    pub fn alloc(&self, device: DeviceId, size: usize) -> Result<DeviceBuffer, MemoryError> {
        let mut allocators = self.allocators.write().unwrap();
        let alloc = allocators
            .get_mut(&device)
            .ok_or(MemoryError::DeviceNotFound(device))?;

        alloc.alloc(size);

        // Use real CUDA allocation when available
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_dev) = self.cuda_devices.get(&device) {
                return DeviceBuffer::new_cuda(device, size, cuda_dev);
            }
        }

        // Fallback to simulated allocation
        Ok(DeviceBuffer::new_simulated(device, size))
    }

    /// Allocate typed buffer on a specific device
    pub fn alloc_typed<T>(&self, device: DeviceId, count: usize) -> Result<DeviceBuffer, MemoryError> {
        self.alloc(device, count * std::mem::size_of::<T>())
    }

    /// Free a device buffer
    pub fn free(&self, buffer: DeviceBuffer) -> Result<(), MemoryError> {
        if !buffer.is_valid() {
            return Err(MemoryError::InvalidState);
        }

        let mut allocators = self.allocators.write().unwrap();
        if let Some(alloc) = allocators.get_mut(&buffer.device) {
            alloc.free(buffer.ptr, buffer.size);
        }

        Ok(())
    }

    /// Allocate unified memory accessible from all devices
    pub fn alloc_unified(
        &self,
        size: usize,
        coherence: CoherenceMode,
    ) -> Result<UnifiedBuffer, MemoryError> {
        // In real implementation: cuMemAllocManaged
        let mut allocated = self.unified_allocated.write().unwrap();
        *allocated += size;

        Ok(UnifiedBuffer::new(size, coherence))
    }

    /// Free unified buffer
    pub fn free_unified(&self, buffer: UnifiedBuffer) -> Result<(), MemoryError> {
        let mut allocated = self.unified_allocated.write().unwrap();
        *allocated = allocated.saturating_sub(buffer.size);
        Ok(())
    }

    /// Copy data between devices
    pub fn copy(
        &self,
        src_device: DeviceId,
        src_ptr: u64,
        dst_device: DeviceId,
        dst_ptr: u64,
        size: usize,
    ) -> Result<(), MemoryError> {
        let path = self.optimal_transfer_path(src_device, dst_device);

        match path {
            TransferPath::SameDevice => Ok(()),
            TransferPath::NVLink | TransferPath::PCIeDirect => {
                self.p2p_copy(src_device, src_ptr, dst_device, dst_ptr, size)
            }
            TransferPath::StagedHost => {
                self.staged_copy(src_device, src_ptr, dst_device, dst_ptr, size)
            }
        }
    }

    /// Copy buffer to another device
    pub fn copy_buffer(
        &self,
        src: &DeviceBuffer,
        dst_device: DeviceId,
    ) -> Result<DeviceBuffer, MemoryError> {
        let dst = self.alloc(dst_device, src.size)?;
        self.copy(src.device, src.ptr, dst_device, dst.ptr, src.size)?;
        Ok(dst)
    }

    /// Prefetch unified buffer to a device
    pub fn prefetch(&self, buffer: &UnifiedBuffer, device: DeviceId) -> Result<(), MemoryError> {
        if !self.pool.devices().iter().any(|d| d.id == device) {
            return Err(MemoryError::DeviceNotFound(device));
        }

        // In real implementation: cuMemPrefetchAsync
        buffer.add_view(device);
        Ok(())
    }

    /// Get optimal transfer path between devices
    pub fn optimal_transfer_path(&self, src: DeviceId, dst: DeviceId) -> TransferPath {
        if src == dst {
            return TransferPath::SameDevice;
        }

        let p2p_enabled = self.p2p_enabled.read().unwrap();
        if !p2p_enabled.get(&(src, dst)).copied().unwrap_or(false) {
            return TransferPath::StagedHost;
        }

        // Check for NVLink
        if let Some(p2p) = self.pool.p2p_capability(src, dst) {
            if p2p.has_nvlink() {
                return TransferPath::NVLink;
            }
        }

        TransferPath::PCIeDirect
    }

    /// Check if P2P is enabled between devices
    pub fn has_p2p(&self, src: DeviceId, dst: DeviceId) -> bool {
        let p2p_enabled = self.p2p_enabled.read().unwrap();
        p2p_enabled.get(&(src, dst)).copied().unwrap_or(false)
    }

    /// Enable P2P access between devices
    pub fn enable_p2p(&self, src: DeviceId, dst: DeviceId) -> Result<bool, MemoryError> {
        if !self.pool.can_p2p(src, dst) {
            return Ok(false);
        }

        // In real implementation: cuCtxEnablePeerAccess
        let mut p2p_enabled = self.p2p_enabled.write().unwrap();
        p2p_enabled.insert((src, dst), true);
        Ok(true)
    }

    /// Disable P2P access between devices
    pub fn disable_p2p(&self, src: DeviceId, dst: DeviceId) {
        let mut p2p_enabled = self.p2p_enabled.write().unwrap();
        p2p_enabled.insert((src, dst), false);
    }

    /// Get memory info for a device
    pub fn memory_info(&self, device: DeviceId) -> Option<MemoryInfo> {
        let allocators = self.allocators.read().unwrap();
        let alloc = allocators.get(&device)?;
        let device_info = self.pool.get(device)?;

        Some(MemoryInfo {
            device,
            total: device_info.memory_bytes,
            allocated: alloc.allocated,
            peak: alloc.peak,
            free: device_info.memory_bytes.saturating_sub(alloc.allocated),
        })
    }

    /// Get total unified memory allocated
    pub fn unified_memory_allocated(&self) -> usize {
        *self.unified_allocated.read().unwrap()
    }

    // === Internal transfer implementations ===

    fn p2p_copy(
        &self,
        _src_device: DeviceId,
        _src_ptr: u64,
        _dst_device: DeviceId,
        _dst_ptr: u64,
        _size: usize,
    ) -> Result<(), MemoryError> {
        // In real implementation: cuMemcpyPeerAsync
        Ok(())
    }

    fn staged_copy(
        &self,
        _src_device: DeviceId,
        _src_ptr: u64,
        _dst_device: DeviceId,
        _dst_ptr: u64,
        _size: usize,
    ) -> Result<(), MemoryError> {
        // In real implementation:
        // 1. cuMemcpyDtoH to host staging buffer
        // 2. cuMemcpyHtoD to destination
        Ok(())
    }
}

/// Memory information for a device
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Device ID
    pub device: DeviceId,
    /// Total memory (bytes)
    pub total: usize,
    /// Currently allocated (bytes)
    pub allocated: usize,
    /// Peak allocation (bytes)
    pub peak: usize,
    /// Free memory (bytes)
    pub free: usize,
}

impl MemoryInfo {
    /// Utilization as fraction (0.0 - 1.0)
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.allocated as f64 / self.total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> HeterogeneousMemoryManager {
        let pool = DevicePool::discover().unwrap();
        HeterogeneousMemoryManager::new(Arc::new(pool))
    }

    #[test]
    fn test_alloc_and_free() {
        let manager = create_test_manager();
        let buffer = manager.alloc(DeviceId(0), 1024).unwrap();

        assert_eq!(buffer.device, DeviceId(0));
        assert_eq!(buffer.size, 1024);
        assert!(buffer.is_valid());

        manager.free(buffer).unwrap();
    }

    #[test]
    fn test_alloc_typed() {
        let manager = create_test_manager();
        let buffer = manager.alloc_typed::<f32>(DeviceId(0), 256).unwrap();

        assert_eq!(buffer.size, 256 * 4); // f32 is 4 bytes
    }

    #[test]
    fn test_unified_memory() {
        let manager = create_test_manager();
        let buffer = manager.alloc_unified(4096, CoherenceMode::Managed).unwrap();

        assert_eq!(buffer.size, 4096);
        assert_eq!(buffer.coherence, CoherenceMode::Managed);

        manager.prefetch(&buffer, DeviceId(0)).unwrap();
        assert!(buffer.views().contains(&DeviceId(0)));

        manager.free_unified(buffer).unwrap();
    }

    #[test]
    fn test_transfer_path() {
        let manager = create_test_manager();

        // Same device
        assert_eq!(
            manager.optimal_transfer_path(DeviceId(0), DeviceId(0)),
            TransferPath::SameDevice
        );

        // Different devices - should be P2P or staged
        let path = manager.optimal_transfer_path(DeviceId(0), DeviceId(1));
        assert!(matches!(
            path,
            TransferPath::NVLink | TransferPath::PCIeDirect | TransferPath::StagedHost
        ));
    }

    #[test]
    fn test_copy_buffer() {
        let manager = create_test_manager();
        let src = manager.alloc(DeviceId(0), 1024).unwrap();
        let dst = manager.copy_buffer(&src, DeviceId(1)).unwrap();

        assert_eq!(dst.device, DeviceId(1));
        assert_eq!(dst.size, src.size);
    }

    #[test]
    fn test_memory_info() {
        let manager = create_test_manager();

        // Allocate some memory
        let _buffer = manager.alloc(DeviceId(0), 4096).unwrap();

        let info = manager.memory_info(DeviceId(0)).unwrap();
        assert_eq!(info.device, DeviceId(0));
        assert!(info.allocated >= 4096);
        assert!(info.total > 0);
    }

    #[test]
    fn test_p2p_enable_disable() {
        let manager = create_test_manager();

        let has_p2p = manager.has_p2p(DeviceId(0), DeviceId(1));
        if has_p2p {
            manager.disable_p2p(DeviceId(0), DeviceId(1));
            assert!(!manager.has_p2p(DeviceId(0), DeviceId(1)));

            manager.enable_p2p(DeviceId(0), DeviceId(1)).unwrap();
            assert!(manager.has_p2p(DeviceId(0), DeviceId(1)));
        }
    }

    #[test]
    fn test_transfer_bandwidth() {
        assert!(TransferPath::NVLink.bandwidth_gb_s() > TransferPath::PCIeDirect.bandwidth_gb_s());
        assert!(TransferPath::PCIeDirect.bandwidth_gb_s() > TransferPath::StagedHost.bandwidth_gb_s());
        assert!(TransferPath::SameDevice.bandwidth_gb_s().is_infinite());
    }
}
