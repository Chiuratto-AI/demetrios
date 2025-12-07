//! Kernel Launch Configuration
//!
//! Provides types and utilities for configuring and launching GPU kernels.

use super::buffer::GpuBuffer;
use super::device::Device;
use super::kernel::{Kernel, KernelError};
use super::stream::Stream;
use std::any::Any;
use std::fmt;
use thiserror::Error;

/// Launch-related errors
#[derive(Debug, Error)]
pub enum LaunchError {
    #[error("Invalid grid dimensions: {0}")]
    InvalidGrid(String),

    #[error("Invalid block dimensions: {0}")]
    InvalidBlock(String),

    #[error("Shared memory exceeds limit: requested {requested}, available {available}")]
    SharedMemoryExceeded { requested: usize, available: usize },

    #[error("Too many threads per block: requested {requested}, max {max}")]
    TooManyThreads { requested: u32, max: u32 },

    #[error("Argument count mismatch: expected {expected}, got {actual}")]
    ArgumentCountMismatch { expected: usize, actual: usize },

    #[error("Kernel error: {0}")]
    Kernel(#[from] KernelError),
}

/// 3D dimensions for grid and block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Create a 1D dimension
    pub fn x(x: u32) -> Self {
        Dim3 { x, y: 1, z: 1 }
    }

    /// Create a 2D dimension
    pub fn xy(x: u32, y: u32) -> Self {
        Dim3 { x, y, z: 1 }
    }

    /// Create a 3D dimension
    pub fn xyz(x: u32, y: u32, z: u32) -> Self {
        Dim3 { x, y, z }
    }

    /// Get total number of elements
    pub fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }

    /// Check if all dimensions are positive
    pub fn is_valid(&self) -> bool {
        self.x > 0 && self.y > 0 && self.z > 0
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Dim3 { x: 1, y: 1, z: 1 }
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Dim3::x(x)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Dim3::xy(x, y)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Dim3::xyz(x, y, z)
    }
}

impl fmt::Display for Dim3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.z == 1 && self.y == 1 {
            write!(f, "{}", self.x)
        } else if self.z == 1 {
            write!(f, "({}, {})", self.x, self.y)
        } else {
            write!(f, "({}, {}, {})", self.x, self.y, self.z)
        }
    }
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (number of blocks)
    pub grid: Dim3,
    /// Block dimensions (threads per block)
    pub block: Dim3,
    /// Dynamic shared memory size in bytes
    pub shared_mem: usize,
}

impl LaunchConfig {
    /// Create a new launch configuration
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        LaunchConfig {
            grid: grid.into(),
            block: block.into(),
            shared_mem: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_mem(mut self, size: usize) -> Self {
        self.shared_mem = size;
        self
    }

    /// Create a 1D launch configuration for N elements
    pub fn for_elements(n: usize, threads_per_block: u32) -> Self {
        let blocks = ((n as u64 + threads_per_block as u64 - 1) / threads_per_block as u64) as u32;
        LaunchConfig::new(blocks, threads_per_block)
    }

    /// Create a 2D launch configuration
    pub fn for_2d(width: u32, height: u32, block_width: u32, block_height: u32) -> Self {
        let grid_x = (width + block_width - 1) / block_width;
        let grid_y = (height + block_height - 1) / block_height;
        LaunchConfig::new((grid_x, grid_y), (block_width, block_height))
    }

    /// Get total number of threads
    pub fn total_threads(&self) -> u64 {
        self.grid.total() * self.block.total()
    }

    /// Get threads per block
    pub fn threads_per_block(&self) -> u64 {
        self.block.total()
    }

    /// Validate the launch configuration
    pub fn validate(&self, device: &Device) -> Result<(), LaunchError> {
        if !self.grid.is_valid() {
            return Err(LaunchError::InvalidGrid(format!(
                "Grid dimensions must be positive: {}",
                self.grid
            )));
        }

        if !self.block.is_valid() {
            return Err(LaunchError::InvalidBlock(format!(
                "Block dimensions must be positive: {}",
                self.block
            )));
        }

        let max_threads = device.max_threads_per_block();
        let threads = self.block.total() as u32;
        if threads > max_threads {
            return Err(LaunchError::TooManyThreads {
                requested: threads,
                max: max_threads,
            });
        }

        let max_shared = device.properties().memory.shared_per_block;
        if self.shared_mem > max_shared {
            return Err(LaunchError::SharedMemoryExceeded {
                requested: self.shared_mem,
                available: max_shared,
            });
        }

        Ok(())
    }
}

impl fmt::Display for LaunchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<<<{}, {}>>>", self.grid, self.block)?;
        if self.shared_mem > 0 {
            write!(f, " (shared: {} bytes)", self.shared_mem)?;
        }
        Ok(())
    }
}

/// Kernel argument (type-erased)
#[derive(Debug)]
pub enum KernelArg {
    /// 32-bit integer
    I32(i32),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit integer
    I64(i64),
    /// 64-bit unsigned integer
    U64(u64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Device pointer
    DevicePtr(*const u8),
    /// Mutable device pointer
    DevicePtrMut(*mut u8),
}

impl KernelArg {
    /// Create from a device buffer (read-only)
    pub fn from_buffer<T: Copy>(buffer: &GpuBuffer<T>) -> Self {
        KernelArg::DevicePtr(buffer.as_device_ptr() as *const u8)
    }

    /// Create from a device buffer (read-write)
    pub fn from_buffer_mut<T: Copy>(buffer: &mut GpuBuffer<T>) -> Self {
        KernelArg::DevicePtrMut(buffer.as_device_ptr_mut() as *mut u8)
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            KernelArg::I32(_) | KernelArg::U32(_) | KernelArg::F32(_) => 4,
            KernelArg::I64(_) | KernelArg::U64(_) | KernelArg::F64(_) => 8,
            KernelArg::DevicePtr(_) | KernelArg::DevicePtrMut(_) => 8,
        }
    }

    /// Get as raw bytes
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            KernelArg::I32(v) => v.to_le_bytes().to_vec(),
            KernelArg::U32(v) => v.to_le_bytes().to_vec(),
            KernelArg::I64(v) => v.to_le_bytes().to_vec(),
            KernelArg::U64(v) => v.to_le_bytes().to_vec(),
            KernelArg::F32(v) => v.to_le_bytes().to_vec(),
            KernelArg::F64(v) => v.to_le_bytes().to_vec(),
            KernelArg::DevicePtr(p) => (*p as u64).to_le_bytes().to_vec(),
            KernelArg::DevicePtrMut(p) => (*p as u64).to_le_bytes().to_vec(),
        }
    }
}

impl From<i32> for KernelArg {
    fn from(v: i32) -> Self {
        KernelArg::I32(v)
    }
}

impl From<u32> for KernelArg {
    fn from(v: u32) -> Self {
        KernelArg::U32(v)
    }
}

impl From<i64> for KernelArg {
    fn from(v: i64) -> Self {
        KernelArg::I64(v)
    }
}

impl From<u64> for KernelArg {
    fn from(v: u64) -> Self {
        KernelArg::U64(v)
    }
}

impl From<f32> for KernelArg {
    fn from(v: f32) -> Self {
        KernelArg::F32(v)
    }
}

impl From<f64> for KernelArg {
    fn from(v: f64) -> Self {
        KernelArg::F64(v)
    }
}

/// Kernel launcher
pub struct KernelLauncher<'a> {
    kernel: &'a Kernel,
    config: LaunchConfig,
    args: Vec<KernelArg>,
    stream: Option<&'a Stream>,
}

impl<'a> KernelLauncher<'a> {
    /// Create a new kernel launcher
    pub fn new(kernel: &'a Kernel, config: LaunchConfig) -> Self {
        KernelLauncher {
            kernel,
            config,
            args: Vec::new(),
            stream: None,
        }
    }

    /// Add an argument
    pub fn arg(mut self, arg: impl Into<KernelArg>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add a buffer argument (read-only)
    pub fn buffer<T: Copy>(mut self, buffer: &GpuBuffer<T>) -> Self {
        self.args.push(KernelArg::from_buffer(buffer));
        self
    }

    /// Add a buffer argument (read-write)
    pub fn buffer_mut<T: Copy>(mut self, buffer: &mut GpuBuffer<T>) -> Self {
        self.args.push(KernelArg::from_buffer_mut(buffer));
        self
    }

    /// Set the stream
    pub fn stream(mut self, stream: &'a Stream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Launch the kernel (CPU simulation)
    pub fn launch(self, device: &Device) -> Result<(), LaunchError> {
        // Validate configuration
        self.config.validate(device)?;

        // Validate argument count
        if self.args.len() != self.kernel.num_params() {
            return Err(LaunchError::ArgumentCountMismatch {
                expected: self.kernel.num_params(),
                actual: self.args.len(),
            });
        }

        // For CPU simulation, we would interpret the kernel here
        // In a real implementation, this would call the GPU driver

        if let Some(stream) = self.stream {
            stream.begin_operation();
        }

        // Simulate kernel execution
        // In practice, this would iterate over all threads and execute the kernel logic
        let total_threads = self.config.total_threads();
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "Simulating kernel '{}' with {} threads {}",
                self.kernel.name(),
                total_threads,
                self.config
            );
        }

        if let Some(stream) = self.stream {
            stream.end_operation();
        }

        Ok(())
    }
}

/// Compute optimal launch configuration
pub fn compute_launch_config(device: &Device, elements: usize, kernel: &Kernel) -> LaunchConfig {
    // Use kernel's max threads hint or device default
    let threads_per_block = kernel
        .max_threads_per_block()
        .unwrap_or(256)
        .min(device.max_threads_per_block());

    LaunchConfig::for_elements(elements, threads_per_block)
}

/// Occupancy calculator
pub struct OccupancyCalculator<'a> {
    device: &'a Device,
    kernel: &'a Kernel,
}

impl<'a> OccupancyCalculator<'a> {
    pub fn new(device: &'a Device, kernel: &'a Kernel) -> Self {
        OccupancyCalculator { device, kernel }
    }

    /// Calculate theoretical occupancy
    pub fn theoretical_occupancy(&self, threads_per_block: u32, shared_mem: usize) -> f32 {
        let props = self.device.properties();

        // Threads limited by block size
        let threads = threads_per_block.min(props.limits.max_threads_per_block);

        // Warps per block
        let warps_per_block = (threads + props.limits.warp_size - 1) / props.limits.warp_size;

        // Blocks limited by shared memory
        let blocks_by_shared = if shared_mem > 0 {
            props.limits.max_shared_per_sm / shared_mem
        } else {
            props.limits.max_blocks_per_sm as usize
        };

        // Active blocks per SM
        let active_blocks = (props.limits.max_blocks_per_sm as usize).min(blocks_by_shared);

        // Active warps per SM
        let max_warps_per_sm = props.limits.max_threads_per_block / props.limits.warp_size
            * props.limits.multiprocessor_count;
        let active_warps = active_blocks as u32 * warps_per_block;

        // Occupancy
        active_warps as f32 / max_warps_per_sm as f32
    }

    /// Find optimal block size for maximum occupancy
    pub fn optimal_block_size(&self, dynamic_shared_mem: usize) -> u32 {
        let mut best_occupancy = 0.0f32;
        let mut best_block_size = 32u32;

        // Try multiples of warp size
        let warp_size = self.device.warp_size();
        for warps in 1..=(self.device.max_threads_per_block() / warp_size) {
            let block_size = warps * warp_size;
            let occupancy = self.theoretical_occupancy(block_size, dynamic_shared_mem);

            if occupancy > best_occupancy {
                best_occupancy = occupancy;
                best_block_size = block_size;
            }
        }

        best_block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim3() {
        assert_eq!(Dim3::x(10).total(), 10);
        assert_eq!(Dim3::xy(10, 20).total(), 200);
        assert_eq!(Dim3::xyz(10, 20, 30).total(), 6000);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::for_elements(1000, 256);
        assert_eq!(config.grid.x, 4);
        assert_eq!(config.block.x, 256);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::for_2d(1920, 1080, 16, 16);
        assert_eq!(config.grid.x, 120);
        assert_eq!(config.grid.y, 68);
    }

    #[test]
    fn test_kernel_arg() {
        let arg = KernelArg::from(42i32);
        assert_eq!(arg.size(), 4);

        let arg = KernelArg::from(3.14f64);
        assert_eq!(arg.size(), 8);
    }

    #[test]
    fn test_launch_validation() {
        let device = Device::cpu();
        let config = LaunchConfig::new(100, 256);
        assert!(config.validate(&device).is_ok());

        let config = LaunchConfig::new(100, 2048);
        assert!(config.validate(&device).is_err());
    }
}
