//! Heterogeneous Device Manager
//!
//! This module provides the main interface for managing heterogeneous GPU execution
//! across mixed-architecture clusters. It integrates device discovery, compilation,
//! scheduling, and memory management into a unified API.

use super::architecture::GpuArchitecture;
use super::compiler::{
    CompileError, CompiledKernelSet, HeterogeneousCompiler, KernelSignature, KernelSource,
    LaunchConfigHint, ParamType,
};
use super::device_pool::{DevicePool, DevicePoolError, HeterogeneousDevice};
use super::memory::{
    CoherenceMode, DeviceBuffer, HeterogeneousMemoryManager, MemoryError, MemoryInfo, UnifiedBuffer,
};
use super::scheduler::{
    DistributedLaunchConfig, DistributionStrategy, HeterogeneousScheduler, WorkDistribution,
};
use crate::optimize::pool::DeviceId;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors from the heterogeneous device manager
#[derive(Debug, Error)]
pub enum HeterogeneousError {
    #[error("Device pool error: {0}")]
    PoolError(#[from] DevicePoolError),

    #[error("Compilation error: {0}")]
    CompileError(#[from] CompileError),

    #[error("Memory error: {0}")]
    MemoryError(#[from] MemoryError),

    #[error("Launch error: {0}")]
    LaunchError(String),

    #[error("Device {0:?} not found")]
    DeviceNotFound(DeviceId),

    #[error("No devices available")]
    NoDevices,

    #[error("Synchronization failed: {0}")]
    SyncFailed(String),
}

/// Result of a kernel execution on a device
#[derive(Debug)]
pub struct DeviceResult {
    /// Device that executed
    pub device: DeviceId,
    /// Work units processed
    pub work_units: usize,
    /// Execution time
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Kernel argument for launch
#[derive(Debug, Clone)]
pub enum KernelArg {
    /// Device buffer pointer
    Buffer(u64),
    /// Unified buffer pointer
    UnifiedBuffer(u64),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit unsigned integer
    U64(u64),
    /// 32-bit signed integer
    I32(i32),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
}

impl KernelArg {
    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            KernelArg::Buffer(_) | KernelArg::UnifiedBuffer(_) | KernelArg::U64(_) => 8,
            KernelArg::U32(_) | KernelArg::I32(_) | KernelArg::F32(_) => 4,
            KernelArg::F64(_) => 8,
        }
    }
}

/// Launch configuration for kernel execution
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid: (u32, u32, u32),
    /// Block dimensions (threads)
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes
    pub shared_mem: u32,
}

impl LaunchConfig {
    /// Create 1D launch config
    pub fn new_1d(blocks: u32, threads: u32) -> Self {
        Self {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        }
    }

    /// Create 2D launch config
    pub fn new_2d(grid: (u32, u32), block: (u32, u32)) -> Self {
        Self {
            grid: (grid.0, grid.1, 1),
            block: (block.0, block.1, 1),
            shared_mem: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Get total threads
    pub fn total_threads(&self) -> u64 {
        self.grid.0 as u64
            * self.grid.1 as u64
            * self.grid.2 as u64
            * self.block.0 as u64
            * self.block.1 as u64
            * self.block.2 as u64
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self::new_1d(1, 256)
    }
}

impl From<LaunchConfigHint> for LaunchConfig {
    fn from(hint: LaunchConfigHint) -> Self {
        Self {
            grid: (1, 1, 1),
            block: (hint.threads_per_block, 1, 1),
            shared_mem: hint.shared_mem_bytes,
        }
    }
}

/// Central manager for heterogeneous GPU execution
pub struct HeterogeneousDeviceManager {
    /// Device pool
    pool: Arc<DevicePool>,
    /// Kernel compiler
    compiler: HeterogeneousCompiler,
    /// Work scheduler
    scheduler: HeterogeneousScheduler,
    /// Memory manager
    memory: HeterogeneousMemoryManager,
}

impl HeterogeneousDeviceManager {
    /// Initialize with automatic device discovery
    pub fn new() -> Result<Self, HeterogeneousError> {
        let pool = Arc::new(DevicePool::discover()?);

        if pool.is_empty() {
            return Err(HeterogeneousError::NoDevices);
        }

        Ok(Self {
            compiler: HeterogeneousCompiler::new(),
            scheduler: HeterogeneousScheduler::new(pool.clone()),
            memory: HeterogeneousMemoryManager::new(pool.clone()),
            pool,
        })
    }

    /// Initialize with specific device IDs
    pub fn with_devices(device_ids: &[u32]) -> Result<Self, HeterogeneousError> {
        let pool = Arc::new(DevicePool::with_devices(device_ids)?);

        if pool.is_empty() {
            return Err(HeterogeneousError::NoDevices);
        }

        Ok(Self {
            compiler: HeterogeneousCompiler::new(),
            scheduler: HeterogeneousScheduler::new(pool.clone()),
            memory: HeterogeneousMemoryManager::new(pool.clone()),
            pool,
        })
    }

    /// Get the device pool
    pub fn pool(&self) -> &DevicePool {
        &self.pool
    }

    /// Get the memory manager
    pub fn memory(&self) -> &HeterogeneousMemoryManager {
        &self.memory
    }

    /// Compile a kernel for all architectures in the pool
    pub fn compile_kernel(
        &self,
        kernel_name: &str,
        source: &KernelSource,
    ) -> Result<CompiledKernelSet, HeterogeneousError> {
        Ok(self.compiler.compile_for_pool(kernel_name, source, &self.pool)?)
    }

    /// Compile with specific parameter types
    pub fn compile_simple(
        &self,
        kernel_name: &str,
        params: Vec<ParamType>,
    ) -> Result<CompiledKernelSet, HeterogeneousError> {
        let source = KernelSource::simple(params);
        self.compile_kernel(kernel_name, &source)
    }

    /// Launch kernel on a specific device
    pub fn launch(
        &self,
        kernel: &CompiledKernelSet,
        device: DeviceId,
        config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<DeviceResult, HeterogeneousError> {
        let dev = self
            .pool
            .get(device)
            .ok_or(HeterogeneousError::DeviceNotFound(device))?;

        let compiled = kernel
            .for_device(dev)
            .ok_or_else(|| HeterogeneousError::LaunchError("No kernel for device architecture".to_string()))?;

        let start = Instant::now();

        // In real implementation, this would:
        // 1. Get or create module handle
        // 2. Get function pointer
        // 3. Set up kernel arguments
        // 4. Call cuLaunchKernel

        let _module = compiled.get_module(device)?;

        // Simulate execution
        let duration = start.elapsed();

        Ok(DeviceResult {
            device,
            work_units: config.total_threads() as usize,
            duration,
            success: true,
            error: None,
        })
    }

    /// Launch kernel across multiple devices with automatic work distribution
    pub fn launch_distributed(
        &self,
        kernel: &CompiledKernelSet,
        dist_config: &DistributedLaunchConfig,
        launch_config: &LaunchConfig,
        args: &[KernelArg],
    ) -> Result<Vec<DeviceResult>, HeterogeneousError> {
        // Compute work distribution
        let distribution = self.scheduler.compute_distribution(dist_config);

        let mut results = Vec::new();

        for assignment in &distribution.assignments {
            if assignment.is_empty() {
                continue;
            }

            // Adjust launch config for this device's work portion
            let device_config = self.adjust_config_for_assignment(launch_config, assignment);

            // Launch on device
            let result = self.launch(kernel, assignment.device, &device_config, args)?;

            // Record for adaptive scheduling
            self.scheduler.record_execution(assignment.device, assignment.count, result.duration);

            results.push(result);
        }

        Ok(results)
    }

    /// Allocate buffer on a specific device
    pub fn alloc(&self, device: DeviceId, size: usize) -> Result<DeviceBuffer, HeterogeneousError> {
        Ok(self.memory.alloc(device, size)?)
    }

    /// Allocate typed buffer on a specific device
    pub fn alloc_typed<T>(&self, device: DeviceId, count: usize) -> Result<DeviceBuffer, HeterogeneousError> {
        Ok(self.memory.alloc_typed::<T>(device, count)?)
    }

    /// Allocate unified memory accessible from all devices
    pub fn alloc_unified(&self, size: usize) -> Result<UnifiedBuffer, HeterogeneousError> {
        Ok(self.memory.alloc_unified(size, CoherenceMode::Managed)?)
    }

    /// Allocate unified memory with specific coherence mode
    pub fn alloc_unified_with_coherence(
        &self,
        size: usize,
        coherence: CoherenceMode,
    ) -> Result<UnifiedBuffer, HeterogeneousError> {
        Ok(self.memory.alloc_unified(size, coherence)?)
    }

    /// Free a device buffer
    pub fn free(&self, buffer: DeviceBuffer) -> Result<(), HeterogeneousError> {
        Ok(self.memory.free(buffer)?)
    }

    /// Free a unified buffer
    pub fn free_unified(&self, buffer: UnifiedBuffer) -> Result<(), HeterogeneousError> {
        Ok(self.memory.free_unified(buffer)?)
    }

    /// Copy buffer to another device
    pub fn copy_to_device(
        &self,
        src: &DeviceBuffer,
        dst_device: DeviceId,
    ) -> Result<DeviceBuffer, HeterogeneousError> {
        Ok(self.memory.copy_buffer(src, dst_device)?)
    }

    /// Prefetch unified buffer to a device
    pub fn prefetch(&self, buffer: &UnifiedBuffer, device: DeviceId) -> Result<(), HeterogeneousError> {
        Ok(self.memory.prefetch(buffer, device)?)
    }

    /// Synchronize a specific device
    pub fn synchronize(&self, device: DeviceId) -> Result<(), HeterogeneousError> {
        // In real implementation: cuCtxSynchronize after setting context
        if self.pool.get(device).is_none() {
            return Err(HeterogeneousError::DeviceNotFound(device));
        }
        Ok(())
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<(), HeterogeneousError> {
        for device in self.pool.devices() {
            self.synchronize(device.id)?;
        }
        Ok(())
    }

    /// Get memory info for a device
    pub fn memory_info(&self, device: DeviceId) -> Option<MemoryInfo> {
        self.memory.memory_info(device)
    }

    /// Get all unique architectures
    pub fn architectures(&self) -> Vec<GpuArchitecture> {
        self.pool.architectures()
    }

    /// Get devices by architecture
    pub fn devices_by_architecture(&self, arch: GpuArchitecture) -> Vec<&HeterogeneousDevice> {
        self.pool
            .by_architecture(arch)
            .iter()
            .filter_map(|id| self.pool.get(*id))
            .collect()
    }

    /// Get the fastest device
    pub fn fastest_device(&self) -> Option<&HeterogeneousDevice> {
        self.pool.fastest_device()
    }

    /// Compute optimal work distribution for a workload
    pub fn compute_distribution(&self, config: &DistributedLaunchConfig) -> WorkDistribution {
        self.scheduler.compute_distribution(config)
    }

    /// Get current adaptive scheduling weights
    pub fn scheduling_weights(&self) -> std::collections::HashMap<DeviceId, f64> {
        self.scheduler.get_current_weights()
    }

    // === Internal helpers ===

    fn adjust_config_for_assignment(
        &self,
        base: &LaunchConfig,
        assignment: &super::scheduler::WorkAssignment,
    ) -> LaunchConfig {
        // Scale grid based on work assignment
        let scale = assignment.weight;
        let scaled_blocks = ((base.grid.0 as f64 * scale).ceil() as u32).max(1);

        LaunchConfig {
            grid: (scaled_blocks, base.grid.1, base.grid.2),
            block: base.block,
            shared_mem: base.shared_mem,
        }
    }
}

/// Summary statistics for heterogeneous execution
#[derive(Debug)]
pub struct ExecutionSummary {
    /// Total execution time
    pub total_time: Duration,
    /// Per-device times
    pub device_times: Vec<(DeviceId, Duration)>,
    /// Total work units
    pub total_work: usize,
    /// Overall throughput (work/second)
    pub throughput: f64,
    /// Load balance score (0-1, 1 = perfect)
    pub load_balance: f64,
}

impl ExecutionSummary {
    /// Create from device results
    pub fn from_results(results: &[DeviceResult]) -> Self {
        let device_times: Vec<_> = results.iter().map(|r| (r.device, r.duration)).collect();

        let total_time = device_times
            .iter()
            .map(|(_, d)| *d)
            .max()
            .unwrap_or(Duration::ZERO);

        let total_work: usize = results.iter().map(|r| r.work_units).sum();

        let throughput = if total_time.as_secs_f64() > 0.0 {
            total_work as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        // Load balance: ratio of min to max time
        let min_time = device_times
            .iter()
            .map(|(_, d)| d.as_secs_f64())
            .fold(f64::INFINITY, f64::min);
        let max_time = device_times
            .iter()
            .map(|(_, d)| d.as_secs_f64())
            .fold(0.0, f64::max);

        let load_balance = if max_time > 0.0 {
            min_time / max_time
        } else {
            1.0
        };

        Self {
            total_time,
            device_times,
            total_work,
            throughput,
            load_balance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        assert!(!manager.pool().is_empty());
    }

    #[test]
    fn test_compile_kernel() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let kernel = manager
            .compile_simple("test_kernel", vec![ParamType::Pointer, ParamType::U32])
            .unwrap();

        // Should have compiled for all architectures
        for arch in manager.architectures() {
            assert!(kernel.has_architecture(arch));
        }
    }

    #[test]
    fn test_launch_single() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let kernel = manager
            .compile_simple("launch_test", vec![ParamType::Pointer])
            .unwrap();

        let device = manager.pool().devices()[0].id;
        let config = LaunchConfig::new_1d(10, 256);

        let result = manager
            .launch(&kernel, device, &config, &[KernelArg::Buffer(0x1000)])
            .unwrap();

        assert!(result.success);
        assert_eq!(result.device, device);
    }

    #[test]
    fn test_launch_distributed() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let kernel = manager
            .compile_simple("distributed_test", vec![ParamType::Pointer, ParamType::U32])
            .unwrap();

        let dist_config = DistributedLaunchConfig::with_work(10000)
            .strategy(DistributionStrategy::ComputeWeighted);

        let launch_config = LaunchConfig::new_1d(100, 256);

        let results = manager
            .launch_distributed(&kernel, &dist_config, &launch_config, &[
                KernelArg::Buffer(0x1000),
                KernelArg::U32(10000),
            ])
            .unwrap();

        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_memory_allocation() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let device = manager.pool().devices()[0].id;

        let buffer = manager.alloc(device, 4096).unwrap();
        assert_eq!(buffer.device, device);
        assert_eq!(buffer.size, 4096);

        manager.free(buffer).unwrap();
    }

    #[test]
    fn test_unified_memory() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let buffer = manager.alloc_unified(8192).unwrap();

        assert_eq!(buffer.size, 8192);

        // Prefetch to all devices
        for device in manager.pool().devices() {
            manager.prefetch(&buffer, device.id).unwrap();
        }

        manager.free_unified(buffer).unwrap();
    }

    #[test]
    fn test_copy_to_device() {
        let manager = HeterogeneousDeviceManager::new().unwrap();

        if manager.pool().len() >= 2 {
            let d0 = manager.pool().devices()[0].id;
            let d1 = manager.pool().devices()[1].id;

            let src = manager.alloc(d0, 1024).unwrap();
            let dst = manager.copy_to_device(&src, d1).unwrap();

            assert_eq!(dst.device, d1);
            assert_eq!(dst.size, src.size);
        }
    }

    #[test]
    fn test_execution_summary() {
        let results = vec![
            DeviceResult {
                device: DeviceId(0),
                work_units: 1000,
                duration: Duration::from_millis(100),
                success: true,
                error: None,
            },
            DeviceResult {
                device: DeviceId(1),
                work_units: 1000,
                duration: Duration::from_millis(110),
                success: true,
                error: None,
            },
        ];

        let summary = ExecutionSummary::from_results(&results);

        assert_eq!(summary.total_work, 2000);
        assert!(summary.throughput > 0.0);
        assert!(summary.load_balance > 0.0 && summary.load_balance <= 1.0);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::new_2d((10, 10), (16, 16));
        assert_eq!(config.total_threads(), 10 * 10 * 16 * 16);

        let config = config.with_shared_mem(1024);
        assert_eq!(config.shared_mem, 1024);
    }

    #[test]
    fn test_scheduling_weights() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let weights = manager.scheduling_weights();

        // All devices should have weights
        for device in manager.pool().devices() {
            assert!(weights.contains_key(&device.id));
        }

        // Weights should sum to ~1.0
        let sum: f64 = weights.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_architectures() {
        let manager = HeterogeneousDeviceManager::new().unwrap();
        let archs = manager.architectures();

        assert!(!archs.is_empty());

        for arch in &archs {
            let devices = manager.devices_by_architecture(*arch);
            assert!(!devices.is_empty());
        }
    }
}
