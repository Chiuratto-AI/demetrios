//! Heterogeneous Multi-GPU Support
//!
//! This module provides support for running mixed-architecture GPU clusters
//! (Volta + Ampere + Ada + Hopper + Blackwell) with automatic device discovery,
//! multi-architecture compilation, and performance-weighted work distribution.
//!
//! # Overview
//!
//! The heterogeneous module enables:
//! - **Device Discovery**: Enumerate all CUDA devices and classify by architecture
//! - **Multi-Architecture Compilation**: Compile one kernel source to optimal PTX for each GPU type
//! - **Performance-Weighted Scheduling**: Distribute work based on device TFLOPS/bandwidth
//! - **Cross-Device Memory**: P2P transfers where available, unified memory support
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::heterogeneous::*;
//!
//! // Initialize with automatic device discovery
//! let manager = HeterogeneousDeviceManager::new()?;
//!
//! println!("Found {} devices across {} architectures",
//!     manager.pool().len(),
//!     manager.architectures().len());
//!
//! // Compile kernel for all architectures
//! let kernel = manager.compile_simple(
//!     "vector_add",
//!     vec![ParamType::Pointer, ParamType::Pointer, ParamType::Pointer, ParamType::U32],
//! )?;
//!
//! // Allocate unified memory
//! let a = manager.alloc_unified(n * 4)?;
//! let b = manager.alloc_unified(n * 4)?;
//! let c = manager.alloc_unified(n * 4)?;
//!
//! // Distributed launch with performance weighting
//! let dist_config = DistributedLaunchConfig::with_work(n)
//!     .strategy(DistributionStrategy::ComputeWeighted)
//!     .shard_input(0)
//!     .gather_output(DeviceId(0));
//!
//! let results = manager.launch_distributed(
//!     &kernel,
//!     &dist_config,
//!     &LaunchConfig::new_1d((n / 256) as u32, 256),
//!     &[KernelArg::UnifiedBuffer(a.ptr()),
//!       KernelArg::UnifiedBuffer(b.ptr()),
//!       KernelArg::UnifiedBuffer(c.ptr()),
//!       KernelArg::U32(n as u32)],
//! )?;
//!
//! // Check results
//! let summary = ExecutionSummary::from_results(&results);
//! println!("Throughput: {:.2} elements/sec", summary.throughput);
//! println!("Load balance: {:.1}%", summary.load_balance * 100.0);
//! ```
//!
//! # Architecture Support
//!
//! | Architecture | Compute Capability | Key Features |
//! |--------------|-------------------|--------------|
//! | Volta | sm_70, sm_72 | Tensor Cores (Gen 1) |
//! | Turing | sm_75 | RT Cores, Tensor Cores (Gen 2) |
//! | Ampere | sm_80, sm_86, sm_87 | BF16, Async Copy, Tensor Cores (Gen 3) |
//! | Ada | sm_89 | FP8, Tensor Cores (Gen 4) |
//! | Hopper | sm_90 | TMA, Clusters, Tensor Cores (Gen 4) |
//! | Blackwell | sm_100 | FP4, Transformer Engine v2, Tensor Cores (Gen 5) |
//!
//! # Distribution Strategies
//!
//! - **Even**: Equal distribution regardless of device performance
//! - **ComputeWeighted**: Proportional to FP32 TFLOPS
//! - **BandwidthWeighted**: Proportional to memory bandwidth
//! - **TensorWeighted**: Proportional to tensor core performance
//! - **Adaptive**: Adjusts based on runtime measurements
//! - **Custom**: User-specified weights

pub mod architecture;
pub mod compiler;
pub mod device_manager;
pub mod device_pool;
pub mod discovery;
pub mod memory;
pub mod scheduler;

// Re-exports
pub use architecture::{ArchCapabilities, ArchFeature, ArchFeatures, GpuArchitecture};
pub use compiler::{
    CompileError, CompiledKernel, CompiledKernelSet, HeterogeneousCompiler, KernelSignature,
    KernelSource, LaunchConfigHint, ModuleHandle, OptSettings, ParamType,
};
pub use device_manager::{
    DeviceResult, ExecutionSummary, HeterogeneousDeviceManager, HeterogeneousError, KernelArg,
    LaunchConfig,
};
pub use device_pool::{DevicePool, DevicePoolError, HeterogeneousDevice};
pub use discovery::{DeviceDiscovery, DiscoveredDevice, DiscoveryError, NVLinkInfo, P2PCapability};
pub use memory::{
    CoherenceMode, DeviceBuffer, HeterogeneousMemoryManager, MemoryError, MemoryInfo,
    TransferPath, UnifiedBuffer,
};
pub use scheduler::{
    DistributedLaunchConfig, DistributionStrategy, HeterogeneousScheduler, InputMode, OutputMode,
    ReduceOp, SyncMode, WorkAssignment, WorkDistribution,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // Create manager
        let manager = HeterogeneousDeviceManager::new().unwrap();

        // Check devices
        assert!(!manager.pool().is_empty());
        println!(
            "Found {} devices: {:?}",
            manager.pool().len(),
            manager.architectures()
        );

        // Compile kernel
        let kernel = manager
            .compile_simple("workflow_test", vec![ParamType::Pointer, ParamType::U32])
            .unwrap();

        // All architectures should be covered
        for arch in manager.architectures() {
            assert!(kernel.has_architecture(arch));
        }

        // Distributed launch
        let dist_config = DistributedLaunchConfig::with_work(10000)
            .strategy(DistributionStrategy::ComputeWeighted);

        let launch_config = LaunchConfig::new_1d(100, 256);

        let results = manager
            .launch_distributed(
                &kernel,
                &dist_config,
                &launch_config,
                &[KernelArg::Buffer(0x1000), KernelArg::U32(10000)],
            )
            .unwrap();

        // All launches should succeed
        assert!(results.iter().all(|r| r.success));

        // Summary
        let summary = ExecutionSummary::from_results(&results);
        assert!(summary.throughput > 0.0);
    }

    #[test]
    fn test_memory_workflow() {
        let manager = HeterogeneousDeviceManager::new().unwrap();

        // Allocate on device 0
        let d0 = manager.pool().devices()[0].id;
        let buf0 = manager.alloc(d0, 4096).unwrap();

        // Copy to device 1 if available
        if manager.pool().len() >= 2 {
            let d1 = manager.pool().devices()[1].id;
            let buf1 = manager.copy_to_device(&buf0, d1).unwrap();
            assert_eq!(buf1.device, d1);
            manager.free(buf1).unwrap();
        }

        manager.free(buf0).unwrap();

        // Unified memory
        let unified = manager.alloc_unified(8192).unwrap();
        for device in manager.pool().devices() {
            manager.prefetch(&unified, device.id).unwrap();
        }
        manager.free_unified(unified).unwrap();
    }

    #[test]
    fn test_architecture_features() {
        // Volta
        let volta = GpuArchitecture::Volta;
        assert!(!volta.supports(ArchFeature::BF16));
        assert!(!volta.supports(ArchFeature::TMA));

        // Ampere
        let ampere = GpuArchitecture::Ampere;
        assert!(ampere.supports(ArchFeature::BF16));
        assert!(ampere.supports(ArchFeature::AsyncCopy));
        assert!(!ampere.supports(ArchFeature::TMA));

        // Hopper
        let hopper = GpuArchitecture::Hopper;
        assert!(hopper.supports(ArchFeature::TMA));
        assert!(hopper.supports(ArchFeature::Clusters));

        // Blackwell
        let blackwell = GpuArchitecture::Blackwell;
        assert!(blackwell.supports(ArchFeature::FP4));
        assert!(blackwell.supports(ArchFeature::TransformerEngineV2));
    }
}
