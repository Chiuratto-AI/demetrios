//! Low-Level Primitives for Distributed GPU Computing
//!
//! This module provides the fundamental building blocks for distributed GPU computing:
//!
//! - **Memory Consistency**: GPU memory fences, atomics, sync flags, barriers
//! - **P2P Communication**: Point-to-point transfers, copy engines, one-sided operations
//! - **Reduction Kernels**: PTX generation for warp/block/device reductions
//!
//! # Theoretical Foundation
//!
//! These primitives implement the α-β model for communication:
//!
//! ```text
//! T(n) = α + β·n
//!
//! where:
//!   α = latency (startup cost)
//!   β = inverse bandwidth (per-byte cost)
//!   n = message size in bytes
//! ```
//!
//! # Memory Hierarchy
//!
//! GPU memory scopes form a hierarchy:
//!
//! ```text
//! Thread → Warp (32 threads) → Block (up to 1024 threads) → Device → System
//! ```
//!
//! Each scope requires appropriate fence and synchronization primitives.
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::primitives::{
//!     MemoryScope, MemoryOrder, MemFence,
//!     P2PCopier, CopyEngine,
//!     generate_warp_reduce, ReduceOpType, DataType,
//! };
//!
//! // Memory fence at device scope
//! let fence = MemFence {
//!     scope: MemoryScope::Device,
//!     order: MemoryOrder::SeqCst,
//! };
//! let ptx = fence.to_ptx();
//!
//! // P2P copy between devices
//! let copier = P2PCopier::new();
//! let desc = copier.create_p2p_copy(src, dst, 1024);
//!
//! // Generate warp reduction kernel
//! let kernel = generate_warp_reduce(ReduceOpType::Sum, DataType::F32);
//! ```

pub mod memory;
pub mod p2p;
pub mod reduce_kernels;

// Re-export memory consistency primitives
pub use memory::{
    AtomicInstruction, AtomicOp, DistributedBarrier, GpuSyncFlag, MemFence, MemoryOrder,
    MemoryScope, SequenceNumber, SyncFlag, WorkStealQueue,
};

// Re-export P2P communication primitives
pub use p2p::{
    CopyDescriptor, CopyEngine, DevicePtr, Exchange, MemoryRegion, OneSided, P2PCapability,
    P2PCopier, RecvRequest, SendRecv, SendRequest,
};

// Re-export reduction kernel generation
pub use reduce_kernels::{
    generate_atomic_reduce, generate_block_reduce, generate_lookback_scan,
    generate_ring_step_kernel, generate_vectorized_copy, generate_warp_reduce, DataType,
    KernelConfig, ReduceOpType,
};

/// Typical network parameters for common interconnects
pub mod network_params {
    /// NVLink 3.0 (A100): 600 GB/s bidirectional
    pub const NVLINK3_BANDWIDTH: f64 = 300e9; // 300 GB/s per direction
    pub const NVLINK3_LATENCY: f64 = 1e-6; // ~1 μs

    /// NVLink 4.0 (H100): 900 GB/s bidirectional
    pub const NVLINK4_BANDWIDTH: f64 = 450e9; // 450 GB/s per direction
    pub const NVLINK4_LATENCY: f64 = 0.7e-6; // ~0.7 μs

    /// PCIe 4.0 x16: ~32 GB/s per direction
    pub const PCIE4_BANDWIDTH: f64 = 32e9;
    pub const PCIE4_LATENCY: f64 = 2e-6; // ~2 μs

    /// PCIe 5.0 x16: ~64 GB/s per direction
    pub const PCIE5_BANDWIDTH: f64 = 64e9;
    pub const PCIE5_LATENCY: f64 = 2e-6; // ~2 μs

    /// InfiniBand HDR: 200 Gb/s = 25 GB/s
    pub const IB_HDR_BANDWIDTH: f64 = 25e9;
    pub const IB_HDR_LATENCY: f64 = 1e-6; // ~1 μs

    /// InfiniBand NDR: 400 Gb/s = 50 GB/s
    pub const IB_NDR_BANDWIDTH: f64 = 50e9;
    pub const IB_NDR_LATENCY: f64 = 0.8e-6; // ~0.8 μs
}

/// Constants for GPU reduction operations
pub mod reduction_constants {
    /// Warp size on NVIDIA GPUs
    pub const WARP_SIZE: usize = 32;

    /// Maximum block size
    pub const MAX_BLOCK_SIZE: usize = 1024;

    /// Typical shared memory per SM
    pub const SHARED_MEMORY_PER_SM: usize = 48 * 1024; // 48 KB

    /// L2 cache line size
    pub const L2_CACHE_LINE: usize = 128;

    /// Optimal vector width for memory operations
    pub const VECTOR_WIDTH: usize = 16; // 4x float4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_fence_generation() {
        let fence = MemFence {
            scope: MemoryScope::Device,
            order: MemoryOrder::SeqCst,
        };
        let ptx = fence.to_ptx();
        assert!(ptx.contains("membar")); // PTX uses membar instruction
        assert!(ptx.contains(".gpu")); // Device scope = .gpu
        assert!(ptx.contains(".acq_rel")); // SeqCst = .acq_rel
    }

    #[test]
    fn test_sync_flag_creation() {
        let flag = SyncFlag::new();
        assert!(!flag.try_wait()); // Not signaled initially
        flag.signal();
        assert!(flag.try_wait()); // Now signaled
        flag.reset();
        assert!(!flag.try_wait()); // Reset
    }

    #[test]
    fn test_copy_descriptor() {
        use crate::optimize::pool::DeviceId;

        let desc = CopyDescriptor::new(
            DeviceId(0),
            DevicePtr(0x1000),
            DeviceId(1),
            DevicePtr(0x2000),
            1024,
        );
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.src_ptr, DevicePtr(0x1000));
        assert_eq!(desc.dst_ptr, DevicePtr(0x2000));
    }

    #[test]
    fn test_warp_reduce_generation() {
        let kernel = generate_warp_reduce(ReduceOpType::Sum, DataType::F32);
        assert!(kernel.contains("shfl.sync.bfly"));
    }

    #[test]
    fn test_block_reduce_generation() {
        let kernel = generate_block_reduce(ReduceOpType::Max, DataType::F64, 256);
        assert!(kernel.contains("shared"));
    }

    #[test]
    fn test_network_params() {
        // NVLink should be faster than PCIe
        assert!(network_params::NVLINK4_BANDWIDTH > network_params::PCIE5_BANDWIDTH);
        // NVLink should have lower latency
        assert!(network_params::NVLINK4_LATENCY < network_params::PCIE4_LATENCY);
    }

    #[test]
    fn test_reduction_constants() {
        assert_eq!(reduction_constants::WARP_SIZE, 32);
        assert_eq!(reduction_constants::MAX_BLOCK_SIZE, 1024);
    }
}
