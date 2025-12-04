//! GPU Code Generation for Demetrios
//!
//! Supports:
//! - PTX (NVIDIA CUDA)
//! - SPIR-V (Vulkan, OpenCL)
//!
//! Architecture:
//! ```text
//! HLIR -> GpuIR -> PTX/SPIR-V -> Driver -> GPU Execution
//! ```

pub mod intrinsics;
pub mod ir;
pub mod ptx;
pub mod runtime;
#[cfg(feature = "gpu")]
pub mod spirv;

pub use intrinsics::{GpuIntrinsic, all_intrinsics, get_intrinsic, is_gpu_intrinsic};
pub use ir::{
    BlockId, GpuBlock, GpuConstValue, GpuConstant, GpuFunction, GpuKernel, GpuModule, GpuOp,
    GpuParam, GpuTarget, GpuTerminator, GpuType, MemorySpace, SharedMemDecl, ValueId, WarpReduceOp,
    WarpVoteOp,
};
pub use ptx::PtxCodegen;
pub use runtime::{
    DeviceBuffer, GpuBackend, GpuError, GpuRuntime, Kernel, KernelArg, LaunchConfig,
};
#[cfg(feature = "gpu")]
pub use spirv::SpirvCodegen;
