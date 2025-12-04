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

pub mod ir;
pub mod ptx;
#[cfg(feature = "gpu")]
pub mod spirv;
pub mod runtime;
pub mod intrinsics;

pub use ir::{
    BlockId, GpuBlock, GpuConstValue, GpuConstant, GpuFunction, GpuKernel, GpuModule, GpuOp,
    GpuParam, GpuTarget, GpuTerminator, GpuType, MemorySpace, SharedMemDecl, ValueId,
    WarpReduceOp, WarpVoteOp,
};
pub use ptx::PtxCodegen;
#[cfg(feature = "gpu")]
pub use spirv::SpirvCodegen;
pub use runtime::{DeviceBuffer, GpuBackend, GpuError, GpuRuntime, Kernel, KernelArg, LaunchConfig};
pub use intrinsics::{all_intrinsics, get_intrinsic, is_gpu_intrinsic, GpuIntrinsic};
