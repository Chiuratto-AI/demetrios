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
//!
//! # Epistemic GPU Computing
//!
//! Demetrios is the first language to track epistemic state through GPU computation:
//! - Shadow registers for uncertainty (ε)
//! - Validity predicates
//! - Provenance tracking
//! - Tensor Core operations with uncertainty propagation
//!
//! # Usage
//!
//! ```ignore
//! use demetrios::codegen::gpu::{hlir_to_gpu, PtxCodegen, GpuTarget};
//!
//! let gpu_module = hlir_to_gpu::lower(&hlir, GpuTarget::Cuda { compute_capability: (8, 0) });
//! let ptx = PtxCodegen::new((8, 0)).generate(&gpu_module);
//! ```

pub mod counterfactual;
pub mod epistemic_ptx;
pub mod hlir_to_gpu;
pub mod intrinsics;
pub mod ir;
pub mod ptx;
pub mod runtime;
#[cfg(feature = "gpu")]
pub mod spirv;
pub mod tensor_epistemic;

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
pub use tensor_epistemic::{
    EpistemicTensorCategory, EpistemicTensorIntrinsic, EpsilonPropagationRule, TensorCoreOp,
    all_epistemic_tensor_intrinsics, get_epistemic_intrinsic, is_epistemic_tensor_intrinsic,
};

// HLIR to GPU lowering - the critical bridge
pub use hlir_to_gpu::{
    LoweringConfig, compile_to_ptx, compile_to_ptx_epistemic, lower, lower_with_config,
};

// Epistemic PTX emission - shadow registers for uncertainty tracking
pub use epistemic_ptx::{
    EpistemicPtxConfig, EpistemicPtxEmitter, EpistemicShadowRegs, WarpEpsilonOp,
};

// Counterfactual GPU execution - Pearl's do-calculus as GPU primitives
pub use counterfactual::{
    CounterfactualContext, CounterfactualPtxConfig, CounterfactualPtxEmitter, CounterfactualValue,
    Intervention, StructuralEqType, WorldDivergence, WorldId, WorldSnapshot,
};
