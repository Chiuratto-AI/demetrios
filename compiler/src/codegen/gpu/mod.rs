//! GPU Code Generation for Demetrios
//!
//! Supports:
//! - PTX (NVIDIA CUDA)
//! - SPIR-V (Vulkan, OpenCL)
//! - MSL (Apple Metal)
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

pub mod bio;
pub mod cooperative;
pub mod counterfactual;
pub mod graph;
pub mod counterfactual_metal;
pub mod epistemic_ptx;
pub mod hlir_to_gpu;
pub mod intrinsics;
pub mod ir;
pub mod metal;
pub mod metal_runtime;
pub mod portable;
pub mod ptx;
pub mod runtime;
#[cfg(feature = "gpu")]
pub mod spirv;
pub mod tensor_epistemic;

pub use intrinsics::{GpuIntrinsic, all_intrinsics, get_intrinsic, is_gpu_intrinsic};
pub use ir::{
    BlockId, CoopReduceOp, CooperativeGroupId, CooperativeScope, Fp8Format, GpuBlock, GpuConstValue,
    GpuConstant, GpuFunction, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTarget, GpuTerminator,
    GpuType, MemorySpace, MetalGpuFamily, PartitionType, QuantizeMode, SharedMemDecl, ValueId,
    WarpReduceOp, WarpVoteOp,
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

// Metal Shading Language (MSL) codegen - native Apple GPU support
pub use metal::{MetalCodegen, MetalCodegenConfig, compile_to_msl, compile_to_msl_epistemic};

// Counterfactual Metal execution - Pearl's do-calculus for Apple Silicon
pub use counterfactual_metal::{
    CounterfactualMetalConfig, CounterfactualMetalEmitter, compile_counterfactual_metal,
    generate_counterfactual_metal_library,
};

// Metal runtime - native Apple GPU execution
pub use metal_runtime::{
    EpistemicMetalRunner, MetalBuffer, MetalCommandBuffer, MetalDeviceInfo, MetalDispatchSize,
    MetalError, MetalKernel, MetalLibrary, MetalResourceOptions, MetalRuntime, MetalStorageMode,
};

// Bio/Quaternion GPU kernels - from "The Quaternionic Syntax of Existence"
pub use bio::{
    add_bio_kernels, gen_dna_complement_kernel, gen_gf4_add_kernel, gen_quaternion_mul_kernel,
    gen_quaternion_normalize_kernel, gen_quaternion_slerp_kernel, gen_transmission_compose_kernel,
};

// Cooperative Groups kernel generators (CUDA 9.0+ / PTX 6.0+)
pub use cooperative::{
    add_cooperative_kernels, gen_ballot_count_kernel, gen_block_reduce_kernel,
    gen_warp_broadcast_kernel, gen_warp_inclusive_scan_kernel, gen_warp_reduce_sum_kernel,
};

// CUDA Graphs with dynamic control flow
pub use graph::{
    BufferId, BufferInfo, BufferLocation, ConditionType, ConditionalNode, GraphExecConfig,
    GraphKernelArg, GraphNode, GraphNodeId, GraphNodeType, GpuGraph, KernelNode, LoopNode,
    MemcpyNode, MemsetNode, StreamId,
};

// Cross-platform portable GPU IR (write-once, compile-anywhere)
pub use portable::{
    AvailableBackends, BackendCapabilities, Capability, CompileError, CompileResult,
    CompiledKernel, Dimension, PortableGpuOp, PortableMemorySpace, PortableType, UnifiedCompiler,
    UnifiedKernel, UnifiedParam, UnifiedSharedMem, compile_kernel, compile_to_all,
};
