//! GPU Intermediate Representation
//!
//! This module provides a typed IR for GPU code that maps closely to PTX/SPIR-V.

pub mod builder;
pub mod effects;
pub mod function;
pub mod inst;
pub mod types;

pub use builder::{FunctionBuilder, ModuleBuilder};
pub use effects::{EffectSet, GpuEffect};
pub use function::{BasicBlock, FunctionKind, GpuFunction, GpuModule, Parameter};
pub use inst::{
    AtomicOp, BarrierScope, BinOp, BlockId, CmpOp, Constant, Dimension, Instruction, MemoryOrder,
    UnaryOp, ValueId, WarpVoteKind,
};
pub use types::{AddressSpace, GpuType, ScalarType, StructType, VectorWidth};
