//! Demetrios GPU Backend
//!
//! This crate provides GPU computing support for the Demetrios programming language,
//! including:
//!
//! - **IR Module**: Typed intermediate representation for GPU code
//! - **Runtime Module**: Device management, buffers, streams, and kernel launching
//! - **Codegen Module**: PTX generation and CPU simulation
//! - **Backend Module**: Abstraction layer for different GPU backends
//! - **Memory Module**: Memory space abstractions and access pattern optimization
//! - **Integrate Module**: Integration with Demetrios type system (linear types, effects, units, epistemic)
//! - **Optimize Module**: Performance analysis and optimization tools
//! - **Distributed Module**: Multi-GPU and distributed computing support
//!
//! # Example
//!
//! ```rust
//! use demetrios_gpu::ir::{FunctionBuilder, Dimension, ScalarType, AddressSpace};
//! use demetrios_gpu::ir::types::common;
//! use demetrios_gpu::runtime::{Device, LaunchConfig};
//! use demetrios_gpu::codegen::PtxGenerator;
//!
//! // Build a simple vector addition kernel
//! let mut builder = FunctionBuilder::kernel("vector_add");
//! let a = builder.param("a", common::f32_ptr());
//! let b = builder.param("b", common::f32_ptr());
//! let c = builder.param("c", common::f32_ptr());
//! let n = builder.param("n", common::u32());
//!
//! // Compute global thread ID
//! let idx = builder.global_thread_id(Dimension::X);
//! let in_bounds = builder.lt(idx, n, ScalarType::U32);
//!
//! let then_block = builder.new_block();
//! let exit_block = builder.new_block();
//! builder.cond_branch(in_bounds, then_block, exit_block);
//!
//! builder.switch_to(then_block);
//! // Load, add, store
//! let idx_i64 = builder.convert(idx, ScalarType::U32, ScalarType::I64);
//! let a_ptr = builder.gep(a, vec![idx_i64], common::f32());
//! let b_ptr = builder.gep(b, vec![idx_i64], common::f32());
//! let c_ptr = builder.gep(c, vec![idx_i64], common::f32());
//!
//! let a_val = builder.load(a_ptr, common::f32(), AddressSpace::Global);
//! let b_val = builder.load(b_ptr, common::f32(), AddressSpace::Global);
//! let sum = builder.fadd(a_val, b_val, ScalarType::F32);
//! builder.store(c_ptr, sum, common::f32(), AddressSpace::Global);
//! builder.branch(exit_block);
//!
//! builder.switch_to(exit_block);
//! builder.ret_void();
//!
//! let func = builder.build();
//!
//! // Generate PTX
//! let mut gen = PtxGenerator::new();
//! let ptx = gen.generate_function(&func).unwrap();
//! assert!(ptx.contains(".entry vector_add"));
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod backend;
pub mod codegen;
pub mod deep;
pub mod distributed;
pub mod integrate;
pub mod ir;
pub mod memory;
pub mod optimize;
pub mod runtime;
pub mod substrate;

// Re-export commonly used types
pub use backend::{BackendError, CpuBackend, GpuBackend};
pub use codegen::{CpuInterpreter, PtxGenerator};
pub use distributed::{
    CollectiveOp, DataParallel, DistributedContext, Distribution, GpuTopology, PipelineExecutor,
    PipelineSchedule, ScalingAdvisor, ScalingRecommendation,
};
pub use integrate::{
    Confidence, Epistemic, EpistemicBuffer, LinearGpuBuffer, Quantity, Unit, UnitBuffer,
};
pub use ir::{FunctionBuilder, GpuFunction, GpuModule, GpuType, ModuleBuilder, ScalarType};
pub use memory::{Global, MemorySpace, Shared};
pub use optimize::{
    AnalysisPipeline, ArchLimits, CoalesceAnalyzer, KernelAnalysis, KernelResources, MemoryPool,
    OccupancyCalculator, OccupancyReport, RegisterPressureAnalyzer,
};
pub use runtime::{Device, GpuBuffer, Kernel, LaunchConfig, Stream};
pub use substrate::{
    AdaptivePrecision,
    CellList,
    ConservationChecker,
    // Conservation law verification
    ConservationLaw,
    Dimensions,
    EnergyConservation,
    EpistemicExecution,
    // Epistemic execution model
    EpistemicValue,
    EquilibriumFinder,
    Equivariant,
    GibbsMinimization,
    HamiltonPrinciple,
    HilbertCurve,
    Invariant,
    // Symmetry and Lie groups
    LieGroup,
    MassConservation,
    MortonCurve,
    // Physical quantities with semantic meaning
    PhysicalQuantity,
    // Physical memory topology
    PhysicalSpace,
    QuantityKind,
    RayleighRitz,
    SecondLawChecker,
    Space3D,
    SubstrateType,
    SymmetryChecker,
    // Thermodynamic consistency
    ThermodynamicState,
    // Variational principles
    VariationalPrinciple,
    SE3,
    SO3,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::backend::{GpuBackend, KernelArgument};
    pub use crate::codegen::{CpuInterpreter, PtxGenerator};
    pub use crate::integrate::{
        Confidence, Epistemic, EpistemicBuffer, LinearGpuBuffer, Quantity, Unit, UnitBuffer,
    };
    pub use crate::ir::{
        AddressSpace, BinOp, CmpOp, Dimension, FunctionBuilder, GpuFunction, GpuModule, GpuType,
        ModuleBuilder, ScalarType, UnaryOp,
    };
    pub use crate::memory::{AccessMode, AccessPattern, Global, MemorySpace, Shared};
    pub use crate::runtime::{
        Device, Dim3, GpuBuffer, Kernel, KernelLauncher, LaunchConfig, LinearBuffer, Stream,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_full_pipeline() {
        // Create a simple kernel
        let mut builder = FunctionBuilder::kernel("saxpy");
        let _x = builder.param("x", crate::ir::types::common::f32_ptr());
        let _y = builder.param("y", crate::ir::types::common::f32_ptr());
        let _a = builder.param("a", crate::ir::types::common::f32());
        let _n = builder.param("n", crate::ir::types::common::u32());
        builder.ret_void();

        let func = builder.build();
        assert_eq!(func.name, "saxpy");
        assert_eq!(func.params.len(), 4);

        // Generate PTX
        let mut gen = PtxGenerator::new();
        let ptx = gen.generate_function(&func).unwrap();
        assert!(ptx.contains(".entry saxpy"));
    }

    #[test]
    fn test_cpu_simulation() {
        let device = Device::cpu();
        assert_eq!(device.device_type(), crate::runtime::DeviceType::Cpu);
    }

    #[test]
    fn test_linear_buffer() {
        let device = Device::cpu();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = LinearGpuBuffer::from_slice(&data, &device).unwrap();
        let result = buffer.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_unit_buffer() {
        use crate::integrate::units::{Meter, MeterBuffer};

        let device = Device::cpu();
        let mut buffer: MeterBuffer<f64> = UnitBuffer::new(4, &device).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        buffer.upload_raw(&data).unwrap();

        let result = buffer.download_raw().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_epistemic_buffer() {
        use crate::integrate::epistemic::KnowledgeSource;

        let device = Device::cpu();
        let data = vec![
            Epistemic::observed(1.0f64),
            Epistemic::observed(2.0),
            Epistemic::computed(3.0),
        ];

        let buffer = EpistemicBuffer::from_epistemic(&data, &device).unwrap();
        assert_eq!(buffer.len(), 3);

        let downloaded = buffer.download().unwrap();
        assert_eq!(downloaded[0].value(), 1.0);
    }
}
