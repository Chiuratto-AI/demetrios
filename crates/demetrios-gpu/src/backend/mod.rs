//! GPU Backends
//!
//! Provides abstraction layer for different GPU backends.

pub mod cpu;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use cpu::{create_cpu_backend, CpuBackend};
pub use traits::{
    BackendCapabilities, BackendError, BackendRegistry, BackendResult, CompiledKernel,
    CompiledModule, DeviceAllocation, GpuBackend, KernelArgument,
};

#[cfg(feature = "cuda")]
pub use cuda::{create_cuda_backend, CudaBackend};
