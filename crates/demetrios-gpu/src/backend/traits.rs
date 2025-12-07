//! GPU Backend Traits
//!
//! Defines the abstract interface for GPU backends (CUDA, ROCm, Metal, CPU, etc.)

use crate::ir::{GpuFunction, GpuModule};
use crate::runtime::{
    BufferError, Device, DeviceError, GpuBuffer, Kernel, KernelError, LaunchConfig, Stream,
    StreamError,
};
use std::any::Any;
use thiserror::Error;

/// Backend errors
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Backend not available: {0}")]
    NotAvailable(String),

    #[error("Compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Launch failed: {0}")]
    LaunchFailed(String),

    #[error("Memory error: {0}")]
    Memory(#[from] BufferError),

    #[error("Device error: {0}")]
    Device(#[from] DeviceError),

    #[error("Kernel error: {0}")]
    Kernel(#[from] KernelError),

    #[error("Stream error: {0}")]
    Stream(#[from] StreamError),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Capability flags for backends
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCapabilities {
    /// Supports async operations
    pub async_operations: bool,
    /// Supports unified memory
    pub unified_memory: bool,
    /// Supports cooperative groups
    pub cooperative_groups: bool,
    /// Supports tensor cores
    pub tensor_cores: bool,
    /// Supports dynamic parallelism
    pub dynamic_parallelism: bool,
    /// Maximum compute capability (SM version)
    pub max_compute_capability: u32,
}

/// Abstract GPU backend trait
pub trait GpuBackend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Check if the backend is available
    fn is_available(&self) -> bool;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Initialize the backend
    fn initialize(&mut self) -> BackendResult<()>;

    /// Shutdown the backend
    fn shutdown(&mut self) -> BackendResult<()>;

    /// Get available devices
    fn devices(&self) -> BackendResult<Vec<Device>>;

    /// Get the default device
    fn default_device(&self) -> BackendResult<Device>;

    /// Compile a GPU module
    fn compile_module(&self, module: &GpuModule) -> BackendResult<CompiledModule>;

    /// Compile a single function
    fn compile_function(&self, func: &GpuFunction) -> BackendResult<CompiledKernel>;

    /// Allocate device memory
    fn allocate(&self, size: usize, device: &Device) -> BackendResult<DeviceAllocation>;

    /// Free device memory
    fn free(&self, alloc: DeviceAllocation) -> BackendResult<()>;

    /// Copy host to device
    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &DeviceAllocation,
        stream: Option<&Stream>,
    ) -> BackendResult<()>;

    /// Copy device to host
    fn copy_to_host(
        &self,
        src: &DeviceAllocation,
        dst: &mut [u8],
        stream: Option<&Stream>,
    ) -> BackendResult<()>;

    /// Copy device to device
    fn copy_device_to_device(
        &self,
        src: &DeviceAllocation,
        dst: &DeviceAllocation,
        stream: Option<&Stream>,
    ) -> BackendResult<()>;

    /// Launch a kernel
    fn launch(
        &self,
        kernel: &CompiledKernel,
        config: &LaunchConfig,
        args: &[KernelArgument],
        stream: Option<&Stream>,
    ) -> BackendResult<()>;

    /// Synchronize all operations
    fn synchronize(&self, device: &Device) -> BackendResult<()>;

    /// Create a stream
    fn create_stream(&self, device: &Device) -> BackendResult<Stream>;

    /// Destroy a stream
    fn destroy_stream(&self, stream: Stream) -> BackendResult<()>;

    /// Get backend-specific data (for extensions)
    fn as_any(&self) -> &dyn Any;
}

/// Device memory allocation handle
#[derive(Debug)]
pub struct DeviceAllocation {
    /// Pointer to device memory
    pub ptr: *mut u8,
    /// Size in bytes
    pub size: usize,
    /// Device index
    pub device_index: usize,
    /// Backend-specific handle
    pub handle: u64,
}

unsafe impl Send for DeviceAllocation {}
unsafe impl Sync for DeviceAllocation {}

impl DeviceAllocation {
    pub fn new(ptr: *mut u8, size: usize, device_index: usize) -> Self {
        DeviceAllocation {
            ptr,
            size,
            device_index,
            handle: 0,
        }
    }

    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

/// Compiled module handle
#[derive(Debug)]
pub struct CompiledModule {
    /// Module name
    pub name: String,
    /// Compiled kernels
    pub kernels: Vec<CompiledKernel>,
    /// Backend-specific handle
    pub handle: u64,
}

impl CompiledModule {
    pub fn new(name: impl Into<String>) -> Self {
        CompiledModule {
            name: name.into(),
            kernels: Vec::new(),
            handle: 0,
        }
    }

    pub fn get_kernel(&self, name: &str) -> Option<&CompiledKernel> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

/// Compiled kernel handle
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Kernel name
    pub name: String,
    /// Number of parameters
    pub num_params: usize,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Static shared memory usage
    pub static_shared_mem: usize,
    /// Register usage per thread
    pub registers_per_thread: u32,
    /// Backend-specific handle
    pub handle: u64,
}

impl CompiledKernel {
    pub fn new(name: impl Into<String>, num_params: usize) -> Self {
        CompiledKernel {
            name: name.into(),
            num_params,
            max_threads_per_block: 1024,
            static_shared_mem: 0,
            registers_per_thread: 0,
            handle: 0,
        }
    }
}

/// Kernel argument for launch
#[derive(Debug, Clone)]
pub enum KernelArgument {
    /// 32-bit integer
    I32(i32),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit integer
    I64(i64),
    /// 64-bit unsigned integer
    U64(u64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Device pointer
    DevicePtr(*const u8),
    /// Mutable device pointer
    DevicePtrMut(*mut u8),
}

impl KernelArgument {
    /// Size of the argument in bytes
    pub fn size(&self) -> usize {
        match self {
            KernelArgument::I32(_) | KernelArgument::U32(_) | KernelArgument::F32(_) => 4,
            KernelArgument::I64(_)
            | KernelArgument::U64(_)
            | KernelArgument::F64(_)
            | KernelArgument::DevicePtr(_)
            | KernelArgument::DevicePtrMut(_) => 8,
        }
    }

    /// Get as raw bytes
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            KernelArgument::I32(v) => v.to_le_bytes().to_vec(),
            KernelArgument::U32(v) => v.to_le_bytes().to_vec(),
            KernelArgument::I64(v) => v.to_le_bytes().to_vec(),
            KernelArgument::U64(v) => v.to_le_bytes().to_vec(),
            KernelArgument::F32(v) => v.to_le_bytes().to_vec(),
            KernelArgument::F64(v) => v.to_le_bytes().to_vec(),
            KernelArgument::DevicePtr(p) => (*p as u64).to_le_bytes().to_vec(),
            KernelArgument::DevicePtrMut(p) => (*p as u64).to_le_bytes().to_vec(),
        }
    }
}

impl From<i32> for KernelArgument {
    fn from(v: i32) -> Self {
        KernelArgument::I32(v)
    }
}

impl From<u32> for KernelArgument {
    fn from(v: u32) -> Self {
        KernelArgument::U32(v)
    }
}

impl From<i64> for KernelArgument {
    fn from(v: i64) -> Self {
        KernelArgument::I64(v)
    }
}

impl From<u64> for KernelArgument {
    fn from(v: u64) -> Self {
        KernelArgument::U64(v)
    }
}

impl From<f32> for KernelArgument {
    fn from(v: f32) -> Self {
        KernelArgument::F32(v)
    }
}

impl From<f64> for KernelArgument {
    fn from(v: f64) -> Self {
        KernelArgument::F64(v)
    }
}

/// Backend registry for managing multiple backends
pub struct BackendRegistry {
    backends: Vec<Box<dyn GpuBackend>>,
    default_backend: Option<usize>,
}

impl BackendRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        BackendRegistry {
            backends: Vec::new(),
            default_backend: None,
        }
    }

    /// Register a backend
    pub fn register(&mut self, backend: Box<dyn GpuBackend>) {
        if self.default_backend.is_none() && backend.is_available() {
            self.default_backend = Some(self.backends.len());
        }
        self.backends.push(backend);
    }

    /// Get a backend by name
    pub fn get(&self, name: &str) -> Option<&dyn GpuBackend> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .map(|b| b.as_ref())
    }

    /// Get the default backend
    pub fn default_backend(&self) -> Option<&dyn GpuBackend> {
        self.default_backend.map(|i| self.backends[i].as_ref())
    }

    /// Set the default backend by name
    pub fn set_default(&mut self, name: &str) -> bool {
        if let Some(idx) = self.backends.iter().position(|b| b.name() == name) {
            self.default_backend = Some(idx);
            true
        } else {
            false
        }
    }

    /// Get all available backends
    pub fn available_backends(&self) -> Vec<&str> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .map(|b| b.name())
            .collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_argument() {
        let arg = KernelArgument::from(42i32);
        assert_eq!(arg.size(), 4);

        let arg = KernelArgument::from(3.14f64);
        assert_eq!(arg.size(), 8);
    }

    #[test]
    fn test_compiled_module() {
        let mut module = CompiledModule::new("test");
        module.kernels.push(CompiledKernel::new("kernel1", 2));
        module.kernels.push(CompiledKernel::new("kernel2", 3));

        assert!(module.get_kernel("kernel1").is_some());
        assert!(module.get_kernel("kernel2").is_some());
        assert!(module.get_kernel("kernel3").is_none());
    }

    #[test]
    fn test_backend_registry() {
        let registry = BackendRegistry::new();
        assert!(registry.available_backends().is_empty());
    }
}
