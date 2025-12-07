//! GPU Kernel Abstraction
//!
//! Provides kernel compilation, caching, and management.

use super::device::{Device, DeviceError};
use crate::ir::{GpuFunction, GpuModule};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Kernel-related errors
#[derive(Debug, Error)]
pub enum KernelError {
    #[error("Kernel compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Kernel not found: {0}")]
    NotFound(String),

    #[error("Invalid kernel argument: {0}")]
    InvalidArgument(String),

    #[error("Launch failed: {0}")]
    LaunchFailed(String),

    #[error("Device error: {0}")]
    Device(#[from] DeviceError),
}

/// Compiled kernel handle
#[derive(Clone)]
pub struct Kernel {
    /// Kernel name
    name: String,
    /// Compiled code (for CPU simulation, this is a function pointer or IR)
    compiled: CompiledKernel,
    /// Kernel metadata
    metadata: KernelMetadata,
}

/// Kernel compilation result
#[derive(Clone)]
enum CompiledKernel {
    /// CPU simulation kernel
    CpuSimulation(Arc<GpuFunction>),
    /// PTX code (for CUDA)
    Ptx(String),
    /// SPIR-V binary (for Vulkan/OpenCL)
    SpirV(Vec<u8>),
}

impl fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompiledKernel::CpuSimulation(_) => write!(f, "CpuSimulation(...)"),
            CompiledKernel::Ptx(code) => write!(f, "Ptx({} bytes)", code.len()),
            CompiledKernel::SpirV(code) => write!(f, "SpirV({} bytes)", code.len()),
        }
    }
}

/// Kernel metadata
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Number of parameters
    pub num_params: usize,
    /// Parameter types (for validation)
    pub param_types: Vec<KernelParamType>,
    /// Maximum threads per block
    pub max_threads_per_block: Option<u32>,
    /// Minimum blocks per SM
    pub min_blocks_per_sm: Option<u32>,
    /// Static shared memory size
    pub static_shared_mem: usize,
    /// Required registers per thread
    pub registers_per_thread: Option<u32>,
}

/// Kernel parameter type
#[derive(Debug, Clone)]
pub enum KernelParamType {
    /// Scalar value
    Scalar(ScalarParamType),
    /// Pointer to device memory
    DevicePtr,
    /// Pointer to constant memory
    ConstPtr,
}

/// Scalar parameter types
#[derive(Debug, Clone, Copy)]
pub enum ScalarParamType {
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Kernel {
    /// Create a kernel from a GPU function (for CPU simulation)
    pub fn from_function(func: GpuFunction) -> Self {
        let metadata = KernelMetadata {
            num_params: func.params.len(),
            param_types: func
                .params
                .iter()
                .map(|p| {
                    if p.ty.is_pointer() {
                        KernelParamType::DevicePtr
                    } else {
                        KernelParamType::Scalar(ScalarParamType::U64)
                    }
                })
                .collect(),
            max_threads_per_block: func.max_threads,
            min_blocks_per_sm: func.min_blocks,
            static_shared_mem: func
                .shared_mem
                .iter()
                .map(|s| s.ty.size_bytes().unwrap_or(0) * s.size)
                .sum(),
            registers_per_thread: None,
        };

        Kernel {
            name: func.name.clone(),
            compiled: CompiledKernel::CpuSimulation(Arc::new(func)),
            metadata,
        }
    }

    /// Create a kernel from PTX code
    pub fn from_ptx(name: impl Into<String>, ptx: String, metadata: KernelMetadata) -> Self {
        Kernel {
            name: name.into(),
            compiled: CompiledKernel::Ptx(ptx),
            metadata,
        }
    }

    /// Get kernel name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get kernel metadata
    pub fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        self.metadata.num_params
    }

    /// Get maximum threads per block
    pub fn max_threads_per_block(&self) -> Option<u32> {
        self.metadata.max_threads_per_block
    }

    /// Get static shared memory size
    pub fn static_shared_mem(&self) -> usize {
        self.metadata.static_shared_mem
    }

    /// Check if this is a CPU simulation kernel
    pub fn is_cpu_simulation(&self) -> bool {
        matches!(self.compiled, CompiledKernel::CpuSimulation(_))
    }

    /// Get the GPU function (for CPU simulation)
    pub fn as_function(&self) -> Option<&GpuFunction> {
        match &self.compiled {
            CompiledKernel::CpuSimulation(func) => Some(func),
            _ => None,
        }
    }

    /// Get PTX code (if available)
    pub fn as_ptx(&self) -> Option<&str> {
        match &self.compiled {
            CompiledKernel::Ptx(ptx) => Some(ptx),
            _ => None,
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.name)
            .field("compiled", &self.compiled)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Kernel cache for avoiding recompilation
pub struct KernelCache {
    /// Cached kernels by name
    kernels: HashMap<String, Kernel>,
    /// Device index
    device_index: usize,
}

impl KernelCache {
    /// Create a new kernel cache
    pub fn new(device: &Device) -> Self {
        KernelCache {
            kernels: HashMap::new(),
            device_index: device.index(),
        }
    }

    /// Get a kernel from the cache
    pub fn get(&self, name: &str) -> Option<&Kernel> {
        self.kernels.get(name)
    }

    /// Insert a kernel into the cache
    pub fn insert(&mut self, kernel: Kernel) {
        self.kernels.insert(kernel.name.clone(), kernel);
    }

    /// Check if a kernel is cached
    pub fn contains(&self, name: &str) -> bool {
        self.kernels.contains_key(name)
    }

    /// Get or compile a kernel
    pub fn get_or_compile<F>(&mut self, name: &str, compile: F) -> Result<&Kernel, KernelError>
    where
        F: FnOnce() -> Result<Kernel, KernelError>,
    {
        if !self.kernels.contains_key(name) {
            let kernel = compile()?;
            self.kernels.insert(name.to_string(), kernel);
        }
        Ok(self.kernels.get(name).unwrap())
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.kernels.clear();
    }

    /// Get number of cached kernels
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Module containing multiple kernels
pub struct KernelModule {
    /// Module name
    name: String,
    /// Kernels in this module
    kernels: HashMap<String, Kernel>,
    /// Source module (for recompilation)
    source: Option<GpuModule>,
}

impl KernelModule {
    /// Create a new kernel module
    pub fn new(name: impl Into<String>) -> Self {
        KernelModule {
            name: name.into(),
            kernels: HashMap::new(),
            source: None,
        }
    }

    /// Create from a GPU module
    pub fn from_gpu_module(module: GpuModule) -> Self {
        let mut km = KernelModule {
            name: module.name.clone(),
            kernels: HashMap::new(),
            source: Some(module.clone()),
        };

        for (name, func) in module.functions {
            km.kernels.insert(name, Kernel::from_function(func));
        }

        km
    }

    /// Get module name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a kernel by name
    pub fn get_kernel(&self, name: &str) -> Option<&Kernel> {
        self.kernels.get(name)
    }

    /// Get all kernel names
    pub fn kernel_names(&self) -> impl Iterator<Item = &str> {
        self.kernels.keys().map(|s| s.as_str())
    }

    /// Get number of kernels
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if module is empty
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Add a kernel
    pub fn add_kernel(&mut self, kernel: Kernel) {
        self.kernels.insert(kernel.name.clone(), kernel);
    }
}

impl fmt::Debug for KernelModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelModule")
            .field("name", &self.name)
            .field("kernels", &self.kernels.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::FunctionBuilder;

    #[test]
    fn test_kernel_from_function() {
        let mut builder = FunctionBuilder::kernel("test_kernel");
        builder.param("data", crate::ir::types::common::f32_ptr());
        builder.param("n", crate::ir::types::common::u32());
        builder.ret_void();

        let func = builder.build();
        let kernel = Kernel::from_function(func);

        assert_eq!(kernel.name(), "test_kernel");
        assert_eq!(kernel.num_params(), 2);
        assert!(kernel.is_cpu_simulation());
    }

    #[test]
    fn test_kernel_cache() {
        let device = Device::cpu();
        let mut cache = KernelCache::new(&device);

        let mut builder = FunctionBuilder::kernel("cached_kernel");
        builder.ret_void();
        let kernel = Kernel::from_function(builder.build());

        cache.insert(kernel);
        assert!(cache.contains("cached_kernel"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_kernel_module() {
        let mut module = KernelModule::new("test_module");

        let mut builder = FunctionBuilder::kernel("kernel1");
        builder.ret_void();
        module.add_kernel(Kernel::from_function(builder.build()));

        let mut builder = FunctionBuilder::kernel("kernel2");
        builder.ret_void();
        module.add_kernel(Kernel::from_function(builder.build()));

        assert_eq!(module.len(), 2);
        assert!(module.get_kernel("kernel1").is_some());
        assert!(module.get_kernel("kernel2").is_some());
    }
}
