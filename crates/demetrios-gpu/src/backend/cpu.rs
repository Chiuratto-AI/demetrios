//! CPU Backend Implementation
//!
//! Provides a CPU-based backend for testing and fallback execution.

use super::traits::*;
use crate::codegen::{CpuInterpreter, Value};
use crate::ir::{GpuFunction, GpuModule};
use crate::runtime::{Device, DeviceProperties, DeviceType, LaunchConfig, Stream};
use std::alloc::{alloc, dealloc, Layout};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CPU backend implementation
pub struct CpuBackend {
    /// Whether the backend is initialized
    initialized: bool,
    /// CPU interpreter for execution
    interpreter: Arc<Mutex<CpuInterpreter>>,
    /// Cached functions
    functions: HashMap<String, Arc<GpuFunction>>,
    /// Allocations tracking
    allocations: Vec<DeviceAllocation>,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        CpuBackend {
            initialized: false,
            interpreter: Arc::new(Mutex::new(CpuInterpreter::new())),
            functions: HashMap::new(),
            allocations: Vec::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }

    fn is_available(&self) -> bool {
        true // CPU backend is always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            async_operations: false,
            unified_memory: true,
            cooperative_groups: false,
            tensor_cores: false,
            dynamic_parallelism: false,
            max_compute_capability: 80, // Simulated SM 8.0
        }
    }

    fn initialize(&mut self) -> BackendResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> BackendResult<()> {
        // Free any remaining allocations
        for alloc in self.allocations.drain(..) {
            if !alloc.ptr.is_null() && alloc.size > 0 {
                let layout = Layout::from_size_align(alloc.size, 16).unwrap();
                unsafe {
                    dealloc(alloc.ptr, layout);
                }
            }
        }
        self.initialized = false;
        Ok(())
    }

    fn devices(&self) -> BackendResult<Vec<Device>> {
        Ok(vec![Device::cpu()])
    }

    fn default_device(&self) -> BackendResult<Device> {
        Ok(Device::cpu())
    }

    fn compile_module(&self, module: &GpuModule) -> BackendResult<CompiledModule> {
        let mut compiled = CompiledModule::new(&module.name);

        for (name, func) in &module.functions {
            let kernel = self.compile_function_inner(func)?;
            compiled.kernels.push(kernel);
        }

        Ok(compiled)
    }

    fn compile_function(&self, func: &GpuFunction) -> BackendResult<CompiledKernel> {
        self.compile_function_inner(func)
    }

    fn allocate(&self, size: usize, _device: &Device) -> BackendResult<DeviceAllocation> {
        if size == 0 {
            return Ok(DeviceAllocation::new(std::ptr::null_mut(), 0, 0));
        }

        let layout = Layout::from_size_align(size, 16).map_err(|e| {
            BackendError::Memory(crate::runtime::BufferError::AllocationFailed(e.to_string()))
        })?;

        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(BackendError::Memory(
                crate::runtime::BufferError::AllocationFailed("CPU allocation failed".to_string()),
            ));
        }

        Ok(DeviceAllocation::new(ptr, size, 0))
    }

    fn free(&self, alloc: DeviceAllocation) -> BackendResult<()> {
        if !alloc.ptr.is_null() && alloc.size > 0 {
            let layout = Layout::from_size_align(alloc.size, 16).unwrap();
            unsafe {
                dealloc(alloc.ptr, layout);
            }
        }
        Ok(())
    }

    fn copy_to_device(
        &self,
        src: &[u8],
        dst: &DeviceAllocation,
        _stream: Option<&Stream>,
    ) -> BackendResult<()> {
        if src.len() != dst.size {
            return Err(BackendError::Memory(
                crate::runtime::BufferError::SizeMismatch {
                    expected: dst.size,
                    actual: src.len(),
                },
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr, src.len());
        }

        Ok(())
    }

    fn copy_to_host(
        &self,
        src: &DeviceAllocation,
        dst: &mut [u8],
        _stream: Option<&Stream>,
    ) -> BackendResult<()> {
        if dst.len() != src.size {
            return Err(BackendError::Memory(
                crate::runtime::BufferError::SizeMismatch {
                    expected: src.size,
                    actual: dst.len(),
                },
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr, dst.as_mut_ptr(), src.size);
        }

        Ok(())
    }

    fn copy_device_to_device(
        &self,
        src: &DeviceAllocation,
        dst: &DeviceAllocation,
        _stream: Option<&Stream>,
    ) -> BackendResult<()> {
        if src.size != dst.size {
            return Err(BackendError::Memory(
                crate::runtime::BufferError::SizeMismatch {
                    expected: dst.size,
                    actual: src.size,
                },
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src.ptr, dst.ptr, src.size);
        }

        Ok(())
    }

    fn launch(
        &self,
        kernel: &CompiledKernel,
        config: &LaunchConfig,
        args: &[KernelArgument],
        _stream: Option<&Stream>,
    ) -> BackendResult<()> {
        // Find the cached function
        let func = self
            .functions
            .get(&kernel.name)
            .ok_or_else(|| BackendError::Internal(format!("Kernel {} not found", kernel.name)))?
            .clone();

        // Convert arguments to Values
        let values: Vec<Value> = args
            .iter()
            .map(|arg| match arg {
                KernelArgument::I32(v) => Value::I32(*v),
                KernelArgument::U32(v) => Value::U32(*v),
                KernelArgument::I64(v) => Value::I64(*v),
                KernelArgument::U64(v) => Value::U64(*v),
                KernelArgument::F32(v) => Value::F32(*v),
                KernelArgument::F64(v) => Value::F64(*v),
                KernelArgument::DevicePtr(p) => Value::Ptr(*p as *mut u8),
                KernelArgument::DevicePtrMut(p) => Value::Ptr(*p),
            })
            .collect();

        // Execute using the interpreter
        let mut interpreter = self.interpreter.lock().unwrap();
        interpreter
            .execute(&func, config, &values)
            .map_err(|e| BackendError::LaunchFailed(e.to_string()))?;

        Ok(())
    }

    fn synchronize(&self, _device: &Device) -> BackendResult<()> {
        // CPU execution is synchronous
        Ok(())
    }

    fn create_stream(&self, device: &Device) -> BackendResult<Stream> {
        Ok(Stream::default_stream(device))
    }

    fn destroy_stream(&self, _stream: Stream) -> BackendResult<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CpuBackend {
    fn compile_function_inner(&self, func: &GpuFunction) -> BackendResult<CompiledKernel> {
        // For CPU backend, we don't actually compile - just store the function
        let kernel = CompiledKernel {
            name: func.name.clone(),
            num_params: func.params.len(),
            max_threads_per_block: func.max_threads.unwrap_or(1024),
            static_shared_mem: func
                .shared_mem
                .iter()
                .map(|s| s.ty.size_bytes().unwrap_or(0) * s.size)
                .sum(),
            registers_per_thread: 0,
            handle: 0,
        };

        Ok(kernel)
    }

    /// Register a function for later execution
    pub fn register_function(&mut self, func: GpuFunction) {
        self.functions.insert(func.name.clone(), Arc::new(func));
    }
}

/// Create a global CPU backend instance
pub fn create_cpu_backend() -> Box<dyn GpuBackend> {
    Box::new(CpuBackend::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::types::common;
    use crate::ir::{Dimension, ScalarType};

    #[test]
    fn test_cpu_backend_available() {
        let backend = CpuBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_devices() {
        let backend = CpuBackend::new();
        let devices = backend.devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_cpu_backend_allocate() {
        let backend = CpuBackend::new();
        let device = Device::cpu();

        let alloc = backend.allocate(1024, &device).unwrap();
        assert!(!alloc.is_null());
        assert_eq!(alloc.size, 1024);

        backend.free(alloc).unwrap();
    }

    #[test]
    fn test_cpu_backend_copy() {
        let backend = CpuBackend::new();
        let device = Device::cpu();

        let src_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let alloc = backend.allocate(src_data.len(), &device).unwrap();

        backend.copy_to_device(&src_data, &alloc, None).unwrap();

        let mut dst_data = vec![0u8; 8];
        backend.copy_to_host(&alloc, &mut dst_data, None).unwrap();

        assert_eq!(src_data, dst_data);

        backend.free(alloc).unwrap();
    }

    #[test]
    fn test_cpu_backend_compile() {
        let mut backend = CpuBackend::new();

        let mut builder = FunctionBuilder::kernel("test_kernel");
        builder.param("data", common::f32_ptr());
        builder.param("n", common::u32());
        builder.ret_void();

        let func = builder.build();
        backend.register_function(func.clone());

        let kernel = backend.compile_function(&func).unwrap();
        assert_eq!(kernel.name, "test_kernel");
        assert_eq!(kernel.num_params, 2);
    }
}
