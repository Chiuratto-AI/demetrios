//! Multi-Architecture Kernel Compilation
//!
//! This module provides compilation of GPU kernels for multiple architectures,
//! enabling heterogeneous execution across mixed GPU clusters.

use super::architecture::GpuArchitecture;
use super::device_pool::DevicePool;
use crate::optimize::pool::DeviceId;
use std::collections::HashMap;
use std::sync::Mutex;
use thiserror::Error;

/// Errors during heterogeneous compilation
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("No architectures specified")]
    NoArchitectures,

    #[error("Compilation failed for {arch:?}: {message}")]
    CompilationFailed {
        arch: GpuArchitecture,
        message: String,
    },

    #[error("Kernel '{0}' not found in module")]
    KernelNotFound(String),

    #[error("Module is empty")]
    EmptyModule,

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

/// A kernel compiled for multiple architectures
#[derive(Debug)]
pub struct CompiledKernelSet {
    /// Kernel name
    pub name: String,
    /// Compiled kernels per architecture
    kernels: HashMap<GpuArchitecture, CompiledKernel>,
    /// Fallback architecture (lowest common)
    fallback_arch: GpuArchitecture,
    /// Kernel signature for validation
    signature: KernelSignature,
}

impl CompiledKernelSet {
    /// Create a new compiled kernel set
    pub fn new(name: String, fallback_arch: GpuArchitecture, signature: KernelSignature) -> Self {
        Self {
            name,
            kernels: HashMap::new(),
            fallback_arch,
            signature,
        }
    }

    /// Add a compiled kernel for an architecture
    pub fn add(&mut self, kernel: CompiledKernel) {
        self.kernels.insert(kernel.architecture, kernel);
    }

    /// Get kernel for a specific architecture (falls back if not available)
    pub fn for_architecture(&self, arch: GpuArchitecture) -> Option<&CompiledKernel> {
        self.kernels
            .get(&arch)
            .or_else(|| self.kernels.get(&self.fallback_arch))
    }

    /// Get kernel for a specific device
    pub fn for_device(&self, device: &super::device_pool::HeterogeneousDevice) -> Option<&CompiledKernel> {
        self.for_architecture(device.architecture)
    }

    /// Get all compiled architectures
    pub fn architectures(&self) -> Vec<GpuArchitecture> {
        self.kernels.keys().copied().collect()
    }

    /// Get kernel signature
    pub fn signature(&self) -> &KernelSignature {
        &self.signature
    }

    /// Check if kernel is compiled for an architecture
    pub fn has_architecture(&self, arch: GpuArchitecture) -> bool {
        self.kernels.contains_key(&arch)
    }
}

/// A single compiled kernel
#[derive(Debug)]
pub struct CompiledKernel {
    /// Target architecture
    pub architecture: GpuArchitecture,
    /// Generated PTX code
    pub ptx: String,
    /// Recommended launch configuration
    pub recommended_config: LaunchConfigHint,
    /// JIT module handles (per device)
    module_handles: Mutex<HashMap<DeviceId, ModuleHandle>>,
}

impl CompiledKernel {
    /// Create a new compiled kernel
    pub fn new(
        architecture: GpuArchitecture,
        ptx: String,
        recommended_config: LaunchConfigHint,
    ) -> Self {
        Self {
            architecture,
            ptx,
            recommended_config,
            module_handles: Mutex::new(HashMap::new()),
        }
    }

    /// Get PTX code
    pub fn ptx(&self) -> &str {
        &self.ptx
    }

    /// Get or create module handle for a device
    pub fn get_module(&self, device_id: DeviceId) -> Result<ModuleHandle, CompileError> {
        let mut handles = self.module_handles.lock().unwrap();

        if let Some(handle) = handles.get(&device_id) {
            return Ok(*handle);
        }

        // In real implementation, this would call cuModuleLoadData
        let handle = ModuleHandle(device_id.0 as u64 * 1000 + self.architecture.sm_version() as u64);
        handles.insert(device_id, handle);

        Ok(handle)
    }
}

/// JIT module handle (opaque pointer in real implementation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleHandle(u64);

/// Kernel signature for validation
#[derive(Debug, Clone)]
pub struct KernelSignature {
    /// Parameter types
    pub params: Vec<ParamType>,
    /// Return type (usually void for kernels)
    pub return_type: Option<ParamType>,
    /// Required shared memory (static)
    pub static_shared_mem: u32,
}

impl KernelSignature {
    /// Create an empty signature
    pub fn empty() -> Self {
        Self {
            params: Vec::new(),
            return_type: None,
            static_shared_mem: 0,
        }
    }

    /// Create with parameter types
    pub fn with_params(params: Vec<ParamType>) -> Self {
        Self {
            params,
            return_type: None,
            static_shared_mem: 0,
        }
    }
}

/// Parameter type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamType {
    Pointer,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    F16,
    BF16,
}

/// Launch configuration hints
#[derive(Debug, Clone)]
pub struct LaunchConfigHint {
    /// Recommended threads per block
    pub threads_per_block: u32,
    /// Recommended shared memory per block
    pub shared_mem_bytes: u32,
    /// Maximum blocks per SM for optimal occupancy
    pub max_blocks_per_sm: u32,
}

impl Default for LaunchConfigHint {
    fn default() -> Self {
        Self {
            threads_per_block: 256,
            shared_mem_bytes: 0,
            max_blocks_per_sm: 4,
        }
    }
}

/// Multi-architecture compiler
pub struct HeterogeneousCompiler {
    /// Optimization settings per architecture
    opt_settings: HashMap<GpuArchitecture, OptSettings>,
}

impl HeterogeneousCompiler {
    /// Create a new heterogeneous compiler
    pub fn new() -> Self {
        Self {
            opt_settings: Self::default_opt_settings(),
        }
    }

    /// Create with custom optimization settings
    pub fn with_settings(opt_settings: HashMap<GpuArchitecture, OptSettings>) -> Self {
        Self { opt_settings }
    }

    /// Compile kernel for all architectures in a device pool
    pub fn compile_for_pool(
        &self,
        kernel_name: &str,
        kernel_source: &KernelSource,
        pool: &DevicePool,
    ) -> Result<CompiledKernelSet, CompileError> {
        let archs = pool.architectures();
        self.compile_for_architectures(kernel_name, kernel_source, &archs)
    }

    /// Compile kernel for specific architectures
    pub fn compile_for_architectures(
        &self,
        kernel_name: &str,
        kernel_source: &KernelSource,
        architectures: &[GpuArchitecture],
    ) -> Result<CompiledKernelSet, CompileError> {
        if architectures.is_empty() {
            return Err(CompileError::NoArchitectures);
        }

        let fallback = *architectures.iter().min().unwrap();
        let signature = kernel_source.signature.clone();
        let mut kernel_set = CompiledKernelSet::new(kernel_name.to_string(), fallback, signature);

        for &arch in architectures {
            let kernel = self.compile_single(kernel_name, kernel_source, arch)?;
            kernel_set.add(kernel);
        }

        Ok(kernel_set)
    }

    /// Compile for a single architecture
    fn compile_single(
        &self,
        kernel_name: &str,
        source: &KernelSource,
        arch: GpuArchitecture,
    ) -> Result<CompiledKernel, CompileError> {
        let settings = self.opt_settings.get(&arch).cloned().unwrap_or_default();

        // Generate architecture-specific PTX
        let ptx = self.generate_ptx(kernel_name, source, arch, &settings)?;

        // Compute recommended launch config
        let recommended_config = self.compute_launch_config(source, arch, &settings);

        Ok(CompiledKernel::new(arch, ptx, recommended_config))
    }

    /// Generate PTX for an architecture
    fn generate_ptx(
        &self,
        kernel_name: &str,
        source: &KernelSource,
        arch: GpuArchitecture,
        settings: &OptSettings,
    ) -> Result<String, CompileError> {
        let (ptx_major, ptx_minor) = arch.ptx_version();
        let sm = arch.sm_string();

        // Build PTX header
        let mut ptx = format!(
            ".version {}.{}\n\
             .target {}\n\
             .address_size 64\n\n",
            ptx_major, ptx_minor, sm
        );

        // Add architecture-specific features
        if arch.features().bf16 && settings.use_bf16 {
            ptx.push_str("// BF16 enabled\n");
        }
        if arch.features().tma && settings.use_tma {
            ptx.push_str("// TMA enabled\n");
        }

        // Generate kernel entry
        ptx.push_str(&format!("\n.entry {}(\n", kernel_name));

        // Add parameters
        for (i, param) in source.signature.params.iter().enumerate() {
            let param_type = match param {
                ParamType::Pointer => ".u64",
                ParamType::U32 | ParamType::I32 => ".u32",
                ParamType::U64 | ParamType::I64 => ".u64",
                ParamType::F32 => ".f32",
                ParamType::F64 => ".f64",
                ParamType::F16 => ".f16",
                ParamType::BF16 => ".b16",
            };
            let comma = if i < source.signature.params.len() - 1 {
                ","
            } else {
                ""
            };
            ptx.push_str(&format!("    .param {} param_{}{}\n", param_type, i, comma));
        }
        ptx.push_str(")\n");

        // Add shared memory if needed
        if source.signature.static_shared_mem > 0 {
            ptx.push_str(&format!(
                ".shared .align 16 .b8 shared_mem[{}];\n",
                source.signature.static_shared_mem
            ));
        }

        // Kernel body (placeholder - real implementation would lower IR)
        ptx.push_str("{\n");
        ptx.push_str("    // Kernel body placeholder\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");

        Ok(ptx)
    }

    /// Compute optimal launch configuration for an architecture
    fn compute_launch_config(
        &self,
        source: &KernelSource,
        arch: GpuArchitecture,
        _settings: &OptSettings,
    ) -> LaunchConfigHint {
        let features = arch.features();

        // Adjust threads based on architecture
        let threads_per_block = match arch {
            GpuArchitecture::Volta | GpuArchitecture::Turing => 256,
            GpuArchitecture::Ampere | GpuArchitecture::Ada => 256,
            GpuArchitecture::Hopper | GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => 512,
        };

        let shared_mem = source.signature.static_shared_mem;
        let max_blocks = (features.max_shared_memory_kb * 1024) / shared_mem.max(1);

        LaunchConfigHint {
            threads_per_block,
            shared_mem_bytes: shared_mem,
            max_blocks_per_sm: max_blocks.min(32),
        }
    }

    /// Default optimization settings per architecture
    fn default_opt_settings() -> HashMap<GpuArchitecture, OptSettings> {
        let mut settings = HashMap::new();

        for arch in GpuArchitecture::all() {
            let features = arch.features();
            settings.insert(
                *arch,
                OptSettings {
                    use_bf16: features.bf16,
                    use_fp8: features.fp8,
                    use_tma: features.tma,
                    use_async_copy: features.async_copy,
                    fast_math: true,
                    unroll_factor: 4,
                },
            );
        }

        settings
    }
}

impl Default for HeterogeneousCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization settings for compilation
#[derive(Debug, Clone, Default)]
pub struct OptSettings {
    /// Use BF16 where beneficial
    pub use_bf16: bool,
    /// Use FP8 where beneficial
    pub use_fp8: bool,
    /// Use TMA for bulk transfers
    pub use_tma: bool,
    /// Use async copy
    pub use_async_copy: bool,
    /// Enable fast math
    pub fast_math: bool,
    /// Loop unroll factor
    pub unroll_factor: u32,
}

/// Kernel source (abstract representation)
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Kernel signature
    pub signature: KernelSignature,
    /// Source IR (placeholder)
    pub ir: Vec<u8>,
}

impl KernelSource {
    /// Create a simple kernel source for testing
    pub fn simple(params: Vec<ParamType>) -> Self {
        Self {
            signature: KernelSignature::with_params(params),
            ir: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_for_single_arch() {
        let compiler = HeterogeneousCompiler::new();
        let source = KernelSource::simple(vec![ParamType::Pointer, ParamType::Pointer, ParamType::U32]);

        let result = compiler.compile_for_architectures(
            "test_kernel",
            &source,
            &[GpuArchitecture::Ampere],
        );

        assert!(result.is_ok());
        let kernel_set = result.unwrap();
        assert!(kernel_set.has_architecture(GpuArchitecture::Ampere));
    }

    #[test]
    fn test_compile_for_multiple_arch() {
        let compiler = HeterogeneousCompiler::new();
        let source = KernelSource::simple(vec![ParamType::Pointer, ParamType::U32]);

        let archs = vec![
            GpuArchitecture::Ampere,
            GpuArchitecture::Hopper,
            GpuArchitecture::Blackwell,
        ];

        let kernel_set = compiler
            .compile_for_architectures("multi_arch_kernel", &source, &archs)
            .unwrap();

        for arch in &archs {
            assert!(kernel_set.has_architecture(*arch));
            let kernel = kernel_set.for_architecture(*arch).unwrap();
            assert!(kernel.ptx.contains(arch.sm_string()));
        }
    }

    #[test]
    fn test_fallback_architecture() {
        let compiler = HeterogeneousCompiler::new();
        let source = KernelSource::simple(vec![ParamType::Pointer]);

        let kernel_set = compiler
            .compile_for_architectures(
                "fallback_test",
                &source,
                &[GpuArchitecture::Ampere, GpuArchitecture::Hopper],
            )
            .unwrap();

        // Ada not compiled, should fallback to Ampere (lowest)
        let kernel = kernel_set.for_architecture(GpuArchitecture::Ada);
        assert!(kernel.is_some());
        assert_eq!(kernel.unwrap().architecture, GpuArchitecture::Ampere);
    }

    #[test]
    fn test_ptx_generation() {
        let compiler = HeterogeneousCompiler::new();
        let source = KernelSource::simple(vec![ParamType::Pointer, ParamType::F32]);

        let kernel_set = compiler
            .compile_for_architectures("ptx_test", &source, &[GpuArchitecture::Hopper])
            .unwrap();

        let kernel = kernel_set.for_architecture(GpuArchitecture::Hopper).unwrap();
        assert!(kernel.ptx.contains(".version 8.3"));
        assert!(kernel.ptx.contains(".target sm_90"));
        assert!(kernel.ptx.contains(".entry ptx_test"));
    }

    #[test]
    fn test_module_handle_caching() {
        let kernel = CompiledKernel::new(
            GpuArchitecture::Ampere,
            "// test ptx".to_string(),
            LaunchConfigHint::default(),
        );

        let handle1 = kernel.get_module(DeviceId(0)).unwrap();
        let handle2 = kernel.get_module(DeviceId(0)).unwrap();

        assert_eq!(handle1, handle2);
    }

    #[test]
    fn test_no_architectures_error() {
        let compiler = HeterogeneousCompiler::new();
        let source = KernelSource::simple(vec![]);

        let result = compiler.compile_for_architectures("empty", &source, &[]);
        assert!(matches!(result, Err(CompileError::NoArchitectures)));
    }
}
