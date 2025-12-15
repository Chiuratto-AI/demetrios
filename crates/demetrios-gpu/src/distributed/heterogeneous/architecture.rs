//! GPU Architecture Classification for Heterogeneous Systems
//!
//! This module defines GPU architecture types and capabilities for mixed-architecture
//! GPU clusters (Volta + Ampere + Ada + Hopper + Blackwell).

use std::fmt;

/// GPU architecture generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GpuArchitecture {
    /// Volta (V100, sm_70, sm_72) - First Tensor Core architecture
    Volta,
    /// Turing (RTX 20xx, T4, sm_75) - RT Cores introduced
    Turing,
    /// Ampere (A100, RTX 30xx, sm_80, sm_86, sm_87) - 3rd gen Tensor Cores
    Ampere,
    /// Ada Lovelace (RTX 40xx, L40, sm_89) - FP8 support
    Ada,
    /// Hopper (H100, sm_90, sm_90a) - TMA, Thread Block Clusters
    Hopper,
    /// Blackwell (B100, B200, sm_100, sm_100a) - FP4, 5th gen Tensor Cores
    Blackwell,
    /// Blackwell Ultra (sm_120) - Enhanced Blackwell
    BlackwellUltra,
}

impl GpuArchitecture {
    /// Create from CUDA compute capability (major, minor)
    pub fn from_compute_capability(cc: (u32, u32)) -> Self {
        match (cc.0, cc.1) {
            (7, 0) | (7, 2) => GpuArchitecture::Volta,
            (7, 5) => GpuArchitecture::Turing,
            (8, 0) | (8, 6) | (8, 7) => GpuArchitecture::Ampere,
            (8, 9) => GpuArchitecture::Ada,
            (9, 0) => GpuArchitecture::Hopper,
            (10, 0) => GpuArchitecture::Blackwell,
            (12, 0) => GpuArchitecture::BlackwellUltra,
            // Unknown - default to Volta for safety
            _ if cc.0 < 7 => GpuArchitecture::Volta,
            _ if cc.0 >= 12 => GpuArchitecture::BlackwellUltra,
            _ => GpuArchitecture::Ampere,
        }
    }

    /// Get compute capability for this architecture (primary variant)
    pub fn compute_capability(&self) -> (u32, u32) {
        match self {
            GpuArchitecture::Volta => (7, 0),
            GpuArchitecture::Turing => (7, 5),
            GpuArchitecture::Ampere => (8, 0),
            GpuArchitecture::Ada => (8, 9),
            GpuArchitecture::Hopper => (9, 0),
            GpuArchitecture::Blackwell => (10, 0),
            GpuArchitecture::BlackwellUltra => (12, 0),
        }
    }

    /// Get SM version number (major * 10 + minor)
    pub fn sm_version(&self) -> u32 {
        let (major, minor) = self.compute_capability();
        major * 10 + minor
    }

    /// Get SM version string for PTX target
    pub fn sm_string(&self) -> &'static str {
        match self {
            GpuArchitecture::Volta => "sm_70",
            GpuArchitecture::Turing => "sm_75",
            GpuArchitecture::Ampere => "sm_80",
            GpuArchitecture::Ada => "sm_89",
            GpuArchitecture::Hopper => "sm_90",
            GpuArchitecture::Blackwell => "sm_100",
            GpuArchitecture::BlackwellUltra => "sm_120",
        }
    }

    /// Get recommended PTX version for this architecture
    pub fn ptx_version(&self) -> (u32, u32) {
        match self {
            GpuArchitecture::Volta => (6, 1),
            GpuArchitecture::Turing => (6, 4),
            GpuArchitecture::Ampere => (7, 1),
            GpuArchitecture::Ada => (8, 1),
            GpuArchitecture::Hopper => (8, 3),
            GpuArchitecture::Blackwell => (8, 5),
            GpuArchitecture::BlackwellUltra => (8, 6),
        }
    }

    /// Get Tensor Core generation (None if no tensor cores)
    pub fn tensor_core_gen(&self) -> Option<u32> {
        match self {
            GpuArchitecture::Volta => Some(1),
            GpuArchitecture::Turing => Some(2),
            GpuArchitecture::Ampere => Some(3),
            GpuArchitecture::Ada | GpuArchitecture::Hopper => Some(4),
            GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => Some(5),
        }
    }

    /// Get architecture features
    pub fn features(&self) -> ArchFeatures {
        ArchFeatures::for_architecture(*self)
    }

    /// Check if this architecture supports a specific feature
    pub fn supports(&self, feature: ArchFeature) -> bool {
        self.features().supports(feature)
    }

    /// Get typical FP32 TFLOPS for this architecture class
    /// (representative flagship GPU)
    pub fn typical_fp32_tflops(&self) -> f64 {
        match self {
            GpuArchitecture::Volta => 15.7,         // V100
            GpuArchitecture::Turing => 16.3,        // RTX 2080 Ti
            GpuArchitecture::Ampere => 19.5,        // A100
            GpuArchitecture::Ada => 82.6,           // RTX 4090
            GpuArchitecture::Hopper => 67.0,        // H100 SXM
            GpuArchitecture::Blackwell => 90.0,     // B100 (estimated)
            GpuArchitecture::BlackwellUltra => 120.0, // B200 (estimated)
        }
    }

    /// Get typical memory bandwidth (GB/s) for this architecture class
    pub fn typical_memory_bandwidth(&self) -> f64 {
        match self {
            GpuArchitecture::Volta => 900.0,        // V100 HBM2
            GpuArchitecture::Turing => 616.0,       // RTX 2080 Ti GDDR6
            GpuArchitecture::Ampere => 2039.0,      // A100 HBM2e
            GpuArchitecture::Ada => 1008.0,         // RTX 4090 GDDR6X
            GpuArchitecture::Hopper => 3352.0,      // H100 HBM3
            GpuArchitecture::Blackwell => 4000.0,   // B100 HBM3e (estimated)
            GpuArchitecture::BlackwellUltra => 5000.0, // B200 (estimated)
        }
    }

    /// Get all architectures in order from oldest to newest
    pub fn all() -> &'static [GpuArchitecture] {
        &[
            GpuArchitecture::Volta,
            GpuArchitecture::Turing,
            GpuArchitecture::Ampere,
            GpuArchitecture::Ada,
            GpuArchitecture::Hopper,
            GpuArchitecture::Blackwell,
            GpuArchitecture::BlackwellUltra,
        ]
    }
}

impl fmt::Display for GpuArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuArchitecture::Volta => write!(f, "Volta"),
            GpuArchitecture::Turing => write!(f, "Turing"),
            GpuArchitecture::Ampere => write!(f, "Ampere"),
            GpuArchitecture::Ada => write!(f, "Ada Lovelace"),
            GpuArchitecture::Hopper => write!(f, "Hopper"),
            GpuArchitecture::Blackwell => write!(f, "Blackwell"),
            GpuArchitecture::BlackwellUltra => write!(f, "Blackwell Ultra"),
        }
    }
}

/// Individual architecture feature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchFeature {
    /// BFloat16 support (sm_80+)
    BF16,
    /// FP8 formats E4M3/E5M2 (sm_89+)
    FP8,
    /// FP4 quantization (sm_100+)
    FP4,
    /// Tensor Memory Accelerator (sm_90+)
    TMA,
    /// Thread Block Clusters (sm_90+)
    Clusters,
    /// Distributed Shared Memory (sm_90+)
    DistributedSharedMemory,
    /// Asynchronous copy cp.async (sm_80+)
    AsyncCopy,
    /// Warp reduce operations redux (sm_80+)
    Redux,
    /// Multithreaded barrier mbarrier (sm_80+)
    MBarrier,
    /// Tensor Cores of specific generation
    TensorCores(u32),
    /// NVLink of specific version
    NVLink(u32),
    /// Dynamic shared memory carveout
    DynamicSharedMemory,
    /// Ray tracing cores
    RTCores,
    /// Transformer Engine v2 (sm_100+)
    TransformerEngineV2,
    /// Hardware decompression engine (sm_100+)
    DecompressionEngine,
}

/// Feature flags for an architecture
#[derive(Debug, Clone, Default)]
pub struct ArchFeatures {
    /// BFloat16 support
    pub bf16: bool,
    /// FP8 formats (E4M3, E5M2)
    pub fp8: bool,
    /// FP4 quantization
    pub fp4: bool,
    /// Tensor Memory Accelerator
    pub tma: bool,
    /// Thread Block Clusters
    pub clusters: bool,
    /// Distributed Shared Memory
    pub distributed_shared_memory: bool,
    /// Asynchronous copy (cp.async)
    pub async_copy: bool,
    /// Warp reduce (redux)
    pub redux: bool,
    /// Multithreaded barrier
    pub mbarrier: bool,
    /// Dynamic shared memory
    pub dynamic_shared_memory: bool,
    /// Ray tracing cores
    pub rt_cores: bool,
    /// Transformer Engine v2
    pub transformer_engine_v2: bool,
    /// Hardware decompression
    pub decompression_engine: bool,
    /// Tensor Core generation (0 = none)
    pub tensor_core_gen: u32,
    /// NVLink version (0 = none)
    pub nvlink_version: u32,
    /// Maximum shared memory per block (KB)
    pub max_shared_memory_kb: u32,
    /// Maximum registers per thread
    pub max_registers_per_thread: u32,
    /// Warp size
    pub warp_size: u32,
}

impl ArchFeatures {
    /// Create features for a specific architecture
    pub fn for_architecture(arch: GpuArchitecture) -> Self {
        let sm = arch.sm_version();

        Self {
            bf16: sm >= 80,
            fp8: sm >= 89,
            fp4: sm >= 100,
            tma: sm >= 90,
            clusters: sm >= 90,
            distributed_shared_memory: sm >= 90,
            async_copy: sm >= 80,
            redux: sm >= 80,
            mbarrier: sm >= 80,
            dynamic_shared_memory: sm >= 70,
            rt_cores: sm >= 75,
            transformer_engine_v2: sm >= 100,
            decompression_engine: sm >= 100,
            tensor_core_gen: arch.tensor_core_gen().unwrap_or(0),
            nvlink_version: Self::nvlink_version_for(arch),
            max_shared_memory_kb: Self::max_shared_memory_for(arch),
            max_registers_per_thread: 255,
            warp_size: 32,
        }
    }

    /// Get NVLink version for architecture
    fn nvlink_version_for(arch: GpuArchitecture) -> u32 {
        match arch {
            GpuArchitecture::Volta => 2,
            GpuArchitecture::Turing => 0, // Consumer Turing has no NVLink
            GpuArchitecture::Ampere => 3,
            GpuArchitecture::Ada => 4,
            GpuArchitecture::Hopper => 4,
            GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => 5,
        }
    }

    /// Get max shared memory per block for architecture
    fn max_shared_memory_for(arch: GpuArchitecture) -> u32 {
        match arch {
            GpuArchitecture::Volta => 96,
            GpuArchitecture::Turing => 64,
            GpuArchitecture::Ampere => 164,
            GpuArchitecture::Ada => 100,
            GpuArchitecture::Hopper => 228,
            GpuArchitecture::Blackwell | GpuArchitecture::BlackwellUltra => 256,
        }
    }

    /// Check if a specific feature is supported
    pub fn supports(&self, feature: ArchFeature) -> bool {
        match feature {
            ArchFeature::BF16 => self.bf16,
            ArchFeature::FP8 => self.fp8,
            ArchFeature::FP4 => self.fp4,
            ArchFeature::TMA => self.tma,
            ArchFeature::Clusters => self.clusters,
            ArchFeature::DistributedSharedMemory => self.distributed_shared_memory,
            ArchFeature::AsyncCopy => self.async_copy,
            ArchFeature::Redux => self.redux,
            ArchFeature::MBarrier => self.mbarrier,
            ArchFeature::DynamicSharedMemory => self.dynamic_shared_memory,
            ArchFeature::RTCores => self.rt_cores,
            ArchFeature::TransformerEngineV2 => self.transformer_engine_v2,
            ArchFeature::DecompressionEngine => self.decompression_engine,
            ArchFeature::TensorCores(gen) => self.tensor_core_gen >= gen,
            ArchFeature::NVLink(ver) => self.nvlink_version >= ver,
        }
    }
}

/// Full capabilities for a GPU device
#[derive(Debug, Clone)]
pub struct ArchCapabilities {
    /// Architecture generation
    pub architecture: GpuArchitecture,
    /// CUDA compute capability (major, minor)
    pub compute_capability: (u32, u32),
    /// Architecture features
    pub features: ArchFeatures,
    /// FP32 peak TFLOPS
    pub fp32_tflops: f64,
    /// FP16 peak TFLOPS
    pub fp16_tflops: f64,
    /// Tensor TFLOPS (FP16 input, FP32 accumulate)
    pub tensor_tflops: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gb_s: f64,
    /// Total device memory (bytes)
    pub memory_bytes: usize,
    /// Number of streaming multiprocessors
    pub sm_count: u32,
    /// Clock rate (MHz)
    pub clock_mhz: u32,
}

impl ArchCapabilities {
    /// Create default capabilities for an architecture (representative values)
    pub fn default_for(arch: GpuArchitecture) -> Self {
        let features = ArchFeatures::for_architecture(arch);
        let (cc_major, cc_minor) = arch.compute_capability();

        Self {
            architecture: arch,
            compute_capability: (cc_major, cc_minor),
            features,
            fp32_tflops: arch.typical_fp32_tflops(),
            fp16_tflops: arch.typical_fp32_tflops() * 2.0,
            tensor_tflops: arch.typical_fp32_tflops() * 8.0,
            memory_bandwidth_gb_s: arch.typical_memory_bandwidth(),
            memory_bytes: Self::typical_memory_for(arch),
            sm_count: Self::typical_sm_count_for(arch),
            clock_mhz: Self::typical_clock_for(arch),
        }
    }

    /// Typical memory for architecture class
    fn typical_memory_for(arch: GpuArchitecture) -> usize {
        match arch {
            GpuArchitecture::Volta => 32 * 1024 * 1024 * 1024,     // 32 GB
            GpuArchitecture::Turing => 11 * 1024 * 1024 * 1024,    // 11 GB
            GpuArchitecture::Ampere => 80 * 1024 * 1024 * 1024,    // 80 GB
            GpuArchitecture::Ada => 24 * 1024 * 1024 * 1024,       // 24 GB
            GpuArchitecture::Hopper => 80 * 1024 * 1024 * 1024,    // 80 GB
            GpuArchitecture::Blackwell => 192 * 1024 * 1024 * 1024, // 192 GB
            GpuArchitecture::BlackwellUltra => 192 * 1024 * 1024 * 1024,
        }
    }

    /// Typical SM count for architecture class
    fn typical_sm_count_for(arch: GpuArchitecture) -> u32 {
        match arch {
            GpuArchitecture::Volta => 80,
            GpuArchitecture::Turing => 68,
            GpuArchitecture::Ampere => 108,
            GpuArchitecture::Ada => 128,
            GpuArchitecture::Hopper => 132,
            GpuArchitecture::Blackwell => 144,
            GpuArchitecture::BlackwellUltra => 160,
        }
    }

    /// Typical clock for architecture class (MHz)
    fn typical_clock_for(arch: GpuArchitecture) -> u32 {
        match arch {
            GpuArchitecture::Volta => 1530,
            GpuArchitecture::Turing => 1545,
            GpuArchitecture::Ampere => 1410,
            GpuArchitecture::Ada => 2520,
            GpuArchitecture::Hopper => 1830,
            GpuArchitecture::Blackwell => 2100,
            GpuArchitecture::BlackwellUltra => 2200,
        }
    }

    /// Compute a normalized performance score (0.0 - 1.0)
    /// relative to the fastest architecture
    pub fn performance_score(&self) -> f64 {
        // Use tensor TFLOPS as primary metric for ML workloads
        let max_tensor = GpuArchitecture::BlackwellUltra.typical_fp32_tflops() * 8.0;
        (self.tensor_tflops / max_tensor).min(1.0)
    }

    /// Compute memory-bound performance score
    pub fn memory_score(&self) -> f64 {
        let max_bw = 5000.0; // BlackwellUltra estimated
        (self.memory_bandwidth_gb_s / max_bw).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_cc() {
        assert_eq!(GpuArchitecture::from_compute_capability((7, 0)), GpuArchitecture::Volta);
        assert_eq!(GpuArchitecture::from_compute_capability((7, 5)), GpuArchitecture::Turing);
        assert_eq!(GpuArchitecture::from_compute_capability((8, 0)), GpuArchitecture::Ampere);
        assert_eq!(GpuArchitecture::from_compute_capability((8, 6)), GpuArchitecture::Ampere);
        assert_eq!(GpuArchitecture::from_compute_capability((8, 9)), GpuArchitecture::Ada);
        assert_eq!(GpuArchitecture::from_compute_capability((9, 0)), GpuArchitecture::Hopper);
        assert_eq!(GpuArchitecture::from_compute_capability((10, 0)), GpuArchitecture::Blackwell);
        assert_eq!(GpuArchitecture::from_compute_capability((12, 0)), GpuArchitecture::BlackwellUltra);
    }

    #[test]
    fn test_architecture_ordering() {
        assert!(GpuArchitecture::Volta < GpuArchitecture::Turing);
        assert!(GpuArchitecture::Turing < GpuArchitecture::Ampere);
        assert!(GpuArchitecture::Ampere < GpuArchitecture::Ada);
        assert!(GpuArchitecture::Ada < GpuArchitecture::Hopper);
        assert!(GpuArchitecture::Hopper < GpuArchitecture::Blackwell);
        assert!(GpuArchitecture::Blackwell < GpuArchitecture::BlackwellUltra);
    }

    #[test]
    fn test_feature_support() {
        let volta = GpuArchitecture::Volta;
        let ampere = GpuArchitecture::Ampere;
        let hopper = GpuArchitecture::Hopper;
        let blackwell = GpuArchitecture::Blackwell;

        // BF16: Ampere+
        assert!(!volta.supports(ArchFeature::BF16));
        assert!(ampere.supports(ArchFeature::BF16));

        // TMA: Hopper+
        assert!(!ampere.supports(ArchFeature::TMA));
        assert!(hopper.supports(ArchFeature::TMA));

        // FP4: Blackwell+
        assert!(!hopper.supports(ArchFeature::FP4));
        assert!(blackwell.supports(ArchFeature::FP4));
    }

    #[test]
    fn test_sm_string() {
        assert_eq!(GpuArchitecture::Volta.sm_string(), "sm_70");
        assert_eq!(GpuArchitecture::Ampere.sm_string(), "sm_80");
        assert_eq!(GpuArchitecture::Blackwell.sm_string(), "sm_100");
    }

    #[test]
    fn test_tensor_core_gen() {
        assert_eq!(GpuArchitecture::Volta.tensor_core_gen(), Some(1));
        assert_eq!(GpuArchitecture::Ampere.tensor_core_gen(), Some(3));
        assert_eq!(GpuArchitecture::Blackwell.tensor_core_gen(), Some(5));
    }

    #[test]
    fn test_capabilities_default() {
        let caps = ArchCapabilities::default_for(GpuArchitecture::Ampere);
        assert_eq!(caps.architecture, GpuArchitecture::Ampere);
        assert_eq!(caps.compute_capability, (8, 0));
        assert!(caps.features.bf16);
        assert!(!caps.features.tma);
    }

    #[test]
    fn test_performance_score() {
        let volta_caps = ArchCapabilities::default_for(GpuArchitecture::Volta);
        let blackwell_caps = ArchCapabilities::default_for(GpuArchitecture::Blackwell);

        // Blackwell should have higher score
        assert!(blackwell_caps.performance_score() > volta_caps.performance_score());
    }

    #[test]
    fn test_all_architectures() {
        let all = GpuArchitecture::all();
        assert_eq!(all.len(), 7);
        assert_eq!(all[0], GpuArchitecture::Volta);
        assert_eq!(all[6], GpuArchitecture::BlackwellUltra);
    }
}
