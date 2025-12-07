//! Hardware-level GPU infrastructure and ISA specifications
//!
//! This module provides the bedrock, subatomic, and quantum hardware layers for distributed GPU computing:
//!
//! ## Bedrock Layer
//! - **PTX ISA**: Complete instruction set reference for NVIDIA GPUs
//! - **Memory Model**: Formal CUDA/PTX memory model with scoped synchronization
//! - **Interconnect**: NVLink and PCIe protocol specifications
//! - **Warp Execution**: SIMT execution model with divergence handling
//!
//! ## Subatomic Layer (Microarchitecture)
//! - **Register File**: Banking, operand collectors, register allocation
//! - **Cache Hierarchy**: L1/L2 caches, memory controller, texture cache
//! - **Tensor Cores**: MMA operations, fragment layout, mixed precision
//! - **Unified Memory**: Page tables, fault handling, migration
//! - **SM Model**: Complete SM simulation and performance prediction
//!
//! ## Quantum Layer (Fundamentals)
//! - **ECC Memory**: Error correcting codes, SECDED, memory scrubbing
//! - **Power Management**: DVFS, power domains, thermal modeling
//! - **NVSwitch Fabric**: Multi-GPU topology, collective scheduling
//! - **Formal Verification**: Temporal logic, model checking, runtime monitors
//!
//! ## Planck Layer (Irreducible Foundations)
//! - **Warp Divergence Deep**: SIMT stack, ITS scheduling, sync hazards
//! - **Precision Analysis**: FP formats, error bounds, reproducibility
//! - **DRAM Internals**: HBM/GDDR organization, timing, bandwidth modeling
//! - **Performance Counters**: Counter specs, profiling, roofline model
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Hardware Abstraction Layer                    │
//! ├─────────────────┬─────────────────┬─────────────────────────────┤
//! │   PTX ISA       │  Memory Model   │     Interconnect            │
//! │  (ptx_isa.rs)   │ (memory_model)  │   (interconnect.rs)         │
//! ├─────────────────┴─────────────────┴─────────────────────────────┤
//! │                    Warp Execution Model                          │
//! │                    (warp_execution.rs)                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    Microarchitecture Layer                       │
//! ├─────────────────┬─────────────────┬─────────────────────────────┤
//! │  Register File  │ Cache Hierarchy │    Tensor Cores             │
//! ├─────────────────┴─────────────────┴─────────────────────────────┤
//! │            Unified Memory  │  SM Model                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    Quantum Fundamentals Layer                    │
//! ├─────────────────┬─────────────────┬─────────────────────────────┤
//! │   ECC Memory    │ Power Mgmt/DVFS │    NVSwitch Fabric          │
//! ├─────────────────┴─────────────────┴─────────────────────────────┤
//! │                   Formal Verification                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    Planck Fundamentals Layer                     │
//! ├─────────────────┬─────────────────┬─────────────────────────────┤
//! │ Divergence Deep │ Precision/FP    │    DRAM Internals           │
//! ├─────────────────┴─────────────────┴─────────────────────────────┤
//! │                   Performance Counters                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::hardware::{
//!     PtxTarget, RegType, StateSpace, MemoryOrdering,
//!     NVLinkSpec, PCIeSpec, GPUDirectRDMA,
//!     ThreadMask, WarpScheduler, DivergenceAnalyzer,
//! };
//!
//! // PTX instruction generation
//! let target = PtxTarget::sm90();
//! let ld = LdInstruction::global_acquire(RegType::F32);
//!
//! // Memory model verification
//! let checker = MemoryModelChecker::new();
//! let result = checker.validate();
//!
//! // Interconnect selection
//! let nvlink = NVLinkSpec::nvlink4();
//! let bandwidth = nvlink.bandwidth_per_direction();
//!
//! // Warp scheduling
//! let scheduler = WarpScheduler::new(4, SchedulerType::Gto);
//! ```

// Bedrock layer modules
pub mod interconnect;
pub mod memory_model;
pub mod ptx_isa;
pub mod warp_execution;

// Subatomic layer modules (microarchitecture)
pub mod cache_hierarchy;
pub mod register_file;
pub mod sm_model;
pub mod tensor_cores;
pub mod unified_memory;

// Quantum layer modules (fundamentals)
pub mod ecc;
pub mod nvswitch;
pub mod power_management;
pub mod verification;

// Planck layer modules (irreducible foundations)
pub mod divergence_deep;
pub mod dram_internals;
pub mod performance_counters;
pub mod precision_analysis;

// Re-export PTX ISA types
pub use ptx_isa::{
    AsyncCopy, AtomInstruction, AtomicOp, BarrierType, CacheOp, ExecutionUnit, FenceInstruction,
    InstructionTiming, LdInstruction, MemoryOrdering, MemoryScope, PtxTarget, RegType,
    ShflInstruction, ShuffleMode, StInstruction, StateSpace, VoteOp,
};

// Re-export memory model types
pub use memory_model::{
    DataRace, ExecutionRelations, LitmusInstruction, LitmusTest, LitmusThread, Location,
    MemoryEvent, MemoryModelChecker, OpType, OrderingTag, Scope, ThreadId, ValidationResult,
};

// Re-export interconnect types
pub use interconnect::{
    GPUDirectRDMA, InterconnectComparison, NVLinkFlowControl, NVLinkPacket, NVLinkPacketType,
    NVLinkSpec, PCIeFlowControl, PCIePath, PCIeSpec, PCIeSwitch, RdmaOpcode, RdmaVerb, TLPType,
    TLP,
};

// Re-export warp execution types
pub use warp_execution::{
    BasicBlock, ControlFlowInst, DivergenceAnalyzer, DivergenceReport, DivergenceStack,
    SchedulerType, StallReason, ThreadMask, TraceEvent, WarpScheduler, WarpState, WarpTrace,
};

// Re-export register file types
pub use register_file::{
    BankConflictResult, BankSelectFunction, GpuRegisterAllocator, InterferenceGraph, LiveRange,
    OperandCollector, RegisterFileSpec, RegisterPressureAnalysis,
};

// Re-export cache hierarchy types
pub use cache_hierarchy::{
    CacheAccessResult, CacheLine, CacheLineState, CacheStats, DirectoryEntry, DirectoryState,
    DramTimingSpec, L1Cache, L1CacheSpec, L2AccessResult, L2Cache, L2CacheSlice, L2CacheSpec,
    MemoryController, MemoryControllerStats, MemoryRequest, MemoryType, MortonCode, MshrEntry,
    TexCoord, TextureCache, TextureCacheSpec, TextureDimensions, WritePolicy,
};

// Re-export tensor core types
pub use tensor_cores::{
    FragmentLayout, FragmentMapping, FragmentType, MmaConfig, MmaDimensions, MmaOperation,
    TensorCoreGen, TensorCoreSpec, TensorCoreStats, TensorCoreUnit, TensorDataType, WmmaOperation,
};

// Re-export unified memory types
pub use unified_memory::{
    AccessHistory, AccessPattern, AccessPatternDetector, FaultHandlerStats, GpuPageTable,
    MemoryLocation, MigrationEngine, MigrationInFlight, MigrationPolicy, MigrationRequest,
    MigrationStats, PageFault, PageFaultHandler, PageFaultResolution, PageFaultType, PageSize,
    PageTableEntry, PageTableStats, PrefetchStats, ResolutionAction, UnifiedMemoryManager,
    UvmAllocation, UvmPrefetcher,
};

// Re-export SM model types
pub use sm_model::{
    KernelCharacteristics, OccupancyCalculator, OccupancyLimiter, OccupancyResult,
    PerformanceBottleneck, PerformancePrediction, PerformancePredictor, SchedulerStats,
    SchedulingPolicy, SmArchitecture, SmSimulator, SmSpec, SmStats, ThreadBlock, Warp,
    WarpScheduler as SmWarpScheduler, WarpState as SmWarpState,
};

// Re-export ECC types
pub use ecc::{
    ChipkillCode, EccConfig, EccMemoryController, EccResult, EccStats, ErrorInjector,
    ErrorRateModel, HammingCode, MemoryError, MemoryScrubber,
};

// Re-export power management types
pub use power_management::{
    DvfsController, DvfsPolicy, FrequencyChangeReason, FrequencyRecord, FrequencyStep,
    GpuPowerBreakdown, KernelPowerModel, PowerAllocationPolicy, PowerBudgetManager,
    PowerConnectors, PowerDomain, PowerManagementSystem, PowerSpec, ThermalModel, WorkloadActivity,
    WorkloadType,
};

// Re-export NVSwitch types
pub use nvswitch::{
    CollectiveAlgorithm, CollectiveScheduler, CollectiveStep, CrossbarStats, CrossbarTransfer,
    FabricTopology, InterNodeType, MultiNodeFabric, NvLinkConnection, NvSwitchCrossbar,
    NvSwitchSpec, TopologyType,
};

// Re-export verification types
pub use verification::{
    CtlFormula, CtlModelChecker, KripkeStructure, LtlFormula, MonitorResult, RuntimeMonitor,
    SmtLibGenerator, State, Transition, VerificationResult,
};

// Re-export divergence deep types
pub use divergence_deep::{
    DivergenceImpactModel, IndependentThreadScheduler, SimtStack, SimtStackEntry, ThreadState,
    WarpSyncHazard, WarpSyncHazardDetector,
};

// Re-export precision analysis types
pub use precision_analysis::{
    ErrorAnalysis, ErrorBound, FailureMode, FloatFormat, MixedPrecisionAnalysis,
    NumericalStabilityChecker, ReproducibilityAnalysis, ReproducibilityCost,
};

// Re-export DRAM internals types
pub use dram_internals::{
    AccessPattern as DramAccessPattern, DramAccessAnalyzer, DramOrganization, DramStats,
    DramTiming, HonestBandwidthModel, RowBufferPolicyAnalyzer,
};

// Re-export performance counter types
pub use performance_counters::{
    CounterInterpreter, CounterSpec, MeasurementType, ProfilingAdvisor, RooflineModel,
    SimulatedCounterCollector,
};

/// Hardware capability detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// PTX target architecture
    pub target: PtxTarget,
    /// Supports async copy
    pub async_copy: bool,
    /// Supports cluster-level sync
    pub cluster_sync: bool,
    /// NVLink specification (if available)
    pub nvlink: Option<NVLinkSpec>,
    /// PCIe specification
    pub pcie: PCIeSpec,
    /// Warp size
    pub warp_size: usize,
    /// Max warps per SM
    pub max_warps_per_sm: usize,
    /// Shared memory per SM (bytes)
    pub shared_memory_per_sm: usize,
    /// L2 cache size (bytes)
    pub l2_cache_size: usize,
}

impl HardwareCapabilities {
    /// Create capabilities for SM 8.0 (Ampere)
    pub fn ampere() -> Self {
        Self {
            target: PtxTarget::sm80(),
            async_copy: true,
            cluster_sync: false,
            nvlink: Some(NVLinkSpec::nvlink3()),
            pcie: PCIeSpec::gen4_x16(),
            warp_size: 32,
            max_warps_per_sm: 64,
            shared_memory_per_sm: 164 * 1024,
            l2_cache_size: 40 * 1024 * 1024,
        }
    }

    /// Create capabilities for SM 8.9 (Ada Lovelace)
    pub fn ada() -> Self {
        Self {
            target: PtxTarget::sm89(),
            async_copy: true,
            cluster_sync: false,
            nvlink: None, // Consumer cards
            pcie: PCIeSpec::gen4_x16(),
            warp_size: 32,
            max_warps_per_sm: 48,
            shared_memory_per_sm: 100 * 1024,
            l2_cache_size: 72 * 1024 * 1024,
        }
    }

    /// Create capabilities for SM 9.0 (Hopper)
    pub fn hopper() -> Self {
        Self {
            target: PtxTarget::sm90(),
            async_copy: true,
            cluster_sync: true,
            nvlink: Some(NVLinkSpec::nvlink4()),
            pcie: PCIeSpec::gen5_x16(),
            warp_size: 32,
            max_warps_per_sm: 64,
            shared_memory_per_sm: 228 * 1024,
            l2_cache_size: 50 * 1024 * 1024,
        }
    }

    /// Check if architecture supports a feature
    pub fn supports_feature(&self, feature: HardwareFeature) -> bool {
        match feature {
            HardwareFeature::AsyncCopy => self.async_copy,
            HardwareFeature::ClusterSync => self.cluster_sync,
            HardwareFeature::NVLink => self.nvlink.is_some(),
            HardwareFeature::TensorCores => true, // All modern archs
            HardwareFeature::FP64 => true,
            HardwareFeature::AtomicFloat => {
                // SM 8.0+ supports atomic float
                self.target.sm >= 80
            }
        }
    }

    /// Get effective bandwidth for device-to-device transfers (bytes/sec)
    pub fn d2d_bandwidth(&self) -> f64 {
        if let Some(ref nvlink) = self.nvlink {
            nvlink.bandwidth_per_direction()
        } else {
            self.pcie.bandwidth()
        }
    }

    /// Get memory model scope supported
    pub fn max_memory_scope(&self) -> Scope {
        // All current GPUs support system scope
        // Cluster scope is handled separately via cluster_sync
        Scope::Sys
    }
}

/// Hardware features for capability checking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareFeature {
    /// SM 8.0+ async memory copy
    AsyncCopy,
    /// SM 9.0+ cluster synchronization
    ClusterSync,
    /// NVLink interconnect
    NVLink,
    /// Tensor cores
    TensorCores,
    /// FP64 support
    FP64,
    /// Atomic float operations
    AtomicFloat,
}

/// Hardware configuration for kernel optimization
#[derive(Debug, Clone)]
pub struct KernelHardwareConfig {
    /// Target architecture
    pub target: PtxTarget,
    /// Block size (threads per block)
    pub block_size: usize,
    /// Grid size (blocks)
    pub grid_size: usize,
    /// Shared memory per block
    pub shared_mem_per_block: usize,
    /// Registers per thread
    pub registers_per_thread: usize,
    /// Memory ordering for loads
    pub load_ordering: MemoryOrdering,
    /// Memory ordering for stores
    pub store_ordering: MemoryOrdering,
    /// Warp scheduler type
    pub scheduler: SchedulerType,
}

impl Default for KernelHardwareConfig {
    fn default() -> Self {
        Self {
            target: PtxTarget::sm80(),
            block_size: 256,
            grid_size: 1,
            shared_mem_per_block: 0,
            registers_per_thread: 32,
            load_ordering: MemoryOrdering::Relaxed,
            store_ordering: MemoryOrdering::Relaxed,
            scheduler: SchedulerType::Gto,
        }
    }
}

impl KernelHardwareConfig {
    /// Calculate theoretical occupancy
    pub fn theoretical_occupancy(&self, caps: &HardwareCapabilities) -> f64 {
        let warps_per_block = (self.block_size + caps.warp_size - 1) / caps.warp_size;
        let max_blocks_by_warps = caps.max_warps_per_sm / warps_per_block;

        let max_blocks_by_shmem = if self.shared_mem_per_block > 0 {
            caps.shared_memory_per_sm / self.shared_mem_per_block
        } else {
            usize::MAX
        };

        let active_blocks = max_blocks_by_warps.min(max_blocks_by_shmem);
        let active_warps = active_blocks * warps_per_block;

        active_warps as f64 / caps.max_warps_per_sm as f64
    }

    /// Estimate warp execution efficiency given divergence
    pub fn estimate_simd_efficiency(&self, divergence_rate: f64) -> f64 {
        // Simple model: efficiency decreases with divergence
        1.0 - (divergence_rate * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_capabilities_ampere() {
        let caps = HardwareCapabilities::ampere();

        assert!(caps.supports_feature(HardwareFeature::AsyncCopy));
        assert!(!caps.supports_feature(HardwareFeature::ClusterSync));
        assert!(caps.supports_feature(HardwareFeature::NVLink));
        assert_eq!(caps.warp_size, 32);
    }

    #[test]
    fn test_hardware_capabilities_hopper() {
        let caps = HardwareCapabilities::hopper();

        assert!(caps.supports_feature(HardwareFeature::AsyncCopy));
        assert!(caps.supports_feature(HardwareFeature::ClusterSync));
        assert!(caps.supports_feature(HardwareFeature::NVLink));
        assert_eq!(caps.max_memory_scope(), Scope::Sys);
    }

    #[test]
    fn test_d2d_bandwidth() {
        let ampere = HardwareCapabilities::ampere();
        let ada = HardwareCapabilities::ada();

        // Ampere with NVLink should have higher bandwidth
        assert!(ampere.d2d_bandwidth() > ada.d2d_bandwidth());
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelHardwareConfig::default();

        assert_eq!(config.block_size, 256);
        assert_eq!(config.target.sm, 80);
    }

    #[test]
    fn test_theoretical_occupancy() {
        let caps = HardwareCapabilities::ampere();
        let config = KernelHardwareConfig {
            block_size: 256,
            ..Default::default()
        };

        let occupancy = config.theoretical_occupancy(&caps);
        assert!(occupancy > 0.0 && occupancy <= 1.0);
    }

    #[test]
    fn test_simd_efficiency() {
        let config = KernelHardwareConfig::default();

        // No divergence = full efficiency
        assert!((config.estimate_simd_efficiency(0.0) - 1.0).abs() < 0.001);

        // 50% divergence
        assert!((config.estimate_simd_efficiency(0.5) - 0.75).abs() < 0.001);
    }
}
