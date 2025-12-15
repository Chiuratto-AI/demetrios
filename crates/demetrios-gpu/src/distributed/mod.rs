//! Multi-GPU & Distributed Scalability Module
//!
//! This module provides SOTA multi-GPU and distributed computing capabilities:
//!
//! - **Topology Discovery**: GPU interconnect detection and modeling
//! - **Collectives**: All-reduce, all-gather, broadcast, etc.
//! - **Data Parallelism**: Type-safe sharding with unit preservation
//! - **Pipeline Parallelism**: GPipe and 1F1B scheduling
//! - **Fault Tolerance**: Phi-accrual failure detection and checkpointing
//!
//! # Example
//!
//! ```ignore
//! use demetrios_gpu::distributed::{
//!     DistributedContext, DataParallel, Distribution,
//!     ShardCorrelation, CollectiveOp,
//! };
//!
//! // Initialize distributed context
//! let ctx = DistributedContext::init()?;
//!
//! // Create data-parallel executor
//! let parallel = DataParallel::new(ctx.my_devices(), ctx.topology.clone());
//!
//! // Shard data across GPUs
//! let distributed = parallel.shard(input, Distribution::EvenSplit)?;
//!
//! // Process in parallel
//! let result = parallel.map(&distributed, "kernel", |shard| {
//!     // Process each shard
//!     Ok(shard.clone())
//! })?;
//!
//! // Gather results
//! let output = parallel.gather(result, DeviceId(0))?;
//! ```

pub mod collectives;
pub mod data_parallel;
pub mod fault_tolerance;
pub mod formal;
pub mod hardware;
pub mod heterogeneous;
pub mod model;
pub mod pipeline;
pub mod primitives;
pub mod topology;
pub mod verify;

// Re-exports
pub use topology::{
    AllReduceTree, DeviceGroup, GpuDevice, GpuTopology, InterconnectType, NodeId, NodeInfo,
};

pub use collectives::{
    BufferShard, CollectiveManager, CollectiveOp, Communicator, DistributedGpuBuffer, Distribution,
    ShardSpec,
};

pub use data_parallel::{
    DataParallel, DistributedEpistemicBuffer, DistributedUnitBuffer, EpistemicDataParallel,
    EpistemicShard, ShardCorrelation, UnitShard,
};

pub use pipeline::{PipelineExecutor, PipelinePartitioner, PipelineSchedule, PipelineStage};

pub use fault_tolerance::{
    Checkpoint, CheckpointId, CheckpointManager, DeviceHealth, ElasticExecutor, FailureDetector,
    RecoveryAction, ReplicatedBuffer,
};

// Heterogeneous multi-GPU support (Volta + Ampere + Ada + Hopper + Blackwell)
pub use heterogeneous::{
    ArchCapabilities, ArchFeature, ArchFeatures, CoherenceMode, CompileError, CompiledKernel,
    CompiledKernelSet, DeviceBuffer, DevicePool, DevicePoolError, DeviceResult,
    DiscoveredDevice, DiscoveryError, DistributedLaunchConfig, DistributionStrategy,
    ExecutionSummary, GpuArchitecture, HeterogeneousCompiler, HeterogeneousDevice,
    HeterogeneousDeviceManager, HeterogeneousError, HeterogeneousMemoryManager,
    HeterogeneousScheduler, InputMode, KernelArg, KernelSignature, KernelSource,
    LaunchConfig, LaunchConfigHint, MemoryError, MemoryInfo, NVLinkInfo, OptSettings,
    OutputMode, P2PCapability, ParamType, ReduceOp, SyncMode, TransferPath, UnifiedBuffer,
    WorkAssignment, WorkDistribution,
};

use crate::backend::CpuBackend;
use crate::optimize::pool::DeviceId;
use crate::runtime::BufferError;
use std::sync::Arc;

/// Complete distributed execution context
pub struct DistributedContext {
    /// Cluster topology
    pub topology: Arc<GpuTopology>,
    /// Collective manager
    pub collective: CollectiveManager,
    /// Failure detector
    pub failure_detector: FailureDetector,
    /// This process's rank
    pub rank: usize,
    /// World size
    pub world_size: usize,
}

impl DistributedContext {
    /// Initialize distributed context
    pub fn init() -> Result<Self, BufferError> {
        let topology = Arc::new(GpuTopology::discover()?);
        let backend = Arc::new(CpuBackend::new());

        let collective = CollectiveManager::new(topology.clone(), backend);
        let failure_detector = FailureDetector::new(10.0);

        Ok(Self {
            topology,
            collective,
            failure_detector,
            rank: 0,
            world_size: 1,
        })
    }

    /// Initialize with specific rank
    pub fn with_rank(mut self, rank: usize, world_size: usize) -> Self {
        self.rank = rank;
        self.world_size = world_size;
        self
    }

    /// Get devices for this rank
    pub fn my_devices(&self) -> Vec<DeviceId> {
        self.topology.devices().map(|d| d.id).collect()
    }

    /// Total GPU memory available
    pub fn total_memory(&self) -> usize {
        self.topology.total_memory()
    }

    /// Total compute in TFLOPS
    pub fn total_compute(&self) -> f64 {
        self.topology.total_compute()
    }

    /// Number of devices
    pub fn num_devices(&self) -> usize {
        self.topology.num_devices()
    }

    /// Check if this is root rank
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Barrier synchronization (placeholder)
    pub async fn barrier(&self) -> Result<(), BufferError> {
        // Would use collective barrier
        Ok(())
    }
}

/// Scaling strategy recommendation
#[derive(Debug)]
pub enum ScalingRecommendation {
    /// Single GPU is sufficient
    SingleGpu { device: DeviceId, reason: String },
    /// Data parallel across devices
    DataParallel {
        devices: Vec<DeviceId>,
        distribution: Distribution,
        reason: String,
    },
    /// Pipeline parallel
    PipelineParallel {
        stages: usize,
        microbatches: usize,
        reason: String,
    },
    /// Hybrid data + pipeline
    Hybrid {
        data_parallel_factor: usize,
        pipeline_stages: usize,
        reason: String,
    },
}

impl ScalingRecommendation {
    /// Get recommendation reason
    pub fn reason(&self) -> &str {
        match self {
            Self::SingleGpu { reason, .. } => reason,
            Self::DataParallel { reason, .. } => reason,
            Self::PipelineParallel { reason, .. } => reason,
            Self::Hybrid { reason, .. } => reason,
        }
    }

    /// Check if multi-GPU is recommended
    pub fn is_multi_gpu(&self) -> bool {
        !matches!(self, Self::SingleGpu { .. })
    }
}

/// Scaling advisor for workload analysis
pub struct ScalingAdvisor {
    topology: Arc<GpuTopology>,
}

impl ScalingAdvisor {
    pub fn new(topology: Arc<GpuTopology>) -> Self {
        Self { topology }
    }

    /// Recommend scaling strategy based on workload
    pub fn recommend(
        &self,
        data_size: usize,
        compute_ops: usize,
        memory_per_sample: usize,
    ) -> ScalingRecommendation {
        let devices: Vec<_> = self.topology.devices().collect();
        let num_devices = devices.len();

        if num_devices == 0 {
            return ScalingRecommendation::SingleGpu {
                device: DeviceId(0),
                reason: "No GPU devices found".into(),
            };
        }

        let total_memory: usize = devices.iter().map(|d| d.memory_bytes).sum();
        let max_device_memory = devices.iter().map(|d| d.memory_bytes).max().unwrap_or(0);

        let memory_needed = data_size * memory_per_sample;
        let fits_single = memory_needed <= max_device_memory;

        // Compute intensity (ops per byte)
        let compute_intensity = if data_size > 0 {
            compute_ops as f64 / data_size as f64
        } else {
            0.0
        };

        // Memory-bound: prefer data parallelism
        // Compute-bound: pipeline/tensor parallelism may help
        let is_memory_bound = compute_intensity < 10.0;

        if fits_single {
            return ScalingRecommendation::SingleGpu {
                device: devices[0].id,
                reason: "Data fits in single GPU memory".into(),
            };
        }

        if is_memory_bound && memory_needed <= total_memory {
            return ScalingRecommendation::DataParallel {
                devices: devices.iter().map(|d| d.id).collect(),
                distribution: Distribution::MemoryWeighted,
                reason: "Memory-bound workload, data parallelism optimal".into(),
            };
        }

        // Check for pipeline suitability
        if num_devices >= 2 && !is_memory_bound {
            return ScalingRecommendation::PipelineParallel {
                stages: num_devices,
                microbatches: num_devices * 4,
                reason: "Compute-bound with sequential dependencies".into(),
            };
        }

        // Hybrid for very large workloads
        if num_devices >= 4 {
            ScalingRecommendation::Hybrid {
                data_parallel_factor: num_devices / 2,
                pipeline_stages: 2,
                reason: "Large scale requires hybrid approach".into(),
            }
        } else {
            ScalingRecommendation::DataParallel {
                devices: devices.iter().map(|d| d.id).collect(),
                distribution: Distribution::EvenSplit,
                reason: "Default to data parallelism".into(),
            }
        }
    }

    /// Estimate execution time for a scaling strategy
    pub fn estimate_time(
        &self,
        recommendation: &ScalingRecommendation,
        total_compute_time: f64,
        total_data_bytes: usize,
    ) -> f64 {
        match recommendation {
            ScalingRecommendation::SingleGpu { .. } => total_compute_time,
            ScalingRecommendation::DataParallel { devices, .. } => {
                let n = devices.len() as f64;
                let compute_time = total_compute_time / n;
                let comm_time = self.topology.min_bandwidth(devices);
                let allreduce_time = if comm_time.is_finite() {
                    total_data_bytes as f64 / comm_time
                } else {
                    0.0
                };
                compute_time + allreduce_time
            }
            ScalingRecommendation::PipelineParallel {
                stages,
                microbatches,
                ..
            } => {
                let stage_time = total_compute_time / *stages as f64;
                let bubbles = stages - 1;
                (bubbles + microbatches) as f64 * stage_time
            }
            ScalingRecommendation::Hybrid {
                data_parallel_factor,
                pipeline_stages,
                ..
            } => {
                let dp_time = total_compute_time / *data_parallel_factor as f64;
                let pp_time = dp_time / *pipeline_stages as f64;
                let bubbles = pipeline_stages - 1;
                let microbatches = 4;
                (bubbles + microbatches) as f64 * pp_time
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_context() {
        let ctx = DistributedContext::init().unwrap();

        assert!(ctx.total_memory() > 0);
        assert!(!ctx.my_devices().is_empty());
        assert!(ctx.is_root());
    }

    #[test]
    fn test_distributed_context_with_rank() {
        let ctx = DistributedContext::init().unwrap().with_rank(1, 4);

        assert_eq!(ctx.rank, 1);
        assert_eq!(ctx.world_size, 4);
        assert!(!ctx.is_root());
    }

    #[test]
    fn test_scaling_advisor_small() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let advisor = ScalingAdvisor::new(topology);

        // Small workload
        let rec = advisor.recommend(1000, 10000, 4);

        // Should recommend single GPU for small data
        assert!(matches!(rec, ScalingRecommendation::SingleGpu { .. }));
    }

    #[test]
    fn test_scaling_recommendation_reason() {
        let rec = ScalingRecommendation::SingleGpu {
            device: DeviceId(0),
            reason: "Test reason".into(),
        };

        assert_eq!(rec.reason(), "Test reason");
        assert!(!rec.is_multi_gpu());
    }

    #[test]
    fn test_scaling_advisor_large() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let devices: Vec<_> = topology.devices().collect();

        if devices.len() >= 2 {
            let advisor = ScalingAdvisor::new(Arc::new(GpuTopology::discover().unwrap()));

            // Large workload that doesn't fit in single GPU
            let total_mem: usize = devices.iter().map(|d| d.memory_bytes).sum();
            let rec = advisor.recommend(total_mem, 1000, 1);

            // Should recommend multi-GPU
            assert!(rec.is_multi_gpu());
        }
    }

    #[test]
    fn test_estimate_time() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let advisor = ScalingAdvisor::new(topology);

        let rec = ScalingRecommendation::SingleGpu {
            device: DeviceId(0),
            reason: "Test".into(),
        };

        let time = advisor.estimate_time(&rec, 1.0, 1000);
        assert!((time - 1.0).abs() < 0.001);
    }
}
