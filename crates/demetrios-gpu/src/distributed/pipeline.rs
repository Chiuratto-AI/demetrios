//! Pipeline Parallelism
//!
//! This module provides pipeline parallelism for sequential computations:
//!
//! - GPipe-style fill-drain scheduling
//! - 1F1B interleaved scheduling (reduced memory)
//! - Automatic stage partitioning
//! - Pipeline visualization

use super::topology::{get_device, GpuTopology};
use crate::ir::effects::EffectSet;
use crate::optimize::pool::DeviceId;
use crate::runtime::{BufferError, GpuBuffer};
use std::collections::VecDeque;
use std::sync::Arc;

/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage index
    pub index: usize,
    /// Device for this stage
    pub device: DeviceId,
    /// Kernel to execute
    pub kernel_name: String,
    /// Effects of this stage
    pub effects: EffectSet,
    /// Input buffer size
    pub input_size: usize,
    /// Output buffer size
    pub output_size: usize,
}

impl PipelineStage {
    /// Create a new pipeline stage
    pub fn new(
        index: usize,
        device: DeviceId,
        kernel_name: impl Into<String>,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            index,
            device,
            kernel_name: kernel_name.into(),
            effects: EffectSet::default(),
            input_size,
            output_size,
        }
    }

    /// Add effects to this stage
    pub fn with_effects(mut self, effects: EffectSet) -> Self {
        self.effects = effects;
        self
    }
}

/// Pipeline schedule type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// Simple fill-drain (GPipe style)
    GPipe,
    /// Interleaved 1F1B (reduces memory)
    Interleaved1F1B,
    /// Virtual pipeline (Megatron-LM style)
    Virtual { num_virtual_stages: usize },
}

impl PipelineSchedule {
    /// Get schedule name
    pub fn name(&self) -> &'static str {
        match self {
            Self::GPipe => "GPipe",
            Self::Interleaved1F1B => "1F1B",
            Self::Virtual { .. } => "Virtual",
        }
    }
}

/// Pipeline parallelism executor
///
/// Splits computation into stages, each on different GPU.
/// Overlaps execution of different microbatches.
pub struct PipelineExecutor {
    stages: Vec<PipelineStage>,
    topology: Arc<GpuTopology>,
    /// Number of microbatches
    num_microbatches: usize,
    /// Schedule type
    schedule: PipelineSchedule,
}

impl PipelineExecutor {
    pub fn new(
        stages: Vec<PipelineStage>,
        topology: Arc<GpuTopology>,
        num_microbatches: usize,
    ) -> Self {
        Self {
            stages,
            topology,
            num_microbatches,
            schedule: PipelineSchedule::GPipe,
        }
    }

    pub fn with_schedule(mut self, schedule: PipelineSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Get stages
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Get number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get number of microbatches
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    /// Get schedule type
    pub fn schedule(&self) -> PipelineSchedule {
        self.schedule
    }

    /// Pipeline bubble overhead (fraction of time wasted)
    pub fn bubble_overhead(&self) -> f64 {
        let depth = self.stages.len();
        if depth == 0 || self.num_microbatches == 0 {
            return 0.0;
        }
        let bubbles = depth - 1;
        bubbles as f64 / (self.num_microbatches + bubbles) as f64
    }

    /// Theoretical efficiency
    pub fn efficiency(&self) -> f64 {
        1.0 - self.bubble_overhead()
    }

    /// Execute pipeline
    pub async fn execute<T: Copy + Default + Send + Sync>(
        &self,
        inputs: Vec<GpuBuffer<T>>,
    ) -> Result<Vec<GpuBuffer<T>>, BufferError> {
        match self.schedule {
            PipelineSchedule::GPipe => self.execute_gpipe(inputs).await,
            PipelineSchedule::Interleaved1F1B => self.execute_1f1b(inputs).await,
            PipelineSchedule::Virtual { num_virtual_stages } => {
                self.execute_virtual(inputs, num_virtual_stages).await
            }
        }
    }

    /// GPipe schedule: fill, steady state, drain
    async fn execute_gpipe<T: Copy + Default + Send + Sync>(
        &self,
        inputs: Vec<GpuBuffer<T>>,
    ) -> Result<Vec<GpuBuffer<T>>, BufferError> {
        let num_stages = self.stages.len();
        let num_microbatches = inputs.len();

        if num_stages == 0 {
            return Ok(inputs);
        }

        // Activation storage
        let mut activations: Vec<VecDeque<Option<Vec<T>>>> = vec![VecDeque::new(); num_stages + 1];

        // Initialize with inputs
        for input in inputs {
            let data = input.download()?;
            activations[0].push_back(Some(data));
        }
        // Add padding for pipeline drain
        for _ in 0..num_stages {
            activations[0].push_back(None);
        }

        let total_steps = num_microbatches + num_stages - 1;

        for step in 0..total_steps {
            // Each stage processes in parallel
            for stage_idx in 0..num_stages {
                let microbatch_idx = step as i32 - stage_idx as i32;

                if microbatch_idx >= 0 && microbatch_idx < num_microbatches as i32 {
                    // This stage has work to do
                    if let Some(Some(input)) = activations[stage_idx].pop_front() {
                        // Execute stage kernel (placeholder - would call actual kernel)
                        let output = input; // Pass through for now

                        // Store output activation
                        activations[stage_idx + 1].push_back(Some(output));
                    }
                }
            }
        }

        // Collect outputs
        let mut outputs = Vec::new();
        let output_device = self.stages.last().map(|s| s.device).unwrap_or(DeviceId(0));

        for opt_data in activations[num_stages].drain(..) {
            if let Some(data) = opt_data {
                let dev = get_device(output_device);
                outputs.push(GpuBuffer::from_slice(&data, &dev)?);
            }
        }

        Ok(outputs)
    }

    /// 1F1B schedule: reduces peak memory
    async fn execute_1f1b<T: Copy + Default + Send + Sync>(
        &self,
        inputs: Vec<GpuBuffer<T>>,
    ) -> Result<Vec<GpuBuffer<T>>, BufferError> {
        // For now, fall back to GPipe
        // Full 1F1B implementation would alternate forward/backward passes
        self.execute_gpipe(inputs).await
    }

    /// Virtual pipeline parallelism
    async fn execute_virtual<T: Copy + Default + Send + Sync>(
        &self,
        inputs: Vec<GpuBuffer<T>>,
        _num_virtual_stages: usize,
    ) -> Result<Vec<GpuBuffer<T>>, BufferError> {
        // Virtual stages allow same GPU to handle multiple stages
        // For now, fall back to GPipe
        self.execute_gpipe(inputs).await
    }

    /// Memory required per stage (peak)
    pub fn memory_per_stage(&self) -> Vec<usize> {
        self.stages
            .iter()
            .map(|s| {
                // Need to store activations for warmup phase
                let activations_per_stage = match self.schedule {
                    PipelineSchedule::GPipe => self.num_microbatches,
                    PipelineSchedule::Interleaved1F1B => self.stages.len(),
                    PipelineSchedule::Virtual { num_virtual_stages } => {
                        self.num_microbatches / num_virtual_stages.max(1)
                    }
                };
                s.output_size * activations_per_stage
            })
            .collect()
    }

    /// Total memory required across all stages
    pub fn total_memory(&self) -> usize {
        self.memory_per_stage().iter().sum()
    }

    /// Generate pipeline schedule visualization
    pub fn visualize_schedule(&self) -> String {
        let num_stages = self.stages.len();
        let num_mb = self.num_microbatches;
        let total_steps = num_mb + num_stages.saturating_sub(1);

        let mut lines = Vec::new();
        lines.push(format!(
            "Pipeline Schedule ({} stages, {} microbatches):",
            num_stages, num_mb
        ));
        lines.push(format!("Schedule: {}", self.schedule.name()));
        lines.push(format!("Efficiency: {:.1}%", self.efficiency() * 100.0));
        lines.push(format!(
            "Bubble overhead: {:.1}%",
            self.bubble_overhead() * 100.0
        ));
        lines.push(String::new());

        // Header
        let mut header = "Time:   ".to_string();
        for step in 0..total_steps {
            header.push_str(&format!("{:4}", step));
        }
        lines.push(header);

        // Each stage row
        for stage in 0..num_stages {
            let mut row = format!("GPU {:2}: ", stage);
            for step in 0..total_steps {
                let mb = step as i32 - stage as i32;
                if mb >= 0 && mb < num_mb as i32 {
                    row.push_str(&format!("[F{}]", mb));
                } else {
                    row.push_str("    ");
                }
            }
            lines.push(row);
        }

        lines.join("\n")
    }
}

/// Automatic pipeline stage assignment
pub struct PipelinePartitioner {
    topology: Arc<GpuTopology>,
}

impl PipelinePartitioner {
    pub fn new(topology: Arc<GpuTopology>) -> Self {
        Self { topology }
    }

    /// Partition computation graph into pipeline stages
    ///
    /// Goals:
    /// 1. Balance compute across stages
    /// 2. Minimize communication between stages
    /// 3. Respect memory constraints
    pub fn partition(
        &self,
        compute_costs: &[f64],
        memory_costs: &[usize],
        devices: &[DeviceId],
    ) -> Vec<Vec<usize>> {
        let num_ops = compute_costs.len();
        let num_stages = devices.len();

        if num_stages == 0 || num_ops == 0 {
            return vec![];
        }

        if num_stages == 1 {
            return vec![(0..num_ops).collect()];
        }

        if num_stages >= num_ops {
            // More stages than ops, assign one op per stage
            return (0..num_ops).map(|i| vec![i]).collect();
        }

        // Dynamic programming for optimal partitioning
        // dp[i][j] = minimum max-stage-cost to assign ops 0..i to stages 0..j
        let mut dp = vec![vec![f64::INFINITY; num_stages + 1]; num_ops + 1];
        let mut split = vec![vec![0usize; num_stages + 1]; num_ops + 1];

        dp[0][0] = 0.0;

        for i in 1..=num_ops {
            for j in 1..=num_stages {
                for k in (j - 1)..i {
                    // Cost of assigning ops k..i to stage j-1
                    let stage_compute: f64 = compute_costs[k..i].iter().sum();
                    let stage_memory: usize = memory_costs[k..i].iter().sum();

                    // Check memory constraint
                    let device = devices[j - 1];
                    let device_memory = self
                        .topology
                        .get_device(device)
                        .map(|d| d.memory_bytes)
                        .unwrap_or(usize::MAX);

                    if stage_memory > device_memory {
                        continue;
                    }

                    // Cost = max of previous stages and current stage
                    let prev_cost = dp[k][j - 1];
                    if prev_cost == f64::INFINITY {
                        continue;
                    }

                    let cost = prev_cost.max(stage_compute);

                    if cost < dp[i][j] {
                        dp[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }

        // Check if partitioning is possible
        if dp[num_ops][num_stages] == f64::INFINITY {
            // Fall back to even split
            return self.even_partition(num_ops, num_stages);
        }

        // Reconstruct partition
        let mut partitions = Vec::new();
        let mut i = num_ops;
        let mut j = num_stages;

        while j > 0 && i > 0 {
            let k = split[i][j];
            partitions.push((k..i).collect());
            i = k;
            j -= 1;
        }

        partitions.reverse();
        partitions
    }

    /// Even partition fallback
    fn even_partition(&self, num_ops: usize, num_stages: usize) -> Vec<Vec<usize>> {
        let chunk = num_ops / num_stages;
        let remainder = num_ops % num_stages;

        let mut result = Vec::new();
        let mut offset = 0;

        for i in 0..num_stages {
            let size = chunk + if i < remainder { 1 } else { 0 };
            result.push((offset..offset + size).collect());
            offset += size;
        }

        result
    }

    /// Estimate total execution time
    pub fn estimate_time(
        &self,
        partitions: &[Vec<usize>],
        compute_costs: &[f64],
        num_microbatches: usize,
    ) -> f64 {
        if partitions.is_empty() {
            return 0.0;
        }

        // Find max stage compute time
        let max_stage_time: f64 = partitions
            .iter()
            .map(|ops| ops.iter().map(|&i| compute_costs[i]).sum::<f64>())
            .fold(0.0, f64::max);

        // Total time = (bubbles + microbatches) * max_stage_time
        let bubbles = partitions.len() - 1;
        (bubbles + num_microbatches) as f64 * max_stage_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_efficiency() {
        let stages = vec![
            PipelineStage::new(0, DeviceId(0), "stage0", 1024, 1024),
            PipelineStage::new(1, DeviceId(1), "stage1", 1024, 1024),
        ];

        let topology = Arc::new(GpuTopology::discover().unwrap());
        let executor = PipelineExecutor::new(stages, topology, 8);

        // 2 stages, 8 microbatches: efficiency = 8/(8+1) = 88.9%
        assert!(executor.efficiency() > 0.85);
        assert!(executor.bubble_overhead() < 0.15);
    }

    #[test]
    fn test_pipeline_visualization() {
        let stages = vec![
            PipelineStage::new(0, DeviceId(0), "stage0", 1024, 1024),
            PipelineStage::new(1, DeviceId(1), "stage1", 1024, 1024),
            PipelineStage::new(2, DeviceId(2), "stage2", 1024, 1024),
        ];

        let topology = Arc::new(GpuTopology::discover().unwrap());
        let executor = PipelineExecutor::new(stages, topology, 4);

        let viz = executor.visualize_schedule();
        assert!(viz.contains("GPU"));
        assert!(viz.contains("[F0]"));
        assert!(viz.contains("Efficiency"));
    }

    #[test]
    fn test_pipeline_partitioner() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let partitioner = PipelinePartitioner::new(topology.clone());

        let compute_costs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let memory_costs = vec![100, 200, 300, 400, 500, 600];
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        if devices.len() >= 2 {
            let partitions = partitioner.partition(&compute_costs, &memory_costs, &devices[..2]);

            assert_eq!(partitions.len(), 2);

            // All ops should be covered
            let all_ops: Vec<usize> = partitions.iter().flatten().copied().collect();
            assert_eq!(all_ops.len(), 6);
        }
    }

    #[test]
    fn test_pipeline_single_stage() {
        let stages = vec![PipelineStage::new(0, DeviceId(0), "stage0", 1024, 1024)];

        let topology = Arc::new(GpuTopology::discover().unwrap());
        let executor = PipelineExecutor::new(stages, topology, 4);

        // Single stage has 100% efficiency
        assert!((executor.efficiency() - 1.0).abs() < 0.001);
        assert!((executor.bubble_overhead() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_per_stage() {
        let stages = vec![
            PipelineStage::new(0, DeviceId(0), "stage0", 1024, 2048),
            PipelineStage::new(1, DeviceId(1), "stage1", 2048, 1024),
        ];

        let topology = Arc::new(GpuTopology::discover().unwrap());
        let executor = PipelineExecutor::new(stages, topology, 4);

        let memory = executor.memory_per_stage();
        assert_eq!(memory.len(), 2);
        assert_eq!(memory[0], 2048 * 4); // output_size * num_microbatches
        assert_eq!(memory[1], 1024 * 4);
    }
}
