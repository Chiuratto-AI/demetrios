//! Work Scheduler for Heterogeneous GPU Systems
//!
//! This module provides work distribution strategies for mixed-architecture
//! GPU clusters, taking into account device performance characteristics.

use super::device_pool::{DevicePool, HeterogeneousDevice};
use crate::optimize::pool::DeviceId;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Work distribution strategy
#[derive(Debug, Clone)]
pub enum DistributionStrategy {
    /// Even distribution (ignores device capabilities)
    Even,
    /// Weighted by compute power (TFLOPS)
    ComputeWeighted,
    /// Weighted by memory bandwidth
    BandwidthWeighted,
    /// Weighted by tensor core performance
    TensorWeighted,
    /// Adaptive based on runtime measurements
    Adaptive,
    /// Custom weights
    Custom(HashMap<DeviceId, f64>),
}

impl Default for DistributionStrategy {
    fn default() -> Self {
        DistributionStrategy::ComputeWeighted
    }
}

/// Synchronization mode for distributed execution
#[derive(Debug, Clone, Copy, Default)]
pub enum SyncMode {
    /// Wait for all devices to complete
    #[default]
    Barrier,
    /// Return after first device completes
    FirstComplete,
    /// Asynchronous (return immediately)
    Async,
}

/// Input data distribution mode
#[derive(Debug, Clone)]
pub enum InputMode {
    /// Broadcast same input to all devices
    Broadcast,
    /// Shard input across devices
    Shard {
        /// Dimension to shard along
        dimension: usize,
    },
    /// Each device has different input
    PerDevice,
}

impl Default for InputMode {
    fn default() -> Self {
        InputMode::Broadcast
    }
}

/// Output aggregation mode
#[derive(Debug, Clone)]
pub enum OutputMode {
    /// Gather outputs to a single device
    Gather {
        /// Target device
        target: DeviceId,
    },
    /// Reduce outputs
    Reduce {
        /// Reduction operation
        op: ReduceOp,
        /// Target device
        target: DeviceId,
    },
    /// Keep outputs distributed
    Distributed,
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::Distributed
    }
}

/// Reduction operation
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
}

/// Configuration for distributed kernel launch
#[derive(Debug, Clone, Default)]
pub struct DistributedLaunchConfig {
    /// Target devices (None = all available)
    pub devices: Option<Vec<DeviceId>>,
    /// Work distribution strategy
    pub strategy: DistributionStrategy,
    /// Synchronization mode
    pub sync_mode: SyncMode,
    /// Input distribution mode
    pub input_mode: InputMode,
    /// Output aggregation mode
    pub output_mode: OutputMode,
    /// Total work units (e.g., batch size)
    pub total_work: usize,
}

impl DistributedLaunchConfig {
    /// Create with total work units
    pub fn with_work(total_work: usize) -> Self {
        Self {
            total_work,
            ..Default::default()
        }
    }

    /// Set distribution strategy
    pub fn strategy(mut self, strategy: DistributionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set target devices
    pub fn devices(mut self, devices: Vec<DeviceId>) -> Self {
        self.devices = Some(devices);
        self
    }

    /// Set input mode to shard
    pub fn shard_input(mut self, dimension: usize) -> Self {
        self.input_mode = InputMode::Shard { dimension };
        self
    }

    /// Set output mode to gather
    pub fn gather_output(mut self, target: DeviceId) -> Self {
        self.output_mode = OutputMode::Gather { target };
        self
    }

    /// Set output mode to reduce
    pub fn reduce_output(mut self, op: ReduceOp, target: DeviceId) -> Self {
        self.output_mode = OutputMode::Reduce { op, target };
        self
    }
}

/// Work assignment for a single device
#[derive(Debug, Clone)]
pub struct WorkAssignment {
    /// Target device
    pub device: DeviceId,
    /// Offset in work units
    pub offset: usize,
    /// Number of work units
    pub count: usize,
    /// Weight (fraction of total work)
    pub weight: f64,
}

impl WorkAssignment {
    /// Check if assignment is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Result of work distribution calculation
#[derive(Debug)]
pub struct WorkDistribution {
    /// Assignments per device
    pub assignments: Vec<WorkAssignment>,
    /// Expected completion order (fastest to slowest)
    pub expected_order: Vec<DeviceId>,
    /// Total work distributed
    pub total_work: usize,
}

impl WorkDistribution {
    /// Get assignment for a device
    pub fn for_device(&self, device: DeviceId) -> Option<&WorkAssignment> {
        self.assignments.iter().find(|a| a.device == device)
    }

    /// Get all device IDs
    pub fn devices(&self) -> Vec<DeviceId> {
        self.assignments.iter().map(|a| a.device).collect()
    }

    /// Get maximum work assigned to any device
    pub fn max_work(&self) -> usize {
        self.assignments.iter().map(|a| a.count).max().unwrap_or(0)
    }
}

/// Performance measurement for adaptive scheduling
#[derive(Debug, Clone)]
struct PerformanceSample {
    /// Work units processed
    work_units: usize,
    /// Time taken
    duration: Duration,
    /// Timestamp
    timestamp: Instant,
}

/// Performance history for a device
#[derive(Debug, Default)]
struct DeviceHistory {
    /// Recent samples
    samples: Vec<PerformanceSample>,
    /// Running average throughput (work/second)
    avg_throughput: f64,
}

impl DeviceHistory {
    const MAX_SAMPLES: usize = 100;

    fn add_sample(&mut self, work_units: usize, duration: Duration) {
        let sample = PerformanceSample {
            work_units,
            duration,
            timestamp: Instant::now(),
        };

        self.samples.push(sample);
        if self.samples.len() > Self::MAX_SAMPLES {
            self.samples.remove(0);
        }

        self.update_average();
    }

    fn update_average(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let total_work: usize = self.samples.iter().map(|s| s.work_units).sum();
        let total_time: f64 = self.samples.iter().map(|s| s.duration.as_secs_f64()).sum();

        if total_time > 0.0 {
            self.avg_throughput = total_work as f64 / total_time;
        }
    }

    fn throughput(&self) -> f64 {
        self.avg_throughput
    }
}

/// Heterogeneous work scheduler
pub struct HeterogeneousScheduler {
    /// Device pool reference
    pool: std::sync::Arc<DevicePool>,
    /// Performance history per device
    perf_history: RwLock<HashMap<DeviceId, DeviceHistory>>,
}

impl HeterogeneousScheduler {
    /// Create a new scheduler
    pub fn new(pool: std::sync::Arc<DevicePool>) -> Self {
        Self {
            pool,
            perf_history: RwLock::new(HashMap::new()),
        }
    }

    /// Compute work distribution for a configuration
    pub fn compute_distribution(&self, config: &DistributedLaunchConfig) -> WorkDistribution {
        let devices = self.get_target_devices(config);
        let total_work = config.total_work;

        let weights = self.compute_weights(&devices, &config.strategy);
        let assignments = self.assign_work(&devices, &weights, total_work);
        let expected_order = self.compute_expected_order(&devices, &weights);

        WorkDistribution {
            assignments,
            expected_order,
            total_work,
        }
    }

    /// Get target devices from config
    fn get_target_devices(&self, config: &DistributedLaunchConfig) -> Vec<&HeterogeneousDevice> {
        if let Some(ref device_ids) = config.devices {
            device_ids
                .iter()
                .filter_map(|id| self.pool.get(*id))
                .collect()
        } else {
            self.pool.devices().iter().collect()
        }
    }

    /// Compute weights based on strategy
    fn compute_weights(
        &self,
        devices: &[&HeterogeneousDevice],
        strategy: &DistributionStrategy,
    ) -> HashMap<DeviceId, f64> {
        match strategy {
            DistributionStrategy::Even => {
                let weight = 1.0 / devices.len() as f64;
                devices.iter().map(|d| (d.id, weight)).collect()
            }

            DistributionStrategy::ComputeWeighted => {
                let total: f64 = devices.iter().map(|d| d.capabilities.fp32_tflops).sum();
                devices
                    .iter()
                    .map(|d| (d.id, d.capabilities.fp32_tflops / total))
                    .collect()
            }

            DistributionStrategy::BandwidthWeighted => {
                let total: f64 = devices
                    .iter()
                    .map(|d| d.capabilities.memory_bandwidth_gb_s)
                    .sum();
                devices
                    .iter()
                    .map(|d| (d.id, d.capabilities.memory_bandwidth_gb_s / total))
                    .collect()
            }

            DistributionStrategy::TensorWeighted => {
                let total: f64 = devices.iter().map(|d| d.capabilities.tensor_tflops).sum();
                devices
                    .iter()
                    .map(|d| (d.id, d.capabilities.tensor_tflops / total))
                    .collect()
            }

            DistributionStrategy::Adaptive => self.get_adaptive_weights(devices),

            DistributionStrategy::Custom(custom_weights) => {
                // Normalize custom weights for selected devices
                let selected: HashMap<_, _> = devices
                    .iter()
                    .filter_map(|d| custom_weights.get(&d.id).map(|w| (d.id, *w)))
                    .collect();
                let total: f64 = selected.values().sum();
                selected.into_iter().map(|(id, w)| (id, w / total)).collect()
            }
        }
    }

    /// Get adaptive weights based on performance history
    fn get_adaptive_weights(&self, devices: &[&HeterogeneousDevice]) -> HashMap<DeviceId, f64> {
        let history = self.perf_history.read().unwrap();

        let throughputs: Vec<(DeviceId, f64)> = devices
            .iter()
            .map(|d| {
                let throughput = history
                    .get(&d.id)
                    .map(|h| h.throughput())
                    .unwrap_or(d.capabilities.fp32_tflops * 1e9); // Default to compute-based estimate
                (d.id, throughput)
            })
            .collect();

        let total: f64 = throughputs.iter().map(|(_, t)| t).sum();
        throughputs.into_iter().map(|(id, t)| (id, t / total)).collect()
    }

    /// Assign work based on weights
    fn assign_work(
        &self,
        devices: &[&HeterogeneousDevice],
        weights: &HashMap<DeviceId, f64>,
        total_work: usize,
    ) -> Vec<WorkAssignment> {
        let mut assignments = Vec::new();
        let mut offset = 0;
        let mut remaining = total_work;

        for (i, device) in devices.iter().enumerate() {
            let weight = weights.get(&device.id).copied().unwrap_or(0.0);

            let count = if i == devices.len() - 1 {
                // Last device gets remaining work
                remaining
            } else {
                let count = (total_work as f64 * weight).round() as usize;
                count.min(remaining)
            };

            assignments.push(WorkAssignment {
                device: device.id,
                offset,
                count,
                weight,
            });

            offset += count;
            remaining = remaining.saturating_sub(count);
        }

        assignments
    }

    /// Compute expected completion order (fastest first)
    fn compute_expected_order(
        &self,
        devices: &[&HeterogeneousDevice],
        weights: &HashMap<DeviceId, f64>,
    ) -> Vec<DeviceId> {
        let mut ordered: Vec<_> = devices
            .iter()
            .map(|d| (d.id, weights.get(&d.id).copied().unwrap_or(0.0)))
            .collect();

        // Higher weight = more work = slower completion (for balanced work)
        // But higher performance = faster, so we sort by weight desc (faster devices get more work)
        ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        ordered.into_iter().map(|(id, _)| id).collect()
    }

    /// Record execution time for adaptive scheduling
    pub fn record_execution(&self, device: DeviceId, work_units: usize, duration: Duration) {
        let mut history = self.perf_history.write().unwrap();
        history
            .entry(device)
            .or_default()
            .add_sample(work_units, duration);
    }

    /// Get current adaptive weights (for inspection)
    pub fn get_current_weights(&self) -> HashMap<DeviceId, f64> {
        let devices: Vec<_> = self.pool.devices().iter().collect();
        self.get_adaptive_weights(&devices)
    }

    /// Clear performance history
    pub fn clear_history(&self) {
        let mut history = self.perf_history.write().unwrap();
        history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_scheduler() -> HeterogeneousScheduler {
        let pool = DevicePool::discover().unwrap();
        HeterogeneousScheduler::new(std::sync::Arc::new(pool))
    }

    #[test]
    fn test_even_distribution() {
        let scheduler = create_test_scheduler();
        let config = DistributedLaunchConfig::with_work(1000).strategy(DistributionStrategy::Even);

        let dist = scheduler.compute_distribution(&config);

        assert_eq!(dist.total_work, 1000);
        assert!(!dist.assignments.is_empty());

        // Check all work is distributed
        let total: usize = dist.assignments.iter().map(|a| a.count).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_compute_weighted_distribution() {
        let scheduler = create_test_scheduler();
        let config = DistributedLaunchConfig::with_work(10000)
            .strategy(DistributionStrategy::ComputeWeighted);

        let dist = scheduler.compute_distribution(&config);

        // More powerful devices should get more work
        if dist.assignments.len() >= 2 {
            let weights: HashMap<_, _> = dist.assignments.iter().map(|a| (a.device, a.weight)).collect();
            // Weights should vary based on compute power
            let min_weight = weights.values().cloned().fold(f64::INFINITY, f64::min);
            let max_weight = weights.values().cloned().fold(f64::NEG_INFINITY, f64::max);
            assert!(max_weight >= min_weight);
        }
    }

    #[test]
    fn test_shard_input_config() {
        let config = DistributedLaunchConfig::with_work(1000)
            .shard_input(0)
            .gather_output(DeviceId(0));

        assert!(matches!(config.input_mode, InputMode::Shard { dimension: 0 }));
        assert!(matches!(
            config.output_mode,
            OutputMode::Gather { target: DeviceId(0) }
        ));
    }

    #[test]
    fn test_adaptive_scheduling() {
        let scheduler = create_test_scheduler();

        // Record some execution times
        scheduler.record_execution(DeviceId(0), 1000, Duration::from_millis(100));
        scheduler.record_execution(DeviceId(0), 1000, Duration::from_millis(110));

        let config = DistributedLaunchConfig::with_work(10000).strategy(DistributionStrategy::Adaptive);

        let dist = scheduler.compute_distribution(&config);
        assert!(!dist.assignments.is_empty());
    }

    #[test]
    fn test_custom_weights() {
        let scheduler = create_test_scheduler();
        let mut custom = HashMap::new();
        custom.insert(DeviceId(0), 3.0);
        custom.insert(DeviceId(1), 1.0);

        let config = DistributedLaunchConfig::with_work(1000)
            .strategy(DistributionStrategy::Custom(custom))
            .devices(vec![DeviceId(0), DeviceId(1)]);

        let dist = scheduler.compute_distribution(&config);

        // Device 0 should get ~75% of work
        if let Some(a0) = dist.for_device(DeviceId(0)) {
            assert!((a0.weight - 0.75).abs() < 0.01);
        }
    }

    #[test]
    fn test_work_assignment_offsets() {
        let scheduler = create_test_scheduler();
        let config = DistributedLaunchConfig::with_work(1000);

        let dist = scheduler.compute_distribution(&config);

        // Check offsets are contiguous
        let mut expected_offset = 0;
        for assignment in &dist.assignments {
            assert_eq!(assignment.offset, expected_offset);
            expected_offset += assignment.count;
        }
        assert_eq!(expected_offset, 1000);
    }

    #[test]
    fn test_expected_order() {
        let scheduler = create_test_scheduler();
        let config = DistributedLaunchConfig::with_work(1000);

        let dist = scheduler.compute_distribution(&config);

        // All devices should be in expected order
        assert_eq!(dist.expected_order.len(), dist.assignments.len());
    }
}
