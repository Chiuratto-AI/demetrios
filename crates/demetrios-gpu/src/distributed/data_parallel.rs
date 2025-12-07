//! Data Parallelism with Type Safety
//!
//! This module provides type-safe data parallelism for multi-GPU computing:
//!
//! - Unit-preserving distributed buffers
//! - Epistemic-aware aggregation with correlation tracking
//! - Proper confidence propagation across shards

use super::collectives::{CollectiveManager, CollectiveOp, Distribution};
use super::topology::{get_device, GpuTopology};
use crate::integrate::{Confidence, EpistemicBuffer, KnowledgeSource, Unit, UnitBuffer};
use crate::optimize::pool::DeviceId;
use crate::runtime::{BufferError, GpuBuffer};
use std::marker::PhantomData;
use std::sync::Arc;

/// Data-parallel computation with type safety
///
/// Key insight: Data parallelism preserves types across shards
/// - Units are preserved (each shard has same unit)
/// - Confidence must be aggregated correctly across shards
/// - Linear buffers need distributed ownership semantics
pub struct DataParallel<T, U: Unit> {
    /// Participating devices
    devices: Vec<DeviceId>,
    /// Topology information
    topology: Arc<GpuTopology>,
    /// Type markers
    _marker: PhantomData<(T, U)>,
}

impl<T: Copy + Default + Send + Sync, U: Unit> DataParallel<T, U> {
    pub fn new(devices: Vec<DeviceId>, topology: Arc<GpuTopology>) -> Self {
        Self {
            devices,
            topology,
            _marker: PhantomData,
        }
    }

    /// Get devices
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Get topology
    pub fn topology(&self) -> &GpuTopology {
        &self.topology
    }

    /// Shard input data across devices
    pub fn shard(
        &self,
        input: &UnitBuffer<T, U>,
        distribution: Distribution,
    ) -> Result<DistributedUnitBuffer<T, U>, BufferError> {
        let data = input.download_raw()?;
        let n = data.len();

        let shard_sizes = distribution.compute_sizes(n, &self.devices, &self.topology);

        // Create shards
        let mut shards = Vec::new();
        let mut offset = 0;

        for (i, &device) in self.devices.iter().enumerate() {
            let size = shard_sizes[i];
            let shard_data = &data[offset..offset + size];

            let device_obj = get_device(device);
            let buffer = UnitBuffer::from_raw(shard_data, &device_obj)?;

            shards.push(UnitShard {
                device,
                buffer,
                start_index: offset,
                count: size,
            });

            offset += size;
        }

        Ok(DistributedUnitBuffer {
            shards,
            total_elements: n,
            distribution,
            _unit: PhantomData,
        })
    }

    /// Gather shards back to single buffer
    pub fn gather(
        &self,
        distributed: DistributedUnitBuffer<T, U>,
        target_device: DeviceId,
    ) -> Result<UnitBuffer<T, U>, BufferError> {
        let mut all_data = Vec::with_capacity(distributed.total_elements);

        // Collect data from all shards
        for shard in distributed.shards {
            let data = shard.buffer.download_raw()?;
            all_data.extend(data);
        }

        let device = get_device(target_device);
        UnitBuffer::from_raw(&all_data, &device)
    }

    /// Apply map operation in parallel across shards
    ///
    /// Type safety: Result has same unit as input (for unit-preserving ops)
    pub fn map<F>(
        &self,
        input: &DistributedUnitBuffer<T, U>,
        _kernel_name: &str,
        f: F,
    ) -> Result<DistributedUnitBuffer<T, U>, BufferError>
    where
        F: Fn(&UnitBuffer<T, U>) -> Result<UnitBuffer<T, U>, BufferError>,
    {
        let mut output_shards = Vec::new();

        for shard in &input.shards {
            let output = f(&shard.buffer)?;

            output_shards.push(UnitShard {
                device: shard.device,
                buffer: output,
                start_index: shard.start_index,
                count: shard.count,
            });
        }

        Ok(DistributedUnitBuffer {
            shards: output_shards,
            total_elements: input.total_elements,
            distribution: input.distribution.clone(),
            _unit: PhantomData,
        })
    }

    /// All-reduce across shards with proper unit preservation
    pub fn reduce(
        &self,
        input: &DistributedUnitBuffer<T, U>,
        op: CollectiveOp,
        _collective: &CollectiveManager,
    ) -> Result<T, BufferError>
    where
        T: std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<u32>,
    {
        // Each shard computes local reduction
        let local_results: Vec<T> = input
            .shards
            .iter()
            .map(|shard| {
                let data = shard.buffer.download_raw().unwrap();
                data.iter().copied().fold(T::default(), |a, b| a + b)
            })
            .collect();

        // Global reduction
        let global_result = local_results
            .iter()
            .copied()
            .fold(T::default(), |a, b| a + b);

        let final_result = match op {
            CollectiveOp::Sum => global_result,
            CollectiveOp::Average => {
                let count = T::from(input.total_elements as u32);
                global_result / count
            }
            _ => global_result,
        };

        Ok(final_result)
    }
}

/// Unit shard with type information
#[derive(Debug)]
pub struct UnitShard<T: Copy, U: Unit> {
    pub device: DeviceId,
    pub buffer: UnitBuffer<T, U>,
    pub start_index: usize,
    pub count: usize,
}

/// Distributed buffer with units preserved
#[derive(Debug)]
pub struct DistributedUnitBuffer<T: Copy, U: Unit> {
    shards: Vec<UnitShard<T, U>>,
    total_elements: usize,
    distribution: Distribution,
    _unit: PhantomData<U>,
}

impl<T: Copy, U: Unit> DistributedUnitBuffer<T, U> {
    pub fn shards(&self) -> &[UnitShard<T, U>] {
        &self.shards
    }

    pub fn shards_mut(&mut self) -> &mut [UnitShard<T, U>] {
        &mut self.shards
    }

    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    pub fn distribution(&self) -> &Distribution {
        &self.distribution
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

/// Shard correlation type for epistemic aggregation
#[derive(Debug, Clone, Copy)]
pub enum ShardCorrelation {
    /// Shards are independent (different patients, experiments)
    Independent,
    /// Shards are correlated (same patient, different time points)
    Correlated(f32), // Correlation coefficient
    /// Shards have unknown correlation (conservative)
    Unknown,
}

impl ShardCorrelation {
    /// Compute combined confidence from multiple shards
    pub fn combine_confidences(&self, confidences: &[Confidence]) -> Confidence {
        if confidences.is_empty() {
            return Confidence::new(0.0);
        }

        match self {
            Self::Independent => {
                // Independent: confidence increases with more data
                // Using: combined_variance = sum(1/variance) where variance = (1-c)^2
                // Then combined_confidence = 1 - 1/sqrt(sum(1/variance))
                // Simplified: C_combined = 1 - product(1 - Ci)
                let product: f64 = confidences.iter().map(|c| 1.0 - c.value()).product();
                Confidence::new((1.0 - product).min(1.0))
            }
            Self::Correlated(rho) => {
                // Correlated: weighted average with correlation factor
                let avg_conf: f64 =
                    confidences.iter().map(|c| c.value()).sum::<f64>() / confidences.len() as f64;

                // Reduce confidence based on correlation
                Confidence::new(avg_conf * (1.0 - *rho as f64 * 0.5))
            }
            Self::Unknown => {
                // Conservative: take minimum
                let min_conf = confidences
                    .iter()
                    .map(|c| c.value())
                    .fold(f64::INFINITY, f64::min);
                Confidence::new(min_conf)
            }
        }
    }
}

/// Epistemic-aware data parallelism
///
/// Key insight: Confidence aggregation depends on whether shards
/// contain independent or correlated data
pub struct EpistemicDataParallel<T> {
    devices: Vec<DeviceId>,
    /// How confidence combines across shards
    correlation: ShardCorrelation,
    topology: Arc<GpuTopology>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Default + Send + Sync> EpistemicDataParallel<T> {
    pub fn new(
        devices: Vec<DeviceId>,
        correlation: ShardCorrelation,
        topology: Arc<GpuTopology>,
    ) -> Self {
        Self {
            devices,
            correlation,
            topology,
            _marker: PhantomData,
        }
    }

    /// Get correlation mode
    pub fn correlation(&self) -> ShardCorrelation {
        self.correlation
    }

    /// Get devices
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Shard epistemic buffer
    pub fn shard(
        &self,
        input: &EpistemicBuffer<T>,
        distribution: Distribution,
    ) -> Result<DistributedEpistemicBuffer<T>, BufferError> {
        let values = input.download_values()?;
        let confidences = input.download_confidences()?;
        let n = values.len();

        let shard_sizes = distribution.compute_sizes(n, &self.devices, &self.topology);

        let mut shards = Vec::new();
        let mut offset = 0;

        for (i, &device) in self.devices.iter().enumerate() {
            let size = shard_sizes[i];
            let val_data = &values[offset..offset + size];
            let conf_data = &confidences[offset..offset + size];

            let device_obj = get_device(device);
            let val_buffer = GpuBuffer::from_slice(val_data, &device_obj)?;
            let conf_buffer = GpuBuffer::from_slice(conf_data, &device_obj)?;

            shards.push(EpistemicShard {
                device,
                values: val_buffer,
                confidences: conf_buffer,
                start_index: offset,
                count: size,
            });

            offset += size;
        }

        Ok(DistributedEpistemicBuffer {
            shards,
            correlation: self.correlation,
            total_elements: n,
        })
    }

    /// Reduce with confidence propagation
    ///
    /// Note: For Average operation, uses Sum instead since generic division
    /// with count is problematic. Caller should divide result by count if needed.
    pub fn reduce_with_confidence(
        &self,
        shard_results: &[(T, Confidence)],
        _op: CollectiveOp,
    ) -> (T, Confidence)
    where
        T: std::ops::Add<Output = T>,
    {
        if shard_results.is_empty() {
            return (T::default(), Confidence::new(0.0));
        }

        // Compute value reduction (sum)
        let result: T = shard_results
            .iter()
            .map(|(v, _)| *v)
            .fold(T::default(), |a, b| a + b);

        // Compute confidence aggregation based on correlation
        let confidences: Vec<Confidence> = shard_results.iter().map(|(_, c)| *c).collect();

        let combined_confidence = self.correlation.combine_confidences(&confidences);

        (result, combined_confidence)
    }
}

/// Epistemic shard
#[derive(Debug)]
pub struct EpistemicShard<T> {
    pub device: DeviceId,
    pub values: GpuBuffer<T>,
    pub confidences: GpuBuffer<f64>,
    pub start_index: usize,
    pub count: usize,
}

/// Distributed epistemic buffer
#[derive(Debug)]
pub struct DistributedEpistemicBuffer<T> {
    shards: Vec<EpistemicShard<T>>,
    correlation: ShardCorrelation,
    total_elements: usize,
}

impl<T: Copy + Default> DistributedEpistemicBuffer<T> {
    pub fn shards(&self) -> &[EpistemicShard<T>] {
        &self.shards
    }

    pub fn total_elements(&self) -> usize {
        self.total_elements
    }

    pub fn correlation(&self) -> ShardCorrelation {
        self.correlation
    }

    /// Aggregate to single device with proper confidence
    pub fn gather(self, target_device: DeviceId) -> Result<EpistemicBuffer<T>, BufferError> {
        let mut all_values = Vec::new();
        let mut all_confidences = Vec::new();

        // Collect from shards
        for shard in self.shards {
            let values = shard.values.download()?;
            let confs = shard.confidences.download()?;

            all_values.extend(values);
            all_confidences.extend(confs);
        }

        // Adjust confidences based on correlation
        let adjusted_confidences: Vec<f64> = match self.correlation {
            ShardCorrelation::Independent => all_confidences,
            ShardCorrelation::Correlated(rho) => all_confidences
                .iter()
                .map(|&c| c * (1.0 - rho as f64 * 0.2))
                .collect(),
            ShardCorrelation::Unknown => {
                let min = all_confidences
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                vec![min; all_confidences.len()]
            }
        };

        let device = get_device(target_device);

        // Create epistemic buffer using from_values
        let avg_confidence = if adjusted_confidences.is_empty() {
            0.0
        } else {
            adjusted_confidences.iter().sum::<f64>() / adjusted_confidences.len() as f64
        };

        EpistemicBuffer::from_values(
            &all_values,
            Confidence::new(avg_confidence),
            KnowledgeSource::Computation,
            &device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::Kilogram;

    #[test]
    fn test_shard_correlation_independent() {
        let confidences = vec![Confidence::new(0.9), Confidence::new(0.9)];

        let combined = ShardCorrelation::Independent.combine_confidences(&confidences);

        // Independent shards should increase confidence
        assert!(combined.value() > 0.9);
    }

    #[test]
    fn test_shard_correlation_correlated() {
        let confidences = vec![Confidence::new(0.9), Confidence::new(0.9)];

        let combined = ShardCorrelation::Correlated(0.5).combine_confidences(&confidences);

        // Correlated shards should reduce confidence
        assert!(combined.value() < 0.9);
    }

    #[test]
    fn test_shard_correlation_unknown() {
        let confidences = vec![Confidence::new(0.95), Confidence::new(0.85)];

        let combined = ShardCorrelation::Unknown.combine_confidences(&confidences);

        // Unknown should take minimum (conservative)
        assert!((combined.value() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_data_parallel() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        let parallel =
            EpistemicDataParallel::<f32>::new(devices, ShardCorrelation::Independent, topology);

        let shard_results = vec![
            (100.0f32, Confidence::new(0.9)),
            (100.0f32, Confidence::new(0.9)),
        ];

        let (total, conf) = parallel.reduce_with_confidence(&shard_results, CollectiveOp::Sum);

        assert!((total - 200.0).abs() < 0.001);
        assert!(conf.value() > 0.9);
    }

    #[test]
    fn test_data_parallel_creation() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        let parallel: DataParallel<f32, Kilogram> = DataParallel::new(devices.clone(), topology);

        assert_eq!(parallel.devices().len(), devices.len());
    }
}
