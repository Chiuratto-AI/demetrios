//! Fault Tolerance for Distributed GPU Computing
//!
//! This module provides fault tolerance mechanisms:
//!
//! - Phi-accrual failure detection
//! - Checkpoint management
//! - Elastic execution with device replacement
//! - Replicated buffers for critical data

use super::topology::{get_device, GpuTopology};
use crate::optimize::pool::DeviceId;
use crate::runtime::{BufferError, GpuBuffer};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Device health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceHealth {
    /// Device is healthy
    Healthy,
    /// Device is degraded (slow, errors)
    Degraded { error_rate: u32 },
    /// Device is suspected failed
    Suspected,
    /// Device is confirmed failed
    Failed,
    /// Device is recovering
    Recovering,
}

impl DeviceHealth {
    /// Check if device is usable
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            Self::Healthy | Self::Degraded { .. } | Self::Recovering
        )
    }

    /// Check if device needs attention
    pub fn needs_attention(&self) -> bool {
        !matches!(self, Self::Healthy)
    }
}

/// Failure detector using phi-accrual algorithm
///
/// The phi-accrual failure detector provides a continuous suspicion level
/// rather than a binary alive/dead decision. This allows for more nuanced
/// failure detection and adaptation to varying network conditions.
pub struct FailureDetector {
    /// Last heartbeat from each device
    last_heartbeat: HashMap<DeviceId, Instant>,
    /// Heartbeat interval samples for phi calculation
    intervals: HashMap<DeviceId, Vec<Duration>>,
    /// Phi threshold for suspicion (typically 8-12)
    phi_threshold: f64,
    /// Window size for interval tracking
    window_size: usize,
    /// Default expected interval
    default_interval: Duration,
}

impl FailureDetector {
    pub fn new(phi_threshold: f64) -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            intervals: HashMap::new(),
            phi_threshold,
            window_size: 100,
            default_interval: Duration::from_millis(100),
        }
    }

    /// Create with custom window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set default expected interval
    pub fn with_default_interval(mut self, interval: Duration) -> Self {
        self.default_interval = interval;
        self
    }

    /// Get phi threshold
    pub fn phi_threshold(&self) -> f64 {
        self.phi_threshold
    }

    /// Record heartbeat from device
    pub fn heartbeat(&mut self, device: DeviceId) {
        let now = Instant::now();

        if let Some(&last) = self.last_heartbeat.get(&device) {
            let interval = now - last;

            let intervals = self.intervals.entry(device).or_default();
            intervals.push(interval);

            // Keep window bounded
            if intervals.len() > self.window_size {
                intervals.remove(0);
            }
        }

        self.last_heartbeat.insert(device, now);
    }

    /// Check if device is suspected failed
    pub fn is_suspected(&self, device: DeviceId) -> bool {
        self.phi(device) > self.phi_threshold
    }

    /// Calculate phi (suspicion level) for device
    ///
    /// Higher phi means higher suspicion of failure
    /// phi = -log(P(X > time_since_last))
    pub fn phi(&self, device: DeviceId) -> f64 {
        let now = Instant::now();

        let last = match self.last_heartbeat.get(&device) {
            Some(&t) => t,
            None => return f64::INFINITY, // Never seen
        };

        let since_last = (now - last).as_secs_f64();

        let intervals = match self.intervals.get(&device) {
            Some(i) if !i.is_empty() => i,
            _ => {
                // No history, use default interval
                return since_last / self.default_interval.as_secs_f64();
            }
        };

        // Calculate mean and variance of intervals
        let mean: f64 =
            intervals.iter().map(|d| d.as_secs_f64()).sum::<f64>() / intervals.len() as f64;

        let variance: f64 = intervals
            .iter()
            .map(|d| (d.as_secs_f64() - mean).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        let std_dev = variance.sqrt().max(0.001);

        // Phi using exponential distribution approximation
        // P(X > t) = exp(-t/mean)
        // phi = -log(P) = t/mean
        let prob = (-since_last / mean.max(0.001)).exp();
        if prob <= 0.0 {
            f64::INFINITY
        } else {
            -prob.ln()
        }
    }

    /// Get health status of device
    pub fn health(&self, device: DeviceId) -> DeviceHealth {
        let phi = self.phi(device);

        if phi < self.phi_threshold * 0.5 {
            DeviceHealth::Healthy
        } else if phi < self.phi_threshold {
            DeviceHealth::Degraded {
                error_rate: (phi / self.phi_threshold * 100.0) as u32,
            }
        } else if phi < self.phi_threshold * 2.0 {
            DeviceHealth::Suspected
        } else {
            DeviceHealth::Failed
        }
    }

    /// Get all monitored devices
    pub fn monitored_devices(&self) -> Vec<DeviceId> {
        self.last_heartbeat.keys().copied().collect()
    }

    /// Clear history for a device
    pub fn clear(&mut self, device: DeviceId) {
        self.last_heartbeat.remove(&device);
        self.intervals.remove(&device);
    }
}

/// Checkpoint identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub u64);

/// Checkpoint data
pub struct Checkpoint<T> {
    pub id: CheckpointId,
    pub device: DeviceId,
    pub data: Vec<T>,
    pub timestamp: Instant,
    pub iteration: usize,
}

impl<T> Checkpoint<T> {
    /// Age of checkpoint
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

/// Checkpoint manager for fault recovery
pub struct CheckpointManager<T> {
    /// Checkpoint storage
    checkpoints: HashMap<CheckpointId, Checkpoint<T>>,
    /// Device -> latest checkpoint
    device_checkpoints: HashMap<DeviceId, CheckpointId>,
    /// Checkpoint interval
    interval: Duration,
    /// Last checkpoint time
    last_checkpoint: Instant,
    /// Next checkpoint ID
    next_id: u64,
}

impl<T: Clone> CheckpointManager<T> {
    pub fn new(interval: Duration) -> Self {
        Self {
            checkpoints: HashMap::new(),
            device_checkpoints: HashMap::new(),
            interval,
            last_checkpoint: Instant::now(),
            next_id: 0,
        }
    }

    /// Get checkpoint interval
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Set checkpoint interval
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Should we checkpoint now?
    pub fn should_checkpoint(&self) -> bool {
        self.last_checkpoint.elapsed() >= self.interval
    }

    /// Time until next checkpoint
    pub fn time_until_checkpoint(&self) -> Duration {
        let elapsed = self.last_checkpoint.elapsed();
        if elapsed >= self.interval {
            Duration::ZERO
        } else {
            self.interval - elapsed
        }
    }

    /// Save checkpoint
    pub fn checkpoint(
        &mut self,
        device: DeviceId,
        buffer: &GpuBuffer<T>,
        iteration: usize,
    ) -> Result<CheckpointId, BufferError>
    where
        T: Copy + Default,
    {
        let data = buffer.download()?;

        let id = CheckpointId(self.next_id);
        self.next_id += 1;

        self.checkpoints.insert(
            id,
            Checkpoint {
                id,
                device,
                data,
                timestamp: Instant::now(),
                iteration,
            },
        );

        self.device_checkpoints.insert(device, id);
        self.last_checkpoint = Instant::now();

        Ok(id)
    }

    /// Restore from checkpoint
    pub fn restore(&self, device: DeviceId) -> Option<&Checkpoint<T>> {
        let id = self.device_checkpoints.get(&device)?;
        self.checkpoints.get(id)
    }

    /// Get checkpoint by ID
    pub fn get(&self, id: CheckpointId) -> Option<&Checkpoint<T>> {
        self.checkpoints.get(&id)
    }

    /// Number of checkpoints stored
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Clean old checkpoints, keeping only the most recent per device
    pub fn gc(&mut self, keep_last: usize) {
        if keep_last == 0 {
            self.checkpoints.clear();
            return;
        }

        let mut to_remove = Vec::new();

        // Group checkpoints by device
        let mut by_device: HashMap<DeviceId, Vec<(CheckpointId, Instant)>> = HashMap::new();
        for (id, cp) in &self.checkpoints {
            by_device
                .entry(cp.device)
                .or_default()
                .push((*id, cp.timestamp));
        }

        // For each device, keep only the most recent
        for (device, mut cps) in by_device {
            if cps.len() > keep_last {
                // Sort by timestamp, newest first
                cps.sort_by(|a, b| b.1.cmp(&a.1));

                // Mark old ones for removal
                let current = self.device_checkpoints.get(&device);
                for (id, _) in cps.into_iter().skip(keep_last) {
                    // Don't remove the current checkpoint
                    if current != Some(&id) {
                        to_remove.push(id);
                    }
                }
            }
        }

        for id in to_remove {
            self.checkpoints.remove(&id);
        }
    }
}

/// Recovery action
#[derive(Debug)]
pub enum RecoveryAction {
    /// Restore from checkpoint
    RestoreFromCheckpoint {
        failed: DeviceId,
        replacement: DeviceId,
        checkpoint_id: CheckpointId,
        iteration: usize,
    },
    /// No checkpoint, restart from beginning
    RestartFromBeginning {
        failed: DeviceId,
        replacement: DeviceId,
    },
    /// Cannot recover (no healthy devices)
    CannotRecover,
}

impl RecoveryAction {
    /// Check if recovery is possible
    pub fn can_recover(&self) -> bool {
        !matches!(self, Self::CannotRecover)
    }
}

/// Elastic executor that handles device failures
pub struct ElasticExecutor<T> {
    /// Available devices
    devices: Arc<RwLock<Vec<DeviceId>>>,
    /// Failed devices
    failed: Arc<RwLock<HashSet<DeviceId>>>,
    /// Failure detector
    detector: Arc<RwLock<FailureDetector>>,
    /// Checkpoint manager
    checkpoints: Arc<RwLock<CheckpointManager<T>>>,
    /// Topology
    topology: Arc<GpuTopology>,
}

impl<T: Clone + Copy + Default + Send + Sync> ElasticExecutor<T> {
    pub fn new(
        devices: Vec<DeviceId>,
        topology: Arc<GpuTopology>,
        checkpoint_interval: Duration,
    ) -> Self {
        Self {
            devices: Arc::new(RwLock::new(devices)),
            failed: Arc::new(RwLock::new(HashSet::new())),
            detector: Arc::new(RwLock::new(FailureDetector::new(10.0))),
            checkpoints: Arc::new(RwLock::new(CheckpointManager::new(checkpoint_interval))),
            topology,
        }
    }

    /// Check for failures and return newly failed devices
    pub fn check_failures(&self) -> Vec<DeviceId> {
        let detector = self.detector.read().unwrap();
        let devices = self.devices.read().unwrap();
        let mut failed = self.failed.write().unwrap();

        let mut new_failures = Vec::new();

        for &device in devices.iter() {
            if detector.is_suspected(device) && !failed.contains(&device) {
                new_failures.push(device);
                failed.insert(device);
            }
        }

        new_failures
    }

    /// Handle device failure
    pub fn handle_failure(&self, failed_device: DeviceId) -> Result<RecoveryAction, BufferError> {
        let devices = self.devices.read().unwrap();
        let failed = self.failed.read().unwrap();

        // Find replacement device
        let healthy_devices: Vec<_> = devices
            .iter()
            .filter(|&&d| d != failed_device && !failed.contains(&d))
            .copied()
            .collect();

        if healthy_devices.is_empty() {
            return Ok(RecoveryAction::CannotRecover);
        }

        // Pick replacement with best connectivity to failed device
        let replacement = healthy_devices
            .iter()
            .max_by(|&&a, &&b| {
                let bw_a = self
                    .topology
                    .get_interconnect(a, failed_device)
                    .map(|ic| ic.bandwidth_bytes_per_sec())
                    .unwrap_or(0.0);
                let bw_b = self
                    .topology
                    .get_interconnect(b, failed_device)
                    .map(|ic| ic.bandwidth_bytes_per_sec())
                    .unwrap_or(0.0);
                bw_a.partial_cmp(&bw_b).unwrap()
            })
            .copied()
            .unwrap();

        // Get checkpoint for failed device
        let checkpoints = self.checkpoints.read().unwrap();
        let checkpoint = checkpoints.restore(failed_device);

        match checkpoint {
            Some(cp) => Ok(RecoveryAction::RestoreFromCheckpoint {
                failed: failed_device,
                replacement,
                checkpoint_id: cp.id,
                iteration: cp.iteration,
            }),
            None => Ok(RecoveryAction::RestartFromBeginning {
                failed: failed_device,
                replacement,
            }),
        }
    }

    /// Report heartbeat from device
    pub fn heartbeat(&self, device: DeviceId) {
        let mut detector = self.detector.write().unwrap();
        detector.heartbeat(device);
    }

    /// Get all devices (including failed)
    pub fn all_devices(&self) -> Vec<DeviceId> {
        self.devices.read().unwrap().clone()
    }

    /// Get active (non-failed) devices
    pub fn active_devices(&self) -> Vec<DeviceId> {
        let devices = self.devices.read().unwrap();
        let failed = self.failed.read().unwrap();

        devices
            .iter()
            .filter(|d| !failed.contains(d))
            .copied()
            .collect()
    }

    /// Get failed devices
    pub fn failed_devices(&self) -> HashSet<DeviceId> {
        self.failed.read().unwrap().clone()
    }

    /// Mark device as recovered
    pub fn mark_recovered(&self, device: DeviceId) {
        let mut failed = self.failed.write().unwrap();
        failed.remove(&device);

        // Clear failure detector history
        let mut detector = self.detector.write().unwrap();
        detector.clear(device);
    }

    /// Get device health
    pub fn device_health(&self, device: DeviceId) -> DeviceHealth {
        let failed = self.failed.read().unwrap();
        if failed.contains(&device) {
            return DeviceHealth::Failed;
        }

        let detector = self.detector.read().unwrap();
        detector.health(device)
    }

    /// Create checkpoint
    pub fn checkpoint(
        &self,
        device: DeviceId,
        buffer: &GpuBuffer<T>,
        iteration: usize,
    ) -> Result<CheckpointId, BufferError> {
        let mut checkpoints = self.checkpoints.write().unwrap();
        checkpoints.checkpoint(device, buffer, iteration)
    }

    /// Should checkpoint now?
    pub fn should_checkpoint(&self) -> bool {
        let checkpoints = self.checkpoints.read().unwrap();
        checkpoints.should_checkpoint()
    }
}

/// Replication for critical data
pub struct ReplicatedBuffer<T> {
    /// Primary copy
    primary: (DeviceId, GpuBuffer<T>),
    /// Replicas
    replicas: Vec<(DeviceId, GpuBuffer<T>)>,
}

impl<T: Copy + Default + Send + Sync> ReplicatedBuffer<T> {
    /// Create a new replicated buffer
    pub fn new(
        primary: (DeviceId, GpuBuffer<T>),
        replica_devices: Vec<DeviceId>,
    ) -> Result<Self, BufferError> {
        let data = primary.1.download()?;

        let mut replicas = Vec::new();
        for device in replica_devices {
            if device != primary.0 {
                let device_obj = get_device(device);
                let buffer = GpuBuffer::from_slice(&data, &device_obj)?;
                replicas.push((device, buffer));
            }
        }

        Ok(Self { primary, replicas })
    }

    /// Get primary device
    pub fn primary_device(&self) -> DeviceId {
        self.primary.0
    }

    /// Get read access (from primary)
    pub fn read(&self) -> &GpuBuffer<T> {
        &self.primary.1
    }

    /// Replication factor
    pub fn replication_factor(&self) -> usize {
        1 + self.replicas.len()
    }

    /// Sync changes to replicas
    pub fn sync(&mut self) -> Result<(), BufferError> {
        let data = self.primary.1.download()?;

        for (device, buffer) in &mut self.replicas {
            let dev = get_device(*device);
            *buffer = GpuBuffer::from_slice(&data, &dev)?;
        }

        Ok(())
    }

    /// Handle primary failure by promoting a replica
    pub fn promote_replica(&mut self) -> Option<DeviceId> {
        if let Some((device, buffer)) = self.replicas.pop() {
            self.primary = (device, buffer);
            Some(device)
        } else {
            None
        }
    }

    /// Check if any replica is on given device
    pub fn has_replica_on(&self, device: DeviceId) -> bool {
        self.replicas.iter().any(|(d, _)| *d == device)
    }

    /// Get all devices holding data
    pub fn all_devices(&self) -> Vec<DeviceId> {
        let mut devices = vec![self.primary.0];
        devices.extend(self.replicas.iter().map(|(d, _)| *d));
        devices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_failure_detector_healthy() {
        let mut detector = FailureDetector::new(10.0);

        // Regular heartbeats
        for _ in 0..10 {
            detector.heartbeat(DeviceId(0));
            sleep(Duration::from_millis(10));
        }

        // Device should be healthy
        assert!(!detector.is_suspected(DeviceId(0)));
        assert_eq!(detector.health(DeviceId(0)), DeviceHealth::Healthy);
    }

    #[test]
    fn test_failure_detector_never_seen() {
        let detector = FailureDetector::new(10.0);

        // Device never sent heartbeat
        assert!(detector.is_suspected(DeviceId(99)));
        assert_eq!(detector.phi(DeviceId(99)), f64::INFINITY);
    }

    #[test]
    fn test_device_health_states() {
        assert!(DeviceHealth::Healthy.is_usable());
        assert!(DeviceHealth::Degraded { error_rate: 10 }.is_usable());
        assert!(!DeviceHealth::Failed.is_usable());

        assert!(!DeviceHealth::Healthy.needs_attention());
        assert!(DeviceHealth::Degraded { error_rate: 10 }.needs_attention());
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager: CheckpointManager<f32> = CheckpointManager::new(Duration::from_secs(60));

        // Initially should checkpoint
        sleep(Duration::from_millis(1));

        // Create a buffer
        let device = get_device(DeviceId(0));
        let buffer = GpuBuffer::from_slice(&[1.0f32, 2.0, 3.0], &device).unwrap();

        // Checkpoint
        let id = manager.checkpoint(DeviceId(0), &buffer, 0).unwrap();

        // Restore
        let cp = manager.restore(DeviceId(0)).unwrap();
        assert_eq!(cp.id, id);
        assert_eq!(cp.data, vec![1.0f32, 2.0, 3.0]);
        assert_eq!(cp.iteration, 0);
    }

    #[test]
    fn test_checkpoint_gc() {
        let mut manager: CheckpointManager<f32> = CheckpointManager::new(Duration::from_millis(1));

        let device = get_device(DeviceId(0));
        let buffer = GpuBuffer::from_slice(&[1.0f32], &device).unwrap();

        // Create multiple checkpoints
        for i in 0..5 {
            manager.checkpoint(DeviceId(0), &buffer, i).unwrap();
        }

        assert_eq!(manager.num_checkpoints(), 5);

        // GC keeping only 2
        manager.gc(2);

        assert!(manager.num_checkpoints() <= 3);
    }

    #[test]
    fn test_elastic_executor() {
        let topology = Arc::new(GpuTopology::discover().unwrap());
        let devices: Vec<_> = topology.devices().map(|d| d.id).collect();

        let executor: ElasticExecutor<f32> =
            ElasticExecutor::new(devices.clone(), topology, Duration::from_secs(60));

        // All devices should be active initially
        assert_eq!(executor.active_devices().len(), devices.len());
        assert!(executor.failed_devices().is_empty());
    }

    #[test]
    fn test_recovery_action() {
        assert!(RecoveryAction::RestoreFromCheckpoint {
            failed: DeviceId(0),
            replacement: DeviceId(1),
            checkpoint_id: CheckpointId(0),
            iteration: 0,
        }
        .can_recover());

        assert!(!RecoveryAction::CannotRecover.can_recover());
    }

    #[test]
    fn test_replicated_buffer() {
        let device = get_device(DeviceId(0));
        let buffer = GpuBuffer::from_slice(&[1.0f32, 2.0, 3.0], &device).unwrap();

        let replicated = ReplicatedBuffer::new((DeviceId(0), buffer), vec![DeviceId(1)]).unwrap();

        assert_eq!(replicated.primary_device(), DeviceId(0));
        assert_eq!(replicated.replication_factor(), 2);

        let data = replicated.read().download().unwrap();
        assert_eq!(data, vec![1.0f32, 2.0, 3.0]);
    }
}
