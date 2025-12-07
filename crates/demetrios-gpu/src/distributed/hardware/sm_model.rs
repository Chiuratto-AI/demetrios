//! Streaming Multiprocessor (SM) Model
//!
//! This module implements a complete SM simulation that integrates:
//! - Register file with banking
//! - Warp scheduling
//! - L1 cache
//! - Tensor Cores
//! - Occupancy calculation
//! - Performance prediction

use std::collections::{HashMap, HashSet, VecDeque};

use super::cache_hierarchy::{CacheStats, L1Cache, L1CacheSpec};
use super::register_file::{OperandCollector, RegisterFileSpec};
use super::tensor_cores::{MmaConfig, TensorCoreSpec, TensorCoreUnit};

// ============================================================================
// SM Specification
// ============================================================================

/// SM architecture generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmArchitecture {
    /// Ampere (A100)
    Ampere,
    /// Ada Lovelace (L4)
    Ada,
    /// Hopper (H100)
    Hopper,
}

/// SM specification
#[derive(Debug, Clone)]
pub struct SmSpec {
    /// Architecture
    pub architecture: SmArchitecture,
    /// Number of CUDA cores per SM
    pub cuda_cores: u32,
    /// Number of Tensor Cores per SM
    pub tensor_cores: u32,
    /// Register file size (KB)
    pub register_file_kb: u32,
    /// Shared memory size (KB)
    pub shared_memory_kb: u32,
    /// L1 cache size (KB)
    pub l1_cache_kb: u32,
    /// Maximum warps per SM
    pub max_warps: u32,
    /// Maximum blocks per SM
    pub max_blocks: u32,
    /// Maximum threads per SM
    pub max_threads: u32,
    /// Warp schedulers
    pub warp_schedulers: u32,
    /// Instructions issued per scheduler per cycle
    pub issue_width: u32,
    /// Clock frequency (GHz)
    pub clock_ghz: f64,
}

impl SmSpec {
    /// A100 SM spec
    pub fn a100() -> Self {
        Self {
            architecture: SmArchitecture::Ampere,
            cuda_cores: 64,
            tensor_cores: 4,
            register_file_kb: 256,
            shared_memory_kb: 164,
            l1_cache_kb: 192,
            max_warps: 64,
            max_blocks: 32,
            max_threads: 2048,
            warp_schedulers: 4,
            issue_width: 1,
            clock_ghz: 1.41,
        }
    }

    /// H100 SM spec
    pub fn h100() -> Self {
        Self {
            architecture: SmArchitecture::Hopper,
            cuda_cores: 128,
            tensor_cores: 4,
            register_file_kb: 256,
            shared_memory_kb: 228,
            l1_cache_kb: 256,
            max_warps: 64,
            max_blocks: 32,
            max_threads: 2048,
            warp_schedulers: 4,
            issue_width: 1,
            clock_ghz: 1.83,
        }
    }

    /// L4 SM spec
    pub fn l4() -> Self {
        Self {
            architecture: SmArchitecture::Ada,
            cuda_cores: 128,
            tensor_cores: 4,
            register_file_kb: 256,
            shared_memory_kb: 128,
            l1_cache_kb: 128,
            max_warps: 48,
            max_blocks: 24,
            max_threads: 1536,
            warp_schedulers: 4,
            issue_width: 1,
            clock_ghz: 2.04,
        }
    }

    /// Peak FP32 TFLOPS for this SM
    pub fn peak_fp32_tflops(&self) -> f64 {
        // FMA = 2 FLOPS, per core, per cycle
        (self.cuda_cores as f64 * 2.0 * self.clock_ghz) / 1000.0
    }

    /// Registers per thread at max occupancy
    pub fn registers_at_max_occupancy(&self) -> u32 {
        (self.register_file_kb * 1024) / (self.max_threads * 4)
    }
}

// ============================================================================
// Warp State
// ============================================================================

/// Warp execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpState {
    /// Ready to execute
    Ready,
    /// Waiting for memory
    WaitingMemory,
    /// Waiting for barrier
    WaitingBarrier,
    /// Waiting for operands (register bank conflict)
    WaitingOperands,
    /// Executing instruction
    Executing,
    /// Completed
    Completed,
}

/// Warp in the SM
#[derive(Debug, Clone)]
pub struct Warp {
    /// Warp ID
    pub warp_id: u32,
    /// Block ID
    pub block_id: u32,
    /// Current state
    pub state: WarpState,
    /// Program counter
    pub pc: u32,
    /// Active thread mask
    pub active_mask: u32,
    /// Instructions executed
    pub instructions_executed: u64,
    /// Cycles stalled
    pub cycles_stalled: u64,
    /// Register base
    pub register_base: u32,
    /// Registers per thread
    pub registers_per_thread: u32,
}

impl Warp {
    pub fn new(warp_id: u32, block_id: u32, register_base: u32, registers_per_thread: u32) -> Self {
        Self {
            warp_id,
            block_id,
            state: WarpState::Ready,
            pc: 0,
            active_mask: 0xFFFFFFFF, // All threads active
            instructions_executed: 0,
            cycles_stalled: 0,
            register_base,
            registers_per_thread,
        }
    }

    /// Number of active threads
    pub fn active_threads(&self) -> u32 {
        self.active_mask.count_ones()
    }
}

// ============================================================================
// Block State
// ============================================================================

/// Thread block in the SM
#[derive(Debug, Clone)]
pub struct ThreadBlock {
    /// Block ID
    pub block_id: u32,
    /// Warps in this block
    pub warps: Vec<u32>,
    /// Threads per block
    pub threads: u32,
    /// Shared memory used (bytes)
    pub shared_memory_bytes: u32,
    /// Registers per thread
    pub registers_per_thread: u32,
    /// Barrier state (barrier_id -> waiting warps)
    pub barriers: HashMap<u32, HashSet<u32>>,
}

impl ThreadBlock {
    pub fn new(
        block_id: u32,
        threads: u32,
        shared_memory_bytes: u32,
        registers_per_thread: u32,
    ) -> Self {
        let num_warps = (threads + 31) / 32;
        let warps = (0..num_warps).map(|i| block_id * 64 + i).collect();

        Self {
            block_id,
            warps,
            threads,
            shared_memory_bytes,
            registers_per_thread,
            barriers: HashMap::new(),
        }
    }

    /// Number of warps
    pub fn num_warps(&self) -> u32 {
        self.warps.len() as u32
    }
}

// ============================================================================
// Occupancy Calculator
// ============================================================================

/// Occupancy calculation result
#[derive(Debug, Clone)]
pub struct OccupancyResult {
    /// Achieved occupancy (0.0 - 1.0)
    pub occupancy: f64,
    /// Active warps
    pub active_warps: u32,
    /// Maximum warps
    pub max_warps: u32,
    /// Limiting factor
    pub limiting_factor: OccupancyLimiter,
    /// Blocks per SM
    pub blocks_per_sm: u32,
    /// Threads per SM
    pub threads_per_sm: u32,
}

/// What limits occupancy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimiter {
    /// Limited by register usage
    Registers,
    /// Limited by shared memory
    SharedMemory,
    /// Limited by max blocks per SM
    MaxBlocks,
    /// Limited by max warps per SM
    MaxWarps,
    /// No limitation (100% occupancy)
    None,
}

/// Occupancy calculator
#[derive(Debug)]
pub struct OccupancyCalculator {
    /// SM specification
    spec: SmSpec,
}

impl OccupancyCalculator {
    pub fn new(spec: SmSpec) -> Self {
        Self { spec }
    }

    /// Calculate occupancy for a kernel launch
    pub fn calculate(
        &self,
        threads_per_block: u32,
        registers_per_thread: u32,
        shared_memory_bytes: u32,
    ) -> OccupancyResult {
        let warps_per_block = (threads_per_block + 31) / 32;

        // Register limit
        let total_registers = self.spec.register_file_kb * 1024 / 4;
        let registers_per_warp = registers_per_thread * 32;
        let warps_by_registers = if registers_per_warp > 0 {
            total_registers / registers_per_warp
        } else {
            self.spec.max_warps
        };

        // Shared memory limit
        let shared_memory_bytes_total = self.spec.shared_memory_kb * 1024;
        let blocks_by_shared = if shared_memory_bytes > 0 {
            shared_memory_bytes_total / shared_memory_bytes
        } else {
            self.spec.max_blocks
        };
        let warps_by_shared = blocks_by_shared * warps_per_block;

        // Max blocks limit
        let blocks_by_max_blocks = self.spec.max_blocks;
        let warps_by_max_blocks = blocks_by_max_blocks * warps_per_block;

        // Max warps limit
        let warps_by_max_warps = self.spec.max_warps;

        // Find minimum
        let active_warps = warps_by_registers
            .min(warps_by_shared)
            .min(warps_by_max_blocks)
            .min(warps_by_max_warps);

        // Determine limiting factor
        let limiting_factor = if active_warps == warps_by_registers
            && warps_by_registers < self.spec.max_warps
        {
            OccupancyLimiter::Registers
        } else if active_warps == warps_by_shared && warps_by_shared < self.spec.max_warps {
            OccupancyLimiter::SharedMemory
        } else if active_warps == warps_by_max_blocks && warps_by_max_blocks < self.spec.max_warps {
            OccupancyLimiter::MaxBlocks
        } else if active_warps < self.spec.max_warps {
            OccupancyLimiter::MaxWarps
        } else {
            OccupancyLimiter::None
        };

        let blocks_per_sm = active_warps / warps_per_block;
        let threads_per_sm = blocks_per_sm * threads_per_block;
        let occupancy = active_warps as f64 / self.spec.max_warps as f64;

        OccupancyResult {
            occupancy,
            active_warps,
            max_warps: self.spec.max_warps,
            limiting_factor,
            blocks_per_sm,
            threads_per_sm,
        }
    }

    /// Suggest optimal block size
    pub fn suggest_block_size(
        &self,
        registers_per_thread: u32,
        shared_memory_per_block: u32,
    ) -> u32 {
        let mut best_occupancy = 0.0;
        let mut best_block_size = 32;

        // Try different block sizes
        for threads in (32..=1024).step_by(32) {
            let result = self.calculate(threads, registers_per_thread, shared_memory_per_block);
            if result.occupancy > best_occupancy {
                best_occupancy = result.occupancy;
                best_block_size = threads;
            }
        }

        best_block_size
    }
}

// ============================================================================
// Warp Scheduler
// ============================================================================

/// Scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Round-robin
    RoundRobin,
    /// Greedy-then-oldest (GTO)
    Gto,
    /// Loose round-robin
    Lrr,
    /// Two-level scheduler
    TwoLevel,
}

/// Warp scheduler
#[derive(Debug)]
pub struct WarpScheduler {
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Ready warp queue
    ready_queue: VecDeque<u32>,
    /// Last issued warp
    last_issued: Option<u32>,
    /// Statistics
    pub stats: SchedulerStats,
}

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub cycles: u64,
    pub issues: u64,
    pub stalls: u64,
    pub no_ready_warps: u64,
}

impl SchedulerStats {
    pub fn issue_rate(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.issues as f64 / self.cycles as f64
        }
    }
}

impl WarpScheduler {
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self {
            policy,
            ready_queue: VecDeque::new(),
            last_issued: None,
            stats: SchedulerStats::default(),
        }
    }

    /// Add a ready warp
    pub fn add_ready(&mut self, warp_id: u32) {
        if !self.ready_queue.contains(&warp_id) {
            self.ready_queue.push_back(warp_id);
        }
    }

    /// Remove a warp from ready queue
    pub fn remove_ready(&mut self, warp_id: u32) {
        self.ready_queue.retain(|&w| w != warp_id);
    }

    /// Select next warp to issue
    pub fn select(&mut self) -> Option<u32> {
        self.stats.cycles += 1;

        if self.ready_queue.is_empty() {
            self.stats.no_ready_warps += 1;
            return None;
        }

        let selected = match self.policy {
            SchedulingPolicy::RoundRobin => self.ready_queue.pop_front(),
            SchedulingPolicy::Gto => {
                // Greedy: keep issuing same warp if ready
                if let Some(last) = self.last_issued {
                    if self.ready_queue.contains(&last) {
                        self.ready_queue.retain(|&w| w != last);
                        Some(last)
                    } else {
                        self.ready_queue.pop_front()
                    }
                } else {
                    self.ready_queue.pop_front()
                }
            }
            SchedulingPolicy::Lrr | SchedulingPolicy::TwoLevel => {
                // Simplified: same as round-robin
                self.ready_queue.pop_front()
            }
        };

        if selected.is_some() {
            self.stats.issues += 1;
            self.last_issued = selected;
        } else {
            self.stats.stalls += 1;
        }

        selected
    }

    /// Number of ready warps
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }
}

// ============================================================================
// SM Simulator
// ============================================================================

/// SM simulator state
#[derive(Debug)]
pub struct SmSimulator {
    /// SM specification
    pub spec: SmSpec,
    /// SM ID
    pub sm_id: u32,
    /// Active warps
    warps: HashMap<u32, Warp>,
    /// Active blocks
    blocks: HashMap<u32, ThreadBlock>,
    /// Warp schedulers
    schedulers: Vec<WarpScheduler>,
    /// L1 cache
    l1_cache: L1Cache,
    /// Tensor Core units
    tensor_cores: Vec<TensorCoreUnit>,
    /// Operand collector
    operand_collector: OperandCollector,
    /// Current cycle
    current_cycle: u64,
    /// Statistics
    pub stats: SmStats,
}

/// SM statistics
#[derive(Debug, Clone, Default)]
pub struct SmStats {
    pub cycles: u64,
    pub instructions_issued: u64,
    pub memory_instructions: u64,
    pub compute_instructions: u64,
    pub tensor_core_instructions: u64,
    pub stall_cycles: u64,
    pub active_cycles: u64,
}

impl SmStats {
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions_issued as f64 / self.cycles as f64
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.active_cycles as f64 / self.cycles as f64
        }
    }
}

impl SmSimulator {
    pub fn new(spec: SmSpec, sm_id: u32) -> Self {
        let num_schedulers = spec.warp_schedulers as usize;
        let schedulers = (0..num_schedulers)
            .map(|_| WarpScheduler::new(SchedulingPolicy::Gto))
            .collect();

        let l1_spec = match spec.architecture {
            SmArchitecture::Ampere => L1CacheSpec::a100(),
            SmArchitecture::Ada => L1CacheSpec::l4(),
            SmArchitecture::Hopper => L1CacheSpec::h100(),
        };

        let tc_spec = match spec.architecture {
            SmArchitecture::Ampere => TensorCoreSpec::a100(),
            SmArchitecture::Ada => TensorCoreSpec::l4(),
            SmArchitecture::Hopper => TensorCoreSpec::h100(),
        };

        let tensor_cores = (0..spec.tensor_cores)
            .map(|_| TensorCoreUnit::new(tc_spec.clone()))
            .collect();

        let reg_spec = match spec.architecture {
            SmArchitecture::Ampere => RegisterFileSpec::a100(),
            SmArchitecture::Ada => RegisterFileSpec::l4(),
            SmArchitecture::Hopper => RegisterFileSpec::h100(),
        };

        Self {
            spec,
            sm_id,
            warps: HashMap::new(),
            blocks: HashMap::new(),
            schedulers,
            l1_cache: L1Cache::new(l1_spec),
            tensor_cores,
            operand_collector: OperandCollector::new(4),
            current_cycle: 0,
            stats: SmStats::default(),
        }
    }

    /// Launch a thread block on this SM
    pub fn launch_block(
        &mut self,
        block_id: u32,
        threads: u32,
        shared_memory: u32,
        registers_per_thread: u32,
    ) -> Result<(), &'static str> {
        // Check resources
        let new_warps = (threads + 31) / 32;
        let current_warps = self.warps.len() as u32;

        if current_warps + new_warps > self.spec.max_warps {
            return Err("Not enough warp slots");
        }

        let current_blocks = self.blocks.len() as u32;
        if current_blocks + 1 > self.spec.max_blocks {
            return Err("Not enough block slots");
        }

        // Create block
        let block = ThreadBlock::new(block_id, threads, shared_memory, registers_per_thread);

        // Create warps
        let mut register_base = 0u32;
        for warp_offset in 0..new_warps {
            let warp_id = block_id * 64 + warp_offset;
            let warp = Warp::new(warp_id, block_id, register_base, registers_per_thread);
            register_base += registers_per_thread * 32;

            // Add to scheduler (round-robin across schedulers)
            let scheduler_idx = (warp_id as usize) % self.schedulers.len();
            self.schedulers[scheduler_idx].add_ready(warp_id);

            self.warps.insert(warp_id, warp);
        }

        self.blocks.insert(block_id, block);
        Ok(())
    }

    /// Simulate one cycle
    pub fn tick(&mut self) {
        self.current_cycle += 1;
        self.stats.cycles += 1;

        let mut any_active = false;
        let mut selected_warps = Vec::new();
        let mut ready_to_reschedule = Vec::new();

        // First pass: select warps from schedulers
        for (scheduler_idx, scheduler) in self.schedulers.iter_mut().enumerate() {
            if let Some(warp_id) = scheduler.select() {
                selected_warps.push((scheduler_idx, warp_id));
            }
        }

        // Second pass: execute selected warps
        for (scheduler_idx, warp_id) in selected_warps {
            if let Some(warp) = self.warps.get_mut(&warp_id) {
                // Execute inline to avoid borrow issues
                warp.instructions_executed += 1;
                self.stats.instructions_issued += 1;
                warp.pc += 1;

                let instruction_type = warp.pc % 10;
                match instruction_type {
                    0..=5 => {
                        self.stats.compute_instructions += 1;
                    }
                    6..=8 => {
                        self.stats.memory_instructions += 1;
                        let address = (warp.warp_id as u64) * 128 + (warp.pc as u64) * 4;
                        let _ = self.l1_cache.read(address);
                    }
                    _ => {
                        if !self.tensor_cores.is_empty() {
                            self.stats.tensor_core_instructions += 1;
                            let tc_idx = (warp.warp_id as usize) % self.tensor_cores.len();
                            let _ = self.tensor_cores[tc_idx].issue_mma(MmaConfig::fp16_16x16x16());
                        }
                    }
                }

                any_active = true;

                // Check if warp should be rescheduled
                if warp.state == WarpState::Ready {
                    ready_to_reschedule.push((scheduler_idx, warp_id));
                }
            }
        }

        // Third pass: re-add ready warps to schedulers
        for (scheduler_idx, warp_id) in ready_to_reschedule {
            self.schedulers[scheduler_idx].add_ready(warp_id);
        }

        if any_active {
            self.stats.active_cycles += 1;
        } else {
            self.stats.stall_cycles += 1;
        }

        // Tick tensor cores
        for tc in &mut self.tensor_cores {
            tc.tick();
        }
    }

    /// Check if SM is idle
    pub fn is_idle(&self) -> bool {
        self.warps.is_empty() || self.warps.values().all(|w| w.state == WarpState::Completed)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &CacheStats {
        &self.l1_cache.stats
    }

    /// Get occupancy for current load
    pub fn current_occupancy(&self) -> f64 {
        self.warps.len() as f64 / self.spec.max_warps as f64
    }

    /// Complete a block
    pub fn complete_block(&mut self, block_id: u32) {
        if let Some(block) = self.blocks.remove(&block_id) {
            for warp_id in block.warps {
                self.warps.remove(&warp_id);
                for scheduler in &mut self.schedulers {
                    scheduler.remove_ready(warp_id);
                }
            }
        }
    }
}

// ============================================================================
// Performance Predictor
// ============================================================================

/// Kernel characteristics for prediction
#[derive(Debug, Clone)]
pub struct KernelCharacteristics {
    /// Arithmetic intensity (FLOPS/byte)
    pub arithmetic_intensity: f64,
    /// Memory bound ratio (0-1)
    pub memory_bound_ratio: f64,
    /// Compute bound ratio (0-1)
    pub compute_bound_ratio: f64,
    /// Instructions per thread
    pub instructions_per_thread: u32,
    /// Memory transactions per thread
    pub memory_transactions_per_thread: u32,
}

/// Performance prediction
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Predicted execution time (ms)
    pub execution_time_ms: f64,
    /// Predicted throughput (TFLOPS)
    pub throughput_tflops: f64,
    /// Predicted memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Predicted compute utilization
    pub compute_utilization: f64,
    /// Bottleneck
    pub bottleneck: PerformanceBottleneck,
}

/// Performance bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceBottleneck {
    Compute,
    Memory,
    Latency,
    Occupancy,
}

/// Performance predictor using roofline model
#[derive(Debug)]
pub struct PerformancePredictor {
    /// SM specification
    sm_spec: SmSpec,
    /// Number of SMs
    num_sms: u32,
    /// Memory bandwidth (GB/s)
    memory_bandwidth_gbps: f64,
}

impl PerformancePredictor {
    pub fn new(sm_spec: SmSpec, num_sms: u32, memory_bandwidth_gbps: f64) -> Self {
        Self {
            sm_spec,
            num_sms,
            memory_bandwidth_gbps,
        }
    }

    /// Predict kernel performance
    pub fn predict(
        &self,
        kernel: &KernelCharacteristics,
        occupancy: &OccupancyResult,
        total_threads: u64,
    ) -> PerformancePrediction {
        // Peak performance
        let peak_tflops = self.sm_spec.peak_fp32_tflops() * self.num_sms as f64;

        // Roofline: performance = min(peak, bandwidth * AI)
        let memory_limited_tflops =
            self.memory_bandwidth_gbps * kernel.arithmetic_intensity / 1000.0;
        let achievable_tflops = peak_tflops.min(memory_limited_tflops);

        // Apply occupancy penalty
        let effective_tflops = achievable_tflops * occupancy.occupancy;

        // Calculate execution time
        let total_flops = (total_threads as f64) * (kernel.instructions_per_thread as f64) * 2.0;
        let execution_time_s = total_flops / (effective_tflops * 1e12);
        let execution_time_ms = execution_time_s * 1000.0;

        // Determine bottleneck
        let bottleneck =
            if kernel.arithmetic_intensity < (peak_tflops * 1000.0 / self.memory_bandwidth_gbps) {
                PerformanceBottleneck::Memory
            } else if occupancy.occupancy < 0.5 {
                PerformanceBottleneck::Occupancy
            } else {
                PerformanceBottleneck::Compute
            };

        let memory_bandwidth_utilization = if bottleneck == PerformanceBottleneck::Memory {
            0.8 * occupancy.occupancy
        } else {
            kernel.memory_bound_ratio * occupancy.occupancy
        };

        let compute_utilization = if bottleneck == PerformanceBottleneck::Compute {
            0.9 * occupancy.occupancy
        } else {
            kernel.compute_bound_ratio * occupancy.occupancy
        };

        PerformancePrediction {
            execution_time_ms,
            throughput_tflops: effective_tflops,
            memory_bandwidth_utilization,
            compute_utilization,
            bottleneck,
        }
    }

    /// Get roofline intersection point
    pub fn roofline_ridge_point(&self) -> f64 {
        let peak_tflops = self.sm_spec.peak_fp32_tflops() * self.num_sms as f64;
        // Ridge point in FLOPS/byte
        peak_tflops * 1000.0 / self.memory_bandwidth_gbps
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm_spec() {
        let a100 = SmSpec::a100();
        assert_eq!(a100.cuda_cores, 64);
        assert_eq!(a100.tensor_cores, 4);
        assert_eq!(a100.max_warps, 64);
    }

    #[test]
    fn test_sm_peak_performance() {
        let h100 = SmSpec::h100();
        let tflops = h100.peak_fp32_tflops();
        assert!(tflops > 0.0);
        // H100 SM should have higher peak than A100
        assert!(tflops > SmSpec::a100().peak_fp32_tflops());
    }

    #[test]
    fn test_occupancy_calculator() {
        let spec = SmSpec::a100();
        let calc = OccupancyCalculator::new(spec);

        // High occupancy case
        let result = calc.calculate(256, 32, 0);
        assert!(result.occupancy > 0.5);

        // Low occupancy due to registers
        let result = calc.calculate(256, 255, 0);
        assert!(result.occupancy < 0.5);
        assert_eq!(result.limiting_factor, OccupancyLimiter::Registers);
    }

    #[test]
    fn test_occupancy_shared_memory() {
        let spec = SmSpec::a100();
        let calc = OccupancyCalculator::new(spec.clone());

        // High shared memory usage
        let result = calc.calculate(256, 32, 100 * 1024);
        assert!(result.blocks_per_sm <= 1);
    }

    #[test]
    fn test_warp_scheduler() {
        let mut scheduler = WarpScheduler::new(SchedulingPolicy::RoundRobin);

        scheduler.add_ready(0);
        scheduler.add_ready(1);
        scheduler.add_ready(2);

        assert_eq!(scheduler.ready_count(), 3);

        let first = scheduler.select();
        assert_eq!(first, Some(0));

        let second = scheduler.select();
        assert_eq!(second, Some(1));
    }

    #[test]
    fn test_gto_scheduler() {
        let mut scheduler = WarpScheduler::new(SchedulingPolicy::Gto);

        scheduler.add_ready(0);
        scheduler.add_ready(1);

        let first = scheduler.select();
        assert_eq!(first, Some(0));

        // Re-add warp 0
        scheduler.add_ready(0);

        // GTO should prefer warp 0 again
        let second = scheduler.select();
        assert_eq!(second, Some(0));
    }

    #[test]
    fn test_sm_simulator() {
        let spec = SmSpec::a100();
        let mut sm = SmSimulator::new(spec, 0);

        // Launch a block
        let result = sm.launch_block(0, 256, 0, 32);
        assert!(result.is_ok());

        assert!(!sm.is_idle());
        assert!(sm.current_occupancy() > 0.0);
    }

    #[test]
    fn test_sm_tick() {
        let spec = SmSpec::a100();
        let mut sm = SmSimulator::new(spec, 0);

        sm.launch_block(0, 256, 0, 32).unwrap();

        // Run some cycles
        for _ in 0..100 {
            sm.tick();
        }

        assert!(sm.stats.instructions_issued > 0);
        assert!(sm.stats.cycles == 100);
    }

    #[test]
    fn test_thread_block() {
        let block = ThreadBlock::new(0, 256, 1024, 32);

        assert_eq!(block.num_warps(), 8); // 256/32 = 8
        assert_eq!(block.threads, 256);
        assert_eq!(block.shared_memory_bytes, 1024);
    }

    #[test]
    fn test_warp() {
        let warp = Warp::new(0, 0, 0, 32);

        assert_eq!(warp.active_threads(), 32);
        assert_eq!(warp.state, WarpState::Ready);
    }

    #[test]
    fn test_performance_predictor() {
        let spec = SmSpec::a100();
        let predictor = PerformancePredictor::new(spec.clone(), 108, 2039.0);

        let kernel = KernelCharacteristics {
            arithmetic_intensity: 10.0,
            memory_bound_ratio: 0.3,
            compute_bound_ratio: 0.7,
            instructions_per_thread: 1000,
            memory_transactions_per_thread: 100,
        };

        let calc = OccupancyCalculator::new(spec);
        let occupancy = calc.calculate(256, 32, 0);

        let prediction = predictor.predict(&kernel, &occupancy, 1_000_000);

        assert!(prediction.execution_time_ms > 0.0);
        assert!(prediction.throughput_tflops > 0.0);
    }

    #[test]
    fn test_roofline_ridge() {
        let spec = SmSpec::a100();
        let predictor = PerformancePredictor::new(spec, 108, 2039.0);

        let ridge = predictor.roofline_ridge_point();
        assert!(ridge > 0.0);
    }

    #[test]
    fn test_block_completion() {
        let spec = SmSpec::a100();
        let mut sm = SmSimulator::new(spec, 0);

        sm.launch_block(0, 128, 0, 32).unwrap();
        assert_eq!(sm.warps.len(), 4); // 128/32 = 4 warps

        sm.complete_block(0);
        assert_eq!(sm.warps.len(), 0);
        assert!(sm.is_idle());
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = WarpScheduler::new(SchedulingPolicy::RoundRobin);

        // No ready warps - should stall
        scheduler.select();
        assert_eq!(scheduler.stats.no_ready_warps, 1);

        scheduler.add_ready(0);
        scheduler.select();
        assert_eq!(scheduler.stats.issues, 1);
    }

    #[test]
    fn test_suggest_block_size() {
        let spec = SmSpec::a100();
        let calc = OccupancyCalculator::new(spec);

        let suggested = calc.suggest_block_size(32, 0);
        assert!(suggested >= 32);
        assert!(suggested <= 1024);
        assert!(suggested % 32 == 0);
    }

    #[test]
    fn test_sm_stats() {
        let spec = SmSpec::a100();
        let mut sm = SmSimulator::new(spec, 0);

        sm.launch_block(0, 256, 0, 32).unwrap();

        for _ in 0..50 {
            sm.tick();
        }

        let ipc = sm.stats.ipc();
        assert!(ipc > 0.0);

        let util = sm.stats.utilization();
        assert!(util > 0.0);
    }
}
