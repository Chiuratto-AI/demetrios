//! Power Management and DVFS (Dynamic Voltage and Frequency Scaling)
//!
//! This module implements GPU power management including:
//! - Power domain modeling for different GPU components
//! - DVFS controllers for voltage/frequency scaling
//! - Power budgeting for multi-GPU systems
//! - Thermal modeling and throttling
//! - Kernel power estimation

use std::collections::{HashMap, VecDeque};

/// Power specifications for different GPU SKUs
#[derive(Debug, Clone)]
pub struct PowerSpec {
    /// GPU model name
    pub model: String,
    /// TDP (Thermal Design Power) in watts
    pub tdp_watts: f64,
    /// Maximum power limit in watts
    pub max_power_watts: f64,
    /// Minimum power limit in watts
    pub min_power_watts: f64,
    /// Idle power in watts
    pub idle_power_watts: f64,
    /// Base clock in MHz
    pub base_clock_mhz: u32,
    /// Boost clock in MHz
    pub boost_clock_mhz: u32,
    /// Memory clock in MHz
    pub memory_clock_mhz: u32,
    /// Voltage range (min_mv, max_mv)
    pub voltage_range_mv: (u32, u32),
    /// Power connector configuration
    pub power_connectors: PowerConnectors,
}

impl PowerSpec {
    /// NVIDIA A100 PCIe variant
    pub fn a100_pcie() -> Self {
        Self {
            model: "A100-PCIE-40GB".to_string(),
            tdp_watts: 250.0,
            max_power_watts: 300.0,
            min_power_watts: 100.0,
            idle_power_watts: 25.0,
            base_clock_mhz: 765,
            boost_clock_mhz: 1410,
            memory_clock_mhz: 1215,
            voltage_range_mv: (700, 1100),
            power_connectors: PowerConnectors::Dual8Pin,
        }
    }

    /// NVIDIA A100 SXM variant (higher power)
    pub fn a100_sxm() -> Self {
        Self {
            model: "A100-SXM4-80GB".to_string(),
            tdp_watts: 400.0,
            max_power_watts: 500.0,
            min_power_watts: 150.0,
            idle_power_watts: 35.0,
            base_clock_mhz: 1095,
            boost_clock_mhz: 1410,
            memory_clock_mhz: 1593,
            voltage_range_mv: (700, 1150),
            power_connectors: PowerConnectors::SxmModule,
        }
    }

    /// NVIDIA L4 inference GPU
    pub fn l4() -> Self {
        Self {
            model: "L4".to_string(),
            tdp_watts: 72.0,
            max_power_watts: 72.0,
            min_power_watts: 30.0,
            idle_power_watts: 10.0,
            base_clock_mhz: 795,
            boost_clock_mhz: 2040,
            memory_clock_mhz: 6251,
            voltage_range_mv: (650, 1000),
            power_connectors: PowerConnectors::SlotPowered,
        }
    }

    /// NVIDIA H100 SXM variant
    pub fn h100_sxm() -> Self {
        Self {
            model: "H100-SXM5-80GB".to_string(),
            tdp_watts: 700.0,
            max_power_watts: 700.0,
            min_power_watts: 200.0,
            idle_power_watts: 50.0,
            base_clock_mhz: 1095,
            boost_clock_mhz: 1830,
            memory_clock_mhz: 2619,
            voltage_range_mv: (700, 1200),
            power_connectors: PowerConnectors::SxmModule,
        }
    }
}

/// Power connector types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerConnectors {
    /// Single 8-pin connector (150W)
    Single8Pin,
    /// Dual 8-pin connectors (300W)
    Dual8Pin,
    /// Triple 8-pin connectors (450W)
    Triple8Pin,
    /// 12VHPWR connector (600W)
    HighPower12V,
    /// SXM module power
    SxmModule,
    /// Slot-powered only (75W)
    SlotPowered,
}

impl PowerConnectors {
    /// Maximum power delivery in watts
    pub fn max_power(&self) -> f64 {
        match self {
            Self::SlotPowered => 75.0,
            Self::Single8Pin => 150.0 + 75.0,
            Self::Dual8Pin => 300.0 + 75.0,
            Self::Triple8Pin => 450.0 + 75.0,
            Self::HighPower12V => 600.0 + 75.0,
            Self::SxmModule => 700.0,
        }
    }
}

/// Power domain representing different GPU components
#[derive(Debug, Clone)]
pub struct PowerDomain {
    /// Domain name
    pub name: String,
    /// Current power consumption in watts
    pub current_power_watts: f64,
    /// Maximum power for this domain
    pub max_power_watts: f64,
    /// Power efficiency coefficient (0.0 to 1.0)
    pub efficiency: f64,
    /// Whether domain is currently active
    pub active: bool,
    /// Clock gating state
    pub clock_gated: bool,
    /// Power gating state
    pub power_gated: bool,
}

impl PowerDomain {
    /// Create a new power domain
    pub fn new(name: &str, max_power: f64) -> Self {
        Self {
            name: name.to_string(),
            current_power_watts: 0.0,
            max_power_watts: max_power,
            efficiency: 0.85,
            active: true,
            clock_gated: false,
            power_gated: false,
        }
    }

    /// Set domain power based on utilization
    pub fn set_utilization(&mut self, utilization: f64) {
        let util = utilization.clamp(0.0, 1.0);
        if self.power_gated {
            self.current_power_watts = 0.0;
        } else if self.clock_gated {
            self.current_power_watts = self.max_power_watts * 0.05; // Leakage only
        } else {
            // Dynamic power scales with utilization squared (roughly)
            // P = P_static + P_dynamic * util^2
            let static_power = self.max_power_watts * 0.15;
            let dynamic_power = self.max_power_watts * 0.85 * util * util;
            self.current_power_watts = static_power + dynamic_power;
        }
    }

    /// Enable clock gating
    pub fn clock_gate(&mut self) {
        self.clock_gated = true;
        self.current_power_watts = self.max_power_watts * 0.05;
    }

    /// Disable clock gating
    pub fn clock_ungate(&mut self) {
        self.clock_gated = false;
    }

    /// Enable power gating
    pub fn power_gate(&mut self) {
        self.power_gated = true;
        self.clock_gated = true;
        self.current_power_watts = 0.0;
    }

    /// Disable power gating
    pub fn power_ungate(&mut self) {
        self.power_gated = false;
        self.clock_gated = false;
    }
}

/// GPU power breakdown by component
#[derive(Debug, Clone)]
pub struct GpuPowerBreakdown {
    /// SM (Streaming Multiprocessor) domain
    pub sm_domain: PowerDomain,
    /// Memory controller domain
    pub memory_domain: PowerDomain,
    /// L2 cache domain
    pub l2_domain: PowerDomain,
    /// NVDEC video decoder
    pub nvdec_domain: PowerDomain,
    /// NVENC video encoder
    pub nvenc_domain: PowerDomain,
    /// Display engine
    pub display_domain: PowerDomain,
    /// PCIe/NVLink interconnect
    pub interconnect_domain: PowerDomain,
    /// Tensor cores (if separate)
    pub tensor_domain: Option<PowerDomain>,
    /// Voltage regulators overhead
    pub vrm_overhead: f64,
}

impl GpuPowerBreakdown {
    /// Create breakdown for A100
    pub fn a100() -> Self {
        Self {
            sm_domain: PowerDomain::new("SM", 250.0),
            memory_domain: PowerDomain::new("HBM2e", 80.0),
            l2_domain: PowerDomain::new("L2", 30.0),
            nvdec_domain: PowerDomain::new("NVDEC", 10.0),
            nvenc_domain: PowerDomain::new("NVENC", 10.0),
            display_domain: PowerDomain::new("Display", 0.0), // No display
            interconnect_domain: PowerDomain::new("NVLink", 20.0),
            tensor_domain: Some(PowerDomain::new("TensorCores", 150.0)),
            vrm_overhead: 0.1, // 10% VRM losses
        }
    }

    /// Get total power consumption
    pub fn total_power(&self) -> f64 {
        let base_power = self.sm_domain.current_power_watts
            + self.memory_domain.current_power_watts
            + self.l2_domain.current_power_watts
            + self.nvdec_domain.current_power_watts
            + self.nvenc_domain.current_power_watts
            + self.display_domain.current_power_watts
            + self.interconnect_domain.current_power_watts
            + self
                .tensor_domain
                .as_ref()
                .map_or(0.0, |d| d.current_power_watts);

        // Add VRM overhead
        base_power * (1.0 + self.vrm_overhead)
    }

    /// Set utilization for compute workload
    pub fn set_compute_utilization(&mut self, sm_util: f64, memory_util: f64, tensor_util: f64) {
        self.sm_domain.set_utilization(sm_util);
        self.memory_domain.set_utilization(memory_util);
        self.l2_domain.set_utilization(memory_util * 0.8);
        if let Some(ref mut tensor) = self.tensor_domain {
            tensor.set_utilization(tensor_util);
        }
    }

    /// Power gate unused domains
    pub fn gate_unused_domains(&mut self, workload: WorkloadType) {
        match workload {
            WorkloadType::Compute => {
                self.nvdec_domain.power_gate();
                self.nvenc_domain.power_gate();
                self.display_domain.power_gate();
            }
            WorkloadType::Inference => {
                self.nvdec_domain.power_gate();
                self.nvenc_domain.power_gate();
                self.display_domain.power_gate();
                // SM clock gating between batches handled elsewhere
            }
            WorkloadType::VideoEncode => {
                self.nvdec_domain.power_gate();
                self.display_domain.power_gate();
                self.sm_domain.clock_gate();
            }
            WorkloadType::VideoDecode => {
                self.nvenc_domain.power_gate();
                self.display_domain.power_gate();
                self.sm_domain.clock_gate();
            }
            WorkloadType::Graphics => {
                self.nvdec_domain.clock_gate();
                self.nvenc_domain.clock_gate();
            }
            WorkloadType::Idle => {
                self.sm_domain.clock_gate();
                self.nvdec_domain.power_gate();
                self.nvenc_domain.power_gate();
                self.display_domain.power_gate();
                if let Some(ref mut tensor) = self.tensor_domain {
                    tensor.power_gate();
                }
            }
        }
    }
}

/// Workload types for power optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    Compute,
    Inference,
    VideoEncode,
    VideoDecode,
    Graphics,
    Idle,
}

/// DVFS (Dynamic Voltage and Frequency Scaling) controller
#[derive(Debug)]
pub struct DvfsController {
    /// Current frequency in MHz
    pub current_freq_mhz: u32,
    /// Current voltage in mV
    pub current_voltage_mv: u32,
    /// Target frequency
    target_freq_mhz: u32,
    /// Frequency steps available
    freq_steps: Vec<FrequencyStep>,
    /// Current P-state index
    current_pstate: usize,
    /// DVFS policy
    policy: DvfsPolicy,
    /// Frequency history for analysis
    freq_history: VecDeque<FrequencyRecord>,
    /// Maximum history size
    max_history: usize,
}

/// A single frequency/voltage operating point
#[derive(Debug, Clone)]
pub struct FrequencyStep {
    /// Frequency in MHz
    pub freq_mhz: u32,
    /// Voltage in mV
    pub voltage_mv: u32,
    /// Estimated power at this point
    pub power_watts: f64,
    /// Performance level (0-100)
    pub perf_level: u8,
}

/// Frequency change record
#[derive(Debug, Clone)]
pub struct FrequencyRecord {
    /// Timestamp (simulation time)
    pub timestamp: u64,
    /// Frequency at this time
    pub freq_mhz: u32,
    /// Voltage at this time
    pub voltage_mv: u32,
    /// Reason for change
    pub reason: FrequencyChangeReason,
}

/// Reasons for frequency changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyChangeReason {
    ThermalThrottle,
    PowerThrottle,
    UserRequest,
    Boost,
    Idle,
    WorkloadChange,
}

/// DVFS policy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DvfsPolicy {
    /// Maximum performance (boost when possible)
    Performance,
    /// Balance performance and power
    Balanced,
    /// Minimize power consumption
    PowerSave,
    /// Fixed frequency (no scaling)
    Fixed,
}

impl DvfsController {
    /// Create DVFS controller for A100
    pub fn new_a100() -> Self {
        let freq_steps = vec![
            FrequencyStep {
                freq_mhz: 210,
                voltage_mv: 700,
                power_watts: 25.0,
                perf_level: 10,
            },
            FrequencyStep {
                freq_mhz: 420,
                voltage_mv: 750,
                power_watts: 60.0,
                perf_level: 25,
            },
            FrequencyStep {
                freq_mhz: 630,
                voltage_mv: 800,
                power_watts: 100.0,
                perf_level: 40,
            },
            FrequencyStep {
                freq_mhz: 765,
                voltage_mv: 850,
                power_watts: 150.0,
                perf_level: 50,
            },
            FrequencyStep {
                freq_mhz: 900,
                voltage_mv: 900,
                power_watts: 200.0,
                perf_level: 60,
            },
            FrequencyStep {
                freq_mhz: 1050,
                voltage_mv: 950,
                power_watts: 260.0,
                perf_level: 75,
            },
            FrequencyStep {
                freq_mhz: 1200,
                voltage_mv: 1000,
                power_watts: 320.0,
                perf_level: 85,
            },
            FrequencyStep {
                freq_mhz: 1350,
                voltage_mv: 1050,
                power_watts: 380.0,
                perf_level: 95,
            },
            FrequencyStep {
                freq_mhz: 1410,
                voltage_mv: 1100,
                power_watts: 400.0,
                perf_level: 100,
            },
        ];

        Self {
            current_freq_mhz: 765,
            current_voltage_mv: 850,
            target_freq_mhz: 765,
            freq_steps,
            current_pstate: 3,
            policy: DvfsPolicy::Balanced,
            freq_history: VecDeque::new(),
            max_history: 1000,
        }
    }

    /// Set DVFS policy
    pub fn set_policy(&mut self, policy: DvfsPolicy) {
        self.policy = policy;
    }

    /// Request frequency change
    pub fn request_frequency(&mut self, freq_mhz: u32, timestamp: u64) {
        self.target_freq_mhz = freq_mhz;
        self.apply_frequency_change(FrequencyChangeReason::UserRequest, timestamp);
    }

    /// Update frequency based on current conditions
    pub fn update(
        &mut self,
        temperature_c: f64,
        power_watts: f64,
        power_limit: f64,
        utilization: f64,
        timestamp: u64,
    ) {
        // Check for thermal throttling
        if temperature_c > 83.0 {
            self.throttle_thermal(timestamp);
            return;
        }

        // Check for power throttling
        if power_watts > power_limit * 0.95 {
            self.throttle_power(timestamp);
            return;
        }

        // Apply policy-based scaling
        match self.policy {
            DvfsPolicy::Performance => {
                if utilization > 0.1 && temperature_c < 75.0 && power_watts < power_limit * 0.9 {
                    self.boost(timestamp);
                }
            }
            DvfsPolicy::Balanced => {
                if utilization > 0.8 && temperature_c < 75.0 {
                    self.step_up(timestamp);
                } else if utilization < 0.3 {
                    self.step_down(timestamp);
                }
            }
            DvfsPolicy::PowerSave => {
                if utilization < 0.5 {
                    self.step_down(timestamp);
                }
            }
            DvfsPolicy::Fixed => {
                // Do nothing
            }
        }
    }

    /// Thermal throttling
    fn throttle_thermal(&mut self, timestamp: u64) {
        if self.current_pstate > 0 {
            self.current_pstate -= 1;
            self.apply_frequency_change(FrequencyChangeReason::ThermalThrottle, timestamp);
        }
    }

    /// Power throttling
    fn throttle_power(&mut self, timestamp: u64) {
        if self.current_pstate > 0 {
            self.current_pstate -= 1;
            self.apply_frequency_change(FrequencyChangeReason::PowerThrottle, timestamp);
        }
    }

    /// Boost to maximum
    fn boost(&mut self, timestamp: u64) {
        let max_pstate = self.freq_steps.len() - 1;
        if self.current_pstate < max_pstate {
            self.current_pstate = max_pstate;
            self.apply_frequency_change(FrequencyChangeReason::Boost, timestamp);
        }
    }

    /// Step up one P-state
    fn step_up(&mut self, timestamp: u64) {
        let max_pstate = self.freq_steps.len() - 1;
        if self.current_pstate < max_pstate {
            self.current_pstate += 1;
            self.apply_frequency_change(FrequencyChangeReason::WorkloadChange, timestamp);
        }
    }

    /// Step down one P-state
    fn step_down(&mut self, timestamp: u64) {
        if self.current_pstate > 0 {
            self.current_pstate -= 1;
            self.apply_frequency_change(FrequencyChangeReason::WorkloadChange, timestamp);
        }
    }

    /// Apply frequency change
    fn apply_frequency_change(&mut self, reason: FrequencyChangeReason, timestamp: u64) {
        let step = &self.freq_steps[self.current_pstate];
        self.current_freq_mhz = step.freq_mhz;
        self.current_voltage_mv = step.voltage_mv;

        let record = FrequencyRecord {
            timestamp,
            freq_mhz: self.current_freq_mhz,
            voltage_mv: self.current_voltage_mv,
            reason,
        };

        self.freq_history.push_back(record);
        if self.freq_history.len() > self.max_history {
            self.freq_history.pop_front();
        }
    }

    /// Get current power estimate
    pub fn current_power_estimate(&self) -> f64 {
        self.freq_steps[self.current_pstate].power_watts
    }

    /// Get performance level (0-100)
    pub fn performance_level(&self) -> u8 {
        self.freq_steps[self.current_pstate].perf_level
    }
}

/// Workload activity metrics for power modeling
#[derive(Debug, Clone)]
pub struct WorkloadActivity {
    /// SM utilization (0.0 to 1.0)
    pub sm_utilization: f64,
    /// Memory bandwidth utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Tensor core utilization (0.0 to 1.0)
    pub tensor_utilization: f64,
    /// L2 cache hit rate (0.0 to 1.0)
    pub l2_hit_rate: f64,
    /// Instructions per cycle
    pub ipc: f64,
    /// Active warps percentage
    pub active_warps: f64,
}

impl WorkloadActivity {
    /// Create from utilization values
    pub fn new(sm: f64, memory: f64, tensor: f64) -> Self {
        Self {
            sm_utilization: sm.clamp(0.0, 1.0),
            memory_utilization: memory.clamp(0.0, 1.0),
            tensor_utilization: tensor.clamp(0.0, 1.0),
            l2_hit_rate: 0.8,
            ipc: 1.0,
            active_warps: sm.clamp(0.0, 1.0),
        }
    }

    /// Idle workload
    pub fn idle() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Compute-heavy workload
    pub fn compute_heavy() -> Self {
        Self::new(0.95, 0.3, 0.0)
    }

    /// Memory-heavy workload
    pub fn memory_heavy() -> Self {
        Self::new(0.4, 0.95, 0.0)
    }

    /// GEMM workload (tensor cores)
    pub fn gemm() -> Self {
        Self::new(0.7, 0.8, 0.95)
    }
}

/// Power budget manager for multi-GPU systems
#[derive(Debug)]
pub struct PowerBudgetManager {
    /// Total system power budget in watts
    pub total_budget_watts: f64,
    /// Per-GPU power limits
    gpu_limits: HashMap<u32, f64>,
    /// Current power consumption per GPU
    gpu_power: HashMap<u32, f64>,
    /// Power allocation policy
    policy: PowerAllocationPolicy,
    /// Reserved power for system overhead
    system_overhead_watts: f64,
}

/// Power allocation policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerAllocationPolicy {
    /// Equal distribution among GPUs
    Equal,
    /// Proportional to workload
    Proportional,
    /// Priority-based allocation
    Priority,
    /// First-come-first-served
    FirstCome,
}

impl PowerBudgetManager {
    /// Create manager for DGX A100 (8 GPUs, 6.5kW)
    pub fn dgx_a100() -> Self {
        let mut gpu_limits = HashMap::new();
        let mut gpu_power = HashMap::new();
        for i in 0..8 {
            gpu_limits.insert(i, 400.0); // Default 400W per GPU
            gpu_power.insert(i, 0.0);
        }

        Self {
            total_budget_watts: 6500.0,
            gpu_limits,
            gpu_power,
            policy: PowerAllocationPolicy::Equal,
            system_overhead_watts: 300.0, // CPUs, memory, NVSwitch, etc.
        }
    }

    /// Create manager for custom configuration
    pub fn new(total_watts: f64, num_gpus: u32, overhead: f64) -> Self {
        let mut gpu_limits = HashMap::new();
        let mut gpu_power = HashMap::new();
        let per_gpu = (total_watts - overhead) / num_gpus as f64;

        for i in 0..num_gpus {
            gpu_limits.insert(i, per_gpu);
            gpu_power.insert(i, 0.0);
        }

        Self {
            total_budget_watts: total_watts,
            gpu_limits,
            gpu_power,
            policy: PowerAllocationPolicy::Equal,
            system_overhead_watts: overhead,
        }
    }

    /// Set power allocation policy
    pub fn set_policy(&mut self, policy: PowerAllocationPolicy) {
        self.policy = policy;
    }

    /// Update GPU power consumption
    pub fn update_gpu_power(&mut self, gpu_id: u32, power_watts: f64) {
        self.gpu_power.insert(gpu_id, power_watts);
    }

    /// Get power limit for GPU
    pub fn get_limit(&self, gpu_id: u32) -> Option<f64> {
        self.gpu_limits.get(&gpu_id).copied()
    }

    /// Set power limit for GPU
    pub fn set_limit(&mut self, gpu_id: u32, limit_watts: f64) {
        if let Some(existing) = self.gpu_limits.get_mut(&gpu_id) {
            *existing = limit_watts;
        }
    }

    /// Get total current power consumption
    pub fn total_power(&self) -> f64 {
        self.gpu_power.values().sum::<f64>() + self.system_overhead_watts
    }

    /// Get remaining power budget
    pub fn remaining_budget(&self) -> f64 {
        self.total_budget_watts - self.total_power()
    }

    /// Check if system is within budget
    pub fn within_budget(&self) -> bool {
        self.total_power() <= self.total_budget_watts
    }

    /// Rebalance power limits based on policy
    pub fn rebalance(&mut self, workload_weights: &HashMap<u32, f64>) {
        let available = self.total_budget_watts - self.system_overhead_watts;
        let num_gpus = self.gpu_limits.len();

        match self.policy {
            PowerAllocationPolicy::Equal => {
                let per_gpu = available / num_gpus as f64;
                for limit in self.gpu_limits.values_mut() {
                    *limit = per_gpu;
                }
            }
            PowerAllocationPolicy::Proportional => {
                let total_weight: f64 = workload_weights.values().sum();
                if total_weight > 0.0 {
                    for (gpu_id, limit) in &mut self.gpu_limits {
                        let weight = workload_weights.get(gpu_id).copied().unwrap_or(1.0);
                        *limit = available * (weight / total_weight);
                    }
                }
            }
            PowerAllocationPolicy::Priority => {
                // Sort GPUs by workload weight (priority)
                let mut sorted: Vec<_> = workload_weights.iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut remaining = available;
                for (gpu_id, _weight) in sorted {
                    let alloc = (remaining / num_gpus as f64).min(500.0); // Cap at 500W
                    if let Some(limit) = self.gpu_limits.get_mut(gpu_id) {
                        *limit = alloc;
                        remaining -= alloc;
                    }
                }
            }
            PowerAllocationPolicy::FirstCome => {
                // Allocate up to max for each GPU in order
                let mut remaining = available;
                for limit in self.gpu_limits.values_mut() {
                    let alloc = remaining.min(400.0);
                    *limit = alloc;
                    remaining = (remaining - alloc).max(0.0);
                }
            }
        }
    }
}

/// Kernel power model for estimating power consumption
#[derive(Debug, Clone)]
pub struct KernelPowerModel {
    /// Base power coefficient (intercept)
    base_power: f64,
    /// SM activity coefficient
    sm_coeff: f64,
    /// Memory activity coefficient
    memory_coeff: f64,
    /// Tensor core coefficient
    tensor_coeff: f64,
    /// Frequency scaling factor
    freq_factor: f64,
    /// Voltage scaling factor (quadratic)
    voltage_factor: f64,
}

impl KernelPowerModel {
    /// Create model calibrated for A100
    pub fn a100() -> Self {
        Self {
            base_power: 50.0,    // Idle baseline
            sm_coeff: 200.0,     // 200W at full SM utilization
            memory_coeff: 80.0,  // 80W at full memory bandwidth
            tensor_coeff: 150.0, // 150W at full tensor utilization
            freq_factor: 1.0,    // Normalized to base frequency
            voltage_factor: 1.0, // Normalized to base voltage
        }
    }

    /// Estimate power for given activity
    pub fn estimate_power(
        &self,
        activity: &WorkloadActivity,
        freq_mhz: u32,
        voltage_mv: u32,
    ) -> f64 {
        // Normalize frequency and voltage
        let freq_norm = freq_mhz as f64 / 1000.0;
        let voltage_norm = voltage_mv as f64 / 1000.0;

        // P = P_static + P_dynamic
        // P_dynamic = C * V^2 * f * activity
        let static_power = self.base_power;

        let sm_power = self.sm_coeff * activity.sm_utilization * freq_norm * voltage_norm.powi(2);
        let memory_power = self.memory_coeff * activity.memory_utilization;
        let tensor_power =
            self.tensor_coeff * activity.tensor_utilization * freq_norm * voltage_norm.powi(2);

        static_power + sm_power + memory_power + tensor_power
    }

    /// Estimate energy for kernel execution
    pub fn estimate_energy(
        &self,
        activity: &WorkloadActivity,
        freq_mhz: u32,
        voltage_mv: u32,
        duration_ms: f64,
    ) -> f64 {
        let power = self.estimate_power(activity, freq_mhz, voltage_mv);
        power * (duration_ms / 1000.0) // Energy in joules
    }
}

/// Thermal model for GPU
#[derive(Debug)]
pub struct ThermalModel {
    /// Current temperature in Celsius
    pub temperature_c: f64,
    /// Ambient temperature
    pub ambient_c: f64,
    /// Thermal resistance (C/W)
    pub thermal_resistance: f64,
    /// Thermal capacitance (J/C)
    pub thermal_capacitance: f64,
    /// Temperature target
    pub target_temp_c: f64,
    /// Shutdown temperature
    pub shutdown_temp_c: f64,
    /// Throttle temperature
    pub throttle_temp_c: f64,
}

impl ThermalModel {
    /// Create thermal model for A100
    pub fn a100() -> Self {
        Self {
            temperature_c: 35.0,
            ambient_c: 25.0,
            thermal_resistance: 0.15,   // C/W
            thermal_capacitance: 500.0, // J/C
            target_temp_c: 75.0,
            shutdown_temp_c: 95.0,
            throttle_temp_c: 83.0,
        }
    }

    /// Update temperature based on power
    pub fn update(&mut self, power_watts: f64, dt_seconds: f64) {
        // Steady-state temperature
        let steady_state = self.ambient_c + power_watts * self.thermal_resistance;

        // RC thermal model: T(t) = T_ss + (T_0 - T_ss) * e^(-t/RC)
        let tau = self.thermal_resistance * self.thermal_capacitance;
        let alpha = dt_seconds / tau;

        self.temperature_c = steady_state + (self.temperature_c - steady_state) * (-alpha).exp();
    }

    /// Check if thermal throttling needed
    pub fn needs_throttling(&self) -> bool {
        self.temperature_c > self.throttle_temp_c
    }

    /// Check if shutdown required
    pub fn needs_shutdown(&self) -> bool {
        self.temperature_c > self.shutdown_temp_c
    }

    /// Get thermal margin (degrees below throttle)
    pub fn thermal_margin(&self) -> f64 {
        self.throttle_temp_c - self.temperature_c
    }
}

/// Complete power management system
#[derive(Debug)]
pub struct PowerManagementSystem {
    /// Power specification
    pub spec: PowerSpec,
    /// Power breakdown by domain
    pub breakdown: GpuPowerBreakdown,
    /// DVFS controller
    pub dvfs: DvfsController,
    /// Thermal model
    pub thermal: ThermalModel,
    /// Power model
    pub power_model: KernelPowerModel,
    /// Current workload type
    pub workload: WorkloadType,
    /// Simulation timestamp
    timestamp: u64,
}

impl PowerManagementSystem {
    /// Create power management for A100 SXM
    pub fn new_a100_sxm() -> Self {
        Self {
            spec: PowerSpec::a100_sxm(),
            breakdown: GpuPowerBreakdown::a100(),
            dvfs: DvfsController::new_a100(),
            thermal: ThermalModel::a100(),
            power_model: KernelPowerModel::a100(),
            workload: WorkloadType::Idle,
            timestamp: 0,
        }
    }

    /// Set workload type
    pub fn set_workload(&mut self, workload: WorkloadType) {
        self.workload = workload;
        self.breakdown.gate_unused_domains(workload);
    }

    /// Update power management state
    pub fn tick(&mut self, activity: &WorkloadActivity, dt_us: u64) {
        self.timestamp += dt_us;
        let dt_seconds = dt_us as f64 / 1_000_000.0;

        // Update power breakdown
        self.breakdown.set_compute_utilization(
            activity.sm_utilization,
            activity.memory_utilization,
            activity.tensor_utilization,
        );

        let current_power = self.breakdown.total_power();

        // Update thermal model
        self.thermal.update(current_power, dt_seconds);

        // Update DVFS based on conditions
        let sm_util = activity.sm_utilization;
        self.dvfs.update(
            self.thermal.temperature_c,
            current_power,
            self.spec.tdp_watts,
            sm_util,
            self.timestamp,
        );
    }

    /// Get current power consumption
    pub fn current_power(&self) -> f64 {
        self.breakdown.total_power()
    }

    /// Get current temperature
    pub fn current_temperature(&self) -> f64 {
        self.thermal.temperature_c
    }

    /// Get current frequency
    pub fn current_frequency(&self) -> u32 {
        self.dvfs.current_freq_mhz
    }

    /// Check if throttling
    pub fn is_throttling(&self) -> bool {
        self.thermal.needs_throttling() || self.current_power() > self.spec.tdp_watts * 0.95
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_spec_a100() {
        let spec = PowerSpec::a100_pcie();
        assert_eq!(spec.tdp_watts, 250.0);
        assert_eq!(spec.boost_clock_mhz, 1410);
    }

    #[test]
    fn test_power_spec_h100() {
        let spec = PowerSpec::h100_sxm();
        assert_eq!(spec.tdp_watts, 700.0);
        assert!(spec.boost_clock_mhz > 1800);
    }

    #[test]
    fn test_power_connectors() {
        assert_eq!(PowerConnectors::SlotPowered.max_power(), 75.0);
        assert_eq!(PowerConnectors::Dual8Pin.max_power(), 375.0);
        assert_eq!(PowerConnectors::SxmModule.max_power(), 700.0);
    }

    #[test]
    fn test_power_domain() {
        let mut domain = PowerDomain::new("SM", 200.0);

        domain.set_utilization(0.0);
        assert!(domain.current_power_watts > 0.0); // Static power

        domain.set_utilization(1.0);
        assert!(domain.current_power_watts > 150.0);

        domain.power_gate();
        assert_eq!(domain.current_power_watts, 0.0);
    }

    #[test]
    fn test_gpu_power_breakdown() {
        let mut breakdown = GpuPowerBreakdown::a100();

        breakdown.set_compute_utilization(0.9, 0.8, 0.7);
        let power = breakdown.total_power();

        assert!(power > 200.0);
        assert!(power < 600.0);
    }

    #[test]
    fn test_dvfs_controller() {
        let mut dvfs = DvfsController::new_a100();

        // Initial state
        assert_eq!(dvfs.current_freq_mhz, 765);

        // Request higher frequency
        dvfs.request_frequency(1200, 1000);

        // Thermal throttle
        dvfs.update(85.0, 300.0, 400.0, 0.9, 2000);
        assert!(dvfs.current_freq_mhz < 1200);
    }

    #[test]
    fn test_dvfs_boost() {
        let mut dvfs = DvfsController::new_a100();
        dvfs.set_policy(DvfsPolicy::Performance);

        // Good conditions for boost
        dvfs.update(60.0, 200.0, 400.0, 0.9, 1000);
        assert_eq!(dvfs.current_freq_mhz, 1410); // Max boost
    }

    #[test]
    fn test_workload_activity() {
        let idle = WorkloadActivity::idle();
        assert_eq!(idle.sm_utilization, 0.0);

        let gemm = WorkloadActivity::gemm();
        assert!(gemm.tensor_utilization > 0.9);
    }

    #[test]
    fn test_power_budget_manager() {
        let mut manager = PowerBudgetManager::dgx_a100();

        assert_eq!(manager.total_budget_watts, 6500.0);

        // Update power for some GPUs
        manager.update_gpu_power(0, 350.0);
        manager.update_gpu_power(1, 380.0);

        assert!(manager.within_budget());
        assert!(manager.remaining_budget() > 5000.0);
    }

    #[test]
    fn test_power_budget_rebalance() {
        let mut manager = PowerBudgetManager::new(2000.0, 4, 200.0);

        let mut weights = HashMap::new();
        weights.insert(0, 2.0); // High priority
        weights.insert(1, 1.0);
        weights.insert(2, 1.0);
        weights.insert(3, 0.5);

        manager.set_policy(PowerAllocationPolicy::Proportional);
        manager.rebalance(&weights);

        // GPU 0 should have highest limit
        let limit_0 = manager.get_limit(0).unwrap();
        let limit_3 = manager.get_limit(3).unwrap();
        assert!(limit_0 > limit_3);
    }

    #[test]
    fn test_kernel_power_model() {
        let model = KernelPowerModel::a100();

        let idle = WorkloadActivity::idle();
        let idle_power = model.estimate_power(&idle, 765, 850);
        assert!(idle_power < 100.0);

        let gemm = WorkloadActivity::gemm();
        let gemm_power = model.estimate_power(&gemm, 1410, 1100);
        assert!(gemm_power > 300.0);
    }

    #[test]
    fn test_thermal_model() {
        let mut thermal = ThermalModel::a100();

        // High power should increase temperature
        for _ in 0..1000 {
            thermal.update(400.0, 0.1);
        }

        // Temperature should have risen from initial 35°C
        assert!(thermal.temperature_c > 40.0);

        // Check throttling
        thermal.temperature_c = 85.0;
        assert!(thermal.needs_throttling());
    }

    #[test]
    fn test_power_management_system() {
        let mut pms = PowerManagementSystem::new_a100_sxm();

        // Idle state
        pms.set_workload(WorkloadType::Idle);
        let idle_activity = WorkloadActivity::idle();
        pms.tick(&idle_activity, 1000);

        let idle_power = pms.current_power();
        assert!(idle_power < 100.0);

        // Compute workload
        pms.set_workload(WorkloadType::Compute);
        let compute_activity = WorkloadActivity::compute_heavy();
        for _ in 0..100 {
            pms.tick(&compute_activity, 1000);
        }

        let compute_power = pms.current_power();
        assert!(compute_power > idle_power);
    }

    #[test]
    fn test_power_management_throttling() {
        let mut pms = PowerManagementSystem::new_a100_sxm();
        pms.set_workload(WorkloadType::Compute);

        // Force high temperature
        pms.thermal.temperature_c = 85.0;

        let activity = WorkloadActivity::compute_heavy();
        pms.tick(&activity, 1000);

        assert!(pms.is_throttling());
    }
}
