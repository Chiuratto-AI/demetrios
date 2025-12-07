//! Hardware Performance Counters and Profiling
//!
//! SCIENTIFIC HONESTY:
//! - Counters have sampling overhead
//! - Some metrics are derived, not directly measured
//! - Multiplexing required for many counters introduces error
//! - Counter accuracy varies by type
//!
//! Sources:
//! - NVIDIA CUPTI documentation
//! - "CUDA Binary Utilities" (NVIDIA)
//! - "GPU Performance Analysis" (NVIDIA GTC talks)
//! - Empirical validation studies

use std::collections::HashMap;

// ============================================================================
// COUNTER DEFINITIONS
// ============================================================================

/// Performance counter specification
#[derive(Debug, Clone)]
pub struct CounterSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// Is this directly measured or derived?
    pub measurement_type: MeasurementType,
    /// Accuracy level
    pub accuracy: Accuracy,
    /// Overhead when collecting
    pub overhead: Overhead,
    /// Available on which architectures
    pub availability: Vec<&'static str>,
    /// What this counter actually means (honest assessment)
    pub honest_interpretation: &'static str,
}

/// How the counter is measured
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementType {
    /// Directly measured by hardware
    Direct,
    /// Derived from other counters
    Derived,
    /// Sampled (not every event counted)
    Sampled,
    /// Event-based (count of specific events)
    Event,
}

/// Counter accuracy level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accuracy {
    /// Exact count
    Exact,
    /// Very accurate (<1% error)
    High,
    /// Reasonably accurate (1-5% error)
    Medium,
    /// Approximate (5-20% error)
    Low,
    /// Rough estimate (>20% error possible)
    Estimate,
}

/// Counter collection overhead
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overhead {
    /// No measurable overhead
    None,
    /// <1% slowdown
    Minimal,
    /// 1-5% slowdown
    Low,
    /// 5-20% slowdown
    Medium,
    /// >20% slowdown
    High,
    /// Requires serialization
    Serializing,
}

/// Standard GPU performance counters
pub fn standard_counters() -> Vec<CounterSpec> {
    vec![
        // ============================================================
        // INSTRUCTION COUNTERS (Generally Accurate)
        // ============================================================
        CounterSpec {
            name: "smsp__inst_executed.sum",
            description: "Total instructions executed",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::Exact,
            overhead: Overhead::Minimal,
            availability: vec!["Pascal", "Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Accurate count of warp-instructions. \
                                   Note: one warp-instruction = 32 thread-instructions.",
        },
        CounterSpec {
            name: "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
            description: "FP32 instructions executed",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::Exact,
            overhead: Overhead::Minimal,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Accurate count of FP32 ops. Useful for FLOP calculation.",
        },
        CounterSpec {
            name: "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
            description: "FP64 instructions executed",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::Exact,
            overhead: Overhead::Minimal,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Accurate count of FP64 ops.",
        },
        CounterSpec {
            name: "smsp__sass_thread_inst_executed_op_tensor_op.sum",
            description: "Tensor Core instructions",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::Exact,
            overhead: Overhead::Minimal,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Counts Tensor Core MMA instructions. \
                                   Each instruction performs many FMAs.",
        },
        // ============================================================
        // MEMORY COUNTERS (Mostly Accurate)
        // ============================================================
        CounterSpec {
            name: "dram__bytes_read.sum",
            description: "Bytes read from DRAM",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::High,
            overhead: Overhead::Minimal,
            availability: vec!["Pascal", "Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Accurate count of DRAM read bytes. \
                                   Includes cache line fills.",
        },
        CounterSpec {
            name: "dram__bytes_write.sum",
            description: "Bytes written to DRAM",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::High,
            overhead: Overhead::Minimal,
            availability: vec!["Pascal", "Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Accurate count of DRAM write bytes. \
                                   Includes writebacks.",
        },
        CounterSpec {
            name: "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            description: "Global load sectors at L1",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::High,
            overhead: Overhead::Low,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Sectors (32-byte chunks) requested from L1. \
                                   High value relative to achieved indicates poor coalescing.",
        },
        CounterSpec {
            name: "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
            description: "L2 read hits",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::High,
            overhead: Overhead::Minimal,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "L2 cache hits. High hit rate = good data reuse.",
        },
        CounterSpec {
            name: "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            description: "L2 read misses",
            measurement_type: MeasurementType::Direct,
            accuracy: Accuracy::High,
            overhead: Overhead::Minimal,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "L2 cache misses. These go to DRAM.",
        },
        // ============================================================
        // OCCUPANCY AND UTILIZATION (Mostly Derived)
        // ============================================================
        CounterSpec {
            name: "sm__warps_active.avg.pct_of_peak_sustained_active",
            description: "Achieved occupancy",
            measurement_type: MeasurementType::Derived,
            accuracy: Accuracy::Medium,
            overhead: Overhead::Low,
            availability: vec!["Pascal", "Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Average warps active as fraction of maximum. \
                                   CAVEAT: High occupancy != high performance. \
                                   Some kernels run best at lower occupancy.",
        },
        CounterSpec {
            name: "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            description: "Memory throughput percentage",
            measurement_type: MeasurementType::Derived,
            accuracy: Accuracy::Medium,
            overhead: Overhead::Low,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "DERIVED metric. Compares actual to 'peak sustained' \
                                   which is itself an estimate. Use with caution.",
        },
        // ============================================================
        // STALL REASONS (Useful but Approximate)
        // ============================================================
        CounterSpec {
            name: "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            description: "Stalls due to long scoreboard",
            measurement_type: MeasurementType::Sampled,
            accuracy: Accuracy::Medium,
            overhead: Overhead::Medium,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Fraction of issue cycles stalled waiting for \
                                   long-latency operations (typically memory). \
                                   High value indicates memory-bound behavior.",
        },
        CounterSpec {
            name: "smsp__warps_issue_stalled_wait_per_issue_active.ratio",
            description: "Stalls due to __syncthreads or barriers",
            measurement_type: MeasurementType::Sampled,
            accuracy: Accuracy::Medium,
            overhead: Overhead::Medium,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Fraction of issue cycles stalled at barriers. \
                                   High value indicates load imbalance between warps.",
        },
        CounterSpec {
            name: "smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
            description: "Stalls due to short scoreboard",
            measurement_type: MeasurementType::Sampled,
            accuracy: Accuracy::Medium,
            overhead: Overhead::Medium,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "Stalls waiting for short-latency ops (shared memory, \
                                   math). High value may indicate shared memory bank conflicts.",
        },
        // ============================================================
        // PROBLEMATIC METRICS (Use With Great Caution)
        // ============================================================
        CounterSpec {
            name: "sm__cycles_elapsed.avg",
            description: "Average SM cycles",
            measurement_type: MeasurementType::Derived,
            accuracy: Accuracy::Low,
            overhead: Overhead::Low,
            availability: vec!["Pascal", "Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "PROBLEMATIC: Different SMs may have different cycle counts. \
                                   Clock frequency varies with DVFS. \
                                   Use wall-clock time instead for performance measurement.",
        },
        CounterSpec {
            name: "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            description: "SM throughput percentage",
            measurement_type: MeasurementType::Derived,
            accuracy: Accuracy::Low,
            overhead: Overhead::Low,
            availability: vec!["Volta", "Turing", "Ampere", "Hopper"],
            honest_interpretation: "HIGHLY DERIVED: Combines multiple factors. \
                                   'Peak sustained' is a theoretical maximum that \
                                   may not be achievable for any real workload.",
        },
    ]
}

// ============================================================================
// PROFILING STRATEGY
// ============================================================================

/// Profiling goal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingGoal {
    IdentifyBottleneck,
    OptimizeMemory,
    OptimizeCompute,
    DebugCorrectness,
    MeasurePowerEfficiency,
}

/// Profiling strategy advisor
#[derive(Debug)]
pub struct ProfilingAdvisor;

impl ProfilingAdvisor {
    /// Get recommended profiling approach for a goal
    pub fn recommend(goal: ProfilingGoal) -> ProfilingRecommendation {
        match goal {
            ProfilingGoal::IdentifyBottleneck => ProfilingRecommendation {
                approach: "Start with high-level metrics, drill down as needed".to_string(),
                counters: vec![
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed".to_string(),
                    "sm__warps_active.avg.pct_of_peak_sustained_active".to_string(),
                    "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio".to_string(),
                ],
                overhead_estimate: "5-10% slowdown",
                caveats: vec![
                    "High-level metrics can be misleading".to_string(),
                    "Multiple passes may be needed for detailed counters".to_string(),
                ],
                alternative: Some("Use Nsight Compute guided analysis first".to_string()),
            },
            ProfilingGoal::OptimizeMemory => ProfilingRecommendation {
                approach: "Focus on memory transaction efficiency".to_string(),
                counters: vec![
                    "dram__bytes_read.sum".to_string(),
                    "dram__bytes_write.sum".to_string(),
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum".to_string(),
                    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum".to_string(),
                ],
                overhead_estimate: "2-5% slowdown",
                caveats: vec![
                    "Coalescing efficiency requires comparing requested vs achieved".to_string(),
                    "L2 hit rate depends on data size and access pattern".to_string(),
                ],
                alternative: None,
            },
            ProfilingGoal::OptimizeCompute => ProfilingRecommendation {
                approach: "Focus on instruction throughput and utilization".to_string(),
                counters: vec![
                    "smsp__inst_executed.sum".to_string(),
                    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum".to_string(),
                    "smsp__sass_thread_inst_executed_op_tensor_op.sum".to_string(),
                ],
                overhead_estimate: "1-3% slowdown",
                caveats: vec![
                    "Instruction counts don't account for divergence".to_string(),
                    "Tensor Core utilization is affected by data layout".to_string(),
                ],
                alternative: None,
            },
            ProfilingGoal::DebugCorrectness => ProfilingRecommendation {
                approach: "Use memory checker and race detection".to_string(),
                counters: vec![], // Use compute-sanitizer instead
                overhead_estimate: "10-100x slowdown",
                caveats: vec![
                    "Memory checking has extreme overhead".to_string(),
                    "Some race conditions may not be detected".to_string(),
                    "Results may differ between debug and release builds".to_string(),
                ],
                alternative: Some("Use compute-sanitizer --tool memcheck".to_string()),
            },
            ProfilingGoal::MeasurePowerEfficiency => ProfilingRecommendation {
                approach: "Combine performance counters with power monitoring".to_string(),
                counters: vec![
                    "smsp__inst_executed.sum".to_string(),
                    "dram__bytes.sum".to_string(),
                ],
                overhead_estimate: "1-2% slowdown",
                caveats: vec![
                    "Power measurement has ~5W accuracy".to_string(),
                    "DVFS makes cycle-based metrics unreliable".to_string(),
                    "Short kernels have high measurement variance".to_string(),
                ],
                alternative: Some("Use nvidia-smi or NVML for power monitoring".to_string()),
            },
        }
    }

    /// What profiling CAN'T tell you
    pub fn limitations() -> Vec<ProfilingLimitation> {
        vec![
            ProfilingLimitation {
                limitation: "Why a specific warp stalled".to_string(),
                explanation: "Stall reasons are aggregated across all warps. \
                             Cannot identify specific problematic code paths."
                    .to_string(),
                workaround: "Use printf debugging or custom instrumentation".to_string(),
            },
            ProfilingLimitation {
                limitation: "Exact memory bandwidth utilization".to_string(),
                explanation: "Counters measure transactions, not time-based bandwidth. \
                             'Percentage of peak' is derived and approximate."
                    .to_string(),
                workaround: "Calculate bandwidth manually: bytes / time".to_string(),
            },
            ProfilingLimitation {
                limitation: "Per-thread behavior".to_string(),
                explanation: "All counters are warp-level or higher. Cannot see \
                             individual thread performance."
                    .to_string(),
                workaround: "Thread-level analysis requires simulation".to_string(),
            },
            ProfilingLimitation {
                limitation: "Memory access patterns".to_string(),
                explanation: "Counters show aggregate statistics, not actual addresses. \
                             Cannot directly see access patterns."
                    .to_string(),
                workaround: "Use memory checker or custom instrumentation".to_string(),
            },
            ProfilingLimitation {
                limitation: "Cache line state".to_string(),
                explanation: "Cannot see which cache lines are valid/dirty. \
                             Only aggregate hit/miss rates."
                    .to_string(),
                workaround: "Use trace-based simulation for detailed cache analysis".to_string(),
            },
            ProfilingLimitation {
                limitation: "Inter-kernel effects".to_string(),
                explanation: "Counters reset between kernels. Cannot see how \
                             one kernel affects another's cache state."
                    .to_string(),
                workaround: "Profile kernel sequences together".to_string(),
            },
        ]
    }
}

/// Profiling recommendation
#[derive(Debug)]
pub struct ProfilingRecommendation {
    pub approach: String,
    pub counters: Vec<String>,
    pub overhead_estimate: &'static str,
    pub caveats: Vec<String>,
    pub alternative: Option<String>,
}

/// Profiling limitation
#[derive(Debug)]
pub struct ProfilingLimitation {
    pub limitation: String,
    pub explanation: String,
    pub workaround: String,
}

// ============================================================================
// COUNTER INTERPRETATION
// ============================================================================

/// Performance counter interpreter
#[derive(Debug)]
pub struct CounterInterpreter {
    counter_specs: HashMap<String, CounterSpec>,
}

impl CounterInterpreter {
    pub fn new() -> Self {
        let counter_specs = standard_counters()
            .into_iter()
            .map(|c| (c.name.to_string(), c))
            .collect();

        Self { counter_specs }
    }

    /// Interpret a set of counter values
    pub fn interpret(&self, values: &HashMap<String, f64>) -> Interpretation {
        let mut findings = Vec::new();
        let mut confidence = Confidence::High;

        // Check memory throughput
        if let Some(&mem_throughput) =
            values.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        {
            if mem_throughput > 80.0 {
                findings.push(Finding {
                    category: "Memory".to_string(),
                    finding: "Memory-bound: >80% memory throughput".to_string(),
                    confidence: Confidence::Medium,
                    action: "Optimize memory access patterns or reduce data movement".to_string(),
                });
            } else if mem_throughput < 30.0 {
                findings.push(Finding {
                    category: "Memory".to_string(),
                    finding: "Low memory utilization: <30%".to_string(),
                    confidence: Confidence::Medium,
                    action: "May be compute-bound or have inefficient access patterns".to_string(),
                });
            }
            confidence = Confidence::Medium; // This metric is derived
        }

        // Check occupancy
        if let Some(&occupancy) = values.get("sm__warps_active.avg.pct_of_peak_sustained_active") {
            if occupancy < 25.0 {
                findings.push(Finding {
                    category: "Occupancy".to_string(),
                    finding: format!("Low occupancy: {:.1}%", occupancy),
                    confidence: Confidence::High,
                    action: "Check register usage, shared memory, and block size".to_string(),
                });
            }
        }

        // Check stall reasons
        if let Some(&long_scoreboard) =
            values.get("smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio")
        {
            if long_scoreboard > 0.5 {
                findings.push(Finding {
                    category: "Stalls".to_string(),
                    finding: format!("High memory stalls: {:.0}%", long_scoreboard * 100.0),
                    confidence: Confidence::Medium,
                    action: "Increase arithmetic intensity or prefetch data".to_string(),
                });
            }
        }

        // Check L2 hit rate
        if let (Some(&hits), Some(&misses)) = (
            values.get("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"),
            values.get("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"),
        ) {
            let total = hits + misses;
            if total > 0.0 {
                let hit_rate = hits / total;
                if hit_rate < 0.5 {
                    findings.push(Finding {
                        category: "Cache".to_string(),
                        finding: format!("Low L2 hit rate: {:.0}%", hit_rate * 100.0),
                        confidence: Confidence::High,
                        action: "Improve data locality or reduce working set size".to_string(),
                    });
                }
            }
        }

        Interpretation {
            findings,
            overall_confidence: confidence,
            honest_caveats: vec![
                "Counter interpretation is heuristic, not definitive".to_string(),
                "Multiple factors can cause similar symptoms".to_string(),
                "Always validate with targeted experiments".to_string(),
            ],
        }
    }

    /// Get counter specification
    pub fn get_spec(&self, name: &str) -> Option<&CounterSpec> {
        self.counter_specs.get(name)
    }

    /// List all available counters
    pub fn list_counters(&self) -> Vec<&str> {
        self.counter_specs.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for CounterInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// Interpretation confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,
    Medium,
    Low,
    Uncertain,
}

/// A finding from counter interpretation
#[derive(Debug, Clone)]
pub struct Finding {
    pub category: String,
    pub finding: String,
    pub confidence: Confidence,
    pub action: String,
}

/// Complete interpretation result
#[derive(Debug)]
pub struct Interpretation {
    pub findings: Vec<Finding>,
    pub overall_confidence: Confidence,
    pub honest_caveats: Vec<String>,
}

// ============================================================================
// ROOFLINE MODEL
// ============================================================================

/// Roofline model for performance analysis
///
/// SCIENTIFIC HONESTY:
/// The roofline model is a SIMPLIFIED view of performance.
/// Real performance is affected by many factors not captured here.
#[derive(Debug, Clone)]
pub struct RooflineModel {
    /// Peak compute throughput (GFLOPS)
    pub peak_gflops: f64,
    /// Peak memory bandwidth (GB/s)
    pub peak_bandwidth_gbps: f64,
    /// Ridge point (FLOPS/byte)
    pub ridge_point: f64,
}

impl RooflineModel {
    /// Create roofline model for A100
    pub fn a100() -> Self {
        let peak_gflops = 19500.0; // FP32
        let peak_bandwidth = 2039.0;
        Self {
            peak_gflops,
            peak_bandwidth_gbps: peak_bandwidth,
            ridge_point: peak_gflops / peak_bandwidth,
        }
    }

    /// Create roofline model for H100
    pub fn h100() -> Self {
        let peak_gflops = 51200.0; // FP32 with Tensor Cores
        let peak_bandwidth = 3350.0;
        Self {
            peak_gflops,
            peak_bandwidth_gbps: peak_bandwidth,
            ridge_point: peak_gflops / peak_bandwidth,
        }
    }

    /// Compute attainable performance for given arithmetic intensity
    pub fn attainable_gflops(&self, arithmetic_intensity: f64) -> f64 {
        if arithmetic_intensity < self.ridge_point {
            // Memory-bound region
            arithmetic_intensity * self.peak_bandwidth_gbps
        } else {
            // Compute-bound region
            self.peak_gflops
        }
    }

    /// Determine if workload is memory or compute bound
    pub fn bottleneck(&self, arithmetic_intensity: f64) -> BottleneckType {
        if arithmetic_intensity < self.ridge_point * 0.9 {
            BottleneckType::MemoryBound {
                headroom: (self.ridge_point - arithmetic_intensity) / self.ridge_point,
            }
        } else if arithmetic_intensity > self.ridge_point * 1.1 {
            BottleneckType::ComputeBound {
                headroom: (arithmetic_intensity - self.ridge_point) / arithmetic_intensity,
            }
        } else {
            BottleneckType::Balanced
        }
    }

    /// Calculate arithmetic intensity needed for given efficiency
    pub fn required_intensity_for_efficiency(&self, target_efficiency: f64) -> f64 {
        // At ridge point, we achieve peak compute
        // Below ridge point, efficiency = AI / ridge_point
        self.ridge_point * target_efficiency
    }
}

/// Bottleneck type from roofline analysis
#[derive(Debug, Clone)]
pub enum BottleneckType {
    MemoryBound { headroom: f64 },
    ComputeBound { headroom: f64 },
    Balanced,
}

// ============================================================================
// COUNTER COLLECTION SIMULATION
// ============================================================================

/// Simulated counter collector for testing
#[derive(Debug)]
pub struct SimulatedCounterCollector {
    values: HashMap<String, f64>,
}

impl SimulatedCounterCollector {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Set a counter value
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }

    /// Get a counter value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }

    /// Get all values
    pub fn all_values(&self) -> &HashMap<String, f64> {
        &self.values
    }

    /// Simulate memory-bound kernel
    pub fn simulate_memory_bound(&mut self) {
        self.set(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            85.0,
        );
        self.set("sm__warps_active.avg.pct_of_peak_sustained_active", 60.0);
        self.set(
            "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            0.6,
        );
        self.set("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", 1000.0);
        self.set("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", 500.0);
    }

    /// Simulate compute-bound kernel
    pub fn simulate_compute_bound(&mut self) {
        self.set(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            30.0,
        );
        self.set("sm__warps_active.avg.pct_of_peak_sustained_active", 90.0);
        self.set(
            "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            0.1,
        );
        self.set("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", 900.0);
        self.set("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", 100.0);
    }

    /// Simulate low-occupancy kernel
    pub fn simulate_low_occupancy(&mut self) {
        self.set(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            40.0,
        );
        self.set("sm__warps_active.avg.pct_of_peak_sustained_active", 15.0);
        self.set(
            "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
            0.3,
        );
    }
}

impl Default for SimulatedCounterCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_specs() {
        let counters = standard_counters();

        // Should have a reasonable number of counters
        assert!(counters.len() > 10);

        // Check that all have honest interpretations
        for counter in &counters {
            assert!(!counter.honest_interpretation.is_empty());
        }
    }

    #[test]
    fn test_profiling_recommendations() {
        let rec = ProfilingAdvisor::recommend(ProfilingGoal::IdentifyBottleneck);

        assert!(!rec.counters.is_empty());
        assert!(!rec.caveats.is_empty());
    }

    #[test]
    fn test_profiling_limitations() {
        let limits = ProfilingAdvisor::limitations();

        assert!(!limits.is_empty());
        for limit in &limits {
            assert!(!limit.workaround.is_empty());
        }
    }

    #[test]
    fn test_counter_interpretation() {
        let interpreter = CounterInterpreter::new();

        let mut values = HashMap::new();
        values.insert(
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed".to_string(),
            85.0,
        );
        values.insert(
            "sm__warps_active.avg.pct_of_peak_sustained_active".to_string(),
            50.0,
        );

        let interp = interpreter.interpret(&values);

        // Should identify memory-bound behavior
        assert!(interp.findings.iter().any(|f| f.category == "Memory"));
        assert!(!interp.honest_caveats.is_empty());
    }

    #[test]
    fn test_low_occupancy_detection() {
        let interpreter = CounterInterpreter::new();

        let mut values = HashMap::new();
        values.insert(
            "sm__warps_active.avg.pct_of_peak_sustained_active".to_string(),
            15.0,
        );

        let interp = interpreter.interpret(&values);

        // Should identify low occupancy
        assert!(interp.findings.iter().any(|f| f.category == "Occupancy"));
    }

    #[test]
    fn test_roofline_model() {
        let roofline = RooflineModel::a100();

        // Low arithmetic intensity = memory bound
        let ai_low = 1.0; // 1 FLOP/byte
        assert!(roofline.attainable_gflops(ai_low) < roofline.peak_gflops);
        assert!(matches!(
            roofline.bottleneck(ai_low),
            BottleneckType::MemoryBound { .. }
        ));

        // High arithmetic intensity = compute bound
        let ai_high = 100.0; // 100 FLOPS/byte
        assert!((roofline.attainable_gflops(ai_high) - roofline.peak_gflops).abs() < 0.1);
        assert!(matches!(
            roofline.bottleneck(ai_high),
            BottleneckType::ComputeBound { .. }
        ));
    }

    #[test]
    fn test_ridge_point() {
        let a100 = RooflineModel::a100();
        let h100 = RooflineModel::h100();

        // H100 has higher ridge point due to more compute
        assert!(h100.ridge_point > a100.ridge_point);
    }

    #[test]
    fn test_simulated_collector() {
        let mut collector = SimulatedCounterCollector::new();
        collector.simulate_memory_bound();

        let interpreter = CounterInterpreter::new();
        let interp = interpreter.interpret(collector.all_values());

        // Should identify memory-bound
        assert!(interp.findings.iter().any(|f| f.category == "Memory"));
    }

    #[test]
    fn test_counter_lookup() {
        let interpreter = CounterInterpreter::new();

        let spec = interpreter.get_spec("smsp__inst_executed.sum");
        assert!(spec.is_some());
        assert!(matches!(spec.unwrap().accuracy, Accuracy::Exact));
    }
}
