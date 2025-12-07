//! Floating Point Precision Analysis
//!
//! SCIENTIFIC HONESTY:
//! - These are theoretical worst-case error bounds
//! - Actual errors depend on data distribution
//! - "Works in practice" does not equal "mathematically sound"
//! - Loss scaling hides problems, doesn't solve them
//!
//! Sources:
//! - IEEE 754-2019 standard
//! - "What Every Computer Scientist Should Know About Floating-Point"
//! - "Mixed Precision Training" (Micikevicius et al., 2018)
//! - Empirical measurements

use std::collections::HashMap;

// ============================================================================
// FLOATING POINT FORMAT SPECIFICATIONS
// ============================================================================

/// Floating point format specification
#[derive(Debug, Clone)]
pub struct FloatFormat {
    pub name: &'static str,
    /// Total bits
    pub bits: u32,
    /// Sign bits (always 1)
    pub sign_bits: u32,
    /// Exponent bits
    pub exponent_bits: u32,
    /// Mantissa bits (explicit)
    pub mantissa_bits: u32,
    /// Exponent bias
    pub bias: i32,
    /// Maximum representable value
    pub max_value: f64,
    /// Minimum positive normal value
    pub min_normal: f64,
    /// Machine epsilon (smallest x such that 1 + x != 1)
    pub epsilon: f64,
    /// Can represent subnormals?
    pub has_subnormals: bool,
}

impl FloatFormat {
    /// IEEE 754 double precision (FP64)
    pub fn fp64() -> Self {
        Self {
            name: "FP64",
            bits: 64,
            sign_bits: 1,
            exponent_bits: 11,
            mantissa_bits: 52,
            bias: 1023,
            max_value: 1.7976931348623157e308,
            min_normal: 2.2250738585072014e-308,
            epsilon: 2.220446049250313e-16, // 2^-52
            has_subnormals: true,
        }
    }

    /// IEEE 754 single precision (FP32)
    pub fn fp32() -> Self {
        Self {
            name: "FP32",
            bits: 32,
            sign_bits: 1,
            exponent_bits: 8,
            mantissa_bits: 23,
            bias: 127,
            max_value: 3.4028235e38,
            min_normal: 1.17549435e-38,
            epsilon: 1.1920929e-7, // 2^-23
            has_subnormals: true,
        }
    }

    /// IEEE 754 half precision (FP16)
    pub fn fp16() -> Self {
        Self {
            name: "FP16",
            bits: 16,
            sign_bits: 1,
            exponent_bits: 5,
            mantissa_bits: 10,
            bias: 15,
            max_value: 65504.0,
            min_normal: 6.103515625e-5,
            epsilon: 9.765625e-4, // 2^-10
            has_subnormals: true,
        }
    }

    /// Brain Float 16 (BF16)
    pub fn bf16() -> Self {
        Self {
            name: "BF16",
            bits: 16,
            sign_bits: 1,
            exponent_bits: 8,
            mantissa_bits: 7,
            bias: 127,
            max_value: 3.3895314e38,
            min_normal: 1.17549435e-38,
            epsilon: 7.8125e-3, // 2^-7
            has_subnormals: false,
        }
    }

    /// TF32 (Tensor Float 32) - used internally by Tensor Cores
    pub fn tf32() -> Self {
        Self {
            name: "TF32",
            bits: 19, // Stored as 32-bit with reduced mantissa
            sign_bits: 1,
            exponent_bits: 8,
            mantissa_bits: 10,
            bias: 127,
            max_value: 3.4028235e38,
            min_normal: 1.17549435e-38,
            epsilon: 9.765625e-4, // 2^-10, same as FP16
            has_subnormals: true,
        }
    }

    /// FP8 E4M3 format (for inference)
    pub fn fp8_e4m3() -> Self {
        Self {
            name: "FP8_E4M3",
            bits: 8,
            sign_bits: 1,
            exponent_bits: 4,
            mantissa_bits: 3,
            bias: 7,
            max_value: 448.0,
            min_normal: 0.015625, // 2^-6
            epsilon: 0.125,       // 2^-3
            has_subnormals: true,
        }
    }

    /// FP8 E5M2 format (wider range)
    pub fn fp8_e5m2() -> Self {
        Self {
            name: "FP8_E5M2",
            bits: 8,
            sign_bits: 1,
            exponent_bits: 5,
            mantissa_bits: 2,
            bias: 15,
            max_value: 57344.0,
            min_normal: 6.103515625e-5, // 2^-14
            epsilon: 0.25,              // 2^-2
            has_subnormals: true,
        }
    }

    /// Relative error bound for a single operation (unit roundoff)
    pub fn unit_roundoff(&self) -> f64 {
        self.epsilon / 2.0
    }

    /// Number of representable values between 1 and 2
    pub fn values_between_1_and_2(&self) -> u64 {
        1u64 << self.mantissa_bits
    }

    /// Decimal digits of precision
    pub fn decimal_precision(&self) -> f64 {
        self.mantissa_bits as f64 * 0.30103 // log10(2)
    }
}

// ============================================================================
// ERROR ANALYSIS
// ============================================================================

/// Error accumulation analysis
///
/// SCIENTIFIC HONESTY:
/// These are WORST-CASE bounds assuming adversarial rounding.
/// Real-world errors are typically much smaller due to:
/// - Random rounding directions
/// - Cancellation effects
/// - Data distribution
///
/// But worst-case matters for safety-critical applications!
#[derive(Debug)]
pub struct ErrorAnalysis {
    format: FloatFormat,
}

impl ErrorAnalysis {
    pub fn new(format: FloatFormat) -> Self {
        Self { format }
    }

    /// Relative error bound for sum of n numbers
    ///
    /// THEOREM (Higham): For floating-point summation of n numbers,
    /// |computed_sum - exact_sum| <= n * u * |exact_sum| + O(u^2)
    /// where u is the unit roundoff
    ///
    /// CAVEAT: This assumes no overflow/underflow
    pub fn summation_error_bound(&self, n: usize) -> ErrorBound {
        let u = self.format.unit_roundoff();

        // Naive summation: error grows linearly
        let naive_bound = n as f64 * u;

        // Pairwise summation: error grows logarithmically
        let pairwise_bound = (n as f64).log2().ceil() * u;

        // Kahan summation: error is essentially constant
        let kahan_bound = 2.0 * u + n as f64 * u * u;

        ErrorBound {
            naive: naive_bound,
            pairwise: pairwise_bound,
            kahan: kahan_bound,
            condition_dependent: true,
            notes: vec![
                format!(
                    "Bounds assume no overflow (max value: {})",
                    self.format.max_value
                ),
                "Actual error typically much smaller than bound".to_string(),
                "Ill-conditioned sums can exceed bounds".to_string(),
            ],
        }
    }

    /// Error bound for dot product of n-element vectors
    ///
    /// THEOREM: |computed_dot - exact_dot| <= n * u * |x|*|y| + O(u^2)
    /// where |x|*|y| is the dot product of absolute values
    pub fn dot_product_error_bound(&self, n: usize) -> ErrorBound {
        let u = self.format.unit_roundoff();

        ErrorBound {
            naive: n as f64 * u,
            pairwise: (n as f64).log2().ceil() * u,
            kahan: 2.0 * u,
            condition_dependent: true,
            notes: vec![
                "Bound is relative to sum of absolute products".to_string(),
                "Catastrophic cancellation can cause larger relative error".to_string(),
            ],
        }
    }

    /// Error bound for matrix multiplication C = A * B
    /// where A is m*k and B is k*n
    ///
    /// THEOREM: For each element c_ij,
    /// |computed_c_ij - exact_c_ij| <= k * u * sum|a_il||b_lj| + O(u^2)
    pub fn matmul_error_bound(&self, k: usize) -> ErrorBound {
        let u = self.format.unit_roundoff();

        // Standard algorithm
        let standard_bound = k as f64 * u;

        // Strassen-like algorithms can have higher error
        let _strassen_bound = k as f64 * u * 1.5; // Approximate

        ErrorBound {
            naive: standard_bound,
            pairwise: standard_bound, // Matrix multiplication doesn't benefit from pairwise
            kahan: standard_bound * 0.1, // Compensated summation helps
            condition_dependent: true,
            notes: vec![
                format!("Error bound for inner dimension k={}", k),
                "Tensor Cores use TF32 internally (epsilon ~ 10^-3)".to_string(),
                "Accumulation in FP32 reduces error".to_string(),
            ],
        }
    }

    /// When does this format FAIL?
    ///
    /// HONEST ASSESSMENT of failure modes
    pub fn failure_modes(&self) -> Vec<FailureMode> {
        let mut modes = Vec::new();

        // Overflow
        modes.push(FailureMode {
            name: "Overflow".to_string(),
            description: format!("Values > {} become Inf", self.format.max_value),
            likelihood: match self.format.name {
                "FP8_E4M3" => Likelihood::VeryHigh,
                "FP16" => Likelihood::High,
                "BF16" => Likelihood::Low,
                "FP32" | "TF32" => Likelihood::VeryLow,
                "FP64" => Likelihood::Negligible,
                _ => Likelihood::Unknown,
            },
            mitigation: "Loss scaling, gradient clipping".to_string(),
        });

        // Underflow to zero
        modes.push(FailureMode {
            name: "Underflow".to_string(),
            description: format!("Values < {} become 0", self.format.min_normal),
            likelihood: match self.format.name {
                "FP8_E4M3" => Likelihood::VeryHigh,
                "FP8_E5M2" => Likelihood::High,
                "FP16" => Likelihood::Medium,
                "BF16" => Likelihood::Low, // Same range as FP32
                "FP32" | "TF32" => Likelihood::VeryLow,
                "FP64" => Likelihood::Negligible,
                _ => Likelihood::Unknown,
            },
            mitigation: "Loss scaling, epsilon additions".to_string(),
        });

        // Precision loss in accumulation
        modes.push(FailureMode {
            name: "Accumulation Error".to_string(),
            description: format!(
                "Summing N numbers: relative error ~ N * {:.2e}",
                self.format.unit_roundoff()
            ),
            likelihood: match self.format.name {
                "FP8_E4M3" | "FP8_E5M2" => Likelihood::VeryHigh,
                "BF16" => Likelihood::High,
                "FP16" | "TF32" => Likelihood::Medium,
                "FP32" => Likelihood::Low,
                "FP64" => Likelihood::VeryLow,
                _ => Likelihood::Unknown,
            },
            mitigation: "Accumulate in higher precision, Kahan summation".to_string(),
        });

        // Catastrophic cancellation
        modes.push(FailureMode {
            name: "Catastrophic Cancellation".to_string(),
            description: "Subtracting nearly equal numbers loses precision".to_string(),
            likelihood: Likelihood::Medium, // Depends on algorithm
            mitigation: "Reformulate algorithm, use compensated arithmetic".to_string(),
        });

        modes
    }

    /// Estimate if format is suitable for given computation
    pub fn suitability_check(&self, computation: &Computation) -> SuitabilityResult {
        let mut issues = Vec::new();
        let mut score = 100i32;

        // Check range
        if computation.expected_max_value > self.format.max_value * 0.9 {
            issues.push("Value range close to or exceeds format maximum".to_string());
            score -= 50;
        }

        if computation.expected_min_value < self.format.min_normal * 10.0 {
            issues.push("Values may underflow".to_string());
            score -= 30;
        }

        // Check accumulation error
        let accum_error = computation.num_accumulations as f64 * self.format.unit_roundoff();
        if accum_error > 0.01 {
            // 1% relative error
            issues.push(format!(
                "Accumulation error bound: {:.2}% (may be too high)",
                accum_error * 100.0
            ));
            score -= 20;
        }

        // Check precision requirements
        let available_precision = self.format.decimal_precision();
        if computation.required_precision_digits > available_precision {
            issues.push(format!(
                "Required {:.1} decimal digits, format provides {:.1}",
                computation.required_precision_digits, available_precision
            ));
            score -= 40;
        }

        SuitabilityResult {
            format: self.format.name,
            score: score.max(0),
            suitable: score >= 60,
            issues,
            recommendations: self.get_recommendations(computation),
        }
    }

    fn get_recommendations(&self, computation: &Computation) -> Vec<String> {
        let mut recs = Vec::new();

        if computation.num_accumulations > 1000 {
            recs.push("Consider Kahan summation for large reductions".to_string());
        }

        if self.format.name == "FP16" || self.format.name == "BF16" {
            recs.push("Use FP32 accumulator for Tensor Core operations".to_string());
        }

        if self.format.name.starts_with("FP8") {
            recs.push(
                "FP8 requires careful loss scaling and is primarily for inference".to_string(),
            );
        }

        recs
    }
}

/// Error bound information
#[derive(Debug, Clone)]
pub struct ErrorBound {
    /// Naive sequential summation
    pub naive: f64,
    /// Pairwise summation
    pub pairwise: f64,
    /// Kahan (compensated) summation
    pub kahan: f64,
    /// Is the bound condition-dependent?
    pub condition_dependent: bool,
    /// Additional notes
    pub notes: Vec<String>,
}

/// A failure mode description
#[derive(Debug, Clone)]
pub struct FailureMode {
    pub name: String,
    pub description: String,
    pub likelihood: Likelihood,
    pub mitigation: String,
}

/// Likelihood of a failure mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Likelihood {
    Negligible,
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Unknown,
}

/// Computation description for suitability check
#[derive(Debug, Clone)]
pub struct Computation {
    pub expected_max_value: f64,
    pub expected_min_value: f64,
    pub num_accumulations: usize,
    pub required_precision_digits: f64,
}

/// Suitability check result
#[derive(Debug, Clone)]
pub struct SuitabilityResult {
    pub format: &'static str,
    pub score: i32,
    pub suitable: bool,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

// ============================================================================
// MIXED PRECISION ANALYSIS
// ============================================================================

/// Mixed precision training analysis
///
/// SCIENTIFIC HONESTY:
/// Mixed precision "works" for most ML workloads because:
/// 1. Neural networks are robust to small errors
/// 2. SGD adds noise anyway
/// 3. Regularization masks precision issues
///
/// But this doesn't mean the math is sound!
#[derive(Debug, Clone)]
pub struct MixedPrecisionAnalysis {
    /// Storage format
    pub storage: FloatFormat,
    /// Compute format
    pub compute: FloatFormat,
    /// Accumulator format
    pub accumulator: FloatFormat,
}

impl MixedPrecisionAnalysis {
    /// Standard mixed precision (FP16 storage, FP32 accumulator)
    pub fn standard() -> Self {
        Self {
            storage: FloatFormat::fp16(),
            compute: FloatFormat::fp16(),
            accumulator: FloatFormat::fp32(),
        }
    }

    /// BF16 mixed precision
    pub fn bf16() -> Self {
        Self {
            storage: FloatFormat::bf16(),
            compute: FloatFormat::bf16(),
            accumulator: FloatFormat::fp32(),
        }
    }

    /// TF32 (Tensor Core default on Ampere+)
    pub fn tf32() -> Self {
        Self {
            storage: FloatFormat::fp32(),
            compute: FloatFormat::tf32(),
            accumulator: FloatFormat::fp32(),
        }
    }

    /// FP8 mixed precision (inference)
    pub fn fp8_inference() -> Self {
        Self {
            storage: FloatFormat::fp8_e4m3(),
            compute: FloatFormat::fp8_e4m3(),
            accumulator: FloatFormat::fp16(),
        }
    }

    /// Analyze error characteristics
    pub fn analyze(&self) -> MixedPrecisionReport {
        let storage_analysis = ErrorAnalysis::new(self.storage.clone());

        // Quantization error from storage format
        let quantization_error = self.storage.unit_roundoff();

        // Computation error
        let compute_error = self.compute.unit_roundoff();

        // Combined error bound (approximate)
        let combined_error = quantization_error + compute_error;

        MixedPrecisionReport {
            storage_format: self.storage.name,
            compute_format: self.compute.name,
            accumulator_format: self.accumulator.name,
            quantization_error,
            compute_error,
            combined_error_bound: combined_error,
            storage_failures: storage_analysis.failure_modes(),
            requires_loss_scaling: self.storage.max_value < 1e6,
            recommended_loss_scale: self.recommend_loss_scale(),
            honest_assessment: self.honest_assessment(),
        }
    }

    fn recommend_loss_scale(&self) -> f64 {
        // HEURISTIC: Scale so that gradients are in "sweet spot" of format
        match self.storage.name {
            "FP16" => 1024.0,
            "BF16" => 1.0,
            "FP8_E4M3" => 256.0,
            "FP8_E5M2" => 128.0,
            _ => 1.0,
        }
    }

    fn honest_assessment(&self) -> String {
        match (self.storage.name, self.compute.name) {
            ("FP16", "FP16") => "FP16 mixed precision works for most ML training but \
                 can fail on models with large activations, small gradients, \
                 or sensitive loss landscapes. Loss scaling is essential."
                .to_string(),
            ("BF16", "BF16") => "BF16 has same range as FP32 but lower precision. \
                 Less prone to overflow/underflow than FP16 but \
                 accumulation errors can be significant."
                .to_string(),
            ("FP32", "TF32") => "TF32 is a compromise: FP32 range with FP16 precision. \
                 Enabled by default on Ampere+ Tensor Cores. \
                 Generally safe but reduces precision vs pure FP32."
                .to_string(),
            (s, _) if s.starts_with("FP8") => {
                "FP8 is primarily for inference. Training is experimental \
                 and requires careful tuning. Significant quantization error."
                    .to_string()
            }
            _ => "Custom mixed precision configuration. \
                 Carefully analyze error bounds for your specific workload."
                .to_string(),
        }
    }
}

/// Mixed precision analysis report
#[derive(Debug, Clone)]
pub struct MixedPrecisionReport {
    pub storage_format: &'static str,
    pub compute_format: &'static str,
    pub accumulator_format: &'static str,
    pub quantization_error: f64,
    pub compute_error: f64,
    pub combined_error_bound: f64,
    pub storage_failures: Vec<FailureMode>,
    pub requires_loss_scaling: bool,
    pub recommended_loss_scale: f64,
    pub honest_assessment: String,
}

// ============================================================================
// REPRODUCIBILITY ANALYSIS
// ============================================================================

/// Reproducibility issues with floating-point GPU computing
///
/// UNCOMFORTABLE TRUTH:
/// Floating-point GPU computations are NOT reproducible by default.
/// Same code + same data + same GPU != same result (usually)
#[derive(Debug)]
pub struct ReproducibilityAnalysis;

impl ReproducibilityAnalysis {
    /// Sources of non-reproducibility
    pub fn non_reproducibility_sources() -> Vec<NonReproducibilitySource> {
        vec![
            NonReproducibilitySource {
                name: "Non-deterministic atomics".to_string(),
                description: "Order of atomic operations is non-deterministic, \
                             and FP addition is not associative"
                    .to_string(),
                impact: Impact::High,
                can_be_fixed: true,
                fix: "Use deterministic algorithms (e.g., segmented scan instead of atomics)"
                    .to_string(),
            },
            NonReproducibilitySource {
                name: "Warp-level reductions".to_string(),
                description: "Order of partial sums in warp reduction depends on \
                             thread scheduling"
                    .to_string(),
                impact: Impact::Medium,
                can_be_fixed: true,
                fix: "Use fixed reduction order with __syncwarp".to_string(),
            },
            NonReproducibilitySource {
                name: "cuBLAS/cuDNN algorithm selection".to_string(),
                description: "Libraries may select different algorithms based on \
                             runtime conditions"
                    .to_string(),
                impact: Impact::High,
                can_be_fixed: true,
                fix: "Set CUBLAS_WORKSPACE_CONFIG and use deterministic APIs".to_string(),
            },
            NonReproducibilitySource {
                name: "Multi-GPU reductions".to_string(),
                description: "Order of all-reduce contributions varies".to_string(),
                impact: Impact::High,
                can_be_fixed: true, // Partially
                fix: "Use deterministic collective algorithms".to_string(),
            },
            NonReproducibilitySource {
                name: "TF32 rounding".to_string(),
                description: "TF32 truncation from FP32 can vary".to_string(),
                impact: Impact::Low,
                can_be_fixed: true,
                fix: "Disable TF32 with NVIDIA_TF32_OVERRIDE=0".to_string(),
            },
            NonReproducibilitySource {
                name: "Memory allocation patterns".to_string(),
                description: "Different memory layouts can cause different \
                             cache behavior affecting instruction timing"
                    .to_string(),
                impact: Impact::Low,
                can_be_fixed: false,
                fix: "Cannot fully control".to_string(),
            },
        ]
    }

    /// Cost of reproducibility
    pub fn reproducibility_cost() -> ReproducibilityCost {
        ReproducibilityCost {
            performance_overhead: "10-50% slowdown typical".to_string(),
            memory_overhead: "May require additional workspace".to_string(),
            code_complexity: "Requires careful algorithm selection".to_string(),
            limitations: vec![
                "Some operations have no deterministic equivalent".to_string(),
                "Multi-GPU determinism is especially hard".to_string(),
                "Debugging non-reproducibility is extremely difficult".to_string(),
            ],
        }
    }
}

/// Source of non-reproducibility
#[derive(Debug, Clone)]
pub struct NonReproducibilitySource {
    pub name: String,
    pub description: String,
    pub impact: Impact,
    pub can_be_fixed: bool,
    pub fix: String,
}

/// Impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Impact {
    Low,
    Medium,
    High,
    Critical,
}

/// Cost of achieving reproducibility
#[derive(Debug, Clone)]
pub struct ReproducibilityCost {
    pub performance_overhead: String,
    pub memory_overhead: String,
    pub code_complexity: String,
    pub limitations: Vec<String>,
}

// ============================================================================
// NUMERICAL STABILITY CHECKS
// ============================================================================

/// Numerical stability analyzer
#[derive(Debug)]
pub struct NumericalStabilityChecker {
    /// Detected issues
    issues: Vec<StabilityIssue>,
}

/// A numerical stability issue
#[derive(Debug, Clone)]
pub struct StabilityIssue {
    pub category: String,
    pub description: String,
    pub severity: IssueSeverity,
    pub suggestion: String,
}

/// Issue severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl NumericalStabilityChecker {
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Check for potential overflow
    pub fn check_overflow(&mut self, format: &FloatFormat, max_value: f64) {
        if max_value > format.max_value * 0.9 {
            self.issues.push(StabilityIssue {
                category: "Overflow".to_string(),
                description: format!(
                    "Maximum value {:.2e} approaches format limit {:.2e}",
                    max_value, format.max_value
                ),
                severity: if max_value > format.max_value {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::Warning
                },
                suggestion: "Use higher precision or apply scaling".to_string(),
            });
        }
    }

    /// Check for potential underflow
    pub fn check_underflow(&mut self, format: &FloatFormat, min_value: f64) {
        if min_value < format.min_normal * 10.0 && min_value > 0.0 {
            self.issues.push(StabilityIssue {
                category: "Underflow".to_string(),
                description: format!(
                    "Minimum value {:.2e} may underflow (min normal: {:.2e})",
                    min_value, format.min_normal
                ),
                severity: if min_value < format.min_normal {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                },
                suggestion: "Apply loss scaling or use higher precision".to_string(),
            });
        }
    }

    /// Check for accumulation precision loss
    pub fn check_accumulation(&mut self, format: &FloatFormat, num_terms: usize) {
        let error_bound = num_terms as f64 * format.unit_roundoff();
        if error_bound > 0.01 {
            self.issues.push(StabilityIssue {
                category: "Accumulation".to_string(),
                description: format!(
                    "Summing {} terms may have {:.2}% relative error",
                    num_terms,
                    error_bound * 100.0
                ),
                severity: if error_bound > 0.1 {
                    IssueSeverity::Error
                } else {
                    IssueSeverity::Warning
                },
                suggestion: "Use Kahan summation or higher precision accumulator".to_string(),
            });
        }
    }

    /// Get all issues
    pub fn get_issues(&self) -> &[StabilityIssue] {
        &self.issues
    }

    /// Check if any critical issues were found
    pub fn has_critical_issues(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Critical)
    }

    /// Clear issues
    pub fn clear(&mut self) {
        self.issues.clear();
    }
}

impl Default for NumericalStabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_specifications() {
        let fp16 = FloatFormat::fp16();
        assert_eq!(fp16.bits, 16);
        assert!((fp16.epsilon - 9.765625e-4).abs() < 1e-10);

        let bf16 = FloatFormat::bf16();
        assert!(bf16.max_value > 1e38); // Same range as FP32
        assert!(bf16.epsilon > fp16.epsilon); // Lower precision
    }

    #[test]
    fn test_unit_roundoff() {
        let fp32 = FloatFormat::fp32();
        let fp64 = FloatFormat::fp64();

        assert!(fp32.unit_roundoff() > fp64.unit_roundoff());
    }

    #[test]
    fn test_error_bounds() {
        let analysis = ErrorAnalysis::new(FloatFormat::fp32());
        let bound = analysis.summation_error_bound(1000);

        // 1000 additions, each with ~6e-8 relative error
        assert!(bound.naive < 1e-4);
        assert!(bound.pairwise < bound.naive);
        assert!(bound.kahan < bound.pairwise);
    }

    #[test]
    fn test_failure_modes() {
        let fp8_analysis = ErrorAnalysis::new(FloatFormat::fp8_e4m3());
        let failures = fp8_analysis.failure_modes();

        // FP8 should have high overflow likelihood
        let overflow = failures.iter().find(|f| f.name == "Overflow").unwrap();
        assert!(matches!(overflow.likelihood, Likelihood::VeryHigh));
    }

    #[test]
    fn test_suitability() {
        let analysis = ErrorAnalysis::new(FloatFormat::fp16());

        let computation = Computation {
            expected_max_value: 100000.0, // Exceeds FP16 max!
            expected_min_value: 1e-8,
            num_accumulations: 10000,
            required_precision_digits: 4.0,
        };

        let result = analysis.suitability_check(&computation);
        assert!(!result.suitable);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_mixed_precision_analysis() {
        let mp = MixedPrecisionAnalysis::standard();
        let report = mp.analyze();

        assert!(report.requires_loss_scaling);
        assert!(report.recommended_loss_scale > 1.0);
    }

    #[test]
    fn test_reproducibility_sources() {
        let sources = ReproducibilityAnalysis::non_reproducibility_sources();
        assert!(!sources.is_empty());

        // Check that at least one high-impact source exists
        assert!(sources.iter().any(|s| s.impact == Impact::High));
    }

    #[test]
    fn test_stability_checker() {
        let mut checker = NumericalStabilityChecker::new();
        let fp16 = FloatFormat::fp16();

        // Check overflow
        checker.check_overflow(&fp16, 70000.0); // Exceeds FP16 max
        assert!(checker.has_critical_issues());

        checker.clear();

        // Check accumulation
        checker.check_accumulation(&fp16, 100000);
        assert!(!checker.get_issues().is_empty());
    }

    #[test]
    fn test_decimal_precision() {
        let fp32 = FloatFormat::fp32();
        let fp64 = FloatFormat::fp64();

        // FP32 has ~7 decimal digits, FP64 has ~15-16
        assert!(fp32.decimal_precision() > 6.0 && fp32.decimal_precision() < 8.0);
        assert!(fp64.decimal_precision() > 15.0 && fp64.decimal_precision() < 17.0);
    }
}
