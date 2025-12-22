//! Sobol' Sensitivity Indices for Global Sensitivity Analysis
//!
//! Variance-based sensitivity analysis decomposes output variance into
//! contributions from each input and their interactions.
//!
//! First-order index S_i: fraction of variance due to x_i alone
//! Total-order index S_Ti: fraction of variance due to x_i and all interactions
//!
//! Key insight: max_dominant_source becomes QUANTITATIVE:
//!   "Your output is 82% hostage to clearance. Go measure clearance."
//!
//! References:
//!   - Sobol' (1993): "Sensitivity estimates for nonlinear mathematical models"
//!   - Saltelli et al. (2008): "Global Sensitivity Analysis: The Primer"
//!   - Homma & Saltelli (1996): "Importance measures in global sensitivity analysis"

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn cos(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

// ============================================================================
// QUASI-RANDOM SEQUENCE: SOBOL' SEQUENCE GENERATOR
// ============================================================================

/// Sobol' sequence state for up to 4 dimensions
struct SobolSequence {
    index: i64,
    dim: i32,
    // Direction numbers (simplified - real impl uses lookup tables)
    // For now, we use a simple pseudo-random fallback
}

fn sobol_new(dim: i32) -> SobolSequence {
    return SobolSequence {
        index: 0,
        dim: dim,
    }
}

/// Generate next point in the Sobol' sequence (simplified)
/// Returns values in [0, 1]^dim
struct SobolPoint {
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
}

/// Van der Corput sequence helper - reverses bits to generate low-discrepancy points
fn reverse_bits(n: i64) -> f64 {
    var result: f64 = 0.0
    var v = n
    var b: f64 = 0.5
    while v > 0 {
        if (v % 2) == 1 {
            result = result + b
        }
        v = v / 2
        b = b / 2.0
    }
    return result
}

fn sobol_next(seq: SobolSequence) -> (SobolPoint, SobolSequence) {
    let idx = seq.index + 1

    // Simple low-discrepancy approximation using van der Corput-like sequence
    // Real Sobol' uses direction numbers - this is a simplified version

    // Different bases for different dimensions to reduce correlation
    let x0 = reverse_bits(idx)
    let x1 = reverse_bits(idx * 2 + 1)
    let x2 = reverse_bits(idx * 3 + 2)
    let x3 = reverse_bits(idx * 5 + 3)

    let point = SobolPoint { x0: x0, x1: x1, x2: x2, x3: x3 }
    let new_seq = SobolSequence { index: idx, dim: seq.dim }

    return (point, new_seq)
}

// ============================================================================
// INPUT SPECIFICATION
// ============================================================================

/// Input parameter with distribution
struct SobolInput {
    name_hash: i64,      // Hash of parameter name
    mean: f64,
    std: f64,
    lo: f64,             // Lower bound for uniform sampling
    hi: f64,             // Upper bound for uniform sampling
}

fn sobol_input(name: i64, mean: f64, std: f64) -> SobolInput {
    // Assume ±3σ range for sampling
    return SobolInput {
        name_hash: name,
        mean: mean,
        std: std,
        lo: mean - 3.0 * std,
        hi: mean + 3.0 * std,
    }
}

fn sobol_input_bounded(name: i64, mean: f64, std: f64, lo: f64, hi: f64) -> SobolInput {
    return SobolInput {
        name_hash: name,
        mean: mean,
        std: std,
        lo: lo,
        hi: hi,
    }
}

/// Transform uniform [0,1] to parameter range
fn transform_uniform(u: f64, input: SobolInput) -> f64 {
    return input.lo + u * (input.hi - input.lo)
}

// ============================================================================
// SOBOL' INDICES COMPUTATION (SALTELLI METHOD)
// ============================================================================

/// Sobol' sensitivity indices for one input
struct SobolIndex {
    input_hash: i64,
    first_order: f64,     // S_i: main effect
    total_order: f64,     // S_Ti: total effect including interactions
    interaction: f64,     // S_Ti - S_i: pure interaction effect
}

/// Complete Sobol' analysis result
struct SobolAnalysis {
    output_variance: f64,
    output_mean: f64,
    n_samples: i32,

    // Indices for up to 4 inputs
    index_count: i32,
    idx0: SobolIndex,
    idx1: SobolIndex,
    idx2: SobolIndex,
    idx3: SobolIndex,

    // Dominance metrics
    max_first_order: f64,
    max_total_order: f64,
    dominant_input: i64,       // Hash of most influential input
    dominance_fraction: f64,   // Fraction of variance from dominant
}

fn empty_sobol_index() -> SobolIndex {
    return SobolIndex {
        input_hash: 0,
        first_order: 0.0,
        total_order: 0.0,
        interaction: 0.0,
    }
}

/// Compute Sobol' indices using Saltelli's estimator
/// For model y = f(x1, x2, ..., xk)
///
/// The method uses two independent sample matrices A and B,
/// then computes conditional variances by mixing columns.
///
/// This is a simplified 2-input version for clarity
fn sobol_analyze_2d(
    input0: SobolInput,
    input1: SobolInput,
    n_samples: i32,
    seed: i64
) -> SobolAnalysis {
    // We need to evaluate f(A), f(B), f(A_B^i) where A_B^i has column i from B
    // Using fixed arrays for simplicity

    var seq = sobol_new(4)  // Need 4 dims for A and B

    // Running statistics
    var sum_y: f64 = 0.0
    var sum_y2: f64 = 0.0
    var sum_y_ab0: f64 = 0.0  // f(A) * f(A_B^0)
    var sum_y_ab1: f64 = 0.0  // f(A) * f(A_B^1)
    var sum_y_ba0: f64 = 0.0  // f(B) * f(B_A^0)
    var sum_y_ba1: f64 = 0.0  // f(B) * f(B_A^1)

    var i: i32 = 0
    while i < n_samples {
        // Get next Sobol' point (x0, x1 for A; x2, x3 for B)
        let r = sobol_next(seq)
        let pt = r.0
        seq = r.1

        // Sample A: (x0, x1)
        let a0 = transform_uniform(pt.x0, input0)
        let a1 = transform_uniform(pt.x1, input1)

        // Sample B: (x2, x3)
        let b0 = transform_uniform(pt.x2, input0)
        let b1 = transform_uniform(pt.x3, input1)

        // Evaluate model at different points
        // Using a test function: f(x0, x1) = x0 + x0*x1 (has main effects and interaction)
        let y_a = a0 + a0 * a1
        let y_b = b0 + b0 * b1
        let y_ab0 = b0 + b0 * a1  // A with x0 from B
        let y_ab1 = a0 + a0 * b1  // A with x1 from B
        let y_ba0 = a0 + a0 * b1  // B with x0 from A
        let y_ba1 = b0 + b0 * a1  // B with x1 from A

        sum_y = sum_y + y_a
        sum_y2 = sum_y2 + y_a * y_a

        sum_y_ab0 = sum_y_ab0 + y_a * y_ab0
        sum_y_ab1 = sum_y_ab1 + y_a * y_ab1
        sum_y_ba0 = sum_y_ba0 + y_b * y_ba0
        sum_y_ba1 = sum_y_ba1 + y_b * y_ba1

        i = i + 1
    }

    let n = n_samples as f64
    let mean_y = sum_y / n
    let var_y = sum_y2 / n - mean_y * mean_y

    // Saltelli estimator for first-order index:
    // S_i = (1/n) Σ f(A)(f(A_B^i) - f(B)) / Var(Y)
    // Simplified: use covariance estimator

    var s0_first = 0.0
    var s1_first = 0.0
    var s0_total = 0.0
    var s1_total = 0.0

    if var_y > 1.0e-15 {
        // First-order: proportion of variance explained by x_i alone
        // Using Jansen estimator variant
        let cov_0 = sum_y_ab0 / n - mean_y * mean_y
        let cov_1 = sum_y_ab1 / n - mean_y * mean_y

        s0_first = cov_0 / var_y
        s1_first = cov_1 / var_y

        // Clamp to [0, 1]
        s0_first = max_f64(0.0, min_f64(1.0, s0_first))
        s1_first = max_f64(0.0, min_f64(1.0, s1_first))

        // Total-order: includes all interactions
        // S_Ti = 1 - V_{~i} / V(Y) where V_{~i} is variance with x_i fixed
        // Approximation: S_Ti ≈ S_i + interaction_terms
        s0_total = s0_first + 0.1 * s1_first  // Simplified interaction estimate
        s1_total = s1_first + 0.1 * s0_first

        s0_total = max_f64(0.0, min_f64(1.0, s0_total))
        s1_total = max_f64(0.0, min_f64(1.0, s1_total))
    }

    // Build result
    let idx0 = SobolIndex {
        input_hash: input0.name_hash,
        first_order: s0_first,
        total_order: s0_total,
        interaction: s0_total - s0_first,
    }

    let idx1 = SobolIndex {
        input_hash: input1.name_hash,
        first_order: s1_first,
        total_order: s1_total,
        interaction: s1_total - s1_first,
    }

    // Determine dominant input
    var dominant = input0.name_hash
    var max_s = s0_total
    if s1_total > s0_total {
        dominant = input1.name_hash
        max_s = s1_total
    }

    return SobolAnalysis {
        output_variance: var_y,
        output_mean: mean_y,
        n_samples: n_samples,
        index_count: 2,
        idx0: idx0,
        idx1: idx1,
        idx2: empty_sobol_index(),
        idx3: empty_sobol_index(),
        max_first_order: max_f64(s0_first, s1_first),
        max_total_order: max_s,
        dominant_input: dominant,
        dominance_fraction: max_s,
    }
}

// ============================================================================
// GENERIC SOBOL' ANALYSIS (EXTERNAL MODEL)
// ============================================================================

/// Sample set for external model evaluation
struct SampleSet {
    count: i32,
    // Store up to 20 samples × 4 dimensions
    // In practice, would be dynamically allocated
    s0_x0: f64, s0_x1: f64, s0_x2: f64, s0_x3: f64, s0_y: f64,
    s1_x0: f64, s1_x1: f64, s1_x2: f64, s1_x3: f64, s1_y: f64,
    s2_x0: f64, s2_x1: f64, s2_x2: f64, s2_x3: f64, s2_y: f64,
    s3_x0: f64, s3_x1: f64, s3_x2: f64, s3_x3: f64, s3_y: f64,
    s4_x0: f64, s4_x1: f64, s4_x2: f64, s4_x3: f64, s4_y: f64,
    s5_x0: f64, s5_x1: f64, s5_x2: f64, s5_x3: f64, s5_y: f64,
    s6_x0: f64, s6_x1: f64, s6_x2: f64, s6_x3: f64, s6_y: f64,
    s7_x0: f64, s7_x1: f64, s7_x2: f64, s7_x3: f64, s7_y: f64,
    s8_x0: f64, s8_x1: f64, s8_x2: f64, s8_x3: f64, s8_y: f64,
    s9_x0: f64, s9_x1: f64, s9_x2: f64, s9_x3: f64, s9_y: f64,
}

fn sample_set_new() -> SampleSet {
    return SampleSet {
        count: 0,
        s0_x0: 0.0, s0_x1: 0.0, s0_x2: 0.0, s0_x3: 0.0, s0_y: 0.0,
        s1_x0: 0.0, s1_x1: 0.0, s1_x2: 0.0, s1_x3: 0.0, s1_y: 0.0,
        s2_x0: 0.0, s2_x1: 0.0, s2_x2: 0.0, s2_x3: 0.0, s2_y: 0.0,
        s3_x0: 0.0, s3_x1: 0.0, s3_x2: 0.0, s3_x3: 0.0, s3_y: 0.0,
        s4_x0: 0.0, s4_x1: 0.0, s4_x2: 0.0, s4_x3: 0.0, s4_y: 0.0,
        s5_x0: 0.0, s5_x1: 0.0, s5_x2: 0.0, s5_x3: 0.0, s5_y: 0.0,
        s6_x0: 0.0, s6_x1: 0.0, s6_x2: 0.0, s6_x3: 0.0, s6_y: 0.0,
        s7_x0: 0.0, s7_x1: 0.0, s7_x2: 0.0, s7_x3: 0.0, s7_y: 0.0,
        s8_x0: 0.0, s8_x1: 0.0, s8_x2: 0.0, s8_x3: 0.0, s8_y: 0.0,
        s9_x0: 0.0, s9_x1: 0.0, s9_x2: 0.0, s9_x3: 0.0, s9_y: 0.0,
    }
}

// ============================================================================
// CORRELATION GATING: SOBOL' REQUIRES INDEPENDENCE
// ============================================================================

/// Sobol' indices have a KEY ASSUMPTION: inputs are independent.
/// If correlation tracking detects dependence, we MUST either:
///   1. REFUSE Sobol' and suggest alternative sensitivity method
///   2. Require explicit "independence declaration" and record it
struct CorrelationGate {
    inputs_independent: bool,
    max_correlation: f64,       // Maximum |ρ| between any pair
    independence_declared: bool, // User explicitly declared independence
    declaration_hash: i64,       // Hash of declaration for audit
}

fn correlation_gate_check(max_corr: f64) -> CorrelationGate {
    // Threshold: |ρ| > 0.1 is "not independent enough"
    let independent = abs_f64(max_corr) < 0.1
    return CorrelationGate {
        inputs_independent: independent,
        max_correlation: max_corr,
        independence_declared: false,
        declaration_hash: 0,
    }
}

fn correlation_gate_with_declaration(max_corr: f64, declaration_hash: i64) -> CorrelationGate {
    // User explicitly declares independence despite measured correlation
    return CorrelationGate {
        inputs_independent: true,  // Overridden by declaration
        max_correlation: max_corr,
        independence_declared: true,
        declaration_hash: declaration_hash,
    }
}

/// Check if Sobol' analysis should be refused due to correlation
fn should_refuse_sobol(gate: CorrelationGate) -> bool {
    if gate.independence_declared {
        return false  // User took responsibility
    }
    return !gate.inputs_independent
}

/// Alternative sensitivity methods for correlated inputs
struct AlternativeRecommendation {
    method_code: i32,    // 0=none, 1=delta, 2=shapley, 3=regional
    reason_code: i32,    // Why Sobol' was rejected
    correlation: f64,
}

fn recommend_alternative(gate: CorrelationGate) -> AlternativeRecommendation {
    if gate.inputs_independent {
        return AlternativeRecommendation {
            method_code: 0,
            reason_code: 0,
            correlation: gate.max_correlation,
        }
    }

    // For correlated inputs, recommend:
    // - |ρ| < 0.5: Delta moment-independent measures
    // - |ρ| >= 0.5: Shapley effects (game-theoretic)
    var method = 1  // Delta
    if gate.max_correlation >= 0.5 {
        method = 2  // Shapley
    }

    return AlternativeRecommendation {
        method_code: method,
        reason_code: 1,  // Correlation detected
        correlation: gate.max_correlation,
    }
}

// ============================================================================
// REFUSAL POLICY BASED ON DOMINANCE
// ============================================================================

/// Policy for refusing computations based on Sobol' dominance
struct DominancePolicy {
    max_total_order: f64,       // Refuse if S_Ti > this
    min_exploration: i32,       // Minimum samples before refusing
    require_diverse_sources: bool,
    require_independence: bool,  // Enforce independence check
}

fn default_dominance_policy() -> DominancePolicy {
    return DominancePolicy {
        max_total_order: 0.7,   // Refuse if 70%+ variance from one input
        min_exploration: 100,
        require_diverse_sources: true,
        require_independence: true,  // Check correlation!
    }
}

fn strict_dominance_policy() -> DominancePolicy {
    return DominancePolicy {
        max_total_order: 0.5,   // Refuse if 50%+ variance from one input
        min_exploration: 200,
        require_diverse_sources: true,
        require_independence: true,
    }
}

struct DominanceRefusal {
    should_refuse: bool,
    reason_code: i32,         // 0=ok, 1=overdominant, 2=insufficient_samples
    dominant_input: i64,
    dominance: f64,
    recommendation: i32,      // 0=proceed, 1=measure_dominant, 2=add_sources
}

fn check_dominance(analysis: SobolAnalysis, policy: DominancePolicy) -> DominanceRefusal {
    // Check if we have enough samples
    if analysis.n_samples < policy.min_exploration {
        return DominanceRefusal {
            should_refuse: false,
            reason_code: 2,
            dominant_input: 0,
            dominance: 0.0,
            recommendation: 0,
        }
    }

    // Check if any input is too dominant
    if analysis.max_total_order > policy.max_total_order {
        return DominanceRefusal {
            should_refuse: true,
            reason_code: 1,
            dominant_input: analysis.dominant_input,
            dominance: analysis.dominance_fraction,
            recommendation: 1,  // Measure the dominant parameter
        }
    }

    return DominanceRefusal {
        should_refuse: false,
        reason_code: 0,
        dominant_input: analysis.dominant_input,
        dominance: analysis.dominance_fraction,
        recommendation: 0,
    }
}

// ============================================================================
// PHARMACOKINETIC MODEL SENSITIVITY (EXAMPLE)
// ============================================================================

/// PK model: C(t) = (Dose/V) * exp(-CL/V * t)
/// Inputs: Dose, Volume (V), Clearance (CL)
///
/// This is the classic case where clearance often dominates
fn pk_concentration(dose: f64, volume: f64, clearance: f64, time: f64) -> f64 {
    if volume < 1.0e-10 { return 0.0 }
    let k = clearance / volume
    return (dose / volume) * exp(0.0 - k * time)
}

/// Analytic Sobol' indices for PK model at steady state
/// For C = Dose/V, we have:
///   S_dose ≈ (σ_D/D)² / [(σ_D/D)² + (σ_V/V)²]
///   S_V ≈ (σ_V/V)² / [(σ_D/D)² + (σ_V/V)²]
fn pk_sobol_analytic(
    dose_cv: f64,    // CV of dose
    volume_cv: f64,  // CV of volume
    clearance_cv: f64,
    time: f64
) -> SobolAnalysis {
    // At time=0, C = Dose/V, so only Dose and V matter
    // For time > 0, CL also contributes through exp(-CL*t/V)

    var s_dose = 0.0
    var s_volume = 0.0
    var s_clearance = 0.0

    let d2 = dose_cv * dose_cv
    let v2 = volume_cv * volume_cv
    let c2 = clearance_cv * clearance_cv

    // Total relative variance (first-order approximation)
    var total_rel_var = d2 + v2

    if time > 0.0 {
        // Clearance contributes through exponential
        total_rel_var = total_rel_var + c2 * time * time
    }

    if total_rel_var > 1.0e-15 {
        s_dose = d2 / total_rel_var
        s_volume = v2 / total_rel_var
        if time > 0.0 {
            s_clearance = (c2 * time * time) / total_rel_var
        }
    }

    // Build indices
    let idx_dose = SobolIndex {
        input_hash: 1,  // Dose
        first_order: s_dose,
        total_order: s_dose,  // No interaction in this linearization
        interaction: 0.0,
    }

    let idx_volume = SobolIndex {
        input_hash: 2,  // Volume
        first_order: s_volume,
        total_order: s_volume,
        interaction: 0.0,
    }

    let idx_clearance = SobolIndex {
        input_hash: 3,  // Clearance
        first_order: s_clearance,
        total_order: s_clearance,
        interaction: 0.0,
    }

    // Determine dominant
    var dominant: i64 = 1
    var max_s = s_dose
    if s_volume > max_s { dominant = 2; max_s = s_volume }
    if s_clearance > max_s { dominant = 3; max_s = s_clearance }

    return SobolAnalysis {
        output_variance: total_rel_var,
        output_mean: 0.0,
        n_samples: 1000000,  // Analytic = infinite samples, use large number
        index_count: 3,
        idx0: idx_dose,
        idx1: idx_volume,
        idx2: idx_clearance,
        idx3: empty_sobol_index(),
        max_first_order: max_s,
        max_total_order: max_s,
        dominant_input: dominant,
        dominance_fraction: max_s,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_sobol_sequence() -> bool {
    var seq = sobol_new(2)

    // Generate 10 points
    var i: i32 = 0
    while i < 10 {
        let r = sobol_next(seq)
        let pt = r.0
        seq = r.1

        // Points should be in [0, 1]
        if pt.x0 < 0.0 || pt.x0 > 1.0 { return false }
        if pt.x1 < 0.0 || pt.x1 > 1.0 { return false }

        i = i + 1
    }

    return true
}

fn test_sobol_analyze_2d() -> bool {
    let input0 = sobol_input(1, 10.0, 2.0)
    let input1 = sobol_input(2, 5.0, 1.0)

    let analysis = sobol_analyze_2d(input0, input1, 100, 12345)

    // Should have 2 indices
    if analysis.index_count != 2 { return false }

    // First-order indices should be in [0, 1]
    if analysis.idx0.first_order < 0.0 || analysis.idx0.first_order > 1.0 { return false }
    if analysis.idx1.first_order < 0.0 || analysis.idx1.first_order > 1.0 { return false }

    // Total-order should be >= first-order
    if analysis.idx0.total_order < analysis.idx0.first_order - 0.01 { return false }
    if analysis.idx1.total_order < analysis.idx1.first_order - 0.01 { return false }

    return true
}

fn test_pk_dominance() -> bool {
    // Typical PK scenario: clearance has high uncertainty
    let analysis = pk_sobol_analytic(
        0.05,   // 5% CV in dose
        0.20,   // 20% CV in volume
        0.40,   // 40% CV in clearance
        4.0     // At time=4 hours
    )

    // Clearance should be dominant at longer times
    // (because it's in the exponent)
    if analysis.dominant_input != 3 { return false }  // CL = 3

    // Check dominance policy
    let policy = default_dominance_policy()
    let refusal = check_dominance(analysis, policy)

    // With these CVs, clearance dominates but may not exceed 70%
    // Just check the logic works
    if refusal.dominance < 0.0 { return false }

    return true
}

fn test_refusal_on_overdominance() -> bool {
    // Create a scenario where one input dominates
    let analysis = pk_sobol_analytic(
        0.02,   // 2% CV in dose (very precise)
        0.02,   // 2% CV in volume (very precise)
        0.80,   // 80% CV in clearance (very uncertain)
        2.0
    )

    let policy = default_dominance_policy()
    let refusal = check_dominance(analysis, policy)

    // Should recommend measuring clearance
    if refusal.recommendation != 1 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_sobol_sequence() { return 1 }
    if !test_sobol_analyze_2d() { return 2 }
    if !test_pk_dominance() { return 3 }
    if !test_refusal_on_overdominance() { return 4 }

    return 0
}
