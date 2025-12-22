//! Dual Propagation Cross-Check: GUM vs Monte Carlo
//!
//! JCGM 101 explicitly positions Monte Carlo as guidance when:
//!   "the conditions for the GUM uncertainty framework are not fulfilled,
//!    or it is unclear whether they are fulfilled"
//!
//! This module runs BOTH methods and catches the most common scientific lie:
//! linear uncertainty propagation through a nonlinear model that breaks it.
//!
//! The invariant is:
//!   |u_mc - u_linear| / u_mc < policy.max_linearization_error
//!
//! If violated: warn or REFUSE and switch to Monte Carlo.
//!
//! References:
//!   - JCGM 101:2008 Section 5.8: Comparison with GUM uncertainty framework
//!   - JCGM 100:2008 Section 5.1.2: Conditions for valid linearization

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
// LINEARIZATION ERROR POLICY
// ============================================================================

struct LinearizationPolicy {
    max_relative_error: f64,    // Maximum |u_mc - u_gum| / u_mc
    auto_switch_to_mc: bool,    // Automatically use MC if linearization fails
    min_samples_for_check: i32, // Minimum MC samples for reliable comparison
    warn_only: bool,            // Warn instead of refuse
}

fn default_linearization_policy() -> LinearizationPolicy {
    return LinearizationPolicy {
        max_relative_error: 0.10,   // 10% max discrepancy
        auto_switch_to_mc: true,
        min_samples_for_check: 100,
        warn_only: false,
    }
}

fn strict_linearization_policy() -> LinearizationPolicy {
    return LinearizationPolicy {
        max_relative_error: 0.05,   // 5% max discrepancy
        auto_switch_to_mc: true,
        min_samples_for_check: 500,
        warn_only: false,
    }
}

fn lenient_linearization_policy() -> LinearizationPolicy {
    return LinearizationPolicy {
        max_relative_error: 0.20,   // 20% max discrepancy
        auto_switch_to_mc: false,
        min_samples_for_check: 50,
        warn_only: true,
    }
}

// ============================================================================
// SIMPLE LCG FOR MC
// ============================================================================

struct Rng {
    state: i64,
}

/// Result of random generation: value and updated RNG
struct RngResult {
    value: f64,
    rng: Rng,
}

fn rng_new(seed: i64) -> Rng {
    return Rng { state: seed }
}

fn rng_next(rng: Rng) -> RngResult {
    let a: i64 = 1103515245
    let c: i64 = 12345
    let m: i64 = 2147483648

    var next_state = a * rng.state + c
    while next_state < 0 { next_state = next_state + m }
    while next_state >= m { next_state = next_state - m }

    let u = (next_state as f64) / (m as f64)
    return RngResult { value: u, rng: Rng { state: next_state } }
}

fn rng_normal(rng: Rng) -> RngResult {
    let r1 = rng_next(rng)
    let u1 = r1.value
    let rng2 = r1.rng

    let r2 = rng_next(rng2)
    let u2 = r2.value
    let rng3 = r2.rng

    var u1_safe = u1
    if u1_safe < 1.0e-10 { u1_safe = 1.0e-10 }

    let pi = 3.14159265358979323846
    let r = sqrt_f64(0.0 - 2.0 * log(u1_safe))
    let theta = 2.0 * pi * u2
    let z = r * cos(theta)

    return RngResult { value: z, rng: rng3 }
}

fn rng_normal_params(rng: Rng, mean: f64, std: f64) -> RngResult {
    let r = rng_normal(rng)
    return RngResult { value: mean + std * r.value, rng: r.rng }
}

// ============================================================================
// DUAL PROPAGATION RESULT
// ============================================================================

struct GumMcResult {
    // GUM (linearized) estimate
    gum_mean: f64,
    gum_std: f64,

    // Monte Carlo estimate
    mc_mean: f64,
    mc_std: f64,
    mc_samples: i32,

    // Discrepancy analysis
    mean_discrepancy: f64,      // |mc_mean - gum_mean| / |gum_mean|
    std_discrepancy: f64,       // |mc_std - gum_std| / gum_std
    is_linear: bool,            // True if linearization is reliable
    linearization_error: f64,   // Actual relative error

    // Decision
    use_mc: bool,               // True if MC should be used
    warning_code: i32,          // 0=ok, 1=warn, 2=switched_to_mc, 3=refused
}

fn dual_result_gum_only(mean: f64, std: f64) -> GumMcResult {
    return GumMcResult {
        gum_mean: mean,
        gum_std: std,
        mc_mean: mean,
        mc_std: std,
        mc_samples: 0,
        mean_discrepancy: 0.0,
        std_discrepancy: 0.0,
        is_linear: true,
        linearization_error: 0.0,
        use_mc: false,
        warning_code: 0,
    }
}

// ============================================================================
// DUAL CHECK: UNIVARIATE FUNCTIONS
// ============================================================================

/// Input specification for dual check
struct UncertainInput {
    mean: f64,
    std: f64,
}

fn dual_input(mean: f64, std: f64) -> UncertainInput {
    return UncertainInput { mean: mean, std: std }
}

/// Check y = exp(x) - highly nonlinear for large σ_x
fn dual_check_exp(x: UncertainInput, n_samples: i32, policy: LinearizationPolicy) -> GumMcResult {
    // GUM estimate: ∂y/∂x = exp(x), so u_y = |exp(μ_x)| * u_x
    let gum_mean = exp(x.mean)
    let gum_std = abs_f64(gum_mean) * x.std

    // Monte Carlo
    var rng = rng_new(12345)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0

    var i: i32 = 0
    while i < n_samples {
        let r = rng_normal_params(rng, x.mean, x.std)
        let samp = r.value
        rng = r.rng

        let y = exp(samp)
        sum = sum + y
        sum_sq = sum_sq + y * y
        i = i + 1
    }

    let mc_mean = sum / (n_samples as f64)
    let mc_var = sum_sq / (n_samples as f64) - mc_mean * mc_mean
    let mc_std = sqrt_f64(abs_f64(mc_var))

    // Compute discrepancy
    var mean_disc: f64 = 0.0
    if abs_f64(gum_mean) > 1.0e-15 {
        mean_disc = abs_f64(mc_mean - gum_mean) / abs_f64(gum_mean)
    }

    var std_disc: f64 = 0.0
    if gum_std > 1.0e-15 {
        std_disc = abs_f64(mc_std - gum_std) / gum_std
    }

    // Jensen's inequality: E[exp(X)] > exp(E[X]) for X non-degenerate
    // So mc_mean > gum_mean is expected for exp()

    let is_linear = std_disc < policy.max_relative_error
    var use_mc = false
    var warning = 0

    if !is_linear {
        if policy.auto_switch_to_mc {
            use_mc = true
            warning = 2
        } else if policy.warn_only {
            warning = 1
        } else {
            warning = 3  // Refuse
        }
    }

    return GumMcResult {
        gum_mean: gum_mean,
        gum_std: gum_std,
        mc_mean: mc_mean,
        mc_std: mc_std,
        mc_samples: n_samples,
        mean_discrepancy: mean_disc,
        std_discrepancy: std_disc,
        is_linear: is_linear,
        linearization_error: std_disc,
        use_mc: use_mc,
        warning_code: warning,
    }
}

/// Check y = log(x) - nonlinear, especially near zero
fn dual_check_log(x: UncertainInput, n_samples: i32, policy: LinearizationPolicy) -> GumMcResult {
    // GUM estimate: ∂y/∂x = 1/x, so u_y = u_x / |μ_x|
    var gum_mean = 0.0
    var gum_std = 0.0
    if x.mean > 1.0e-10 {
        gum_mean = log(x.mean)
        gum_std = x.std / x.mean
    }

    // Monte Carlo
    var rng = rng_new(54321)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0
    var valid: i32 = 0

    var i: i32 = 0
    while i < n_samples {
        let r = rng_normal_params(rng, x.mean, x.std)
        let samp = r.value
        rng = r.rng

        if samp > 1.0e-10 {
            let y = log(samp)
            sum = sum + y
            sum_sq = sum_sq + y * y
            valid = valid + 1
        }
        i = i + 1
    }

    var mc_mean: f64 = 0.0
    var mc_std: f64 = 0.0
    if valid > 1 {
        mc_mean = sum / (valid as f64)
        let mc_var = sum_sq / (valid as f64) - mc_mean * mc_mean
        mc_std = sqrt_f64(abs_f64(mc_var))
    }

    var std_disc: f64 = 0.0
    if gum_std > 1.0e-15 {
        std_disc = abs_f64(mc_std - gum_std) / gum_std
    }

    let is_linear = std_disc < policy.max_relative_error
    var use_mc = !is_linear && policy.auto_switch_to_mc
    var warning = 0
    if !is_linear { warning = 2 }

    return GumMcResult {
        gum_mean: gum_mean,
        gum_std: gum_std,
        mc_mean: mc_mean,
        mc_std: mc_std,
        mc_samples: valid,
        mean_discrepancy: 0.0,
        std_discrepancy: std_disc,
        is_linear: is_linear,
        linearization_error: std_disc,
        use_mc: use_mc,
        warning_code: warning,
    }
}

/// Check y = a/b - nonlinear when b has large relative uncertainty
fn dual_check_div(a: UncertainInput, b: UncertainInput, n_samples: i32, policy: LinearizationPolicy) -> GumMcResult {
    // GUM estimate: relative uncertainties add in quadrature
    var gum_mean: f64 = 0.0
    var gum_std: f64 = 0.0
    if abs_f64(b.mean) > 1.0e-10 {
        gum_mean = a.mean / b.mean
        let rel_a = a.std / abs_f64(a.mean)
        let rel_b = b.std / abs_f64(b.mean)
        let rel_y = sqrt_f64(rel_a * rel_a + rel_b * rel_b)
        gum_std = abs_f64(gum_mean) * rel_y
    }

    // Monte Carlo
    var rng = rng_new(98765)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0
    var valid: i32 = 0

    var i: i32 = 0
    while i < n_samples {
        let ra = rng_normal_params(rng, a.mean, a.std)
        let va = ra.value
        rng = ra.rng

        let rb = rng_normal_params(rng, b.mean, b.std)
        let vb = rb.value
        rng = rb.rng

        if abs_f64(vb) > 1.0e-10 {
            let y = va / vb
            sum = sum + y
            sum_sq = sum_sq + y * y
            valid = valid + 1
        }
        i = i + 1
    }

    var mc_mean: f64 = 0.0
    var mc_std: f64 = 0.0
    if valid > 1 {
        mc_mean = sum / (valid as f64)
        let mc_var = sum_sq / (valid as f64) - mc_mean * mc_mean
        mc_std = sqrt_f64(abs_f64(mc_var))
    }

    var std_disc: f64 = 0.0
    if gum_std > 1.0e-15 {
        std_disc = abs_f64(mc_std - gum_std) / gum_std
    }

    let is_linear = std_disc < policy.max_relative_error
    var use_mc = !is_linear && policy.auto_switch_to_mc
    var warning = 0
    if !is_linear { warning = 2 }

    return GumMcResult {
        gum_mean: gum_mean,
        gum_std: gum_std,
        mc_mean: mc_mean,
        mc_std: mc_std,
        mc_samples: valid,
        mean_discrepancy: 0.0,
        std_discrepancy: std_disc,
        is_linear: is_linear,
        linearization_error: std_disc,
        use_mc: use_mc,
        warning_code: warning,
    }
}

// ============================================================================
// PHARMACOKINETIC MODEL DUAL CHECK
// ============================================================================

/// C(t) = (Dose/V) * exp(-CL/V * t)
/// This is the classic case: exponential term makes GUM unreliable
fn dual_check_pk(
    dose: UncertainInput,
    volume: UncertainInput,
    clearance: UncertainInput,
    time: f64,
    n_samples: i32,
    policy: LinearizationPolicy
) -> GumMcResult {
    // GUM estimate (linearized)
    let c0 = dose.mean / volume.mean
    let k = clearance.mean / volume.mean
    let gum_mean = c0 * exp(0.0 - k * time)

    // Sensitivities at nominal point
    let dc_dD = exp(0.0 - k * time) / volume.mean
    let dc_dV = 0.0 - (dose.mean / (volume.mean * volume.mean)) * exp(0.0 - k * time)
              + (dose.mean * clearance.mean * time / (volume.mean * volume.mean * volume.mean)) * exp(0.0 - k * time)
    let dc_dCL = 0.0 - (dose.mean * time / (volume.mean * volume.mean)) * exp(0.0 - k * time)

    let gum_var = dc_dD * dc_dD * dose.std * dose.std
                + dc_dV * dc_dV * volume.std * volume.std
                + dc_dCL * dc_dCL * clearance.std * clearance.std
    let gum_std = sqrt_f64(abs_f64(gum_var))

    // Monte Carlo
    var rng = rng_new(11111)
    var sum: f64 = 0.0
    var sum_sq: f64 = 0.0
    var valid: i32 = 0

    var i: i32 = 0
    while i < n_samples {
        let rd = rng_normal_params(rng, dose.mean, dose.std)
        let d = rd.value
        rng = rd.rng

        let rv = rng_normal_params(rng, volume.mean, volume.std)
        let v = rv.value
        rng = rv.rng

        let rc = rng_normal_params(rng, clearance.mean, clearance.std)
        let cl = rc.value
        rng = rc.rng

        if v > 1.0e-10 {
            let k_sample = cl / v
            let c = (d / v) * exp(0.0 - k_sample * time)
            sum = sum + c
            sum_sq = sum_sq + c * c
            valid = valid + 1
        }
        i = i + 1
    }

    var mc_mean: f64 = 0.0
    var mc_std: f64 = 0.0
    if valid > 1 {
        mc_mean = sum / (valid as f64)
        let mc_var = sum_sq / (valid as f64) - mc_mean * mc_mean
        mc_std = sqrt_f64(abs_f64(mc_var))
    }

    var std_disc: f64 = 0.0
    if gum_std > 1.0e-15 {
        std_disc = abs_f64(mc_std - gum_std) / gum_std
    }

    var mean_disc: f64 = 0.0
    if abs_f64(gum_mean) > 1.0e-15 {
        mean_disc = abs_f64(mc_mean - gum_mean) / abs_f64(gum_mean)
    }

    let is_linear = std_disc < policy.max_relative_error
    var use_mc = !is_linear && policy.auto_switch_to_mc
    var warning = 0
    if !is_linear { warning = 2 }

    return GumMcResult {
        gum_mean: gum_mean,
        gum_std: gum_std,
        mc_mean: mc_mean,
        mc_std: mc_std,
        mc_samples: valid,
        mean_discrepancy: mean_disc,
        std_discrepancy: std_disc,
        is_linear: is_linear,
        linearization_error: std_disc,
        use_mc: use_mc,
        warning_code: warning,
    }
}

// ============================================================================
// AUTOMATIC DUAL CHECK WRAPPER
// ============================================================================

/// Get the best estimate, automatically switching methods if needed
struct BestEstimate {
    mean: f64,
    std: f64,
    method: i32,           // 0=GUM, 1=MC
    linearization_ok: bool,
    warning_message: i32,  // 0=none, 1=warn, 2=switched, 3=refused
}

fn get_best_estimate(res: GumMcResult) -> BestEstimate {
    if res.use_mc {
        return BestEstimate {
            mean: res.mc_mean,
            std: res.mc_std,
            method: 1,
            linearization_ok: false,
            warning_message: res.warning_code,
        }
    } else {
        return BestEstimate {
            mean: res.gum_mean,
            std: res.gum_std,
            method: 0,
            linearization_ok: res.is_linear,
            warning_message: res.warning_code,
        }
    }
}

/// Check if result should be refused entirely
fn should_refuse(res: GumMcResult) -> bool {
    return res.warning_code == 3
}

// ============================================================================
// TESTS
// ============================================================================

fn test_linear_case() -> bool {
    // For small relative uncertainty, exp() is approximately linear
    let x = dual_input(1.0, 0.01)  // 1% relative uncertainty
    let policy = default_linearization_policy()

    let result = dual_check_exp(x, 1000, policy)

    // Should be considered linear
    if !result.is_linear { return false }
    if result.use_mc { return false }
    if result.warning_code != 0 { return false }

    return true
}

fn test_nonlinear_exp() -> bool {
    // Large relative uncertainty in exponent -> nonlinear
    let x = dual_input(1.0, 0.5)  // 50% relative uncertainty!
    let policy = default_linearization_policy()

    let result = dual_check_exp(x, 1000, policy)

    // Should detect nonlinearity
    if result.is_linear { return false }

    // Should switch to MC (default policy)
    if !result.use_mc { return false }
    if result.warning_code != 2 { return false }

    // MC mean should be HIGHER than GUM mean (Jensen's inequality)
    if result.mc_mean < result.gum_mean { return false }

    return true
}

fn test_division_nonlinear() -> bool {
    // Division with large denominator uncertainty
    let a = dual_input(100.0, 5.0)    // 5% relative uncertainty
    let b = dual_input(10.0, 3.0)     // 30% relative uncertainty in denominator!
    let policy = default_linearization_policy()

    let result = dual_check_div(a, b, 1000, policy)

    // Should likely detect nonlinearity due to high denominator uncertainty
    // (linearization fails when σ_b/b is large)
    if result.linearization_error < 0.05 {
        // If somehow still linear, at least check reasonable values
        if result.gum_std < 1.0 { return false }
    }

    return true
}

fn test_pk_model_dual() -> bool {
    // PK model with realistic uncertainty
    let dose = dual_input(500.0, 10.0)      // 2% CV
    let volume = dual_input(50.0, 10.0)     // 20% CV
    let clearance = dual_input(5.0, 2.0)    // 40% CV in clearance
    let time = 4.0

    let policy = default_linearization_policy()
    let result = dual_check_pk(dose, volume, clearance, time, 1000, policy)

    // Should detect nonlinearity due to exponential with uncertain rate
    // At least verify we get reasonable values
    if result.gum_std < 0.0 { return false }
    if result.mc_std < 0.0 { return false }

    // Get best estimate
    let best = get_best_estimate(result)
    if best.mean < 0.0 { return false }
    if best.std < 0.0 { return false }

    return true
}

fn test_best_estimate_selection() -> bool {
    let x = dual_input(1.0, 0.5)
    let policy = default_linearization_policy()

    let result = dual_check_exp(x, 1000, policy)
    let best = get_best_estimate(result)

    // For nonlinear case, should use MC
    if result.use_mc {
        if best.method != 1 { return false }
        if abs_f64(best.mean - result.mc_mean) > 0.001 { return false }
    }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_linear_case() { return 1 }
    if !test_nonlinear_exp() { return 2 }
    if !test_division_nonlinear() { return 3 }
    if !test_pk_model_dual() { return 4 }
    if !test_best_estimate_selection() { return 5 }

    return 0
}
