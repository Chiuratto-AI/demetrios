//! ODE Solvers Demo
//!
//! Demonstrates the comprehensive ODE solver capabilities of Demetrios:
//! - Tsit5: Tsitouras 5(4) adaptive solver (recommended for most problems)
//! - DOPRI5: Dormand-Prince 5(4) adaptive solver (robust, widely used)
//! - RK4: Classic 4th-order fixed-step Runge-Kutta
//! - BDF: Backward differentiation formula (for stiff systems)
//!
//! Run with: dc run examples/ode_demo.d

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { return a }
    return b
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { return a }
    return b
}

fn fifth_root(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = 1.0
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    y = y - (y * y * y * y * y - x) / (5.0 * y * y * y * y)
    return y
}

// Approximate exp using Taylor series
fn exp_approx(x: f64) -> f64 {
    let x2 = x * x
    let x3 = x2 * x
    let x4 = x3 * x
    let x5 = x4 * x
    let x6 = x5 * x
    let x7 = x6 * x
    let x8 = x7 * x
    return 1.0 + x + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 + x6/720.0 + x7/5040.0 + x8/40320.0
}

// Analytical solution for exponential decay
fn exp_decay_analytical(u0: f64, k: f64, t: f64) -> f64 {
    return u0 * exp_approx(0.0 - k * t)
}

// ============================================================================
// SOLVER CONFIGURATION
// ============================================================================

struct ODEConfig {
    rtol: f64,
    atol: f64,
    dt_init: f64,
    dt_min: f64,
    dt_max: f64,
    max_steps: i64,
    safety: f64,
    max_growth: f64,
    min_shrink: f64
}

fn default_config() -> ODEConfig {
    return ODEConfig {
        rtol: 0.0001,
        atol: 0.0000001,
        dt_init: 0.1,
        dt_min: 0.000000000001,
        dt_max: 1000000.0,
        max_steps: 1000000,
        safety: 0.9,
        max_growth: 10.0,
        min_shrink: 0.2
    }
}

fn high_accuracy_config() -> ODEConfig {
    return ODEConfig {
        rtol: 0.0000000001,
        atol: 0.000000000001,
        dt_init: 0.0,
        dt_min: 0.000000000000001,
        dt_max: 1.0,
        max_steps: 10000000,
        safety: 0.9,
        max_growth: 5.0,
        min_shrink: 0.1
    }
}

// ============================================================================
// SOLUTION STRUCTURE
// ============================================================================

struct ODESolution {
    success: bool,
    nsteps: i64,
    nfeval: i64,
    nreject: i64,
    t_final: f64,
    u_final: f64
}

// ============================================================================
// TSIT5 SOLVER (Tsitouras 5(4))
// ============================================================================

// Stage times
fn c2() -> f64 { return 0.161 }
fn c3() -> f64 { return 0.327 }
fn c4() -> f64 { return 0.9 }
fn c5() -> f64 { return 0.9800255409045097 }
fn c6() -> f64 { return 1.0 }
fn c7() -> f64 { return 1.0 }

// Row coefficients
fn a21() -> f64 { return 0.161 }
fn a31() -> f64 { return 0.0 - 0.008480655492356989 }
fn a32() -> f64 { return 0.335480655492357 }
fn a41() -> f64 { return 2.8971530571054935 }
fn a42() -> f64 { return 0.0 - 6.359448489975075 }
fn a43() -> f64 { return 4.3622954328695815 }
fn a51() -> f64 { return 5.325864828439257 }
fn a52() -> f64 { return 0.0 - 11.748883564062828 }
fn a53() -> f64 { return 7.4955393428898365 }
fn a54() -> f64 { return 0.0 - 0.09249506636175525 }
fn a61() -> f64 { return 5.86145544294642 }
fn a62() -> f64 { return 0.0 - 12.92096931784711 }
fn a63() -> f64 { return 8.159367898576159 }
fn a64() -> f64 { return 0.0 - 0.071584973281401 }
fn a65() -> f64 { return 0.0 - 0.028269050394068383 }
fn a71() -> f64 { return 0.09646076681806523 }
fn a72() -> f64 { return 0.01 }
fn a73() -> f64 { return 0.4798896504144996 }
fn a74() -> f64 { return 1.379008574103742 }
fn a75() -> f64 { return 0.0 - 3.290069515436081 }
fn a76() -> f64 { return 2.324710524099774 }

// 5th order weights
fn b1() -> f64 { return 0.09646076681806523 }
fn b2() -> f64 { return 0.01 }
fn b3() -> f64 { return 0.4798896504144996 }
fn b4() -> f64 { return 1.379008574103742 }
fn b5() -> f64 { return 0.0 - 3.290069515436081 }
fn b6() -> f64 { return 2.324710524099774 }
fn b7() -> f64 { return 0.0 }

// Error coefficients
fn e1() -> f64 { return 0.00178001105222577714 }
fn e2() -> f64 { return 0.0008164344596567469 }
fn e3() -> f64 { return 0.0 - 0.007880878010261995 }
fn e4() -> f64 { return 0.1447110071732629 }
fn e5() -> f64 { return 0.0 - 0.5823571654525552 }
fn e6() -> f64 { return 0.45808210592918697 }
fn e7() -> f64 { return 0.0 - 0.01515151515151515 }

// ODE: du/dt = -k*u
fn exp_decay_ode(u: f64, t: f64) -> f64 {
    let k = 0.1
    return 0.0 - k * u
}

struct StepResult {
    u_new: f64,
    err: f64
}

fn tsit5_step_exp(u_in: f64, t: f64, dt: f64) -> StepResult {
    let k1 = exp_decay_ode(u_in, t)
    let k2 = exp_decay_ode(u_in + dt * a21() * k1, t + c2() * dt)
    let k3 = exp_decay_ode(u_in + dt * (a31() * k1 + a32() * k2), t + c3() * dt)
    let k4 = exp_decay_ode(u_in + dt * (a41() * k1 + a42() * k2 + a43() * k3), t + c4() * dt)
    let k5 = exp_decay_ode(u_in + dt * (a51() * k1 + a52() * k2 + a53() * k3 + a54() * k4), t + c5() * dt)
    let k6 = exp_decay_ode(u_in + dt * (a61() * k1 + a62() * k2 + a63() * k3 + a64() * k4 + a65() * k5), t + c6() * dt)
    let k7 = exp_decay_ode(u_in + dt * (a71() * k1 + a72() * k2 + a73() * k3 + a74() * k4 + a75() * k5 + a76() * k6), t + c7() * dt)

    let u_new = u_in + dt * (b1() * k1 + b2() * k2 + b3() * k3 + b4() * k4 + b5() * k5 + b6() * k6 + b7() * k7)
    let err = dt * (e1() * k1 + e2() * k2 + e3() * k3 + e4() * k4 + e5() * k5 + e6() * k6 + e7() * k7)

    return StepResult { u_new: u_new, err: abs_f64(err) }
}

fn solve_tsit5_exp(u0: f64, t0: f64, t_end: f64, config: ODEConfig) -> ODESolution {
    let mut u = u0
    let mut t = t0
    let mut dt = config.dt_init

    if dt <= 0.0 {
        dt = 0.01 * (t_end - t0)
    }

    let mut nsteps: i64 = 0
    let mut nfeval: i64 = 1
    let mut nreject: i64 = 0

    while t < t_end && nsteps < config.max_steps {
        let dt_use = if t + dt > t_end { t_end - t } else { dt }

        let result = tsit5_step_exp(u, t, dt_use)
        nfeval = nfeval + 7

        let scale = config.atol + config.rtol * max_f64(abs_f64(u), abs_f64(result.u_new))
        let err_norm = result.err / scale

        if err_norm <= 1.0 {
            t = t + dt_use
            u = result.u_new
            nsteps = nsteps + 1

            if err_norm > 0.0 {
                let factor = config.safety * fifth_root(1.0 / err_norm)
                let factor = min_f64(config.max_growth, factor)
                dt = dt * factor
            } else {
                dt = dt * config.max_growth
            }
        } else {
            nreject = nreject + 1
            let factor = config.safety * fifth_root(1.0 / err_norm)
            let factor = max_f64(config.min_shrink, factor)
            dt = dt * factor
        }

        dt = max_f64(config.dt_min, min_f64(config.dt_max, dt))
    }

    return ODESolution {
        success: t >= t_end - config.dt_min,
        nsteps: nsteps,
        nfeval: nfeval,
        nreject: nreject,
        t_final: t,
        u_final: u
    }
}

// ============================================================================
// DOPRI5 SOLVER (Dormand-Prince 5(4))
// ============================================================================

fn d_c2() -> f64 { 1.0/5.0 }
fn d_c3() -> f64 { 3.0/10.0 }
fn d_c4() -> f64 { 4.0/5.0 }
fn d_c5() -> f64 { 8.0/9.0 }

fn d_a21() -> f64 { 1.0/5.0 }
fn d_a31() -> f64 { 3.0/40.0 }
fn d_a32() -> f64 { 9.0/40.0 }
fn d_a41() -> f64 { 44.0/45.0 }
fn d_a42() -> f64 { 0.0 - 56.0/15.0 }
fn d_a43() -> f64 { 32.0/9.0 }
fn d_a51() -> f64 { 19372.0/6561.0 }
fn d_a52() -> f64 { 0.0 - 25360.0/2187.0 }
fn d_a53() -> f64 { 64448.0/6561.0 }
fn d_a54() -> f64 { 0.0 - 212.0/729.0 }
fn d_a61() -> f64 { 9017.0/3168.0 }
fn d_a62() -> f64 { 0.0 - 355.0/33.0 }
fn d_a63() -> f64 { 46732.0/5247.0 }
fn d_a64() -> f64 { 49.0/176.0 }
fn d_a65() -> f64 { 0.0 - 5103.0/18656.0 }

fn d_b1() -> f64 { 35.0/384.0 }
fn d_b3() -> f64 { 500.0/1113.0 }
fn d_b4() -> f64 { 125.0/192.0 }
fn d_b5() -> f64 { 0.0 - 2187.0/6784.0 }
fn d_b6() -> f64 { 11.0/84.0 }

fn d_e1() -> f64 { 71.0/57600.0 }
fn d_e3() -> f64 { 0.0 - 71.0/16695.0 }
fn d_e4() -> f64 { 71.0/1920.0 }
fn d_e5() -> f64 { 0.0 - 17253.0/339200.0 }
fn d_e6() -> f64 { 22.0/525.0 }
fn d_e7() -> f64 { 0.0 - 1.0/40.0 }

fn dopri5_step_exp(u_in: f64, t: f64, dt: f64) -> StepResult {
    let k = 0.1

    let k1 = 0.0 - k * u_in
    let k2 = 0.0 - k * (u_in + dt * d_a21() * k1)
    let k3 = 0.0 - k * (u_in + dt * (d_a31() * k1 + d_a32() * k2))
    let k4 = 0.0 - k * (u_in + dt * (d_a41() * k1 + d_a42() * k2 + d_a43() * k3))
    let k5 = 0.0 - k * (u_in + dt * (d_a51() * k1 + d_a52() * k2 + d_a53() * k3 + d_a54() * k4))
    let k6 = 0.0 - k * (u_in + dt * (d_a61() * k1 + d_a62() * k2 + d_a63() * k3 + d_a64() * k4 + d_a65() * k5))

    let u_new = u_in + dt * (d_b1() * k1 + d_b3() * k3 + d_b4() * k4 + d_b5() * k5 + d_b6() * k6)
    let k7 = 0.0 - k * u_new
    let err = dt * (d_e1() * k1 + d_e3() * k3 + d_e4() * k4 + d_e5() * k5 + d_e6() * k6 + d_e7() * k7)

    return StepResult { u_new: u_new, err: abs_f64(err) }
}

fn solve_dopri5_exp(u0: f64, t0: f64, t_end: f64, config: ODEConfig) -> ODESolution {
    let mut u = u0
    let mut t = t0
    let mut dt = config.dt_init

    if dt <= 0.0 {
        dt = 0.01 * (t_end - t0)
    }

    let mut nsteps: i64 = 0
    let mut nfeval: i64 = 0
    let mut nreject: i64 = 0

    while t < t_end && nsteps < config.max_steps {
        let dt_use = if t + dt > t_end { t_end - t } else { dt }

        let result = dopri5_step_exp(u, t, dt_use)
        nfeval = nfeval + 7

        let scale = config.atol + config.rtol * abs_f64(u)
        let err_norm = result.err / scale

        if err_norm <= 1.0 {
            t = t + dt_use
            u = result.u_new
            nsteps = nsteps + 1

            if err_norm > 0.0 {
                let factor = config.safety * fifth_root(1.0 / err_norm)
                let factor = min_f64(config.max_growth, factor)
                dt = dt * factor
            } else {
                dt = dt * config.max_growth
            }
        } else {
            nreject = nreject + 1
            let factor = config.safety * fifth_root(1.0 / err_norm)
            let factor = max_f64(config.min_shrink, factor)
            dt = dt * factor
        }

        dt = max_f64(config.dt_min, min_f64(config.dt_max, dt))
    }

    return ODESolution {
        success: t >= t_end - config.dt_min,
        nsteps: nsteps,
        nfeval: nfeval,
        nreject: nreject,
        t_final: t,
        u_final: u
    }
}

// ============================================================================
// RK4 SOLVER (Classic 4th-order Runge-Kutta)
// ============================================================================

fn solve_rk4_exp(u0: f64, t0: f64, t_end: f64, n_steps: i64) -> ODESolution {
    let dt = (t_end - t0) / (n_steps as f64)
    let k = 0.1
    let mut u = u0
    let mut t = t0
    let mut nsteps: i64 = 0

    while nsteps < n_steps {
        let k1 = 0.0 - k * u
        let k2 = 0.0 - k * (u + 0.5 * dt * k1)
        let k3 = 0.0 - k * (u + 0.5 * dt * k2)
        let k4 = 0.0 - k * (u + dt * k3)

        u = u + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
        nsteps = nsteps + 1
    }

    return ODESolution {
        success: true,
        nsteps: nsteps,
        nfeval: nsteps * 4,
        nreject: 0,
        t_final: t,
        u_final: u
    }
}

// ============================================================================
// BDF1 SOLVER (Backward Euler - Implicit)
// ============================================================================

fn solve_bdf1_exp(u0: f64, t0: f64, t_end: f64, n_steps: i64) -> ODESolution {
    let dt = (t_end - t0) / (n_steps as f64)
    let k = 0.1
    let mut u = u0
    let mut t = t0
    let mut nsteps: i64 = 0

    // For du/dt = -k*u, backward Euler:
    // u_{n+1} = u_n / (1 + k*dt)
    let factor = 1.0 / (1.0 + k * dt)

    while nsteps < n_steps {
        u = u * factor
        t = t + dt
        nsteps = nsteps + 1
    }

    return ODESolution {
        success: true,
        nsteps: nsteps,
        nfeval: nsteps,
        nreject: 0,
        t_final: t,
        u_final: u
    }
}

// ============================================================================
// 3-COMPARTMENT PK MODEL
// ============================================================================

struct PKState3 {
    gut: f64,
    central: f64,
    periph: f64,
    t: f64
}

struct PKParams3 {
    ka: f64,
    ke: f64,
    k12: f64,
    k21: f64
}

fn solve_rk4_pk3(s0: PKState3, pk: PKParams3, t_end: f64, n_steps: i64) -> PKState3 {
    let dt = t_end / (n_steps as f64)
    let mut g = s0.gut
    let mut c = s0.central
    let mut p = s0.periph
    let mut t = s0.t

    let mut i: i64 = 0
    while i < n_steps {
        // k1
        let dg1 = 0.0 - pk.ka * g
        let dc1 = pk.ka * g - (pk.ke + pk.k12) * c + pk.k21 * p
        let dp1 = pk.k12 * c - pk.k21 * p

        // k2
        let g2 = g + 0.5 * dt * dg1
        let c2 = c + 0.5 * dt * dc1
        let p2 = p + 0.5 * dt * dp1
        let dg2 = 0.0 - pk.ka * g2
        let dc2 = pk.ka * g2 - (pk.ke + pk.k12) * c2 + pk.k21 * p2
        let dp2 = pk.k12 * c2 - pk.k21 * p2

        // k3
        let g3 = g + 0.5 * dt * dg2
        let c3 = c + 0.5 * dt * dc2
        let p3 = p + 0.5 * dt * dp2
        let dg3 = 0.0 - pk.ka * g3
        let dc3 = pk.ka * g3 - (pk.ke + pk.k12) * c3 + pk.k21 * p3
        let dp3 = pk.k12 * c3 - pk.k21 * p3

        // k4
        let g4 = g + dt * dg3
        let c4 = c + dt * dc3
        let p4 = p + dt * dp3
        let dg4 = 0.0 - pk.ka * g4
        let dc4 = pk.ka * g4 - (pk.ke + pk.k12) * c4 + pk.k21 * p4
        let dp4 = pk.k12 * c4 - pk.k21 * p4

        // Update
        g = g + dt / 6.0 * (dg1 + 2.0 * dg2 + 2.0 * dg3 + dg4)
        c = c + dt / 6.0 * (dc1 + 2.0 * dc2 + 2.0 * dc3 + dc4)
        p = p + dt / 6.0 * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        t = t + dt
        i = i + 1
    }

    return PKState3 { gut: g, central: c, periph: p, t: t }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    println("=== Demetrios ODE Solver Demo ===")
    println("")

    // ========================================================================
    // PART 1: Simple Exponential Decay
    // ========================================================================
    println("--- Part 1: Exponential Decay (du/dt = -0.1*u) ---")
    println("")

    let u0 = 100.0
    let t0 = 0.0
    let t_end = 10.0

    let analytical = exp_decay_analytical(u0, 0.1, t_end)
    println("Analytical solution at t=10: {}", analytical)
    println("")

    // Tsit5
    println("Tsit5 Solver (adaptive, 5th order):")
    let sol_tsit5 = solve_tsit5_exp(u0, t0, t_end, default_config())
    println("  u(10) = {}", sol_tsit5.u_final)
    println("  Error = {}", abs_f64(sol_tsit5.u_final - analytical))
    println("  Steps = {}", sol_tsit5.nsteps)
    println("  Function evals = {}", sol_tsit5.nfeval)
    println("")

    // DOPRI5
    println("DOPRI5 Solver (adaptive, 5th order):")
    let sol_dopri = solve_dopri5_exp(u0, t0, t_end, default_config())
    println("  u(10) = {}", sol_dopri.u_final)
    println("  Error = {}", abs_f64(sol_dopri.u_final - analytical))
    println("  Steps = {}", sol_dopri.nsteps)
    println("  Function evals = {}", sol_dopri.nfeval)
    println("")

    // RK4
    println("RK4 Solver (fixed-step, 4th order, 100 steps):")
    let sol_rk4 = solve_rk4_exp(u0, t0, t_end, 100)
    println("  u(10) = {}", sol_rk4.u_final)
    println("  Error = {}", abs_f64(sol_rk4.u_final - analytical))
    println("  Steps = {}", sol_rk4.nsteps)
    println("")

    // BDF1
    println("BDF1 Solver (implicit, 1st order, 1000 steps):")
    let sol_bdf = solve_bdf1_exp(u0, t0, t_end, 1000)
    println("  u(10) = {}", sol_bdf.u_final)
    println("  Error = {}", abs_f64(sol_bdf.u_final - analytical))
    println("  Steps = {}", sol_bdf.nsteps)
    println("")

    // ========================================================================
    // PART 2: Solver Accuracy Comparison
    // ========================================================================
    println("--- Part 2: Accuracy vs Step Count ---")
    println("")

    println("RK4 with different step counts:")
    let sol_rk4_10 = solve_rk4_exp(u0, t0, t_end, 10)
    let sol_rk4_100 = solve_rk4_exp(u0, t0, t_end, 100)
    let sol_rk4_1000 = solve_rk4_exp(u0, t0, t_end, 1000)
    println("  10 steps:   error = {}", abs_f64(sol_rk4_10.u_final - analytical))
    println("  100 steps:  error = {}", abs_f64(sol_rk4_100.u_final - analytical))
    println("  1000 steps: error = {}", abs_f64(sol_rk4_1000.u_final - analytical))
    println("")

    println("BDF1 with different step counts:")
    let sol_bdf_100 = solve_bdf1_exp(u0, t0, t_end, 100)
    let sol_bdf_1000 = solve_bdf1_exp(u0, t0, t_end, 1000)
    let sol_bdf_10000 = solve_bdf1_exp(u0, t0, t_end, 10000)
    println("  100 steps:   error = {}", abs_f64(sol_bdf_100.u_final - analytical))
    println("  1000 steps:  error = {}", abs_f64(sol_bdf_1000.u_final - analytical))
    println("  10000 steps: error = {}", abs_f64(sol_bdf_10000.u_final - analytical))
    println("")

    // ========================================================================
    // PART 3: 3-Compartment PK Model
    // ========================================================================
    println("--- Part 3: 3-Compartment Pharmacokinetic Model ---")
    println("")

    println("Model: Gut -> Central <-> Peripheral")
    println("Parameters: ka=1.0, ke=0.1, k12=0.3, k21=0.2 h^-1")
    println("Initial: 500 mg in gut")
    println("")

    let pk = PKParams3 { ka: 1.0, ke: 0.1, k12: 0.3, k21: 0.2 }
    let s0 = PKState3 { gut: 500.0, central: 0.0, periph: 0.0, t: 0.0 }

    let pk_sol = solve_rk4_pk3(s0, pk, 24.0, 2400)
    println("After 24 hours:")
    println("  Gut:        {} mg", pk_sol.gut)
    println("  Central:    {} mg", pk_sol.central)
    println("  Peripheral: {} mg", pk_sol.periph)

    let total = pk_sol.gut + pk_sol.central + pk_sol.periph
    let eliminated = 500.0 - total
    println("  Remaining:  {} mg", total)
    println("  Eliminated: {} mg", eliminated)
    println("")

    // ========================================================================
    // PART 4: Solver Selection Guide
    // ========================================================================
    println("--- Part 4: Solver Selection Guide ---")
    println("")
    println("| Problem Type        | Solver | Why                      |")
    println("|---------------------|--------|--------------------------|")
    println("| Non-stiff, smooth   | Tsit5  | Fast, accurate, adaptive |")
    println("| Non-stiff, general  | DOPRI5 | Robust, widely used      |")
    println("| Real-time/embedded  | RK4    | Predictable, fixed step  |")
    println("| Stiff systems       | BDF    | L-stable, implicit       |")
    println("")

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println("=== Demo Complete ===")
    println("")
    println("Features demonstrated:")
    println("  - Tsit5 adaptive solver (Tsitouras 5(4))")
    println("  - DOPRI5 adaptive solver (Dormand-Prince 5(4))")
    println("  - RK4 fixed-step solver (classic 4th order)")
    println("  - BDF1 implicit solver (backward Euler)")
    println("  - 3-compartment pharmacokinetic model")
    println("")

    return 0
}

