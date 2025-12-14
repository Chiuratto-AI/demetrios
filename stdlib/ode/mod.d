// ode/mod.d - ODE Solver Module for Demetrios
//
// This module provides numerical ODE solvers for scientific computing,
// specifically designed for PBPK (Physiologically-Based Pharmacokinetic) simulations.
//
// Solvers available:
// - tsit5: Tsitouras 5(4) adaptive RK method (recommended for non-stiff ODEs)
// - rk4: Classic 4th-order Runge-Kutta (fixed step, simpler)
//
// Usage:
//   import ode::tsit5;
//   let sol = solve_scalar(my_ode_fn, u0, t0, t_end, default_config());

module ode

// Re-export main solver
pub use tsit5::*;

// Future solvers:
// pub use rk4::*;        // Fixed-step RK4
// pub use dopri5::*;     // Dormand-Prince 5(4)
// pub use rodas::*;      // Rosenbrock for stiff systems
// pub use radau::*;      // Radau IIA for very stiff systems
