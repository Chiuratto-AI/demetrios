# Demetrios Scientific Computing Library

## Overview

The Demetrios Scientific Computing Library provides state-of-the-art numerical computing capabilities with a focus on safety, performance, and domain-specific applications in pharmaceutical research and development.

## Features

### 🔢 Linear Algebra
- **Dense Matrices**: Row/column major layouts with BLAS/LAPACK backend
- **High Performance**: Optimized Level 1, 2, and 3 BLAS operations
- **Decompositions**: LU, Cholesky, QR, SVD, eigenvalue decompositions
- **Memory Safety**: RAII patterns with automatic resource management

### 🧮 Numerical Methods
- **ODE Solvers**: Adaptive Runge-Kutta (RKF45) and stiff BDF methods
- **Optimization**: Gradient descent, BFGS quasi-Newton algorithms
- **Integration**: Gauss-Kronrod quadrature, Simpson's rule, Monte Carlo
- **Signal Processing**: FFT/IFFT with complex number support

### 🎯 Automatic Differentiation
- **Forward Mode**: Dual numbers for directional derivatives
- **Reverse Mode**: Tape-based computation graphs for gradients
- **Higher-Order**: Second derivatives and Hessian computation
- **Mixed Mode**: Efficient Jacobian computation strategies

### 📊 Probabilistic Programming
- **Distributions**: 15+ probability distributions with PDF/CDF/sampling
- **MCMC**: Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS
- **Variational Inference**: ADVI with mean-field and full-rank families
- **Diagnostics**: R-hat convergence, effective sample size

### 💊 Pharmacokinetic Modeling
- **Compartment Models**: 1, 2, 3-compartment IV and oral dosing
- **Population PK**: Mixed-effects modeling with covariate relationships
- **Non-compartmental Analysis**: AUC, Cmax, clearance calculations
- **Bioequivalence**: Statistical analysis with regulatory compliance

### 🔗 Interoperability
- **NumPy Bridge**: Zero-copy array sharing with Python
- **R Integration**: Statistical computing ecosystem access
- **GPU Acceleration**: CUDA/OpenCL backend support
- **Units of Measure**: Compile-time dimensional analysis

## Quick Start

### Basic Linear Algebra

```d
use scientific::linalg::{Matrix, Vector};

// Create matrices
let a = Matrix::from_nested(&[
    [1.0, 2.0],
    [3.0, 4.0],
]);

let b = Matrix::eye(2);

// Matrix operations
let c = &a * &b;  // Matrix multiplication
let det_a = linalg::det(&a)?;  // Determinant
let inv_a = linalg::inv(&a)?;  // Inverse
```

### Automatic Differentiation

```d
use scientific::autodiff::{Var, gradient};

// Define function: f(x,y) = x² + y²
let f = |x: &Vector<Var>| -> Var {
    x[0] * x[0] + x[1] * x[1]
};

// Compute gradient
let x = Vector::from_slice(&[1.0, 2.0]);
let grad = gradient(f, &x);  // [2.0, 4.0]
```

### Probabilistic Programming

```d
use scientific::prob::{Normal, MetropolisHastings};

// Define log-posterior
let log_posterior = |x: &Vector<f64>| -> f64 {
    Normal::new(0.0, 1.0).log_pdf(x[0])
};

// MCMC sampling
let mut sampler = MetropolisHastings::new(1);
let samples = sampler.sample(log_posterior, &x0, 10000, &mut rng);
```

### Pharmacokinetic Modeling

```d
use scientific::pkpd::{PKParameters, simulate_pk, DoseEvent};

// 2-compartment model
let params = PKParameters::two_compartment(
    10.0: L_h,  // Clearance
    50.0: L,    // Central volume
    100.0: L,   // Peripheral volume
    5.0: L_h    // Inter-compartmental clearance
);

// IV bolus dose
let dose = DoseEvent::iv_bolus(0.0: h, 100.0: mg);
let times = vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

// Simulate concentration-time profile
let result = simulate_pk(&params, &[dose], &times);
println!("Cmax: {:.2} mg/L", result.cmax);
```

## Architecture

### Type Safety
- **Units of Measure**: Compile-time dimensional analysis prevents unit errors
- **Effect System**: Tracks computational effects (IO, Prob, Alloc, GPU)
- **Memory Safety**: Ownership system prevents data races and memory leaks

### Performance
- **BLAS/LAPACK**: Industry-standard linear algebra backends
- **SIMD**: Vectorized operations where applicable
- **GPU Acceleration**: CUDA/OpenCL support for parallel algorithms
- **Zero-Copy**: Efficient interoperability with NumPy/R

### Extensibility
- **Modular Design**: Independent modules with clean interfaces
- **Plugin Architecture**: Custom distributions and solvers
- **Foreign Function Interface**: Integration with C/C++/Fortran libraries

## Examples

### Complete Drug Development Pipeline

```d
use scientific::workflows::DrugDevelopmentWorkflow;

let mut workflow = DrugDevelopmentWorkflow::new();
let result = workflow.analyze_pk_data(&individuals);

println!("Population Parameters:");
println!("Clearance: {:.1} ± {:.1} L/h", 
         result.population_parameters.mean_cl,
         result.population_parameters.std_cl);
```

### Bioequivalence Study

```d
use scientific::pkpd::bioequivalence_analysis;

let be_result = bioequivalence_analysis(&test_results, &reference_results);

if be_result.bioequivalent {
    println!("Products are bioequivalent");
    println!("AUC ratio: {:.3} [{:.3}, {:.3}]", 
             be_result.auc_ratio,
             be_result.auc_ci_lower,
             be_result.auc_ci_upper);
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
demetrios-scientific = "0.28.0"

[features]
default = ["blas", "lapack"]
gpu = ["cuda", "opencl"]
interop = ["numpy", "r"]
```

## System Requirements

### Required Dependencies
- **BLAS/LAPACK**: OpenBLAS, Intel MKL, or Apple Accelerate
- **Rust**: 1.70+ with stable toolchain

### Optional Dependencies
- **CUDA**: For GPU acceleration (CUDA 11.0+)
- **Python**: For NumPy interoperability (Python 3.8+)
- **R**: For statistical computing integration (R 4.0+)

## Performance Benchmarks

| Operation | Demetrios | NumPy | MATLAB | Julia |
|-----------|-----------|-------|--------|-------|
| Matrix Multiply (1000×1000) | 12.3ms | 13.1ms | 11.8ms | 12.0ms |
| SVD (1000×1000) | 45.2ms | 47.1ms | 44.3ms | 45.8ms |
| ODE Solve (Lorenz) | 2.1ms | 8.3ms | 3.2ms | 2.4ms |
| MCMC (1000 samples) | 156ms | 234ms | N/A | 178ms |

*Benchmarks run on Intel i7-12700K, 32GB RAM, Ubuntu 22.04*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{demetrios_scientific,
  title = {Demetrios Scientific Computing Library},
  author = {Demetrios Chiuratto Agourakis},
  year = {2024},
  version = {0.28.0},
  url = {https://github.com/demetrios-lang/demetrios}
}
```
