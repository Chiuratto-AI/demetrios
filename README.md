# Demetrios (D)

A novel L0 systems + scientific programming language.

[![Version](https://img.shields.io/badge/version-0.28.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](LICENSE)

## Features

### 🔬 **Scientific Computing Revolution (v0.28.0)**
- **Linear Algebra**: BLAS/LAPACK integration with matrix decompositions
- **Numerical Methods**: ODE solvers, optimization, integration, FFT
- **Automatic Differentiation**: Forward/reverse mode with higher-order derivatives
- **Probabilistic Programming**: MCMC, variational inference, 15+ distributions
- **Pharmacokinetic Modeling**: Compartment models, population PK, NCA
- **Interoperability**: NumPy/R bridges with zero-copy array sharing

### 🚀 **Core Language Features**
- **Novel Syntax**: Designed for scientific and medical computing
- **Full Algebraic Effects**: IO, Mut, Alloc, GPU, Prob with handlers
- **Linear/Affine Types**: Safe resource management
- **Units of Measure**: Compile-time dimensional analysis
- **Refinement Types**: SMT-backed constraint verification
- **GPU-Native**: First-class GPU memory and kernels

### 🛠️ **Developer Experience**
- **IDE Support**: Full LSP server with VS Code extension
- **LLVM Backend**: Native AOT compilation with optimizations
- **Documentation Generator**: HTML docs, mdBook integration, doctests

## Building

```bash
cd compiler
cargo build --release

# With all features
cargo build --release --features "jit,smt,lsp"
```

## Usage

```bash
# Compile
dc compile program.d -o program

# Build native executable (requires --features llvm)
dc build program.d -O2

# Type check only
dc check program.d

# Run with JIT (requires --features jit)
dc run program.d

# REPL
dc repl

# Generate documentation
dc doc --open

# Generate mdBook
dc doc-book

# Run doctests
dc doctest

# Start LSP server (requires --features lsp)
demetrios-lsp --stdio
```

## IDE Support

Demetrios includes a full-featured Language Server Protocol (LSP) implementation:

- **Real-time Diagnostics**: Syntax, type, effect, and ownership errors
- **Hover Information**: Type info, documentation, and effect signatures
- **Go to Definition**: Navigate to function, type, and variable definitions
- **Find References**: Find all usages across the codebase
- **Code Completion**: Context-aware completions with snippets
- **Semantic Highlighting**: Rich syntax highlighting with custom token types

### VS Code Extension

Install the VS Code extension from `editors/vscode/`:

```bash
cd editors/vscode
npm install
npm run compile
# Then install the .vsix or use VS Code's "Developer: Install Extension from Location"
```

See [docs/lsp.md](docs/lsp.md) for detailed LSP documentation.

## Scientific Computing Examples

### Linear Algebra & Automatic Differentiation

```d
use scientific::linalg::{Matrix, Vector};
use scientific::autodiff::{Var, gradient};

fn main() with IO, Alloc {
    // Create matrices
    let a = Matrix::from_nested(&[
        [1.0, 2.0],
        [3.0, 4.0],
    ]);

    // Matrix operations with BLAS backend
    let eigenvals = linalg::eig(&a)?.values_real;
    println!("Eigenvalues: {:?}", eigenvals);

    // Automatic differentiation
    let f = |x: &Vector<Var>| -> Var {
        x[0] * x[0] + x[1] * x[1]  // f(x,y) = x² + y²
    };

    let x = Vector::from_slice(&[1.0, 2.0]);
    let grad = gradient(f, &x);  // [2.0, 4.0]
    println!("Gradient: {:?}", grad);
}
```

### Pharmacokinetic Modeling

```d
use scientific::pkpd::{PKParameters, simulate_pk, DoseEvent};
use units::{mg, L, h, L_h};

fn main() with IO, Alloc {
    // 2-compartment PK model
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
    println!("Cmax: {:.2} mg/L at {:.1} h", result.cmax, result.tmax);
    println!("Half-life: {:.1} h", params.half_life());
}
```

### Probabilistic Programming & MCMC

```d
use scientific::prob::{Normal, MetropolisHastings};

fn main() with IO, Prob {
    // Define log-posterior for Bayesian inference
    let log_posterior = |x: &Vector<f64>| -> f64 {
        // Prior: x ~ Normal(0, 1)
        let prior = Normal::new(0.0, 1.0).log_pdf(x[0]);

        // Likelihood: data ~ Normal(x, 0.5)
        let data = vec![0.5, 1.2, 0.8, 1.1, 0.9];
        let likelihood: f64 = data.iter()
            .map(|&y| Normal::new(x[0], 0.5).log_pdf(y))
            .sum();

        prior + likelihood
    };

    // MCMC sampling
    let mut sampler = MetropolisHastings::new(1);
    let x0 = Vector::from_slice(&[0.0]);
    let mut rng = rand::thread_rng();

    let samples = sampler.sample(log_posterior, &x0, 10000, &!rng);
    println!("Posterior mean: {:.3}", samples.samples.mean());
    println!("Acceptance rate: {:.1}%", samples.acceptance_rate * 100.0);
}
```

## Basic Language Example

```d
module example

let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume

fn simulate(params: PKParams) -> Vec<f64> with Prob, Alloc {
    let eta = sample(Normal(0.0, 0.3))
    // ...
}

kernel fn matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = gpu.thread_id.x
    // ...
}
```

## Architecture

```
Source -> Lexer -> Parser -> AST -> Type Checker -> HIR -> HLIR -> Codegen
                                           |
                                           v
                                    LSP Server (IDE)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Documentation

Demetrios includes a comprehensive documentation generator:

```d
/// Calculate drug concentration from dose and volume.
///
/// @param dose The drug dose in milligrams
/// @param volume The solution volume in milliliters
/// @returns Concentration in mg/mL
///
/// @example
/// ```d
/// let conc = calculate_concentration(500_mg, 10_mL)
/// assert(conc == 50_mg/mL)
/// ```
fn calculate_concentration(dose: mg, volume: mL) -> mg/mL {
    dose / volume
}
```

- **HTML Documentation**: Responsive, themed API documentation
- **mdBook Integration**: Generate readable guides and tutorials
- **Doctests**: Run code examples from documentation as tests
- **Coverage**: Track documentation coverage statistics

## Feature Flags

| Feature | Description |
|---------|-------------|
| `jit`   | Cranelift JIT backend for fast development |
| `llvm`  | LLVM backend for optimized native code |
| `smt`   | Z3 SMT solver for refinement type verification |
| `lsp`   | Language Server Protocol for IDE integration |
| `full`  | Enable all features |

## License

MIT OR Apache-2.0
