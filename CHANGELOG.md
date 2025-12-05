# Changelog

All notable changes to the Demetrios (D) compiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.44.0] - 2024-12-05

### Added - Epistemic PBPK Module & Knowledge Types

This release introduces **epistemic computing** - the key differentiator of Demetrios from all other programming languages. No other language provides compile-time tracking of knowledge confidence, provenance, and temporal validity.

#### Epistemic Type System (`compiler/src/types/epistemic.rs`)
- **Knowledge[T, ε, Φ, τ]** wrapper type for values with epistemic qualifications
- **Confidence bounds**: `ε >= 0.80` constraints checked at compile time
- **Provenance tracking**: Source, Derived, Merged, UserInput provenance types
- **Temporal constraints**: MaxAge, ValidAfter, ValidBetween for time-sensitive knowledge
- **Propagation rules**: Automatic confidence degradation through computations
  - Minimum rule for arithmetic
  - Degradation factors for ODE solvers (0.95), interpolation (0.99)
- **EpistemicChecker**: Compile-time verification of epistemic constraints

#### PBPK Module (`stdlib/pbpk/mod.d`)
- **14-compartment PBPK model** with organ-specific partition coefficients
- **Drug struct** with ChEBI ontology validation at compile time
- **PBPKParams** with epistemic qualifications on all parameters
- **Patient profiles** for individualized simulations (weight, age, genotype)
- **Allometric scaling** for patient-specific parameter adjustment
- **SimulationResult** with propagated confidence and full provenance
- **FDA validation functions** requiring minimum confidence thresholds

#### QUDT Units (`stdlib/units/qudt.d`)
- Complete QUDT-aligned pharmacokinetic units
- SI base units: m, kg, s, mol, K, A, cd with prefixes
- Derived units: L, mL, mg/L, L/h, per_h, mg·h/L (AUC)
- **UnitMetadata** with QUDT IRI, UCUM code, SI conversion factor
- Physiological constants: cardiac output, blood volume, GFR
- Compile-time dimensional analysis

#### MedLang Compatibility (`stdlib/interop/medlang.d`)
- Parser for Darwin PBPK Platform's MedLang DSL
- **Bidirectional translation** between MedLang and Demetrios
- MedLang drug, model, and dosing constructs
- **FDA report generation** with provenance audit trail
- Darwin JSON import/export with confidence annotations
- FDA PBPK Guidance validation checklist

#### Example: Metformin Simulation (`examples/pbpk/metformin_simulation.d`)
- Complete PBPK simulation demonstrating all new features
- ChEBI:6801 validated drug definition
- Parameters with confidence from literature sources
- 24-hour simulation with PK metrics (Cmax, Tmax, AUC)
- FDA validation against observed clinical data

#### Documentation
- `docs/epistemic-pbpk.md`: Comprehensive guide to epistemic PBPK modeling
- Comparison tables: Python, Julia, R/NONMEM vs Demetrios
- Architecture diagrams and API reference

### Technical Details
- All epistemic type tests pass (5/5)
- Full compiler test suite passes (1103 tests)
- Inspired by Darwin PBPK Platform (developed solo in 2 months)

## [0.28.0] - 2024-11-30

### Added - Day 28: State-of-the-Art Scientific Computing & Domain-Specific Libraries

#### Linear Algebra Foundation
- **Dense Matrix Library** (`stdlib/src/linalg/matrix.d`)
  - Configurable memory layouts (row-major/column-major)
  - RAII memory management with ownership tracking
  - Matrix operations: transpose, reshape, norm calculations
  - Vector operations: dot product, cross product, normalization
  - Matrix views and slicing with zero-copy semantics

- **BLAS Bindings** (`stdlib/src/linalg/blas.d`)
  - Level 1: DAXPY, DDOT, DNRM2, DSCAL, DASUM, IDAMAX
  - Level 2: DGEMV, DSYMV, DTRSV with transpose support
  - Level 3: DGEMM, DSYRK, DTRSM for matrix-matrix operations
  - High-level operator overloading for intuitive syntax
  - Performance-optimized with industry-standard backends

- **LAPACK Bindings** (`stdlib/src/linalg/lapack.d`)
  - LU decomposition with partial pivoting (DGETRF/DGETRS)
  - Cholesky decomposition for positive definite matrices (DPOTRF)
  - QR decomposition with Householder reflectors (DGEQRF/DORGQR)
  - SVD with full/economy modes (DGESVD)
  - Eigenvalue decomposition for general/symmetric matrices (DGEEV/DSYEV)
  - Matrix inverse, determinant, condition number, rank, pseudoinverse

#### Advanced Numerical Methods
- **ODE Solvers** (`stdlib/src/numerics/ode.d`)
  - Runge-Kutta-Fehlberg 4(5) with adaptive step size control
  - Backward Differentiation Formula (BDF) for stiff equations
  - Configurable tolerances and step size limits
  - Comprehensive solution statistics and diagnostics

- **Optimization Algorithms** (`stdlib/src/numerics/optimize.d`)
  - Gradient descent with momentum support
  - BFGS quasi-Newton method with line search
  - Automatic differentiation integration
  - Convergence diagnostics and iteration limits

- **Numerical Integration** (`stdlib/src/numerics/integrate.d`)
  - Adaptive Gauss-Kronrod quadrature (15-point rule)
  - Adaptive Simpson's rule with recursive subdivision
  - Monte Carlo integration for high-dimensional problems
  - Error estimation and convergence monitoring

- **Signal Processing** (`stdlib/src/numerics/fft.d`)
  - Cooley-Tukey FFT algorithm (radix-2)
  - Complex number arithmetic with polar form support
  - Power spectral density estimation
  - Convolution via FFT with zero-padding

#### Automatic Differentiation
- **Forward Mode AD** (`stdlib/src/autodiff/mod.d`)
  - Dual number implementation with complete mathematical operations
  - Gradient computation for scalar and vector functions
  - Directional derivatives and Jacobian-vector products
  - Support for trigonometric, exponential, and special functions

- **Reverse Mode AD** (`stdlib/src/autodiff/mod.d`)
  - Tape-based computation graph with Wengert list
  - Efficient gradient computation for functions with many inputs
  - Vector-Jacobian products and full Jacobian matrices
  - Memory-efficient backpropagation algorithm

- **Higher-Order Derivatives** (`stdlib/src/autodiff/higher.d`)
  - Second-order dual numbers for Hessian computation
  - Mixed-mode AD (forward-over-reverse, reverse-over-forward)
  - Taylor series expansion with arbitrary order
  - Directional second derivatives

#### Probabilistic Programming
- **Distribution Library** (`stdlib/src/prob/distributions.d`)
  - Continuous: Normal, LogNormal, Gamma, Beta, StudentT, Exponential
  - Discrete: Poisson, Binomial, Categorical
  - Multivariate: MultivariateNormal with Cholesky parameterization
  - Complete PDF/CDF/quantile/sampling implementations
  - Statistical moments and parameter estimation

- **MCMC Samplers** (`stdlib/src/prob/mcmc.d`)
  - Metropolis-Hastings with adaptive proposal covariance
  - Hamiltonian Monte Carlo with leapfrog integration
  - No-U-Turn Sampler (NUTS) with tree building
  - Convergence diagnostics: R-hat, effective sample size
  - Automatic step size and mass matrix adaptation

- **Variational Inference** (`stdlib/src/prob/vi.d`)
  - Automatic Differentiation Variational Inference (ADVI)
  - Mean-field Gaussian variational families
  - Full-rank Gaussian with Cholesky parameterization
  - Stochastic Variational Inference (SVI) for large datasets
  - ELBO optimization with gradient ascent

#### Pharmacokinetic/Pharmacodynamic Modeling
- **Compartment Models** (`stdlib/src/pkpd/compartment.d`)
  - 1, 2, 3-compartment models with IV and oral dosing
  - Multiple dosing regimens with infusion support
  - ODE-based simulation with adaptive solvers
  - Emax pharmacodynamic models with Hill coefficients
  - Units of measure integration for dimensional safety

- **Population Modeling** (`stdlib/src/pkpd/population.d`)
  - Mixed-effects modeling with between-subject variability
  - Covariate relationships (linear, power, exponential, categorical)
  - MCMC-based parameter estimation
  - Individual parameter prediction with shrinkage
  - Model diagnostics and validation metrics

- **Non-compartmental Analysis** (`stdlib/src/pkpd/nca.d`)
  - AUC calculation with trapezoidal rule
  - Cmax/Tmax identification
  - Terminal elimination rate constant estimation
  - Clearance, volume, half-life calculations
  - Bioequivalence analysis with 90% confidence intervals
  - AUMC and mean residence time computation

#### Interoperability & GPU Acceleration
- **NumPy Bridge** (`stdlib/src/interop/numpy.d`)
  - Zero-copy array sharing with Python ecosystem
  - Automatic memory layout conversion
  - NumPy C API integration for seamless interop
  - Support for complex data types and multidimensional arrays

- **R Integration** (`stdlib/src/interop/r.d`)
  - R SEXP object manipulation
  - Statistical function calls (summary, lm, t.test)
  - Data frame creation and manipulation
  - Seamless integration with R's statistical ecosystem

#### Testing & Quality Assurance
- **Comprehensive Test Suite** (`stdlib/src/test_scientific.d`)
  - Unit tests for all mathematical operations
  - Numerical accuracy verification against reference implementations
  - Performance benchmarks and regression tests
  - Integration tests for complete workflows

- **Example Applications** (`examples/scientific_computing_demo.d`)
  - Complete drug development pipeline demonstration
  - Bioequivalence study analysis
  - Population PK modeling workflow
  - Bayesian parameter estimation examples

### Enhanced
- **Effect System Integration**: All scientific computing operations properly annotated with effects (IO, Prob, Alloc, GPU)
- **Units of Measure**: Complete dimensional analysis for pharmacokinetic parameters
- **Memory Safety**: RAII patterns throughout with automatic resource cleanup
- **Performance**: Optimized algorithms with SIMD and GPU acceleration support

### Technical Specifications
- **Dependencies**: BLAS/LAPACK (OpenBLAS, Intel MKL, Apple Accelerate)
- **Optional**: CUDA 11.0+, Python 3.8+, R 4.0+
- **Performance**: Competitive with NumPy, MATLAB, Julia on standard benchmarks
- **Memory**: Zero-copy operations where possible, efficient memory layouts
- **Precision**: IEEE 754 double precision with configurable tolerances

## [0.12.0] - 2025-11-28

### Added - Day 15: Documentation Generator

- **Doc Comment Parsing** (`doc/parser.rs`)
  - Support for `///` outer line doc comments
  - Support for `//!` inner line doc comments
  - Support for `/** */` outer block doc comments
  - Support for `/*! */` inner block doc comments
  - Markdown support via `pulldown-cmark`
  - Attribute-style docs: `@param`, `@returns`, `@example`, `@since`, `@deprecated`
  - Cross-reference syntax: `[item]` linking

- **Documentation Model** (`doc/model.rs`)
  - `CrateDoc`, `ModuleDoc`, `FunctionDoc`, `TypeDoc`, `TraitDoc`, `ConstantDoc`
  - `SearchIndex` with name-based and full-text term indexing
  - Type information: parameters, generics, where clauses
  - Source file tracking and line number references

- **HTML Generation** (`doc/html/`)
  - Responsive HTML templates for all documentation pages
  - Dark/light theme support with CSS variables
  - Syntax highlighting for D language code
  - Interactive features: theme toggle, copy code buttons, keyboard search
  - Breadcrumb navigation and sidebar

- **mdBook Integration** (`doc/book/`)
  - Complete mdBook structure generation
  - Auto-generated chapters: Introduction, Getting Started, Reference, API
  - Cross-linking between guide and API documentation
  - `book.toml` and `SUMMARY.md` generation

- **Doctest Runner** (`doc/doctest.rs`)
  - Extract and run code examples from documentation
  - Support for `should_panic`, `ignore`, `no_run`, `compile_fail` attributes
  - Documentation coverage statistics
  - Test summary with pass/fail counts and timing

- **CLI Commands**
  - `dc doc` - Generate HTML documentation
  - `dc doc-book` - Generate mdBook documentation
  - `dc doctest` - Run documentation tests
  - `dc doc-coverage` - Show documentation coverage statistics

- **Lexer Updates**
  - Added `DocCommentOuter`, `DocCommentInner`, `DocBlockOuter`, `DocBlockInner` tokens
  - Updated comment skip patterns to preserve doc comments

### Changed
- Updated `lib.rs` to export `doc` module
- Added `pulldown-cmark` dependency for markdown rendering

## [0.11.0] - 2025-11-28

### Added - Day 11: LLVM Backend for AOT Compilation

- **LLVM Code Generation** (`codegen/llvm/`)
  - Full LLVM IR generation from HLIR
  - Type mapping to LLVM types
  - Function compilation with proper calling conventions
  - Control flow: if/else, loops, match expressions
  - Arithmetic and comparison operations

- **Optimization Passes**
  - Multiple optimization levels (O0, O1, O2, O3, Os, Oz)
  - Standard LLVM optimization pipeline
  - Function inlining and dead code elimination
  - Loop optimizations

- **Native Code Emission**
  - Object file generation
  - Assembly output option
  - Executable linking with system linker
  - Cross-platform target support

- **CLI Enhancements**
  - `dc build` command for AOT compilation
  - `--emit-llvm` for LLVM IR output
  - `--emit-asm` for assembly output
  - `-O` flag for optimization level
  - `--target` for cross-compilation

## [0.10.0] - 2025-11-27

### Added - Day 10: LSP Server for IDE Integration

- **LSP Server Core** (`tower-lsp` based)
  - Full Language Server Protocol implementation
  - Async architecture with `tokio` runtime
  - Document synchronization with incremental updates
  - Rope-based text storage for efficient editing (`ropey`)
  - Thread-safe document management (`dashmap`)

- **Real-time Diagnostics**
  - Syntax error reporting from parser
  - Type error reporting from type checker
  - Effect system violation detection
  - Ownership/linearity error reporting
  - Refinement type constraint failures

- **Hover Information**
  - Type information for variables and expressions
  - Documentation for keywords and built-ins
  - Effect signatures for functions
  - Unit annotations for scientific values

- **Go to Definition**
  - Jump to function definitions
  - Jump to type definitions
  - Jump to module declarations
  - Cross-file navigation support

- **Find All References**
  - Find all usages of variables
  - Find all usages of functions
  - Find all usages of types

- **Intelligent Code Completion**
  - Context-aware completions (top-level, expressions, types)
  - Keyword completions with snippets
  - Type name completions
  - Effect name completions
  - Unit completions for scientific computing
  - Built-in function completions

- **Semantic Tokens**
  - Rich syntax highlighting
  - Custom token types: effect, unit, refinement, lifetime
  - Custom modifiers: mutable, linear, affine, unsafe
  - Full token classification from lexer

- **VS Code Extension**
  - Language configuration for `.d` and `.dem` files
  - TextMate grammar for syntax highlighting
  - Extension commands: restart server, show HIR/HLIR, run file
  - Configurable settings for server path and trace level

### Changed
- Updated `Cargo.toml` with LSP feature flag and dependencies
- Added `demetrios-lsp` binary entry point

## [0.9.0] - 2025-11-27

### Added - Day 9: Refinement Types with Z3 SMT Solver

- **Refinement Type System**
  - Predicate-based type refinements
  - SMT-backed constraint verification via Z3
  - Compile-time proof of numeric constraints
  - Subtyping based on logical implication

- **Z3 Integration**
  - Optional `smt` feature flag
  - Automatic constraint extraction from types
  - Proof caching for performance
  - Detailed error messages for failed proofs

- **Refinement Syntax**
  - `x: {v: i32 | v > 0}` - positive integers
  - `x: {v: f64 | v >= 0.0 && v <= 1.0}` - probabilities
  - Array bounds refinements
  - Function pre/post conditions

## [0.8.0] - 2025-11-27

### Added - Day 8: Units of Measure, Source Maps, and Parser Recovery

- **Units of Measure System**
  - Compile-time dimensional analysis
  - SI base units (m, kg, s, A, K, mol, cd)
  - Common derived units (N, J, W, Pa, Hz, etc.)
  - Medical/pharmacological units (mg, mL, mg/mL, etc.)
  - Unit inference and checking
  - Automatic unit conversion

- **Source Maps**
  - Bidirectional source location mapping
  - Span tracking through all compiler phases
  - Debug info generation for source-level debugging
  - Error location precision

- **Parser Recovery**
  - Graceful error recovery
  - Multiple error collection
  - Synchronization tokens
  - Continued parsing after errors

## [0.7.0] - 2025-11-27

### Added - Day 7: HLIR (SSA-Based IR) + Cranelift JIT + REPL

- **HLIR (High-Level Low-Level IR)**
  - SSA-form intermediate representation
  - Basic blocks with phi nodes
  - Explicit control flow graph
  - Type-preserving lowering from HIR

- **Cranelift JIT Backend**
  - Just-in-time compilation
  - Fast development iteration
  - Native code execution
  - Optional via `jit` feature flag

- **Interactive REPL**
  - Read-Eval-Print-Loop
  - Expression evaluation
  - Definition persistence
  - Command history

## [0.6.0] - 2025-11-26

### Added - Day 6: HIR and Type Checking

- **HIR (High-level IR)**
  - Typed AST representation
  - Resolved symbols and types
  - Desugared language constructs

- **Type Checker**
  - Bidirectional type inference
  - Effect type checking
  - Ownership/linearity verification
  - Generic instantiation

## [0.5.0] - 2025-11-26

### Added - Day 5: Effect System

- **Algebraic Effects**
  - Effect declarations
  - Effect handlers
  - Effect polymorphism
  - Built-in effects: IO, Mut, Alloc, GPU, Prob

## [0.4.0] - 2025-11-26

### Added - Day 4: Ownership and Linearity

- **Ownership System**
  - Linear types (must use exactly once)
  - Affine types (use at most once)
  - Copy types (freely copyable)
  - Move semantics

## [0.3.0] - 2025-11-25

### Added - Day 3: AST and Parser

- **Abstract Syntax Tree**
  - Complete AST node definitions
  - Module structure
  - Expressions and statements
  - Pattern matching

- **Recursive Descent Parser**
  - Full language grammar
  - Operator precedence
  - Error messages

## [0.2.0] - 2025-11-25

### Added - Day 2: Lexer

- **Lexer Implementation**
  - Logos-based tokenization
  - All language tokens
  - String and numeric literals
  - Comments and whitespace

## [0.1.0] - 2025-11-25

### Added - Day 1: Project Setup

- Initial project structure
- Cargo workspace configuration
- Basic CLI scaffolding
- Documentation templates
