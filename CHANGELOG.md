# Changelog

All notable changes to Demetrios will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.92.0] - 2025-12-24

### MedLang Integration for Computational Pharmacology

This release adds the MedLang integration module, providing a high-level API
for pharmacokinetic/pharmacodynamic (PK/PD) modeling with native uncertainty
quantification, bridging computational pharmacology and clinical reasoning.

### Added

#### MedLang Module (`stdlib/medlang/`)
- **Core API** (`mod.d`): High-level PK/PD modeling interface
  - One-compartment IV/oral models with analytical solutions
  - Two-compartment models with macro-constant parameterization
  - Emax and sigmoid Emax pharmacodynamic models
  - Effect compartment for hysteresis handling
  - Dosing protocols (bolus, infusion, oral, repeated)
  - NCA metrics (AUC, Cmax, Tmax, t½, MRT)
  - Steady-state calculations with accumulation ratios
  - GUM-compliant parameter uncertainty (CV%, 95% CI)

- **AST** (`ast.d`): Abstract syntax tree for MedLang DSL
  - Parameter nodes with prior distributions (Normal, LogNormal, Beta, Gamma)
  - Compartment nodes (central, peripheral, depot, effect site)
  - Flow nodes (linear, saturable Michaelis-Menten)
  - Dosing nodes with bioavailability and lag time
  - Observation/error models (additive, proportional, combined)
  - Covariate effects (power, linear, exponential)
  - Pre-built model templates (1-comp IV/oral, 2-comp IV)

- **Codegen** (`codegen.d`): ODE generation and simulation
  - ODE systems for standard PK models
  - RK4 solver integration with adaptive stepping
  - Weighted least squares objective function
  - Gradient descent optimizer for parameter estimation
  - GUM uncertainty propagation via finite differences
  - AIC/BIC model comparison metrics
  - Prediction bands with 50%/90% intervals

### Design Philosophy
- Bridge between MedLang clinical DSL and Demetrios numerical infrastructure
- Native integration with ODE solvers and Bayesian inference modules
- First-class support for PK/PD modeling paradigms (NONMEM-like)
- Programmatic model construction for automated workflow generation

### References
- Gabrielsson & Weiner: Pharmacokinetic and Pharmacodynamic Data Analysis
- Bonate: Pharmacokinetic-Pharmacodynamic Modeling and Simulation
- NONMEM User's Guides (Beal & Sheiner)
- GUM: Guide to Expression of Uncertainty in Measurement

## [0.91.0] - 2025-12-24

### Statistics & Bayesian Inference Modules

This release adds comprehensive statistical analysis and Bayesian inference
capabilities, completing the scientific computing triad: simulate → fit → infer.

### Added

#### Statistics Module (`stdlib/stats/`)
- **Descriptive** (`descriptive.d`): GUM-compliant descriptive statistics
  - Mean, variance, std with expanded uncertainty (k=2, 95% coverage)
  - Skewness, kurtosis (Fisher's definition)
  - Correlation with Fisher z-based uncertainty
  - Robust statistics: median, IQR, MAD
- **Inferential** (`inferential.d`): Hypothesis testing with effect sizes
  - One-sample, independent, paired t-tests
  - Welch's t-test for unequal variances
  - One-way ANOVA (3 groups) with eta-squared
  - Mann-Whitney U test with rank-biserial correlation
  - Cohen's d effect size interpretation
- **Resampling** (`resampling.d`): Distribution-free inference
  - Bootstrap CI for mean, median, correlation
  - Permutation test for mean differences
  - Permutation test for correlation significance
  - Percentile-based confidence intervals
- **Multiple Testing** (`multiple_testing.d`): Error rate control
  - Bonferroni correction (FWER control)
  - Holm step-down procedure
  - Benjamini-Hochberg FDR correction
  - Effective number of tests for correlated variables

#### Bayesian Module (`stdlib/bayes/`)
- **Prior** (`prior.d`): Prior distribution library
  - Normal, LogNormal, Uniform, Beta, Gamma priors
  - HalfNormal, HalfCauchy for scale parameters
  - Exponential, Cauchy for heavy-tailed priors
  - Weakly informative prior recommendations (Stan-style)
- **MCMC** (`mcmc.d`): Markov Chain Monte Carlo sampling
  - Metropolis-Hastings with symmetric proposals
  - Adaptive proposal tuning during warmup
  - Posterior mean, std, median, 95% credible intervals
  - Acceptance rate monitoring
- **VI** (`vi.d`): Variational Inference
  - Mean-field Gaussian variational family
  - ELBO optimization with reparameterization trick
  - Gradient ascent with configurable learning rate
  - Convergence monitoring
- **Diagnostics** (`diagnostics.d`): Convergence assessment
  - Gelman-Rubin split R-hat (should be < 1.01)
  - Effective sample size (ESS) via autocorrelation
  - Monte Carlo standard error (MCSE)
  - Autocorrelation at arbitrary lags

### References
- Efron & Tibshirani (1993): Bootstrap methods
- Benjamini & Hochberg (1995): FDR control
- Gelman et al. (2013): Bayesian Data Analysis
- Vehtari et al. (2021): R-hat and ESS improvements

## [0.90.0] - 2025-12-24

### fMRI Processing & Multimodal Fusion Modules

This release adds neuroimaging analysis capabilities for fMRI data processing
and EEG-fMRI multimodal integration, essential for computational psychiatry.

### Added

#### fMRI Module (`stdlib/fmri/`)
- **NIfTI** (`nifti.d`): Neuroimaging file format support
  - NIfTI image data structures for 3D/4D brain volumes
  - Voxel indexing and coordinate transformations
  - Affine matrix operations for MNI space conversion
  - Header metadata handling (dimensions, voxel sizes, TR)
- **Preprocess** (`preprocess.d`): fMRI preprocessing pipeline
  - 6-DOF motion parameters (translation + rotation)
  - Framewise displacement (FD) for motion scrubbing
  - Gaussian smoothing kernels (FWHM-based)
  - Temporal filtering configurations (bandpass 0.01-0.1 Hz)
  - Nuisance regression settings (motion, WM/CSF, global signal)
  - Linear detrending and z-score normalization
- **Connectivity** (`connectivity.d`): Functional connectivity analysis
  - Pearson correlation for ROI-to-ROI connectivity
  - Fisher z-transformation for statistical inference
  - Confidence intervals via z-space standard errors
  - Network metrics (mean FC strength, node degree)
  - FCResult structure with correlation and uncertainty

#### Fusion Module (`stdlib/fusion/`)
- **EEG-fMRI** (`eeg_fmri.d`): Multimodal integration
  - Hemodynamic Response Function (HRF) computation
    - Canonical double-gamma model (Glover, 1999)
    - Configurable peak/undershoot parameters
  - Representational Similarity Analysis (RSA)
    - Pattern dissimilarity (1 - correlation)
    - RDM comparison for cross-modal alignment
  - fMRI-informed EEG source localization
    - Soft-thresholded source priors from t-maps
    - Sigmoid weighting for activation probability

### References
- Esteban et al. (2019): fMRIPrep preprocessing pipeline
- Power et al. (2012): Motion artifact handling
- Biswal et al. (1995): Functional connectivity
- Debener et al. (2006): Trial-by-trial EEG-fMRI coupling
- Cichy et al. (2016): Similarity-based fusion

## [0.89.0] - 2025-12-24

### Signal Processing & Connectivity Modules

This release adds comprehensive biosignal analysis capabilities for EEG, ECG,
and time series data, along with brain connectivity measures.

### Added

#### Signal Module (`stdlib/signal/`)
- **Filter** (`filter.d`): Digital signal filtering
  - Butterworth filter coefficient computation (lowpass, highpass, bandpass)
  - Notch filter for powerline interference (50/60 Hz)
  - FIR filter design with windowed sinc method
  - Zero-phase filtering via forward-backward application
- **Spectral** (`spectral.d`): Frequency domain analysis
  - Radix-2 Cooley-Tukey FFT implementation
  - Periodogram and Welch's method for PSD estimation
  - EEG band power extraction (delta, theta, alpha, beta, gamma)
  - Spectral entropy for signal complexity
  - Hanning window implementation
- **Epoch** (`epoch.d`): Event-related analysis
  - Event-locked segmentation for ERP extraction
  - Baseline correction with pre-stimulus window
  - Artifact rejection by amplitude threshold
  - Trial averaging with standard error of the mean
- **Fractal** (`fractal.d`): Nonlinear dynamics
  - Higuchi Fractal Dimension (HFD) for signal complexity
  - Detrended Fluctuation Analysis (DFA) for long-range correlations
  - Permutation entropy for irregularity quantification

#### Connectivity Module (`stdlib/connectivity/`)
- **Phase** (`phase.d`): Phase synchronization measures
  - Phase Locking Value (PLV) for synchronization strength
  - Phase Lag Index (PLI) for volume conduction robustness
  - Weighted PLI (wPLI) with improved SNR
  - Debiased wPLI (dwPLI) for small sample correction
  - Hilbert transform via FFT for instantaneous phase
  - Connectivity matrix computation for multi-channel data

## [0.88.0] - 2025-12-24

### Numerical Optimization Module

This release adds a comprehensive numerical optimization library with algorithms
for nonlinear least squares, quasi-Newton methods, derivative-free optimization,
global optimization, and GUM-compliant uncertainty quantification.

### Added

#### Optimize Module (`stdlib/optimize/`)
- **Levenberg-Marquardt** (`levenberg_marquardt.d`): Nonlinear least squares
  - Gauss-Newton with adaptive damping (trust region)
  - Finite-difference Jacobian computation
  - Cholesky decomposition for solving damped normal equations
  - Exponential curve fitting example
- **BFGS** (`bfgs.d`): Quasi-Newton optimization
  - BFGS inverse Hessian approximation
  - Backtracking line search with Armijo condition
  - Superlinear convergence for smooth problems
  - Quadratic and Rosenbrock test functions
- **Nelder-Mead** (`nelder_mead.d`): Derivative-free simplex method
  - Reflection, expansion, contraction, and shrink operations
  - Robust for noisy or non-differentiable objectives
  - No gradient computation required
- **Differential Evolution** (`differential_evolution.d`): Global optimization
  - Population-based stochastic search
  - rand/1 mutation strategy
  - Effective for multi-modal and high-dimensional problems
  - Sphere and Rastrigin test functions
- **Uncertainty** (`uncertainty.d`): GUM-compliant uncertainty quantification
  - Covariance estimation from Jacobian and residuals
  - Confidence intervals with t-distribution quantiles
  - Correlation matrix computation
  - Linear uncertainty propagation
  - Relative standard error (%CV) calculation

## [0.87.0] - 2025-12-24

### Random Number Generation Module

This release adds a comprehensive random number generation library with
probability distributions and PK/PD variability support.

### Added

#### Random Module (`stdlib/random/`)
- **RNG** (`rng.d`): High-quality random number generators
  - PCG64: Default generator with excellent statistical quality
  - Xoshiro256++: Fast generator for simulations
  - SplitMix64: For seeding other generators
  - Functional state passing (no global mutable state)
- **Distributions** (`distributions.d`): Probability distributions
  - Continuous: Uniform, Normal, LogNormal, Exponential, Gamma, Beta
  - Discrete: Poisson, Bernoulli
  - Box-Muller transform for Normal sampling
  - Marsaglia & Tsang method for Gamma sampling
- **Sampling** (`sampling.d`): Random sampling utilities
  - `sample_index`, `sample_one_f64`, `sample_one_i64`
  - `sample_weighted_index`: Weighted random selection
  - `shuffle_f64`, `shuffle_i64`: Fisher-Yates shuffle
  - `resample_f64`: Bootstrap resampling
- **PK/PD Variability** (`sampling.d`):
  - `IIV`: Inter-Individual Variability parameters
  - `generate_individual_pk`: Individual PK parameters with IIV
  - `generate_population`: Virtual population generation
  - Residual error models: proportional, additive, combined

## [0.86.0] - 2025-12-24

### DataFrame Module & Causal Inference Reorganization

This release adds a full DataFrame module with epistemic integration and reorganizes
the causal inference modules into their own directory.

### Added

#### DataFrame Module (`stdlib/data/`)
- **Series** (`series.d`): Typed columns (Float, Int, String, Bool, Epistemic)
  - Aggregations: sum, mean, var, std, min, max
  - Epistemic series with uncertainty propagation
  - Element-wise operations and filtering
- **Frame** (`frame.d`): DataFrame with type-erased columns
  - Row/column access, slicing, filtering
  - Aggregations across columns
  - Drop column operations
- **I/O** (`io.d`): CSV parsing with uncertainty notation
  - Value±uncertainty format: `100.0±2.0` or `100.0+-2.0`
  - Paired column detection: `value`, `value_u`, `value_conf`
  - Automatic type inference (numeric vs string)
- **Operations** (`ops.d`): GroupBy and transformations
  - GroupBy with epistemic-aware aggregations
  - Cumsum, cummax, rolling mean
  - Missing value handling: fillna, dropna

#### Causal Inference Module (`stdlib/causal/`)
- Reorganized from `stdlib/epistemic/` into dedicated directory
- **Core** (`core.d`): DAG, do-calculus, interventions
- **Discovery** (`discovery.d`): PC, FCI structure learning
- **Uplift** (`uplift.d`): CATE learners (S/T/X-learner)
- **Refutation** (`refutation.d`): Robustness and sensitivity tests

### Fixed

#### Interpreter
- **Type casting**: Int/Float/Bool to String casts now work properly
  - `42 as String` correctly returns `"42"` instead of raw Int
  - Fixes string concatenation with cast results
- **Scope leak**: Fixed return/break/continue not properly restoring scope
- **String methods**: Added `slice()` and `byte_at()` for string manipulation
- **Escape sequences**: Proper processing of `\n`, `\t`, `\\` in string literals

### Notes

All DataFrame and CSV I/O tests now pass in the interpreter. The causal modules
are reorganized but some use advanced syntax (Counterfactual enum, Knowledge type)
that requires further parser support.

## [0.85.0] - 2025-12-23

### MCTS Module for Game Playing & Optimization

This release adds Monte Carlo Tree Search with UCB1/PUCT selection policies.

### Added

#### MCTS Search (`stdlib/search/mcts/`)
- **Node** (`node.d`): MCTS node structure with state, visits, values, child tracking
- **Policy** (`policy.d`): UCB1 and PUCT (AlphaZero-style) selection policies
- **Core** (`core.d`): Main tree operations, backpropagation, result extraction

#### MCTS Examples
- **Tic-Tac-Toe** (`examples/tictactoe.d`): Game playing with 100 MCTS simulations
- **Variance Tracking** (`examples/uncertainty_demo.d`): GUM-style variance with Welford's algorithm

#### Features
- UCB1: `q + c * sqrt(ln(N)/n)` exploration/exploitation balance
- PUCT: Prior probability integration for neural network guided search
- Arena allocation with flat node structure for cache efficiency
- Adversarial and single-player backpropagation variants
- Online variance estimation for uncertainty-aware values

## [0.84.0] - 2025-12-23

### KEC Pipeline & Stdlib Expansion

This release adds the Knowledge Epistemic Curvature (KEC) pipeline and expands
the standard library with interpreter-compatible utility modules.

### Added

#### Graph Analysis (`stdlib/graph/`)
- **Entropy** (`entropy.d`): Shannon entropy and structural entropy for knowledge graphs
- **Coherence** (`coherence.d`): Degree regularity, connectivity, and balance metrics
- **Forman Curvature** (`curvature.d`): O(1) per-edge curvature as alternative to Ollivier-Ricci

#### GUM Uncertainty (`stdlib/epistemic/gum.d`)
- JCGM 100:2008 compliant uncertainty propagation
- Coverage factors (k) with t-distribution tables for ν=1 to ∞
- Welch-Satterthwaite approximation for combined degrees of freedom
- Type A (statistical) and Type B (a priori) uncertainty components
- Uncertainty propagation through +, -, *, / operations

#### Data & I/O Modules
- **CSV** (`stdlib/csv/mod.d`): Parser and writer, interpreter-compatible
- **Argparse** (`stdlib/io/argparse.d`): Command-line argument parsing
- **Directory Operations** (`stdlib/io/mod.d`): create_dir, remove_dir, read_dir, metadata

#### Statistical Validation (`stdlib/stats/validation.d`)
- Descriptive statistics: mean, variance, std_dev, min, max
- Correlation and R-squared
- Linear regression with fit quality metrics
- Residual analysis: RMSE, MAE, MSE
- Confidence intervals (95% and custom)
- t-statistics for hypothesis testing

### Notes

All new modules are tested and passing in the interpreter. Some interpreter
limitations were discovered:
- `pub const` at module level requires function wrappers
- String methods `.slice()` and `.byte_at()` not yet implemented
- Float comparisons on function returns may cause issues

## [0.83.0] - 2025-12-22

### Trust Gate Week: "Refuses to Compute Lies"

This release implements the epistemic hardening infrastructure that makes Demetrios
a language where invalid computations are **refused**, not just warned about.

### Added

#### Epistemic Refusal Gates (`stdlib/epistemic/`)
- **ROI Refusal** (`roi.d`): `gate_negative_roi()` refuses when `roi_bits < 0`
  - KL divergence for information gain (credit)
  - Landauer-motivated entropy debt tracking
  - Refusal message: "Refused: gained 0.7 bits; erased 3.1 bits. You are polishing noise."
- **Sobol Type-B Dominance** (`sobol.d`): `gate_type_b_dominance()` refuses when relying on priors
  - Type-A (measured) vs Type-B (literature/expert) classification per GUM
  - Refusal message: "Your output is 80% hostage to literature values. Go measure."
- **GUM vs MC Auto-Switch** (`dual_check.d`): JCGM 101 decision tree
  - `decide_method()` implements linearization check
  - `cross_check_should_refuse()` catches linearization lies
  - Auto-switch to Monte Carlo when GUM assumptions fail

#### Property-Based Testing (`stdlib/epistemic/proptest.d`)
- QuickCheck-style tests for 5 epistemic laws under random assault:
  1. Provenance append-only (never shrinks)
  2. Confidence monotone (no silent uplift)
  3. Uncertainty non-contraction (no silent narrowing)
  4. Interval enclosure preserved
  5. Debt monotone (Landauer limit)
- 400 property tests with random generation and ILLEGAL operation detection

#### Fuzzing Infrastructure (`compiler/fuzz/`)
- **INVARIANT**: Lexer and parser must NEVER panic on any input
- `fuzz_lexer`: bytes → tokens (crash-proof)
- `fuzz_parser`: tokens → AST (error recovery)
- `fuzz_full_pipeline`: lexer → parser → type checker
- libFuzzer-based coverage-guided fuzzing

#### CI Hardening (`.github/workflows/ci.yml`)
- Removed `continue-on-error: true` (the lie)
- Added `scripts/run_stdlib_tests.sh` running all 73 stdlib programs
- 61 tests pass, 12 known broken (tracked explicitly, blocking regressions)

### Changed
- CI test suite is now a **hard gate** - failures block merge
- Known broken tests are tracked with explicit reasons, not silently skipped

### References
- JCGM 100:2008 (GUM) - Uncertainty propagation framework
- JCGM 101:2008 - Monte Carlo supplement
- JCGM 102:2011 - Multivariate outputs
- Landauer (1961) - Irreversibility and Heat Generation
- Sobol' (1993) - Sensitivity estimates for nonlinear mathematical models

## [0.82.0] - 2025-12-21

### Added
- Epistemic AD, entropic ledger, and invariant tests

## [0.81.0] - 2025-12-21

### Added
- PROV-DM provenance, budget ledger, and Monte Carlo propagation

## [0.80.0] - 2025-12-21

### Added
- Correlation tracking and policy gating

## [0.79.1] - 2025-12-20

### Added
- FFI imports: `extern "C" { fn ...; }` now type-check and lower into HLIR (with `#[link_name = "..."]` and variadic support).
- Epistemic stdlib expansions: new modules `knowledge`, `propagate`, `meta`, `active`, `merkle` plus early `linalg` and `ode` scaffolding (`stdlib/epistemic/README.md`).
- FFI examples: `examples/ffi_exports.d` with Julia/Python smoke tests.

### Fixed
- Issue #11 pointer indexing: corrected address-of/deref lowering, signed/unsigned casts, and Cranelift GEP indexing so pointer reads/writes work under JIT.
- Cranelift: declaration-only extern imports are treated as imports (no body compilation) and LLVM uses the linker-visible symbol name when `link_name` is set.

## [0.79.0] - 2025-12-20

### ODE Solvers, BLAS Integration & Academic Citation

This release adds numerical ODE solvers, BLAS/LAPACK bindings, and establishes academic citation with a Zenodo DOI.

### Added

#### ODE Solvers (`stdlib/ode/`)
- **Tsit5** - Tsitouras 5(4) adaptive Runge-Kutta solver (recommended for non-stiff problems)
- **DOPRI5** - Dormand-Prince 5(4) adaptive solver with PI step control
- **RK4** - Classic fixed-step 4th-order Runge-Kutta
- **BDF1** - Backward Euler implicit solver for stiff equations
- `ODEConfig` - Configurable tolerances (rtol, atol), step limits, safety factors
- `ODESolution` - Result type with statistics (steps, function evaluations, rejections)
- `examples/ode_demo.d` - Comprehensive demo comparing solver accuracy

#### BLAS/LAPACK Integration (`stdlib/`)
- Dynamic matrix types with heap allocation
- BLAS Level 1: `dscal`, `daxpy`, `ddot`, `dnrm2`
- BLAS Level 2: `dgemv`
- BLAS Level 3: `dgemm`
- LAPACK: `dgetrf`, `dgetrs`, `dgesv`, `dpotrf`, `dgeev`

#### Academic Citation
- `CITATION.cff` - Machine-readable citation metadata
- Zenodo DOI: [10.5281/zenodo.18004435](https://doi.org/10.5281/zenodo.18004435)

### Changed

#### Module System Integration
- Type checker now fully integrates with module-aware symbol tables
- Resolver supports nested modules and cross-module references
- Enhanced import syntax: `use std::math::{sin, cos}`

### Fixed
- Tsit5 solver accuracy issue caused by mutable variable scoping in loops
- LLVM 15 API compatibility for GPU codegen

## [0.58.0] - 2025-12-08

### Day 58: End-to-End Integration Tests

Comprehensive integration test suite proving the semantic type system works
end-to-end with real ontologies and real biomedical scenarios.

### Added

#### Test Infrastructure (`tests/e2e/common/mod.rs`)
- `TestHarness` - Compiler invocation wrapper with fluent API
- `CompileResult` - Rich assertion methods for compilation outcomes
- JSON diagnostic parsing with `Diagnostic`, `Location`, `Suggestion`, `SemanticDistance`
- Golden file comparison utilities with `UPDATE_GOLDEN=1` support
- Test fixtures for ChEBI, GO, HP, MONDO ontology terms

#### Pharmacology Test Suite (`tests/e2e/pharmacology.rs`)
- Drug type declarations and drug-drug interaction modeling
- Metabolite pathway tracking through GO processes
- Phenotype-to-disease mapping with HP and MONDO
- Drug indication and adverse reaction prediction
- Dosage calculations with UO unit types
- Pharmacokinetic modeling (one-compartment model)
- Polypharmacy analysis and clinical trial eligibility

#### Cross-Ontology Test Suite (`tests/e2e/cross_ontology.rs`)
- ChEBI ↔ DrugBank alignment and coercion
- ChEBI ↔ RxNorm clinical drug mapping
- Three-way ontology integration (ChEBI/DrugBank/RxNorm)
- HP ↔ MP cross-species phenotype alignment
- MONDO ↔ OMIM and MONDO ↔ DOID disease alignment
- Transitive coercion chain tests
- Alignment threshold boundary tests

#### Diagnostic Test Suite (`tests/e2e/diagnostics.rs`)
- Error location accuracy tests
- Type mismatch message quality tests
- Semantic distance suggestion tests
- "Did you mean?" suggestion tests
- Threshold adjustment suggestion tests
- JSON vs human-readable output format tests
- Multi-error and cascading error tests

#### Edge Case Test Suite (`tests/e2e/edge_cases.rs`)
- Empty/whitespace/comment-only file handling
- Unicode identifiers and strings
- Numeric boundaries (large term IDs, threshold 0/1)
- Deep nesting (expressions, blocks, types)
- Circular and self-referential type detection
- Conflicting alignment handling
- Error recovery across multiple errors

#### Performance Test Suite (`tests/e2e/performance.rs`)
- Baseline compile time benchmarks
- Function/type/alignment scaling tests
- O(n²) complexity detection
- Large file memory stress tests
- Diagnostic generation performance
- Transitive alignment chain performance

#### Golden File Tests (`tests/e2e/golden.rs`)
- Snapshot tests for error messages
- Warning message golden files
- Suggestion quality golden files
- JSON format golden files
- Help text golden files

### Golden Files Created
- `tests/golden/errors/` - Type mismatch, semantic distance, undefined ontology, duplicates, cascading
- `tests/golden/warnings/` - Unused imports, loose thresholds
- `tests/golden/suggestions/` - "Did you mean?", threshold adjustment, alignment suggestions
- `tests/golden/scenarios/` - Pharmacology field swap, transitive distance chains
- `tests/golden/json/` - JSON diagnostic format examples
- `tests/golden/help/` - CLI help output reference

### Test Coverage
- 80+ integration test cases
- Real pharmacology scenarios (not toy examples)
- Real ontology terms (ChEBI, GO, HP, MONDO, UO)
- Error message quality verification
- Performance regression detection
- Edge case and boundary condition coverage

## [0.57.0] - 2025-12-08

### Day 57: Diagnostic Messages & CLI Integration

Rich diagnostic rendering with semantic annotations and progress reporting.

### Added

#### Diagnostic Rendering (`src/diagnostic/render.rs`)
- `TerminalCaps` - Terminal capability detection (colors, Unicode, width)
- `RichRenderer` - Full diagnostic rendering with source snippets
- `DistanceSuggestion` - "Did you mean?" output for semantic distance
- ANSI color support with NO_COLOR/TERM/CI environment detection
- Unicode box drawing for clean error display

#### Semantic Annotations (`src/diagnostic/semantic.rs`)
- `SemanticContext` - Type mismatch context with reasons and notes
- `TermInfo` - Ontological term info (label, description, branch path)
- `DistanceComponents` - Path/IC/embedding distance breakdown
- `SemanticAnnotator` - Domain-specific explanations for ChEBI, GO, HP, MONDO, UO, PATO
- `SemanticSuggestion` - Distance-aware type suggestions

#### Progress Reporting (`src/diagnostic/progress.rs`)
- `Progress` - Progress bar with Bar/Spinner/Count/Bytes styles
- `StatusLine` - Single-line status updates
- `MultiProgress` - Parallel task tracking
- `CompilationProgress` - Phase-specific compilation reporting

## [0.56.0] - 2025-12-08

### Day 56: Type Checker Integration

Complete type checker with semantic distance-aware unification.

### Added

#### Type Checker Core (`src/typeck/`)
- `mod.rs` - SemanticTypeChecker with distance-aware unification
- `unify_distance.rs` - Distance-aware type unification
- `threshold.rs` - Threshold management for `#[compat]` attribute
- `coercion_insert.rs` - Automatic coercion insertion in HIR
- `suggestions.rs` - Type suggestion generation
- `diagnostics.rs` - Semantic type error diagnostics
- `hooks.rs` - Extension points for custom type rules

## [0.50.0] - 2025-12-07

### 50-Day Milestone Release

This release marks the 50th day of Demetrios development, featuring a complete
ontological type system with 15+ million terms as first-class types.

### Added

#### Ontological Type System (Days 47-49)
- Native ontology loading from BioPortal, OBO Foundry, Schema.org, FHIR
- Three-tier caching architecture (L1 hot / L2 warm / Federated)
- Semantic distance calculation with path, IC, and embedding fusion
- Cross-ontology type compatibility via SSSOM mappings
- Implicit and explicit coercion based on semantic distance
- Confidence propagation through type coercions

#### Embedding Space (Day 48)
- OWL2Vec*-style structural embeddings
- Text embeddings from labels and definitions
- Hybrid fusion strategy
- Memory-mapped storage for 15M+ vectors
- VP-tree ANN index for fast similarity search

#### Developer Experience (Day 50)
- Rich semantic error messages with distance explanations
- Polished CLI with progress indicators and colors
- Compilation profiling (`--profile` flag)
- SIMD-accelerated embedding distance calculations
- Bloom filter optimization for fast term lookup
- Interned IRIs for memory efficiency
- Pre-computed subsumption index

#### Formal Specification
- LaTeX type theory specification (`spec/formal/semantic_types.tex`)
- End-to-end integration tests (27 test cases)
- Academic paper skeleton for ICBO/POPL submission
- Criterion benchmark suite for performance tracking

### Changed
- Type checking now uses metric subtyping (distance-based compatibility)
- Error messages include semantic context and fix suggestions
- CLI output is now colorized by default

### Performance
- L1 cache hit: ~50 ns
- L2 cache hit: ~5 μs
- Distance calculation: ~10 μs
- Embedding similarity (SIMD): ~500 ns for 256-dim vectors
- Bloom filter negative check: ~100 ns

### Documentation
- Formal type theory specification (LaTeX)
- Paper skeleton for ICBO/POPL submission
- Updated user documentation with ontology examples
- Research notes on semantic metric types

## [0.46.0] - 2025-12-05

### Added
- GPU epistemic computing
- Counterfactual execution engine
- Z3 verification integration
- Compute graph visualization

## [0.41.0] - 2025-11-25

### Added
- Epistemic primitives: Knowledge[τ,ε,δ,Φ]
- Provenance tracking
- Multi-agent knowledge composition
- Semantic-physical duality
- Cache locality optimization

## [0.39.0] - 2025-11-22

### Added
- Layout synthesis for semantic-aware struct layout
- Memory layout optimization based on access patterns

## [0.30.0] - 2025-11-15

### Added
- Epistemic paradigm foundations
- Knowledge type with confidence tracking
- Evidence fusion operators

## [0.28.0] - 2025-11-10

### Added
- Scientific computing foundations
- Units of measure with dimensional analysis
- Refinement types with SMT verification

## [0.13.0] - 2025-10-20

### Added
- Complete macro system with hygiene
- LSP server for IDE integration
- Distributed build support

## [0.12.0] - 2025-10-18

### Added
- GPU compute support (PTX/SPIR-V codegen)
- Cranelift JIT backend

## [0.1.0] - 2025-10-08

### Added
- Initial project structure
- Lexer and parser
- Basic type system with effects
- Linear types with ownership
- HIR and LLVM codegen
