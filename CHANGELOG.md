# Changelog

All notable changes to Demetrios will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
