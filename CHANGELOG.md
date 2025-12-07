# Changelog

All notable changes to the Demetrios compiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.46.0] - 2025-12-07

### Added

#### Revolutionary GPU Epistemic Computing (World-First)

This release introduces groundbreaking capabilities that make Demetrios the first programming language to track epistemic state through GPU computation.

##### HLIR to GPU IR Pipeline (`hlir_to_gpu.rs`)
- Complete bridge from high-level SSA IR to GPU-specific IR
- Full lowering with epistemic shadow tracking
- New public API functions:
  - `compile_to_gpu(source, sm_version)` - Compile to PTX
  - `compile_to_gpu_epistemic(source, sm_version)` - Compile with epistemic tracking
- Support for CUDA (PTX) and SPIR-V (Vulkan/OpenCL) targets
- `LoweringConfig` for fine-grained control over GPU code generation

##### Epistemic PTX Emission (`epistemic_ptx.rs`)
- Shadow register architecture for tracking uncertainty through GPU computation:
  - `value: f32/f64` - The actual computed value
  - `epsilon: f32` - Uncertainty bound (shadow register)
  - `validity: pred` - Predicate register for validity
  - `provenance: u64` - Data lineage bitmask
- Quadrature propagation: `epsilon_result = sqrt(epsilon_a^2 + epsilon_b^2)`
- Epistemic operations:
  - `emit_epistemic_add()` / `emit_epistemic_mul()` / `emit_epistemic_div()`
  - `emit_epistemic_fma()` - Fused multiply-add with uncertainty
- Warp-level aggregation:
  - `emit_warp_confidence_vote()` - Ballot of valid lanes
  - `emit_warp_epsilon_reduce()` - Min/Max/Sum reduction across warp
- Confidence-gated execution: `emit_confidence_gate()`
- `EpistemicPtxConfig` for configuring:
  - Confidence threshold
  - Quadrature propagation (slower but more accurate)
  - Provenance tracking granularity

##### Counterfactual GPU Execution (`counterfactual.rs`)
- Pearl's do-calculus as GPU primitives - causal inference at hardware speed
- The Ladder of Causation on GPU:
  - Level 1: Association (Seeing) - Standard GPU execution
  - Level 2: Intervention (Doing) - `do(X=x)` operator
  - Level 3: Counterfactual (Imagining) - "What if X had been x'?"
- Parallel world execution via warp lane assignment
- Core types:
  - `WorldId` - Factual vs counterfactual world identifiers
  - `Intervention` - do-operator application
  - `WorldSnapshot` - World state at a point in time
  - `CounterfactualContext` - Manages multi-world execution
- Treatment effect computation:
  - `emit_divergence_compute()` - Individual Treatment Effect (ITE)
  - `emit_ate_compute()` - Average Treatment Effect across warp
  - `emit_probability_causation()` - P(Y_x=1 | X=0, Y=0)
- Structural equation evaluation:
  - Linear: `Y = beta_0 + sum(beta_i * X_i)`
  - Logistic: `Y = sigmoid(beta_0 + sum(beta_i * X_i))`
  - Multiplicative: `Y = product(X_i^beta_i)`
  - Threshold: `Y = 1 if X > theta else 0`
- Nested interventions with configurable max depth

##### Z3 SMT Verification (`z3_solver.rs`)
- Real Z3 FFI implementation (feature-gated with `smt` feature)
- `Z3Solver` for direct Z3 interaction
- `Z3EpistemicVerifier` for formal verification of epistemic properties:
  - `verify_bounded_uncertainty(epsilon, bound)` - Prove epsilon <= bound
  - `verify_validity_implies_confidence(threshold)` - Prove validity => confidence
  - `verify_provenance_completeness()` - Verify all data sources tracked
  - `verify_quadrature_correctness()` - Verify uncertainty propagation
- Counterexample extraction for failed proofs
- `MockEpistemicVerifier` fallback when Z3 unavailable

##### Integration Tests
- Comprehensive test suite with 48 tests covering:
  - GPU IR construction
  - PTX code generation
  - HLIR lowering
  - Epistemic PTX emission
  - Counterfactual execution
  - World snapshots
  - Full pipeline integration
  - Edge cases and performance characteristics

### Changed
- Updated `lib.rs` to expose GPU compilation pipeline
- Extended `codegen/gpu/mod.rs` with new module exports

### Technical Details

#### Epistemic State as Hardware Resources
```
Knowledge[T, epsilon, delta, Phi] --> {
    value: T,           // The actual data
    epsilon: f32,       // Uncertainty bound (shadow register)
    validity: pred,     // Predicate register for validity
    provenance: u64,    // Bit-packed provenance mask
}
```

#### PTX Code Pattern for Epistemic Add
```ptx
// Value: c = a + b
add.f32 %r_c, %r_a, %r_b;
// Epsilon: epsilon_c = sqrt(epsilon_a^2 + epsilon_b^2)
mul.f32 %r_t1, %r_eps_a, %r_eps_a;
mul.f32 %r_t2, %r_eps_b, %r_eps_b;
add.f32 %r_t3, %r_t1, %r_t2;
sqrt.approx.f32 %r_eps_c, %r_t3;
// Validity: v_c = v_a AND v_b
and.pred %p_valid_c, %p_valid_a, %p_valid_b;
// Provenance: prov_c = prov_a XOR prov_b
xor.b64 %r_prov_c, %r_prov_a, %r_prov_b;
```

#### Counterfactual World Branching
```ptx
// Intervention: do(X = x_cf)
mov.u32 %r_lane, %laneid;
and.b32 %r_is_cf, %r_lane, 1;      // Odd lanes = counterfactual
setp.ne.u32 %p_cf, %r_is_cf, 0;
selp.f32 %x, %x_cf, %x_factual, %p_cf;

// Execute model in both worlds
... compute outcome ...

// Compute treatment effect (divergence between worlds)
shfl.sync.xor.b32 %r_other_outcome, %r_outcome, 1, 0xFFFFFFFF;
sub.f32 %r_ite, %r_outcome, %r_other_outcome;  // Individual Treatment Effect
```

## [0.45.0] - Previous Release

Initial GPU codegen infrastructure and epistemic type system.
