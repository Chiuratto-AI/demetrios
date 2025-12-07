# Epistemic GPU Computing in Demetrios

Demetrios is the **world's first programming language** to track epistemic state through GPU computation. This document provides comprehensive examples and usage patterns.

## Table of Contents

1. [Overview](#overview)
2. [Shadow Register Architecture](#shadow-register-architecture)
3. [Basic Usage](#basic-usage)
4. [Epistemic PTX Emission](#epistemic-ptx-emission)
5. [Counterfactual Execution](#counterfactual-execution)
6. [Z3 SMT Verification](#z3-smt-verification)
7. [Advanced Patterns](#advanced-patterns)
8. [Performance Considerations](#performance-considerations)

---

## Overview

Traditional GPU computing discards uncertainty information during computation. Demetrios preserves and propagates epistemic state through every GPU operation using **shadow registers**.

```
Standard GPU:     value → compute → value
Demetrios GPU:    Knowledge[value, ε, validity, provenance] → compute → Knowledge[value', ε', validity', provenance']
```

### Key Concepts

| Concept | Description | GPU Representation |
|---------|-------------|-------------------|
| **Value** | The actual computed data | `%r_val` (f32/f64) |
| **Epsilon (ε)** | Uncertainty bound | `%r_eps` (f32) |
| **Validity** | Is this value trustworthy? | `%p_valid` (predicate) |
| **Provenance** | Data lineage tracking | `%r_prov` (u64 bitmask) |

---

## Shadow Register Architecture

Every epistemic value in Demetrios is represented by four shadow registers:

```
┌─────────────────────────────────────────────────────────┐
│                    Epistemic Value                       │
├─────────────────────────────────────────────────────────┤
│  %r_value    │  f32/f64  │  The actual computed value   │
│  %r_epsilon  │  f32      │  Uncertainty bound (σ)       │
│  %p_valid    │  pred     │  Validity predicate          │
│  %r_prov     │  u64      │  Provenance bitmask          │
└─────────────────────────────────────────────────────────┘
```

### Rust API

```rust
use demetrios::codegen::gpu::{EpistemicShadowRegs, EpistemicPtxEmitter, EpistemicPtxConfig};

// Create shadow registers for a value
let concentration = EpistemicShadowRegs {
    value: "%r_conc".to_string(),
    epsilon: "%r_conc_eps".to_string(),
    validity: "%p_conc_valid".to_string(),
    provenance: "%r_conc_prov".to_string(),
};

// Or use the convenience constructor
let concentration = EpistemicShadowRegs::new("concentration");
// Creates: %r_concentration, %r_concentration_eps, %p_concentration_valid, %r_concentration_prov
```

---

## Basic Usage

### Compiling Demetrios Code to GPU

```rust
use demetrios::{compile_to_gpu, compile_to_gpu_epistemic};

// Source code with epistemic types
let source = r#"
    module pharmacokinetics
    
    // Drug concentration with uncertainty
    let concentration: Knowledge[mg/L, ε=0.05] = measure_plasma()
    
    // Compute clearance - uncertainty propagates automatically
    let clearance = dose / concentration
"#;

// Compile to standard PTX (SM 8.0 = RTX 3000/4000 series)
let ptx = compile_to_gpu(source, (8, 0))?;

// Compile with epistemic tracking enabled
let epistemic_ptx = compile_to_gpu_epistemic(source, (8, 0))?;
```

### Using the Lowering API Directly

```rust
use demetrios::codegen::gpu::{lower, lower_with_config, LoweringConfig, GpuTarget, PtxCodegen};
use demetrios::hlir::HlirModule;

// Create or obtain HLIR module
let hlir: HlirModule = /* from parser/type checker */;

// Lower with default settings
let gpu_module = lower(&hlir, GpuTarget::Cuda { compute_capability: (8, 0) });

// Or with custom configuration
let config = LoweringConfig {
    target: GpuTarget::Cuda { compute_capability: (8, 0) },
    epistemic_enabled: true,
    counterfactual_enabled: true,
    max_threads_per_block: Some(256),
    shared_memory_hint: 48 * 1024,
    fast_math: true,
    debug_info: false,
};

let gpu_module = lower_with_config(&hlir, &config);

// Generate PTX
let mut codegen = PtxCodegen::new((8, 0));
let ptx = codegen.generate(&gpu_module);
```

---

## Epistemic PTX Emission

### Configuration

```rust
use demetrios::codegen::gpu::{EpistemicPtxConfig, EpistemicPtxEmitter};

let config = EpistemicPtxConfig {
    sm_version: (8, 0),              // Target GPU architecture
    default_epsilon: 0.0,            // Default uncertainty for constants
    confidence_threshold: 0.05,      // Threshold for confidence gating
    quadrature_propagation: true,    // Use sqrt(ε_a² + ε_b²) for addition
    warp_aggregation: true,          // Enable warp-level epistemic ops
    provenance_tracking: true,       // Track data lineage
    provenance_bits: 8,              // Bits per source in provenance mask
};

let mut emitter = EpistemicPtxEmitter::new(config);
```

### Epistemic Addition

When adding two uncertain values, uncertainties combine via quadrature:

```
ε_result = √(ε_a² + ε_b²)
```

```rust
// Define operands
let a = EpistemicShadowRegs::new("a");
let b = EpistemicShadowRegs::new("b");
let result = EpistemicShadowRegs::new("result");

// Emit epistemic addition
emitter.emit_epistemic_add(&result, &a, &b, false); // false = add, true = subtract

// Generated PTX:
// // Epistemic add
// add.f32 %r_result, %r_a, %r_b;
// // Quadrature: ε_c = sqrt(ε_a² + ε_b²)
// mul.f32 %r_eps_t0, %r_a_eps, %r_a_eps;
// mul.f32 %r_eps_t1, %r_b_eps, %r_b_eps;
// add.f32 %r_eps_t2, %r_eps_t0, %r_eps_t1;
// sqrt.approx.f32 %r_result_eps, %r_eps_t2;
// // Validity: v_c = v_a ∧ v_b
// and.pred %p_result_valid, %p_a_valid, %p_b_valid;
// // Provenance: prov_c = prov_a ⊕ prov_b
// xor.b64 %r_result_prov, %r_a_prov, %r_b_prov;
```

### Epistemic Multiplication

For multiplication, uncertainty propagates via first-order approximation:

```
ε_result ≈ |a|·ε_b + |b|·ε_a
```

```rust
emitter.emit_epistemic_mul(&result, &a, &b);

// Generated PTX:
// // Epistemic mul
// mul.f32 %r_result, %r_a, %r_b;
// // Relative error propagation
// abs.f32 %r_abs_a, %r_a;
// abs.f32 %r_abs_b, %r_b;
// mul.f32 %r_t0, %r_abs_a, %r_b_eps;
// mul.f32 %r_t1, %r_abs_b, %r_a_eps;
// add.f32 %r_result_eps, %r_t0, %r_t1;
// and.pred %p_result_valid, %p_a_valid, %p_b_valid;
// xor.b64 %r_result_prov, %r_a_prov, %r_b_prov;
```

### Epistemic Division

Division requires careful handling of uncertainty, especially near zero:

```rust
emitter.emit_epistemic_div(&result, &numerator, &denominator);
```

### Fused Multiply-Add (FMA)

For `result = a * b + c` with uncertainty:

```rust
emitter.emit_epistemic_fma(&result, &a, &b, &c);
```

### Warp-Level Aggregation

Aggregate epistemic state across a warp (32 threads):

```rust
use demetrios::codegen::gpu::WarpEpsilonOp;

// Reduce epsilon across warp (find max uncertainty)
emitter.emit_warp_epsilon_reduce(&shadow, "%r_max_eps", WarpEpsilonOp::Max);

// Count valid values in warp
emitter.emit_warp_confidence_vote(&shadow, "%r_valid_count", "%r_valid_mask");
```

### Confidence-Gated Execution

Only execute expensive operations when confidence is high:

```rust
emitter.emit_confidence_gate(&shadow, 0.05, "high_confidence_path", "low_confidence_fallback");

// Generated PTX:
// // Confidence gate (threshold = 0.05)
// setp.lt.f32 %p_confident, %r_shadow_eps, 0F3D4CCCCD;  // 0.05 in hex
// and.pred %p_confident, %p_confident, %p_shadow_valid;
// @%p_confident bra high_confidence_path;
// @!%p_confident bra low_confidence_fallback;
```

---

## Counterfactual Execution

Demetrios implements Pearl's causal hierarchy on GPU using parallel world execution.

### The Ladder of Causation

```
Level 3: COUNTERFACTUAL  │  "What if X had been x'?"     │  Imagining
         ↑               │                               │
Level 2: INTERVENTION    │  "What if I do X = x?"        │  Doing
         ↑               │                               │
Level 1: ASSOCIATION     │  "What is P(Y|X)?"            │  Seeing
```

### Setting Up Counterfactual Context

```rust
use demetrios::codegen::gpu::{
    CounterfactualContext, CounterfactualValue, WorldId,
    CounterfactualPtxConfig, CounterfactualPtxEmitter
};

// Create context
let mut ctx = CounterfactualContext::new();

// Set factual (observed) values
ctx.set_factual("treatment", CounterfactualValue::F32(0.0));  // No treatment
ctx.set_factual("age", CounterfactualValue::F32(45.0));
ctx.set_factual("outcome", CounterfactualValue::F32(0.3));    // 30% recovery

// Define exogenous variables (can be intervened upon)
ctx.add_exogenous("treatment");
ctx.add_exogenous("dosage");

// Add structural equations
ctx.add_structural_equation("outcome", "0.2 + 0.5*treatment - 0.01*age");
```

### Performing Interventions (do-operator)

```rust
// Intervene: do(treatment = 1.0)
let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

// The counterfactual world now has treatment = 1.0
// while the factual world still has treatment = 0.0

assert!(cf_world.is_counterfactual());
assert_eq!(ctx.get_value("treatment", cf_world).unwrap().as_f32(), Some(1.0));
assert_eq!(ctx.get_value("treatment", WorldId::FACTUAL).unwrap().as_f32(), Some(0.0));
```

### Computing Treatment Effects

```rust
// Compute divergence between worlds
let divergence = ctx.compute_divergence("outcome", cf_world);

if let Some(div) = divergence {
    println!("Absolute effect: {}", div.absolute);
    println!("Relative effect: {}", div.relative);
}
```

### Generating Counterfactual PTX

```rust
let config = CounterfactualPtxConfig {
    sm_version: (8, 0),
    worlds_per_warp: 2,      // Half warp = factual, half = counterfactual
    track_divergence: true,
    track_depth: true,
    max_depth: 8,            // Maximum nested interventions
};

let mut emitter = CounterfactualPtxEmitter::new(config);

// Emit declarations and initialization
emitter.emit_cf_declarations();
emitter.emit_cf_init();

// Emit intervention
emitter.emit_intervention("treatment", "%r_treatment", 1.0, 1);

// Generated PTX:
// // Intervention: do(treatment = 1)
// xor.b64 %r_world_id, %r_world_id, 0xCAFEBABE00000001;
// add.u32 %r_causal_depth, %r_causal_depth, 1;
// // Assign threads to worlds based on lane ID
// mov.u32 %r_cf_temp0, %laneid;
// and.b32 %r_cf_temp1, %r_cf_temp0, 1;
// setp.eq.u32 %p_is_factual, %r_cf_temp1, 0;
// setp.ne.u32 %p_is_cf, %r_cf_temp1, 0;
// // Apply intervention
// mov.f32 %r_cf_ftemp0, 0F3F800000;  // 1.0
// selp.f32 %r_treatment, %r_cf_ftemp0, %r_treatment, %p_is_cf;
```

### Computing Individual Treatment Effect (ITE)

```rust
emitter.emit_divergence_compute("%r_outcome", "%r_ite");

// Generated PTX:
// // Compute world divergence (treatment effect)
// // Exchange outcome between factual/counterfactual pairs
// shfl.sync.xor.b32 %r_cf_ftemp0, %r_outcome, 1, 0xFFFFFFFF;
// // ITE = outcome_cf - outcome_factual
// sub.f32 %r_ite, %r_outcome, %r_cf_ftemp0;
// @%p_is_factual neg.f32 %r_ite, %r_ite;
```

### Computing Average Treatment Effect (ATE)

```rust
emitter.emit_ate_compute("%r_ite", "%r_ate");

// Warp-level reduction of ITEs to compute ATE
```

### Structural Equation Models

```rust
use demetrios::codegen::gpu::StructuralEqType;

// Linear: Y = β₀ + β₁X₁ + β₂X₂
emitter.emit_structural_eq(
    "%r_y",
    StructuralEqType::Linear,
    &["%r_x1", "%r_x2"],
    &[0.5, 2.0, -1.0],  // β₀=0.5, β₁=2.0, β₂=-1.0
);

// Logistic: Y = sigmoid(β₀ + β₁X)
emitter.emit_structural_eq(
    "%r_y",
    StructuralEqType::Logistic,
    &["%r_x"],
    &[0.0, 1.0],
);

// Threshold: Y = 1 if X > θ else 0
emitter.emit_structural_eq(
    "%r_y",
    StructuralEqType::Threshold,
    &["%r_x"],
    &[0.5],  // threshold
);
```

### Probability of Causation

Compute P(Y_x=1 | X=0, Y=0) - the probability that X caused Y:

```rust
emitter.emit_probability_causation("%r_x", "%r_y", "%r_y_cf", "%r_prob_causation");
```

### Nested Interventions

For second-level counterfactuals (nested do-operators):

```rust
emitter.emit_nested_intervention(
    "treatment", 1.0,    // First intervention
    "dosage", 100.0,     // Second intervention
    "%r_treatment",
    "%r_dosage",
);
```

---

## Z3 SMT Verification

Verify epistemic properties formally using Z3.

### Basic Verification

```rust
use demetrios::smt::z3_solver::{Z3EpistemicVerifier, EpistemicProperty};

let verifier = Z3EpistemicVerifier::new();

// Verify that uncertainty is bounded
let result = verifier.verify_bounded_uncertainty(0.05, 0.1);  // ε=0.05, bound=0.1
assert!(result.is_valid);

// Verify validity implies confidence
let result = verifier.verify_validity_implies_confidence(0.95);
if !result.is_valid {
    println!("Counterexample: {:?}", result.counterexample);
}
```

### Available Properties

```rust
// 1. Bounded Uncertainty: ε ≤ bound
verifier.verify_bounded_uncertainty(epsilon, bound);

// 2. Validity Implies Confidence: valid → confidence > threshold
verifier.verify_validity_implies_confidence(threshold);

// 3. Provenance Completeness: all data sources tracked
verifier.verify_provenance_completeness();

// 4. Quadrature Correctness: ε_c² = ε_a² + ε_b² (for addition)
verifier.verify_quadrature_correctness();

// 5. Monotonicity: more data → less uncertainty
verifier.verify_uncertainty_monotonicity();
```

### Using Mock Verifier (when Z3 unavailable)

```rust
use demetrios::smt::z3_solver::MockEpistemicVerifier;

let verifier = MockEpistemicVerifier::new();
// Same API as Z3EpistemicVerifier, but uses heuristics instead of SMT
```

---

## Advanced Patterns

### Pharmacokinetic Simulation with Epistemic Tracking

```rust
// Full PK simulation kernel with uncertainty propagation
let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());

// Declare epistemic registers
let dose = EpistemicShadowRegs::new("dose");
let volume = EpistemicShadowRegs::new("volume");
let clearance = EpistemicShadowRegs::new("clearance");
let concentration = EpistemicShadowRegs::new("concentration");
let time = EpistemicShadowRegs::new("time");
let decay = EpistemicShadowRegs::new("decay");

// C(t) = (Dose/V) * exp(-CL/V * t)
// Step 1: dose / volume
emitter.emit_epistemic_div(&concentration, &dose, &volume);

// Step 2: clearance / volume (for decay rate)
let rate = EpistemicShadowRegs::new("rate");
emitter.emit_epistemic_div(&rate, &clearance, &volume);

// Step 3: rate * time
emitter.emit_epistemic_mul(&decay, &rate, &time);

// Step 4: exp(-decay) - use PTX exponential
emitter.emit("neg.f32 %r_neg_decay, %r_decay;");
emitter.emit("ex2.approx.f32 %r_exp_decay, %r_neg_decay;");

// Step 5: concentration * exp_decay
let exp_decay = EpistemicShadowRegs::new("exp_decay");
emitter.emit_epistemic_mul(&concentration, &concentration, &exp_decay);

// Gate on confidence before expensive lookup
emitter.emit_confidence_gate(&concentration, 0.1, "use_precise", "use_fallback");
```

### Causal Inference for Clinical Trial Analysis

```rust
// Analyze treatment effect with counterfactual reasoning
let mut ctx = CounterfactualContext::new();

// Observed data
ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
ctx.set_factual("age", CounterfactualValue::F32(55.0));
ctx.set_factual("comorbidity", CounterfactualValue::F32(2.0));
ctx.set_factual("outcome", CounterfactualValue::F32(0.4));

// Counterfactual: what if patient received treatment?
let cf_treated = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

// Generate GPU kernel for parallel evaluation
let mut emitter = CounterfactualPtxEmitter::new(CounterfactualPtxConfig::default());

emitter.emit_cf_declarations();
emitter.emit_cf_init();
emitter.emit_parallel_worlds(&ctx);

// Evaluate structural equation: outcome = f(treatment, age, comorbidity)
emitter.emit_structural_eq(
    "%r_outcome",
    StructuralEqType::Logistic,
    &["%r_treatment", "%r_age", "%r_comorbidity"],
    &[-2.0, 1.5, -0.02, -0.3],  // coefficients
);

// Compute treatment effect
emitter.emit_divergence_compute("%r_outcome", "%r_ite");
emitter.emit_ate_compute("%r_ite", "%r_ate");

let ptx = emitter.output();
```

### Warp-Level Epistemic Consensus

```rust
// Aggregate epistemic state across warp for consensus
let measurement = EpistemicShadowRegs::new("measurement");

// Find maximum uncertainty in warp
emitter.emit_warp_epsilon_reduce(&measurement, "%r_max_eps", WarpEpsilonOp::Max);

// Find minimum uncertainty (best measurement)
emitter.emit_warp_epsilon_reduce(&measurement, "%r_min_eps", WarpEpsilonOp::Min);

// Count valid measurements
emitter.emit_warp_confidence_vote(&measurement, "%r_valid_count", "%r_valid_mask");

// Only proceed if majority are valid
emitter.emit("setp.ge.u32 %p_consensus, %r_valid_count, 16;");  // 16/32 = 50%
emitter.emit("@!%p_consensus bra insufficient_consensus;");
```

---

## Performance Considerations

### Register Pressure

Each epistemic value uses 4 registers:
- 1 for value (f32/f64)
- 1 for epsilon (f32)
- 1 for validity (predicate, shared)
- 1 for provenance (u64)

**Recommendation**: For high-occupancy kernels, consider disabling provenance tracking:

```rust
let config = EpistemicPtxConfig {
    provenance_tracking: false,  // Save 1 register per value
    ..Default::default()
};
```

### Warp Divergence

Counterfactual execution uses warp lane assignment. With `worlds_per_warp = 2`:
- Even lanes (0, 2, 4, ...) = factual world
- Odd lanes (1, 3, 5, ...) = counterfactual world

**No warp divergence** because both worlds execute the same instructions with different data.

### Memory Bandwidth

Epistemic values are 4x larger than raw values. Consider:
- Using shared memory for frequently accessed epistemic state
- Coalescing epistemic loads/stores
- Prefetching epsilon values

### Approximations

For maximum performance:

```rust
let config = EpistemicPtxConfig {
    quadrature_propagation: false,  // Use additive instead of quadrature
    fast_math: true,                // Use .approx variants
    ..Default::default()
};
```

---

## Complete Example: Drug Interaction Simulation

```rust
use demetrios::codegen::gpu::*;

fn generate_drug_interaction_kernel() -> String {
    let epistemic_config = EpistemicPtxConfig {
        sm_version: (8, 0),
        confidence_threshold: 0.1,
        quadrature_propagation: true,
        ..Default::default()
    };
    
    let cf_config = CounterfactualPtxConfig {
        worlds_per_warp: 4,  // 4 different dosing scenarios
        track_divergence: true,
        ..Default::default()
    };
    
    let mut epistemic = EpistemicPtxEmitter::new(epistemic_config);
    let mut counterfactual = CounterfactualPtxEmitter::new(cf_config);
    
    // === Epistemic Computation ===
    
    // Drug A concentration with measurement uncertainty
    let drug_a = EpistemicShadowRegs::new("drug_a");
    let drug_b = EpistemicShadowRegs::new("drug_b");
    let interaction = EpistemicShadowRegs::new("interaction");
    
    // Interaction effect: drug_a * drug_b * k_interaction
    epistemic.emit_epistemic_mul(&interaction, &drug_a, &drug_b);
    
    // Gate expensive PK model on confidence
    epistemic.emit_confidence_gate(&interaction, 0.15, "full_model", "simplified_model");
    
    // === Counterfactual Analysis ===
    
    // What if we adjusted the dose?
    counterfactual.emit_cf_declarations();
    counterfactual.emit_cf_init();
    
    // Test 4 scenarios: 50%, 75%, 100%, 125% of standard dose
    counterfactual.emit_intervention("dose_factor", "%r_dose_factor", 0.5, 1);
    
    // Evaluate outcome in all worlds
    counterfactual.emit_divergence_compute("%r_efficacy", "%r_dose_response");
    
    // Combine outputs
    format!(
        "// Drug Interaction Kernel with Epistemic Tracking\n\
         // Generated by Demetrios v0.46.0\n\n\
         {}\n\n\
         // Counterfactual Analysis\n\
         {}",
        epistemic.output(),
        counterfactual.output()
    )
}
```

---

## See Also

- [CHANGELOG.md](../../CHANGELOG.md) - Release notes
- [Counterfactual Execution Deep Dive](./COUNTERFACTUAL_EXECUTION.md)
- [Z3 Verification Guide](./Z3_VERIFICATION.md)
- [Performance Tuning Guide](./PERFORMANCE.md)

---

*Demetrios: Where Uncertainty Meets Performance*
