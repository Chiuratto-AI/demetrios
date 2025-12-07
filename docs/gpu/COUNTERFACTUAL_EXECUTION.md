# Counterfactual Execution on GPU

This document provides an in-depth guide to counterfactual execution in Demetrios, implementing Pearl's causal hierarchy as GPU primitives.

## Table of Contents

1. [Introduction to Causal Inference](#introduction-to-causal-inference)
2. [The Ladder of Causation](#the-ladder-of-causation)
3. [GPU Implementation Strategy](#gpu-implementation-strategy)
4. [World Management](#world-management)
5. [Treatment Effect Estimation](#treatment-effect-estimation)
6. [Structural Causal Models](#structural-causal-models)
7. [Clinical Trial Examples](#clinical-trial-examples)
8. [Economic Policy Analysis](#economic-policy-analysis)

---

## Introduction to Causal Inference

Traditional statistics answers: *"What is the probability of Y given X?"*

Causal inference answers: *"What would happen to Y if we changed X?"*

Demetrios brings causal inference to GPU computing, enabling:
- **Parallel counterfactual evaluation** across thousands of scenarios
- **Real-time treatment effect estimation**
- **Causal discovery** at scale

### Why Causal Reasoning on GPU?

| Use Case | Traditional | Demetrios GPU |
|----------|-------------|---------------|
| Clinical trial simulation | Hours (CPU) | Seconds (GPU) |
| Policy impact analysis | Days | Minutes |
| Personalized medicine | Batch processing | Real-time |
| Economic modeling | Supercomputer | Single GPU |

---

## The Ladder of Causation

Pearl's three-level causal hierarchy, implemented on GPU:

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: COUNTERFACTUAL (Imagining)                             │
│ "What would Y have been if X had been x', given that we        │
│  actually observed X=x and Y=y?"                                │
│                                                                 │
│ GPU: Nested interventions, parallel world branching             │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: INTERVENTION (Doing)                                   │
│ "What will Y be if I do X=x?"                                  │
│                                                                 │
│ GPU: do(X=x) operator, warp-lane world assignment               │
├─────────────────────────────────────────────────────────────────┤
│ Level 1: ASSOCIATION (Seeing)                                   │
│ "What is P(Y|X)?"                                              │
│                                                                 │
│ GPU: Standard computation, conditional probabilities            │
└─────────────────────────────────────────────────────────────────┘
```

---

## GPU Implementation Strategy

### Warp Lane World Assignment

A GPU warp has 32 threads. Demetrios assigns different threads to different causal worlds:

```
Warp (32 threads)
├── Lanes 0-15:  Factual World (what actually happened)
└── Lanes 16-31: Counterfactual World (what-if scenario)

With worlds_per_warp = 4:
├── Lanes 0-7:   World 0 (Factual)
├── Lanes 8-15:  World 1 (do(X=x₁))
├── Lanes 16-23: World 2 (do(X=x₂))
└── Lanes 24-31: World 3 (do(X=x₃))
```

### PTX Implementation

```ptx
// Determine which world this thread belongs to
mov.u32 %r_lane, %laneid;
and.b32 %r_world_idx, %r_lane, 0x1;     // For 2 worlds
setp.eq.u32 %p_is_factual, %r_world_idx, 0;
setp.ne.u32 %p_is_cf, %r_world_idx, 0;

// Apply intervention in counterfactual world only
mov.f32 %r_cf_value, 0F3F800000;        // 1.0 (intervention value)
selp.f32 %r_treatment, %r_cf_value, %r_treatment, %p_is_cf;
```

### No Warp Divergence

The key insight: **both worlds execute the same code path** with different data values. This means:
- No branch divergence penalty
- Full SIMD utilization
- Maximum throughput

---

## World Management

### WorldId

Every causal world has a unique identifier:

```rust
use demetrios::codegen::gpu::WorldId;

// The factual world (what actually happened)
let factual = WorldId::FACTUAL;  // ID = 0

// Counterfactual worlds (what-if scenarios)
let cf1 = WorldId::counterfactual(1);  // ID = 0xCAFEBABE00000001
let cf2 = WorldId::counterfactual(2);  // ID = 0xCAFEBABE00000002

// Check world type
assert!(factual.is_factual());
assert!(cf1.is_counterfactual());
assert_eq!(cf1.intervention_id(), Some(1));
```

### CounterfactualContext

Manages the state of all causal worlds:

```rust
use demetrios::codegen::gpu::{CounterfactualContext, CounterfactualValue, WorldId};

let mut ctx = CounterfactualContext::new();

// Set observed (factual) values
ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
ctx.set_factual("age", CounterfactualValue::F32(65.0));
ctx.set_factual("blood_pressure", CounterfactualValue::F32(140.0));
ctx.set_factual("outcome", CounterfactualValue::F32(0.2));

// Perform intervention: do(treatment = 1.0)
let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

// Access values in different worlds
let factual_treatment = ctx.get_value("treatment", WorldId::FACTUAL);  // 0.0
let cf_treatment = ctx.get_value("treatment", cf_world);               // 1.0
```

### WorldSnapshot

Captures the complete state of a world at a point in time:

```rust
use demetrios::codegen::gpu::WorldSnapshot;

// Create factual snapshot
let snapshot = WorldSnapshot::factual();
assert_eq!(snapshot.depth, 0);  // No interventions

// After intervention, depth increases
let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));
let cf_snapshot = ctx.snapshots.get(&cf_world).unwrap();
assert_eq!(cf_snapshot.depth, 1);  // One intervention from factual
```

### Multiple Interventions

```rust
// Sequential interventions create a tree of worlds
let world_a = ctx.intervene("treatment", CounterfactualValue::F32(1.0));
let world_b = ctx.intervene("dosage", CounterfactualValue::F32(100.0));

// Both branch from factual world
assert!(world_a.is_counterfactual());
assert!(world_b.is_counterfactual());
assert_ne!(world_a, world_b);
```

---

## Treatment Effect Estimation

### Individual Treatment Effect (ITE)

The effect of treatment for a specific individual:

```
ITE = Y(1) - Y(0)
    = Outcome if treated - Outcome if not treated
```

```rust
let mut emitter = CounterfactualPtxEmitter::new(config);

// After computing outcomes in both worlds...
emitter.emit_divergence_compute("%r_outcome", "%r_ite");

// Generated PTX:
// // Exchange outcome with paired thread in other world
// shfl.sync.xor.b32 %r_other_outcome, %r_outcome, 1, 0xFFFFFFFF;
// // ITE = my_outcome - other_outcome
// sub.f32 %r_ite, %r_outcome, %r_other_outcome;
// // For factual threads, negate to get correct sign
// @%p_is_factual neg.f32 %r_ite, %r_ite;
```

### Average Treatment Effect (ATE)

The average effect across a population:

```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

```rust
// Compute ATE via warp reduction
emitter.emit_ate_compute("%r_ite", "%r_ate");

// Generated PTX:
// // Warp-level reduction of ITEs
// mov.f32 %r_ate, %r_ite;
// shfl.sync.down.b32 %r_tmp, %r_ate, 16, 31, 0xFFFFFFFF;
// add.f32 %r_ate, %r_ate, %r_tmp;
// shfl.sync.down.b32 %r_tmp, %r_ate, 8, 31, 0xFFFFFFFF;
// add.f32 %r_ate, %r_ate, %r_tmp;
// ... (continue halving)
// // Divide by number of pairs
// mul.f32 %r_ate, %r_ate, 0.0625;  // 1/16 for 2 worlds
```

### Conditional Average Treatment Effect (CATE)

Treatment effect for a subgroup:

```rust
// CATE for patients over 60
emitter.emit("setp.ge.f32 %p_over_60, %r_age, 60.0;");
emitter.emit("@%p_over_60 mov.f32 %r_cate_contrib, %r_ite;");
emitter.emit("@!%p_over_60 mov.f32 %r_cate_contrib, 0.0;");

// Then reduce only non-zero contributions
```

### Probability of Causation

Three related probabilities:

1. **Probability of Necessity (PN)**: P(Y₀=0 | X=1, Y=1)
   *"Given that treatment was given and patient recovered, would they have recovered without it?"*

2. **Probability of Sufficiency (PS)**: P(Y₁=1 | X=0, Y=0)
   *"Given that no treatment and no recovery, would treatment have caused recovery?"*

3. **Probability of Necessity and Sufficiency (PNS)**:
   *"Treatment is both necessary and sufficient for this outcome"*

```rust
// Compute probability of causation
emitter.emit_probability_causation("%r_x", "%r_y", "%r_y_cf", "%r_prob");

// Generated PTX checks:
// 1. X=0 (no treatment in factual)
// 2. Y=0 (no outcome in factual)
// 3. Y_cf=1 (outcome would occur with treatment)
```

---

## Structural Causal Models

### Defining Structural Equations

A Structural Causal Model (SCM) consists of:
- **Exogenous variables (U)**: External factors
- **Endogenous variables (V)**: Determined by the model
- **Structural equations (F)**: V = f(U, Pa(V))

```rust
let mut ctx = CounterfactualContext::new();

// Define exogenous variables
ctx.add_exogenous("genetics");
ctx.add_exogenous("environment");

// Define structural equations
ctx.add_structural_equation("blood_pressure", "120 + 0.3*genetics + 0.2*environment");
ctx.add_structural_equation("outcome", "sigmoid(-2 + 0.5*treatment - 0.01*blood_pressure)");
```

### Evaluating Structural Equations on GPU

```rust
use demetrios::codegen::gpu::StructuralEqType;

// Linear model: Y = β₀ + β₁X₁ + β₂X₂ + ...
emitter.emit_structural_eq(
    "%r_blood_pressure",
    StructuralEqType::Linear,
    &["%r_genetics", "%r_environment"],
    &[120.0, 0.3, 0.2],  // [β₀, β₁, β₂]
);

// Generated PTX:
// mov.f32 %r_blood_pressure, 0F42F00000;  // 120.0
// mov.f32 %r_t0, 0F3E99999A;              // 0.3
// fma.rn.f32 %r_blood_pressure, %r_t0, %r_genetics, %r_blood_pressure;
// mov.f32 %r_t0, 0F3E4CCCCD;              // 0.2
// fma.rn.f32 %r_blood_pressure, %r_t0, %r_environment, %r_blood_pressure;
```

### Logistic Structural Equation

For binary outcomes:

```rust
// Y = sigmoid(β₀ + β₁X₁ + ...)
emitter.emit_structural_eq(
    "%r_outcome",
    StructuralEqType::Logistic,
    &["%r_treatment", "%r_blood_pressure"],
    &[-2.0, 0.5, -0.01],
);

// Generated PTX includes sigmoid:
// ... compute linear part ...
// neg.f32 %r_t0, %r_linear;
// ex2.approx.f32 %r_t0, %r_t0;    // exp(-x) ≈ 2^(-x/ln2)
// add.f32 %r_t0, %r_t0, 1.0;
// rcp.approx.f32 %r_outcome, %r_t0;  // 1/(1+exp(-x))
```

### Multiplicative Model

For log-linear relationships:

```rust
// Y = ∏ Xᵢ^βᵢ
emitter.emit_structural_eq(
    "%r_risk",
    StructuralEqType::Multiplicative,
    &["%r_age", "%r_bmi"],
    &[1.5, 0.8],  // Age increases risk, BMI decreases
);
```

### Threshold Model

For step-function responses:

```rust
// Y = 1 if X > θ else 0
emitter.emit_structural_eq(
    "%r_hypertensive",
    StructuralEqType::Threshold,
    &["%r_blood_pressure"],
    &[140.0],  // Threshold for hypertension
);
```

---

## Clinical Trial Examples

### Example 1: Drug Efficacy Analysis

```rust
use demetrios::codegen::gpu::*;

fn analyze_drug_efficacy() {
    let mut ctx = CounterfactualContext::new();
    
    // Patient cohort data (would come from trial)
    ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
    ctx.set_factual("age", CounterfactualValue::F32(55.0));
    ctx.set_factual("baseline_severity", CounterfactualValue::F32(7.0));
    ctx.set_factual("outcome_severity", CounterfactualValue::F32(5.0));
    
    // Counterfactual: what if patient received drug?
    let cf_treated = ctx.intervene("treatment", CounterfactualValue::F32(1.0));
    
    // Generate GPU kernel
    let config = CounterfactualPtxConfig::default();
    let mut emitter = CounterfactualPtxEmitter::new(config);
    
    emitter.emit_cf_declarations();
    emitter.emit_cf_init();
    emitter.emit_parallel_worlds(&ctx);
    
    // Model: outcome = baseline - treatment_effect - age_effect
    emitter.emit_structural_eq(
        "%r_outcome",
        StructuralEqType::Linear,
        &["%r_baseline_severity", "%r_treatment", "%r_age"],
        &[0.0, 1.0, -3.0, 0.02],  // Drug reduces severity by 3 points
    );
    
    // Compute treatment effect
    emitter.emit_divergence_compute("%r_outcome", "%r_ite");
    emitter.emit_ate_compute("%r_ite", "%r_ate");
    
    println!("Generated PTX:\n{}", emitter.output());
}
```

### Example 2: Personalized Dosing

```rust
fn personalized_dosing_kernel() {
    let mut ctx = CounterfactualContext::new();
    
    // Patient characteristics
    ctx.set_factual("weight", CounterfactualValue::F32(70.0));
    ctx.set_factual("renal_function", CounterfactualValue::F32(0.8));
    ctx.set_factual("current_dose", CounterfactualValue::F32(100.0));
    
    // Test multiple dose scenarios
    let config = CounterfactualPtxConfig {
        worlds_per_warp: 8,  // Test 8 different doses
        ..Default::default()
    };
    
    let mut emitter = CounterfactualPtxEmitter::new(config);
    
    emitter.emit_cf_declarations();
    emitter.emit_cf_init();
    
    // Each world tests a different dose multiplier
    // World 0: 0.5x, World 1: 0.75x, ..., World 7: 2.0x
    emitter.emit("mov.u32 %r_world, %laneid;");
    emitter.emit("and.b32 %r_world, %r_world, 7;");
    emitter.emit("cvt.rn.f32.u32 %r_mult, %r_world;");
    emitter.emit("fma.rn.f32 %r_mult, %r_mult, 0.214, 0.5;");  // 0.5 to 2.0
    emitter.emit("mul.f32 %r_test_dose, %r_current_dose, %r_mult;");
    
    // PK model: concentration = dose / (volume * clearance)
    // clearance depends on renal function
    emitter.emit_structural_eq(
        "%r_clearance",
        StructuralEqType::Linear,
        &["%r_renal_function"],
        &[5.0, 10.0],  // CL = 5 + 10 * renal_function
    );
    
    // Compute efficacy and toxicity for each dose
    emitter.emit_structural_eq(
        "%r_efficacy",
        StructuralEqType::Logistic,
        &["%r_concentration"],
        &[-2.0, 0.1],  // Efficacy increases with concentration
    );
    
    emitter.emit_structural_eq(
        "%r_toxicity",
        StructuralEqType::Logistic,
        &["%r_concentration"],
        &[-5.0, 0.15],  // Toxicity at higher concentrations
    );
    
    // Find optimal dose (max efficacy - toxicity)
    emitter.emit("sub.f32 %r_utility, %r_efficacy, %r_toxicity;");
    
    // Warp reduction to find best dose
    emitter.emit("redux.sync.max.f32 %r_best_utility, %r_utility, 0xFF;");
}
```

### Example 3: Adverse Event Attribution

```rust
fn adverse_event_attribution() {
    // Did the drug cause the adverse event?
    let mut ctx = CounterfactualContext::new();
    
    // Observed: patient took drug and had adverse event
    ctx.set_factual("drug_exposure", CounterfactualValue::F32(1.0));
    ctx.set_factual("adverse_event", CounterfactualValue::F32(1.0));
    ctx.set_factual("risk_factors", CounterfactualValue::F32(0.3));
    
    // Counterfactual: what if no drug exposure?
    let cf_no_drug = ctx.intervene("drug_exposure", CounterfactualValue::F32(0.0));
    
    let mut emitter = CounterfactualPtxEmitter::new(CounterfactualPtxConfig::default());
    
    emitter.emit_cf_declarations();
    emitter.emit_cf_init();
    emitter.emit_parallel_worlds(&ctx);
    
    // Model adverse event probability
    emitter.emit_structural_eq(
        "%r_ae_prob",
        StructuralEqType::Logistic,
        &["%r_drug_exposure", "%r_risk_factors"],
        &[-3.0, 2.0, 1.5],  // Drug increases AE risk
    );
    
    // Compute probability of causation
    // P(AE=0 | do(drug=0)) when we observed AE=1 with drug=1
    emitter.emit_probability_causation(
        "%r_drug_exposure",
        "%r_adverse_event",
        "%r_ae_cf",
        "%r_prob_caused"
    );
    
    // If prob_caused > 0.5, likely drug-related
}
```

---

## Economic Policy Analysis

### Example: Minimum Wage Impact

```rust
fn minimum_wage_analysis() {
    let mut ctx = CounterfactualContext::new();
    
    // Current economic state
    ctx.set_factual("min_wage", CounterfactualValue::F32(7.25));
    ctx.set_factual("employment_rate", CounterfactualValue::F32(0.95));
    ctx.set_factual("avg_income", CounterfactualValue::F32(35000.0));
    ctx.set_factual("inflation", CounterfactualValue::F32(0.02));
    
    // Test wage increases: $10, $12, $15
    let scenarios = [10.0, 12.0, 15.0];
    
    let config = CounterfactualPtxConfig {
        worlds_per_warp: 4,  // Factual + 3 scenarios
        ..Default::default()
    };
    
    let mut emitter = CounterfactualPtxEmitter::new(config);
    
    emitter.emit_cf_declarations();
    emitter.emit_cf_init();
    
    // Assign scenarios to warp lanes
    for (i, wage) in scenarios.iter().enumerate() {
        emitter.emit_intervention("min_wage", "%r_min_wage", *wage, (i + 1) as u32);
    }
    
    // Economic model (simplified)
    // Employment effect: higher wage -> lower employment (elasticity)
    emitter.emit_structural_eq(
        "%r_employment_effect",
        StructuralEqType::Linear,
        &["%r_min_wage"],
        &[1.0, -0.01],  // 1% decrease per $1 increase
    );
    
    // Income effect for employed workers
    emitter.emit_structural_eq(
        "%r_income_effect",
        StructuralEqType::Linear,
        &["%r_min_wage"],
        &[30000.0, 500.0],  // Base + $500 per $1 wage increase
    );
    
    // Compute net welfare change
    emitter.emit("mul.f32 %r_employed_welfare, %r_employment_effect, %r_income_effect;");
    
    // Compare to factual
    emitter.emit_divergence_compute("%r_employed_welfare", "%r_policy_effect");
}
```

---

## Best Practices

### 1. Choose Appropriate World Count

```rust
// 2 worlds: Simple A/B comparison
let config = CounterfactualPtxConfig { worlds_per_warp: 2, .. };

// 4 worlds: Good for dose-response with 4 levels
let config = CounterfactualPtxConfig { worlds_per_warp: 4, .. };

// 8+ worlds: Many scenarios, but reduces threads per world
let config = CounterfactualPtxConfig { worlds_per_warp: 8, .. };
```

### 2. Limit Intervention Depth

```rust
// Nested interventions increase complexity exponentially
let config = CounterfactualPtxConfig {
    max_depth: 3,  // Reasonable limit
    track_depth: true,
    ..Default::default()
};
```

### 3. Use Appropriate Structural Equations

| Data Type | Recommended Model |
|-----------|------------------|
| Continuous outcome | Linear |
| Binary outcome | Logistic |
| Count data | Multiplicative |
| Threshold effect | Threshold |

### 4. Validate with Known Effects

```rust
// Test with known causal effect before production
ctx.set_factual("x", CounterfactualValue::F32(0.0));
ctx.set_factual("y", CounterfactualValue::F32(0.0));

// Known: treatment effect = 2.0
ctx.add_structural_equation("y", "2.0 * x");

let cf = ctx.intervene("x", CounterfactualValue::F32(1.0));
let div = ctx.compute_divergence("y", cf).unwrap();
assert!((div.absolute - 2.0).abs() < 0.001);  // Should be exactly 2.0
```

---

## See Also

- [Epistemic GPU Computing](./EPISTEMIC_GPU_COMPUTING.md)
- [Z3 Verification Guide](./Z3_VERIFICATION.md)
- [Pearl, J. - Causality (2009)](http://bayes.cs.ucla.edu/BOOK-2K/)
- [Pearl, J. - The Book of Why (2018)](http://bayes.cs.ucla.edu/WHY/)

---

*"Correlation is not causation, but with Demetrios, you can compute causation at GPU speed."*
