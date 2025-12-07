# Z3 SMT Verification for Epistemic Properties

This guide covers formal verification of epistemic properties in Demetrios using the Z3 SMT solver.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup and Configuration](#setup-and-configuration)
3. [Verifiable Properties](#verifiable-properties)
4. [Basic Usage](#basic-usage)
5. [Counterexample Analysis](#counterexample-analysis)
6. [Custom Verification](#custom-verification)
7. [Integration with GPU Code](#integration-with-gpu-code)
8. [Performance Tips](#performance-tips)

---

## Introduction

Demetrios integrates with Z3 to provide **formal guarantees** about epistemic properties. This enables:

- **Compile-time verification** of uncertainty bounds
- **Proof that validity implies confidence**
- **Verification of provenance completeness**
- **Correctness proofs for uncertainty propagation**

### Why Formal Verification?

| Approach | Guarantee | Runtime Cost |
|----------|-----------|--------------|
| Testing | Probabilistic | High |
| Runtime checks | None until failure | Continuous |
| **Z3 Verification** | **Mathematical proof** | **Compile-time only** |

---

## Setup and Configuration

### Enabling Z3 Support

Z3 verification is feature-gated. Enable it in `Cargo.toml`:

```toml
[dependencies]
demetrios = { version = "0.46.0", features = ["smt"] }
```

Or build with the feature flag:

```bash
cargo build --features smt
```

### Checking Z3 Availability

```rust
use demetrios::smt::z3_solver;

#[cfg(feature = "smt")]
fn verify_with_z3() {
    let verifier = z3_solver::Z3EpistemicVerifier::new();
    // Z3 is available
}

#[cfg(not(feature = "smt"))]
fn verify_with_mock() {
    let verifier = z3_solver::MockEpistemicVerifier::new();
    // Using mock verifier (heuristic-based)
}
```

---

## Verifiable Properties

### 1. Bounded Uncertainty

**Property**: Uncertainty (ε) is always less than or equal to a specified bound.

```
∀ values v: v.epsilon ≤ bound
```

**Use Case**: Ensure measurement uncertainty never exceeds acceptable limits.

```rust
let verifier = Z3EpistemicVerifier::new();

// Verify that epsilon ≤ 0.1 (10% uncertainty)
let result = verifier.verify_bounded_uncertainty(0.05, 0.1);

if result.is_valid {
    println!("Proven: uncertainty is bounded by 10%");
} else {
    println!("Violation found: {:?}", result.counterexample);
}
```

### 2. Validity Implies Confidence

**Property**: If a value is marked valid, its confidence exceeds a threshold.

```
∀ values v: v.valid → (1 - v.epsilon) > threshold
```

**Use Case**: Ensure valid data meets quality requirements.

```rust
// Verify that valid values have >95% confidence
let result = verifier.verify_validity_implies_confidence(0.95);

assert!(result.is_valid, "Valid values must have high confidence");
```

### 3. Provenance Completeness

**Property**: All data sources contributing to a value are tracked.

```
∀ values v: popcount(v.provenance) ≥ expected_sources
```

**Use Case**: Audit trail compliance, data lineage verification.

```rust
let result = verifier.verify_provenance_completeness();

if result.is_valid {
    println!("All data sources are tracked");
}
```

### 4. Quadrature Correctness

**Property**: Uncertainty propagation follows quadrature rule for independent errors.

```
∀ a, b, c where c = a + b:
    c.epsilon² = a.epsilon² + b.epsilon²
```

**Use Case**: Verify uncertainty propagation in scientific computations.

```rust
let result = verifier.verify_quadrature_correctness();

assert!(result.is_valid, "Quadrature propagation must be correct");
```

### 5. Uncertainty Monotonicity

**Property**: Adding more data cannot increase uncertainty (in aggregate).

```
∀ computations with n inputs:
    more_inputs → lower_or_equal_aggregate_uncertainty
```

**Use Case**: Verify that data fusion reduces uncertainty.

```rust
let result = verifier.verify_uncertainty_monotonicity();
```

---

## Basic Usage

### Creating a Verifier

```rust
use demetrios::smt::z3_solver::{Z3EpistemicVerifier, VerificationResult};

// Create verifier with default configuration
let verifier = Z3EpistemicVerifier::new();

// Or with custom timeout
let verifier = Z3EpistemicVerifier::with_timeout(Duration::from_secs(30));
```

### Running Verification

```rust
// Single property
let result = verifier.verify_bounded_uncertainty(0.05, 0.1);

// Multiple properties
let properties = vec![
    verifier.verify_bounded_uncertainty(0.05, 0.1),
    verifier.verify_validity_implies_confidence(0.95),
    verifier.verify_provenance_completeness(),
];

for (i, result) in properties.iter().enumerate() {
    if result.is_valid {
        println!("Property {} verified", i);
    } else {
        println!("Property {} failed: {:?}", i, result.counterexample);
    }
}
```

### Verification Result

```rust
pub struct VerificationResult {
    /// Whether the property holds
    pub is_valid: bool,
    
    /// Counterexample if property fails
    pub counterexample: Option<Counterexample>,
    
    /// Time taken for verification
    pub verification_time: Duration,
    
    /// Z3 statistics
    pub stats: VerificationStats,
}

pub struct Counterexample {
    /// Variable assignments that violate the property
    pub assignments: HashMap<String, f64>,
    
    /// Human-readable explanation
    pub explanation: String,
}
```

---

## Counterexample Analysis

When verification fails, Z3 provides a counterexample showing **why** the property doesn't hold.

### Example: Bounded Uncertainty Failure

```rust
let result = verifier.verify_bounded_uncertainty(0.15, 0.1);

if !result.is_valid {
    let ce = result.counterexample.unwrap();
    
    println!("Counterexample found:");
    println!("  epsilon = {}", ce.assignments.get("epsilon").unwrap());
    println!("  bound = {}", ce.assignments.get("bound").unwrap());
    println!("  Explanation: {}", ce.explanation);
    
    // Output:
    // Counterexample found:
    //   epsilon = 0.15
    //   bound = 0.1
    //   Explanation: epsilon (0.15) exceeds bound (0.1)
}
```

### Example: Confidence Threshold Failure

```rust
let result = verifier.verify_validity_implies_confidence(0.99);

if !result.is_valid {
    let ce = result.counterexample.unwrap();
    
    println!("Found valid value with low confidence:");
    println!("  validity = {}", ce.assignments.get("validity").unwrap());
    println!("  epsilon = {}", ce.assignments.get("epsilon").unwrap());
    println!("  confidence = {}", 1.0 - ce.assignments.get("epsilon").unwrap());
    
    // Output:
    // Found valid value with low confidence:
    //   validity = 1.0 (true)
    //   epsilon = 0.05
    //   confidence = 0.95 (< 0.99 threshold)
}
```

### Using Counterexamples for Debugging

```rust
fn debug_epistemic_kernel(verifier: &Z3EpistemicVerifier) {
    // Check all properties
    let checks = vec![
        ("bounded", verifier.verify_bounded_uncertainty(0.05, 0.1)),
        ("confidence", verifier.verify_validity_implies_confidence(0.95)),
        ("provenance", verifier.verify_provenance_completeness()),
        ("quadrature", verifier.verify_quadrature_correctness()),
    ];
    
    let mut failures = Vec::new();
    
    for (name, result) in checks {
        if !result.is_valid {
            failures.push((name, result.counterexample.unwrap()));
        }
    }
    
    if failures.is_empty() {
        println!("All epistemic properties verified!");
    } else {
        println!("Verification failures:");
        for (name, ce) in &failures {
            println!("\n  Property '{}' failed:", name);
            println!("    {}", ce.explanation);
            println!("    Values: {:?}", ce.assignments);
        }
        
        // Suggest fixes
        suggest_fixes(&failures);
    }
}

fn suggest_fixes(failures: &[(&str, Counterexample)]) {
    for (name, ce) in failures {
        match *name {
            "bounded" => {
                let eps = ce.assignments.get("epsilon").unwrap();
                println!("Fix: Reduce uncertainty by improving measurement precision");
                println!("     Current: ε = {:.4}", eps);
            }
            "confidence" => {
                println!("Fix: Mark low-confidence values as invalid");
                println!("     Or lower the confidence threshold");
            }
            "provenance" => {
                println!("Fix: Ensure all data sources set provenance bits");
            }
            "quadrature" => {
                println!("Fix: Enable quadrature_propagation in EpistemicPtxConfig");
            }
            _ => {}
        }
    }
}
```

---

## Custom Verification

### Defining Custom Properties

```rust
use demetrios::smt::z3_solver::{Z3Solver, Formula};

fn verify_custom_property(solver: &Z3Solver) -> VerificationResult {
    // Create Z3 context and solver
    let ctx = solver.context();
    let z3 = solver.solver();
    
    // Define variables
    let epsilon_a = ctx.real_const("epsilon_a");
    let epsilon_b = ctx.real_const("epsilon_b");
    let epsilon_c = ctx.real_const("epsilon_c");
    
    // Define property: ε_c ≤ ε_a + ε_b (triangle inequality for uncertainty)
    let property = epsilon_c.le(&(epsilon_a.clone() + epsilon_b.clone()));
    
    // Add constraints
    z3.assert(&epsilon_a.ge(&ctx.real(0)));
    z3.assert(&epsilon_b.ge(&ctx.real(0)));
    z3.assert(&epsilon_c.ge(&ctx.real(0)));
    
    // Check if negation is satisfiable (counterexample exists)
    z3.assert(&property.not());
    
    match z3.check() {
        z3::SatResult::Unsat => {
            // No counterexample = property holds
            VerificationResult::valid()
        }
        z3::SatResult::Sat => {
            // Counterexample found
            let model = z3.get_model().unwrap();
            let ce = extract_counterexample(&model);
            VerificationResult::invalid(ce)
        }
        z3::SatResult::Unknown => {
            VerificationResult::timeout()
        }
    }
}
```

### Complex Property: Epistemic Consistency

```rust
fn verify_epistemic_consistency(solver: &Z3Solver) -> VerificationResult {
    let ctx = solver.context();
    let z3 = solver.solver();
    
    // Variables for value a
    let val_a = ctx.real_const("val_a");
    let eps_a = ctx.real_const("eps_a");
    let valid_a = ctx.bool_const("valid_a");
    let prov_a = ctx.bv_const("prov_a", 64);
    
    // Variables for value b
    let val_b = ctx.real_const("val_b");
    let eps_b = ctx.real_const("eps_b");
    let valid_b = ctx.bool_const("valid_b");
    let prov_b = ctx.bv_const("prov_b", 64);
    
    // Variables for result c = a + b
    let val_c = ctx.real_const("val_c");
    let eps_c = ctx.real_const("eps_c");
    let valid_c = ctx.bool_const("valid_c");
    let prov_c = ctx.bv_const("prov_c", 64);
    
    // Constraints: epistemic addition rules
    // 1. Value: c = a + b
    z3.assert(&val_c._eq(&(val_a.clone() + val_b.clone())));
    
    // 2. Validity: valid_c = valid_a ∧ valid_b
    z3.assert(&valid_c._eq(&valid_a.and(&[&valid_b])));
    
    // 3. Provenance: prov_c = prov_a ⊕ prov_b
    z3.assert(&prov_c._eq(&prov_a.bvxor(&prov_b)));
    
    // 4. Epsilon: eps_c² = eps_a² + eps_b² (quadrature)
    let eps_a_sq = eps_a.clone() * eps_a.clone();
    let eps_b_sq = eps_b.clone() * eps_b.clone();
    let eps_c_sq = eps_c.clone() * eps_c.clone();
    z3.assert(&eps_c_sq._eq(&(eps_a_sq + eps_b_sq)));
    
    // Property to verify: if both inputs valid, output valid
    let property = valid_a.and(&[&valid_b]).implies(&valid_c);
    
    // Check property
    z3.assert(&property.not());
    
    // ... check and return result
}
```

---

## Integration with GPU Code

### Pre-Compilation Verification

```rust
use demetrios::smt::z3_solver::Z3EpistemicVerifier;
use demetrios::codegen::gpu::{compile_to_gpu_epistemic, EpistemicPtxConfig};

fn compile_verified(source: &str, sm_version: (u32, u32)) -> Result<String, String> {
    // First, verify epistemic properties
    let verifier = Z3EpistemicVerifier::new();
    
    let bounded = verifier.verify_bounded_uncertainty(0.05, 0.1);
    let confidence = verifier.verify_validity_implies_confidence(0.95);
    
    if !bounded.is_valid {
        return Err(format!(
            "Compilation failed: uncertainty bound violated\n{}",
            bounded.counterexample.unwrap().explanation
        ));
    }
    
    if !confidence.is_valid {
        return Err(format!(
            "Compilation failed: confidence requirement not met\n{}",
            confidence.counterexample.unwrap().explanation
        ));
    }
    
    // Properties verified, proceed with compilation
    let ptx = compile_to_gpu_epistemic(source, sm_version)?;
    Ok(ptx)
}
```

### Runtime Verification Hooks

```rust
fn generate_verified_kernel() -> String {
    let mut emitter = EpistemicPtxEmitter::new(EpistemicPtxConfig::default());
    
    // Generate epistemic computation
    let a = EpistemicShadowRegs::new("a");
    let b = EpistemicShadowRegs::new("b");
    let result = EpistemicShadowRegs::new("result");
    
    emitter.emit_epistemic_add(&result, &a, &b, false);
    
    // Add runtime assertion for debug builds
    #[cfg(debug_assertions)]
    {
        // Assert epsilon is bounded
        emitter.emit("// Runtime epsilon check");
        emitter.emit("setp.le.f32 %p_eps_ok, %r_result_eps, 0F3DCCCCCD;");  // 0.1
        emitter.emit("@!%p_eps_ok trap;");  // Halt if violated
    }
    
    emitter.output().to_string()
}
```

### Verified Module Pattern

```rust
/// A GPU module that has passed epistemic verification
pub struct VerifiedGpuModule {
    module: GpuModule,
    verification_results: Vec<VerificationResult>,
}

impl VerifiedGpuModule {
    pub fn new(hlir: &HlirModule, config: &LoweringConfig) -> Result<Self, VerificationError> {
        // Lower to GPU IR
        let module = lower_with_config(hlir, config);
        
        // Verify epistemic properties
        let verifier = Z3EpistemicVerifier::new();
        let results = vec![
            verifier.verify_bounded_uncertainty(0.05, config.epsilon_threshold),
            verifier.verify_validity_implies_confidence(0.95),
            verifier.verify_provenance_completeness(),
            verifier.verify_quadrature_correctness(),
        ];
        
        // Check all passed
        for result in &results {
            if !result.is_valid {
                return Err(VerificationError {
                    property: result.property_name.clone(),
                    counterexample: result.counterexample.clone(),
                });
            }
        }
        
        Ok(Self {
            module,
            verification_results: results,
        })
    }
    
    pub fn generate_ptx(&self, sm_version: (u32, u32)) -> String {
        let mut codegen = PtxCodegen::new(sm_version);
        codegen.generate(&self.module)
    }
    
    /// Returns proof artifacts for audit
    pub fn proofs(&self) -> &[VerificationResult] {
        &self.verification_results
    }
}
```

---

## Performance Tips

### 1. Use Incremental Solving

```rust
let solver = Z3Solver::new_incremental();

// Add base constraints once
solver.push();
solver.add_base_constraints();

// Check multiple properties without re-adding base
for property in properties {
    solver.push();
    solver.assert(&property.negation());
    let result = solver.check();
    solver.pop();
}
```

### 2. Simplify Formulas

```rust
// Before verification, simplify the formula
let simplified = ctx.simplify(&complex_formula);
z3.assert(&simplified);
```

### 3. Set Appropriate Timeouts

```rust
// Short timeout for compile-time checks
let verifier = Z3EpistemicVerifier::with_timeout(Duration::from_secs(5));

// Longer timeout for thorough verification
let verifier = Z3EpistemicVerifier::with_timeout(Duration::from_secs(60));
```

### 4. Cache Verification Results

```rust
use std::collections::HashMap;

struct VerificationCache {
    cache: HashMap<u64, VerificationResult>,
}

impl VerificationCache {
    fn verify_or_cached(
        &mut self,
        verifier: &Z3EpistemicVerifier,
        property_hash: u64,
        verify_fn: impl FnOnce() -> VerificationResult,
    ) -> &VerificationResult {
        self.cache.entry(property_hash).or_insert_with(verify_fn)
    }
}
```

### 5. Use Mock Verifier for Development

```rust
// Fast heuristic checks during development
#[cfg(debug_assertions)]
type Verifier = MockEpistemicVerifier;

// Full Z3 verification in release/CI
#[cfg(not(debug_assertions))]
type Verifier = Z3EpistemicVerifier;

fn verify_properties() {
    let verifier = Verifier::new();
    // Same API, different implementation
}
```

---

## Troubleshooting

### Z3 Not Found

```
Error: Z3 library not found
```

**Solution**: Install Z3 and set `Z3_SYS_Z3_HEADER` environment variable:

```bash
# Ubuntu/Debian
sudo apt install libz3-dev

# macOS
brew install z3

# Windows
# Download from https://github.com/Z3Prover/z3/releases
```

### Verification Timeout

```
Error: Verification timed out after 30s
```

**Solutions**:
1. Increase timeout: `Z3EpistemicVerifier::with_timeout(Duration::from_secs(120))`
2. Simplify the formula
3. Add more constraints to reduce search space
4. Use `MockEpistemicVerifier` for quick checks

### Unexpected Counterexample

If Z3 finds a counterexample you believe is invalid:

1. Check constraint completeness - are all invariants specified?
2. Check for floating-point edge cases (NaN, Inf)
3. Add explicit bounds on variables

```rust
// Add explicit bounds to avoid edge cases
z3.assert(&epsilon.ge(&ctx.real(0)));
z3.assert(&epsilon.le(&ctx.real(1)));
z3.assert(&value.ge(&ctx.real(-1e10)));
z3.assert(&value.le(&ctx.real(1e10)));
```

---

## See Also

- [Epistemic GPU Computing](./EPISTEMIC_GPU_COMPUTING.md)
- [Counterfactual Execution](./COUNTERFACTUAL_EXECUTION.md)
- [Z3 Documentation](https://microsoft.github.io/z3guide/)
- [SMT-LIB Standard](http://smtlib.cs.uiowa.edu/)

---

*"Trust, but verify. With Z3, verification is mathematical certainty."*
