# Demetrios Refinement Types

## Overview

Refinement types extend the base type system with logical predicates that are verified at compile time using an SMT solver (Z3). This enables:

- **Compile-time verification** of safety properties
- **Array bounds checking** without runtime overhead
- **Division-by-zero prevention** statically
- **Medical domain constraints** for safe pharmaceutical computing
- **Automatic inference** via Liquid Types

## Theory Background

### What are Refinement Types?

A refinement type `{ v: T | P }` consists of:
- A **base type** `T` (e.g., `i64`, `f64`)
- A **refinement variable** `v` representing values of the type
- A **predicate** `P` that must hold for all values

Examples:
```
{ x: int | x > 0 }           -- Positive integers
{ x: f64 | 0.0 <= x <= 1.0 } -- Probability
{ dose: mg | dose <= max }   -- Safe medication dose
```

### Subtyping Rule

A refinement `{v: T | P}` is a subtype of `{v: T | Q}` if and only if:

```
forall v. P(v) => Q(v)
```

This implication is checked by the Z3 SMT solver.

### Liquid Types

Demetrios uses the Liquid Types approach for automatic inference:
1. Define a set of **qualifier templates** (e.g., `v > 0`, `v < x`)
2. Infer refinements as conjunctions of these qualifiers
3. Use fixpoint iteration to find the strongest valid refinement

## Predicate Language

### Terms

Terms are expressions that can appear in predicates:

```rust
Term::Var("x")              // Variable reference
Term::Int(42)               // Integer constant
Term::Float(3.14)           // Float constant
Term::Bool(true)            // Boolean constant
Term::add(t1, t2)           // Addition: t1 + t2
Term::sub(t1, t2)           // Subtraction: t1 - t2
Term::mul(t1, t2)           // Multiplication: t1 * t2
Term::div(t1, t2)           // Division: t1 / t2
Term::len(t)                // Array length: len(t)
Term::field(t, "name")      // Field access: t.name
```

### Predicates

Predicates are logical formulas:

```rust
Predicate::True                      // Always true
Predicate::False                     // Always false
Predicate::eq(t1, t2)                // Equality: t1 = t2
Predicate::lt(t1, t2)                // Less than: t1 < t2
Predicate::le(t1, t2)                // Less or equal: t1 <= t2
Predicate::gt(t1, t2)                // Greater than: t1 > t2
Predicate::ge(t1, t2)                // Greater or equal: t1 >= t2
Predicate::ne(t1, t2)                // Not equal: t1 != t2
Predicate::and([p1, p2, ...])        // Conjunction
Predicate::or([p1, p2, ...])         // Disjunction
Predicate::implies(p, q)             // Implication: p => q
Predicate::not(p)                    // Negation: !p
```

### Refinement Types

```rust
// Trivial refinement (predicate = true)
RefinementType::trivial(Type::I64)

// Positive integers: { v | v > 0 }
RefinementType::positive(Type::I64)

// Non-negative: { v | v >= 0 }
RefinementType::non_negative(Type::I64)

// Bounded range: { v | lo <= v <= hi }
RefinementType::bounded(Type::F64, 0.0, 100.0)

// Custom refinement
RefinementType::refined(
    Type::F64,
    "dose",
    Predicate::and([
        Predicate::gt(Term::var("dose"), Term::float(0.0)),
        Predicate::le(Term::var("dose"), Term::float(1000.0)),
    ])
)
```

## Medical Domain Refinements

The `medical` module provides pre-built refinements for pharmacological computing:

### Dose Constraints

```rust
// Positive dose: dose > 0
medical::positive(Type::F64)

// Safe dose: 0 < dose <= max
medical::safe_dose(Type::F64, 1000.0)

// Adjustment factor: 0 < factor <= 1
medical::adjustment_factor(Type::F64)
```

### Physiological Parameters

```rust
// Valid creatinine clearance: 0 < crcl < 200 mL/min
medical::valid_crcl(Type::F64)

// Valid age: 0 <= age <= 150 years
medical::valid_age(Type::F64)

// Valid weight: 0 < weight <= 500 kg
medical::valid_weight(Type::F64)

// Valid serum creatinine: 0.1 <= scr <= 20 mg/dL
medical::valid_serum_creatinine(Type::F64)

// Therapeutic range: min <= conc <= max
medical::therapeutic_range(Type::F64, 10.0, 20.0)
```

### Vital Signs

```rust
// Valid heart rate: 20 <= hr <= 300 bpm
medical::valid_heart_rate(Type::F64)

// Valid systolic BP: 40 <= bp <= 300 mmHg
medical::valid_systolic_bp(Type::F64)

// Valid diastolic BP: 20 <= bp <= 200 mmHg
medical::valid_diastolic_bp(Type::F64)

// Valid temperature: 25 <= temp <= 45 C
medical::valid_temperature(Type::F64)
```

## Constraint Generation

The `ConstraintGenerator` collects verification constraints:

```rust
let mut cg = ConstraintGenerator::new();

// Add variable to environment
cg.push_binding("weight", medical::valid_weight(Type::F64));

// Add subtyping constraint
cg.add_subtype(&actual_type, &expected_type, span);

// Add bounds check: 0 <= index < length
cg.add_bounds_check(Term::var("i"), Term::var("len"), span);

// Add division check: divisor != 0
cg.add_division_check(Term::var("x"), span);

// Add safety constraint
cg.add_safety_constraint(
    Predicate::le(Term::var("dose"), Term::float(1000.0)),
    "dose must not exceed maximum",
    span,
);

// Get all constraints
let constraints = cg.into_constraints();
```

## Subtype Checking

The `SubtypeChecker` verifies refinement constraints:

```rust
let mut checker = SubtypeChecker::new();

// Add assumptions to environment
checker.assume("weight", medical::valid_weight(Type::F64));

// Check subtyping
checker.is_subtype(&sub_type, &super_type, span);

// Check preconditions
checker.check_precondition("calculate_dose", "weight", &arg_type, &param_type, span);

// Check postconditions
checker.check_postcondition("calculate_dose", &result_type, &return_type, span);

// Check bounds
checker.check_bounds(Term::var("i"), Term::var("len"), span);

// Check division safety
checker.check_division(Term::var("divisor"), span);

// Verify all constraints
let result = checker.verify();
if !result.is_valid() {
    for error in result.all_errors() {
        println!("Error: {}", error);
    }
}
```

## Z3 SMT Solver Integration

When compiled with the `smt` feature, constraints are verified using Z3:

```bash
# Build with Z3 support
cargo build --features smt

# Run tests with Z3
cargo test --features smt
```

Without Z3, constraints are checked using simple syntactic rules.

## Liquid Type Qualifiers

### Standard Qualifiers

```rust
// Basic comparisons
"Zero"      // v = 0
"Pos"       // v > 0
"NonNeg"    // v >= 0
"Neg"       // v < 0
"NonZero"   // v != 0

// Variable comparisons
"EqVar"     // v = x
"LtVar"     // v < x
"LeVar"     // v <= x
"GtVar"     // v > x
"GeVar"     // v >= x

// Arithmetic
"Sum"       // v = x + y
"Diff"      // v = x - y
"Prod"      // v = x * y

// Array bounds
"InBounds"  // 0 <= v < x
"IsLen"     // v = len(x)
```

### Medical Qualifiers

```rust
"SafeDose"        // 0 < dose <= max
"ValidConc"       // conc >= 0
"TherapeuticRange" // min <= conc <= max
"ValidCrCl"       // 0 < crcl < 200
"ValidAge"        // 0 <= age <= 150
"ValidWeight"     // 0 < weight <= 500
"AdjustFactor"    // 0 < factor <= 1
"Probability"     // 0 <= p <= 1
```

## Refinement Inference

The `RefinementInference` module automatically infers refinements:

```rust
let mut infer = RefinementInference::with_medical();

// Add bindings
infer.add_binding("x", Type::I64);
infer.add_refined_binding("weight", medical::valid_weight(Type::F64));

// Infer literals
let ty = infer.infer_literal(LiteralValue::Int(42));
// Result: { v | v = 42 }

// Infer binary operations
let result_ty = infer.infer_binary(
    BinaryOp::Add,
    &lhs_type,
    &rhs_type,
    Type::I64,
);

// Solve constraints
let result = infer.solve();
```

## Common Patterns

### Array Index Safety

```rust
let idx_type = patterns::array_index("i", Term::var("len"));
// { i | 0 <= i < len }
```

### Loop Counter

```rust
let counter_type = patterns::loop_counter("i", Term::var("n"));
// { i | 0 <= i <= n }
```

### Safe Division

```rust
let divisor_type = patterns::safe_divisor("x", Type::I64);
// { x | x != 0 }
```

### Dose Calculation

```rust
let dose_type = patterns::dose_result(1000.0);
// { dose | 0 < dose <= 1000 }
```

## Example: Safe Dose Calculation

```d
// Demetrios source code (proposed syntax)

type ValidWeight = { weight: f64<kg> | weight > 0.0 && weight <= 500.0 }
type ValidDosePerKg = { d: f64<mg/kg> | d > 0.0 && d <= 20.0 }
type SafeDose = { dose: f64<mg> | dose > 0.0 && dose <= 1000.0 }

fn calculate_dose(
    weight: ValidWeight,
    dose_per_kg: ValidDosePerKg
) -> SafeDose {
    // Compiler verifies:
    // weight > 0 && dose_per_kg > 0 => dose > 0
    // weight <= 500 && dose_per_kg <= 20 => dose <= 10000
    // Need to add bounds check for SafeDose <= 1000
    let dose = weight * dose_per_kg
    
    if dose > 1000.0_mg {
        return 1000.0_mg  // Cap at maximum
    }
    return dose
}
```

## Architecture

```
refinement/
├── mod.rs          # Module exports
├── predicate.rs    # Predicate language (Term, Predicate, RefinementType)
├── constraint.rs   # Constraint generation (ConstraintGenerator, Constraint)
├── solver.rs       # Z3 integration (Z3Solver, VerifyResult, SimpleChecker)
├── subtype.rs      # Subtype checking (SubtypeChecker, SubtypeResult)
├── qualifiers.rs   # Liquid type qualifiers (Qualifier, QualifierSet)
└── infer.rs        # Refinement inference (RefinementInference)
```

## Future Work

1. **Full Z3 Integration** - Complete Horn clause generation for fixpoint solving
2. **Parser Integration** - Parse refinement type annotations from source
3. **IDE Support** - Show inferred refinements on hover
4. **Abstract Refinements** - Higher-order refinement predicates
5. **Incremental Verification** - Cache and reuse verification results
