# Physical Units in Demetrios

## Overview

Demetrios provides first-class support for **physical units** with compile-time dimensional analysis. This means you can catch unit mismatches at compile time, preventing a whole class of bugs common in scientific computing.

Unlike dynamic type systems, Demetrios validates units during type checking, ensuring that operations like adding meters to seconds will fail before execution.

## Syntax

Units are specified using the `@` operator:

```d
let distance: f64@m = 5.0@m;
let time: f64@s = 2.5@s;
let mass: f64@kg = 10.0@kg;
```

## Base SI Units

Demetrios supports all seven SI base units:

| Unit | Symbol | Demetrios Syntax |
|------|--------|-----------------|
| Meter (length) | m | `@m` |
| Kilogram (mass) | kg | `@kg` |
| Second (time) | s | `@s` |
| Ampere (current) | A | `@A` |
| Kelvin (temperature) | K | `@K` |
| Mole (amount) | mol | `@mol` |
| Candela (luminosity) | cd | `@cd` |

## Derived SI Units

Demetrios also supports common derived units:

| Quantity | Symbol | Demetrios Syntax |
|----------|--------|-----------------|
| Force | N (Newton) | `@N` |
| Energy | J (Joule) | `@J` |
| Power | W (Watt) | `@W` |
| Pressure | Pa (Pascal) | `@Pa` |
| Frequency | Hz (Hertz) | `@Hz` |
| Electric Charge | C (Coulomb) | `@C` |
| Voltage | V (Volt) | `@V` |
| Resistance | Ω (Ohm) | `@Ohm` |

## Medical/Pharmaceutical Units

For healthcare and pharmaceutical applications:

| Unit | Symbol | Demetrios Syntax |
|------|--------|-----------------|
| Milligram | mg | `@mg` |
| Microgram | μg | `@ug` |
| Gram | g | `@g` |
| Milliliter | mL | `@ml` |
| Liter | L | `@l` |
| Microliter | μL | `@ul` |
| Minute | min | `@min` |
| Hour | h | `@h` |
| Day | day | `@day` |

### Concentration Units

For chemistry and pharmacology:

```d
let concentration: f64@(mg/ml) = 5.0@(mg/ml);
let molarity: f64@(mol/l) = 0.1@(mol/l);
let flow_rate: f64@(ml/min) = 10.0@(ml/min);
```

## Unit Arithmetic

### Multiplication

When you multiply two quantities with units, their units multiply:

```d
let length: f64@m = 3.0@m;
let width: f64@m = 4.0@m;
let area: f64@(m*m) = length * width;  // f64@m²
```

### Division

Dividing quantities produces unit ratios:

```d
let distance: f64@m = 100.0@m;
let time: f64@s = 5.0@s;
let velocity: f64@(m/s) = distance / time;  // 20 m/s
```

### Addition and Subtraction

You can only add or subtract quantities with **compatible units**:

```d
let height1: f64@m = 1.0@m;
let height2: f64@cm = 50.0@cm;
let total_height: f64@m = height1 + height2;  // ✓ Compatible units
```

But this fails at compile time:

```d
let distance: f64@m = 5.0@m;
let time: f64@s = 2.0@s;
let result: f64 = distance + time;  // ✗ ERROR: Cannot add m and s
```

### Powers

Taking powers of units:

```d
let side: f64@m = 5.0@m;
let area: f64@(m^2) = side * side;
let volume: f64@(m^3) = area * side;
```

## Unit Conversion

Demetrios automatically handles unit conversions when compatible:

```d
let height_m: f64@m = 1.0@m;
let height_cm: f64@cm = 100.0@cm;

// Automatic conversion for compatible units
let total: f64@m = height_m + height_cm;  // = 1.01 m
```

## Type-Level Verification

The type system ensures unit safety:

```d
fn calculate_kinetic_energy(mass: f64@kg, velocity: f64@(m/s)) -> f64@J {
    // E = (1/2) * m * v²
    return 0.5@dimensionless * mass * velocity * velocity;
}

// ✓ Type-checked:
let m: f64@kg = 2.0@kg;
let v: f64@(m/s) = 10.0@(m/s);
let energy: f64@J = calculate_kinetic_energy(m, v);

// ✗ Type error (would fail at compile time):
let wrong: f64@J = calculate_kinetic_energy(10.0@s, 5.0@m);
```

## Medical Example: Pharmacokinetics

This is where units really shine. Here's a realistic pharmacokinetic model:

```d
fn clearance_rate(
    drug_amount: f64@mg,
    concentration: f64@(mg/ml),
    volume_of_distribution: f64@ml
) -> f64@(mg/min) {
    // Rate = Amount / (Concentration * Volume)
    let rate: f64@(mg/min) = drug_amount / (concentration * volume_of_distribution);
    return rate;
}

// In daily use:
let dose: f64@mg = 500.0@mg;
let conc: f64@(mg/ml) = 2.5@(mg/ml);
let vd: f64@ml = 100.0@ml;

let clearance: f64@(mg/min) = clearance_rate(dose, conc, vd);
println("Clearance rate: {}", clearance);  // 2.0 mg/min
```

The compiler **prevents** unit errors like:

```d
// These would fail at compile time:
let wrong1: f64@(mg/min) = dose / (conc * dose);  // ✗ Wrong dimensions
let wrong2: f64@(mg/min) = clearance_rate(dose, dose, vd);  // ✗ Wrong unit for conc
```

## Dimensionless Quantities

For pure numbers without units:

```d
let dimensionless: f64@dimensionless = 5.0@dimensionless;
let ratio: f64 = 10.0 / 2.0;  // Pure number, no units
```

## Uncertainty with Units

You can combine units with uncertainty:

```d
let mass: uncertain<kg> = (5.0 ± 0.1)@kg;
let velocity: uncertain<(m/s)> = (10.0 ± 0.5)@(m/s);

// Uncertainty propagates through unit calculations
let momentum: uncertain<(kg*m/s)> = mass * velocity;
```

## Best Practices

### 1. Always annotate function signatures

```d
fn force_needed(mass: f64@kg, acceleration: f64@(m/s²)) -> f64@N {
    return mass * acceleration;
}
```

### 2. Use consistent units within functions

```d
fn metabolic_rate(mass: f64@kg, temperature: f64@K) -> f64@(cal/day) {
    // Keep all intermediate units consistent
    let bmr: f64@(cal/day) = 1.2@(cal/(kg*K)) * mass * (temperature - 273.15@K);
    return bmr;
}
```

### 3. Document non-obvious conversions

```d
// Convert half-life from hours to minutes for simulation
let half_life_hours: f64@h = 6.0@h;
let half_life_minutes: f64@min = half_life_hours * 60.0@dimensionless;
```

## Runtime Inspection

While units are compile-time constructs, you can still inspect them at runtime:

```d
let distance: f64@m = 5.0@m;
println("Value: {}", distance);  // Prints: 5
// Units are checked at compile time, not runtime
```

## Common Patterns

### Scientific Constants

```d
const GRAVITATIONAL_CONSTANT: f64@(m³/(kg*s²)) = 6.674e-11@(m³/(kg*s²));
const BOLTZMANN_CONSTANT: f64@(J/K) = 1.381e-23@(J/K);
const AVOGADRO_NUMBER: f64@(mol^-1) = 6.022e23@(mol^-1);
```

### Unit-safe vector operations

```d
let positions: [f64@m, 3] = [1.0@m, 2.0@m, 3.0@m];
let velocities: [f64@(m/s), 3] = [0.1@(m/s), 0.2@(m/s), 0.3@(m/s)];

// Element-wise operations preserve units
for i in 0..3 {
    let momentum: f64@(kg*m/s) = mass * velocities[i];
}
```

## Limitations and Future Work

- **Temperature differences**: Currently, `@K` is absolute temperature. Future versions will support relative temperature differences (`@ΔK`).
- **Logarithmic scales**: pH, decibels, and other logarithmic units are not yet supported.
- **Currency**: While not physical units, currency handling is planned.

## References

- [SI Brochure (BIPM)](https://www.bipm.org/en/publications/si-brochure)
- [Units in Medical Research (NIH)](https://www.nlm.nih.gov/pubs/techreports/lhncbc_tr_2009_001.pdf)
- [Dimensional Analysis in Physics](https://en.wikipedia.org/wiki/Dimensional_analysis)

## Troubleshooting

### "Incompatible units" error

This means you're trying to add/subtract quantities with different units:

```d
let x: f64@m = 5.0@m;
let y: f64@s = 3.0@s;
let z = x + y;  // ✗ ERROR: Cannot add m and s
```

**Solution**: Convert to compatible units first.

### "Unit mismatch in function argument"

```d
fn work(force: f64@N, distance: f64@m) -> f64@J { ... }

let f: f64@kg = 5.0@kg;  // Wrong unit!
let w: f64@J = work(f, 10.0@m);  // ✗ ERROR
```

**Solution**: Ensure argument units match the function signature.

## See Also

- [SCIENTIFIC_COMPUTING.md](SCIENTIFIC_COMPUTING.md) - Scientific primitives
- [AUTODIFF.md](AUTODIFF.md) - Automatic differentiation
- [LINEAR_ALGEBRA.md](LINEAR_ALGEBRA.md) - Tensor operations
