# Demetrios Macro System & Compile-Time Metaprogramming

## Overview

The Demetrios macro system provides state-of-the-art compile-time code generation with:

- **Declarative Macros** (`macro_rules!`) — Pattern-based code generation with hygiene
- **Procedural Macros** — Token stream manipulation for derive, attribute, and function-like macros
- **Compile-Time Function Execution (CTFE)** — Evaluate functions at compile time
- **Scientific Domain-Specific Macros** — Dimensional analysis, automatic differentiation

## Part A: Declarative Macros

### Basic Syntax

```d
macro_rules! vec {
    () => { Vec::new() };
    ($($x:expr),*) => {
        {
            let mut v = Vec::new();
            $(v.push($x);)*
            v
        }
    };
}
```

### Fragment Specifiers

| Specifier | Matches |
|-----------|---------|
| `ident` | Identifier |
| `expr` | Expression |
| `ty` | Type |
| `stmt` | Statement |
| `pat` | Pattern |
| `block` | Block `{ ... }` |
| `item` | Item (fn, struct, etc.) |
| `literal` | Literal value |
| `tt` | Token tree |
| `effect` | Effect annotation |
| `unit` | Unit of measure |

### Repetition

```d
macro_rules! repeat {
    ($($x:expr),*) => { /* zero or more */ };
    ($($x:expr),+) => { /* one or more */ };
    ($($x:expr),?) => { /* zero or one */ };
    ($($x:expr);* sep) => { /* with separator */ };
}
```

### Hygiene

Macro-generated identifiers are automatically scoped to prevent unintended capture:

```d
macro_rules! counter {
    () => {
        {
            let __counter = 0;  // Hygienically scoped
            __counter
        }
    };
}
```

## Part B: Procedural Macros

### Derive Macros

```d
#[derive(Debug, Clone, Serialize)]
pub struct Point {
    x: f64,
    y: f64,
}
```

Generates implementations of `Debug`, `Clone`, and `Serialize` traits.

### Attribute Macros

```d
#[effect(IO, Prob)]
fn simulate() {
    // Automatically adds effect annotations
}
```

### Function-like Macros

```d
let v = vec![1, 2, 3];
let m = matrix![[1, 2], [3, 4]];
```

## Part C: Compile-Time Evaluation

### Const Functions

```d
const fn factorial(n: i32) -> i32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

const FACT_5: i32 = factorial(5);  // Evaluated at compile time
```

### Static Assertions

```d
static_assert!(std::mem::size_of::<u64>() == 8, "u64 must be 8 bytes");
```

### Type-Level Computation

```d
const fn array_len<T, const N: usize>(_: &[T; N]) -> usize {
    N
}
```

## Part D: Scientific Macros

### Dimensional Analysis

```d
use units::*;

let dose: mg = 500.0;
let volume: mL = 10.0;
let concentration: mg/mL = dose / volume;  // Type-checked at compile time

// Unit mismatch caught at compile time:
// let invalid: mg = 10.0: mL;  // ERROR: dimension mismatch
```

### Automatic Differentiation

```d
use autodiff::*;

let f = |x: f64| x * x + 2.0 * x + 1.0;
let grad = gradient(f, 3.0);  // Computes derivative at compile time
```

### Linear Algebra DSL

```d
use linalg::*;

let a = matrix![[1, 2], [3, 4]];
let b = matrix![[5, 6], [7, 8]];
let c = a * b;  // Optimized matrix multiplication
```

## Architecture

### Token Trees

Macros operate on token trees, which preserve structure:

```
TokenTree::Token(identifier)
TokenTree::Delimited(Parenthesis, [tokens], span)
```

### Pattern Matching

Patterns match token trees with:
- Literal tokens
- Metavariables with fragment specifiers
- Repetitions with separators
- Nested groups

### Hygiene

Implemented using syntax contexts and marks:
- Each macro expansion gets a fresh context
- Identifiers carry their definition context
- Name resolution respects context boundaries

### CTFE Engine

Evaluates a subset of D at compile time:
- Arithmetic and logical operations
- Control flow (if/else)
- Function calls (const fn only)
- Array and struct construction

## Examples

### Example 1: Assert Macro

```d
macro_rules! assert {
    ($cond:expr) => {
        if !$cond {
            panic!("assertion failed");
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            panic!("assertion failed: {}", $msg);
        }
    };
}
```

### Example 2: Derive Debug

```d
#[derive(Debug)]
struct Point { x: i32, y: i32 }

// Generates:
// impl Debug for Point {
//     fn fmt(&self, f: &mut Formatter) -> Result {
//         f.debug_struct("Point")
//             .field("x", &self.x)
//             .field("y", &self.y)
//             .finish()
//     }
// }
```

### Example 3: Unit Checking

```d
const fn check_dose(dose: mg, volume: mL) -> mg/mL {
    dose / volume  // Type-checked: mg / mL = mg/mL ✓
}

// Type error caught at compile time:
// const fn invalid() -> mg {
//     10.0: mL  // ERROR: expected mg, found mL
// }
```

## Performance

- **Macro Expansion**: O(n) in token count
- **Pattern Matching**: O(m) in pattern complexity
- **CTFE**: Fuel-limited (1M steps default)
- **Hygiene**: O(1) context operations

## Limitations

- Macros cannot inspect types (only tokens)
- CTFE limited to pure functions
- No recursive macro definitions
- Procedural macros must be in separate crate

## Future Enhancements

- Macro debugging with expansion traces
- Procedural macro plugins
- Const generics with full type-level computation
- Macro-based DSL framework

## References

- "Macros That Work Together" (Flatt et al., 2012)
- "Macro-by-Example Revisited" (Kohlbecker et al., 1986)
- "Macros That Work" (Clinger & Rees, 1991)
- Rust's proc_macro crate design
