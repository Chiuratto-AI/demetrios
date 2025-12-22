# Demetrios Verified Syntax Reference

This document contains ONLY syntax that has been verified to work in the current compiler. Use this as the authoritative reference when writing D code.

---

## Variables

```d
// Immutable binding
let x = 5
let y: i32 = 10

// Mutable binding (two equivalent forms)
var count = 0                    // preferred style
let mut count = 0                // also works

// Assignment (only for mutable bindings)
count = count + 1

// Compile-time constant (at module level)
const PI: f64 = 3.14159265359
```

---

## Functions

```d
// Basic function
fn add(a: i32, b: i32) -> i32 {
    return a + b
}

// Function with explicit return
fn square(x: f64) -> f64 {
    return x * x
}

// Void-returning function
fn greet() {
    return ()
}

// Function with effects
fn read_file(path: string) -> string with IO {
    // ...
}

// Function with multiple effects
fn risky_io() with IO, Panic {
    // ...
}

// Generic function
fn identity<T>(x: T) -> T {
    return x
}
```

---

## Types

### Primitives
```d
i8, i16, i32, i64, i128          // signed integers
u8, u16, u32, u64, u128          // unsigned integers
f32, f64                          // floating point
bool                              // true or false
char                              // Unicode character
string                            // UTF-8 string
()                                // unit type
```

### Arrays and Slices
```d
[T; N]                            // fixed-size array
[T]                               // slice type
Vec<T>                            // dynamic array (generic)
```

### Tuples
```d
(T, U)                            // pair
(T, U, V)                         // triple
(T,)                              // single-element tuple
```

### References
```d
&T                                // shared reference (read-only)
&mut T                            // mutable reference (read-write)
```

**NOTE**: Documentation mentions `&!T` but this is NOT implemented yet. Use `&mut T`.

### Function Types
```d
T -> U                            // single-arg function type
(T, U) -> V                       // multi-arg function type
```

**NOTE**: `fn(T) -> U` syntax does NOT work. Use `T -> U` or `(T, U) -> V`.

---

## Structs

```d
struct Point {
    x: f64,
    y: f64,
}

// Create instance
let p = Point { x: 1.0, y: 2.0 }

// Access fields
let x_val = p.x

// Linear struct (must be used exactly once)
linear struct FileHandle {
    fd: i32,
}

// Struct with methods
impl Point {
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt_f64(dx * dx + dy * dy)
    }

    fn set_x(&mut self, x: f64) {
        self.x = x
    }
}
```

---

## Enums

```d
enum Color {
    Red,
    Green,
    Blue,
}

enum Option<T> {
    Some(T),
    None,
}

// Access variant
let c = Color::Red

// Match on enum
match opt {
    Option::Some(v) => v,
    Option::None => 0,
}
```

---

## Control Flow

### If-Else
```d
if condition {
    // ...
} else if other {
    // ...
} else {
    // ...
}

// If as expression
let max = if a > b { a } else { b }
```

### Match
```d
match value {
    0 => handle_zero(),
    1 | 2 => handle_one_or_two(),
    n if n > 10 => handle_big(n),
    _ => handle_default(),
}
```

### Loops
```d
// For with range
for i in 0..10 {
    // i goes 0, 1, 2, ..., 9
}

// Inclusive range
for i in 0..=10 {
    // i goes 0, 1, 2, ..., 10
}

// While loop
while condition {
    // ...
}

// Infinite loop
loop {
    if done {
        break
    }
}

// Break with value
let result = loop {
    if found {
        break value
    }
}
```

---

## Arrays and Slices

```d
// Fixed-size array
let arr: [i32; 5] = [1, 2, 3, 4, 5]

// Access element
let first = arr[0]

// Slice (subset of array)
let middle = arr[1..4]           // elements 1, 2, 3
let from_start = arr[..3]        // elements 0, 1, 2
let to_end = arr[3..]            // elements 3, 4
let all = arr[..]                // all elements

// Concatenation
let combined = arr1 ++ arr2      // requires Panic effect
```

---

## Tuples

```d
// Create tuple
let pair = (1, 2)

// Type annotation
let pair: (i64, i64) = (1, 2)

// Access elements
let first = pair.0
let second = pair.1
```

**NOTE**: Tuple destructuring `let (a, b) = tuple` is NOT working yet. Use `.0`, `.1` access.

---

## Units of Measure

```d
// Unit literals
let dose = 500.0_mg
let volume = 0.5_L
let concentration = 10.0_mg/L

// Unit type annotations
let mass: f64@kg = 75.0
let time: f64@h = 2.5

// Compound units (shorthand)
let clearance: L/h = 1.5
let velocity: m/s = 10.0

// Arithmetic preserves units
let result = dose / volume       // gives mg/L
```

---

## Effects

```d
// Declare function with effect
fn print_message(msg: string) with IO {
    // IO operations allowed
}

// Multiple effects
fn risky_io() with IO, Panic {
    // Can do IO and may panic
}

// Built-in effects:
// IO     - input/output operations
// Mut    - mutable state
// Alloc  - heap allocation
// Panic  - may panic/abort
// Async  - async operations
// GPU    - GPU operations
// Prob   - probabilistic operations
// Div    - may diverge
```

---

## Closures

```d
// With explicit types
let add_one = |x: i32| -> i32 { x + 1 }

// With type inference
let double = |x| x * 2

// Multi-line
let process = |data: &[f64]| -> f64 {
    var sum = 0.0
    for x in data {
        sum = sum + x
    }
    return sum
}
```

---

## Doc Comments

```d
/// This is a documentation comment
/// It documents the following item
fn documented_function() -> i32 {
    return 42
}

//! This is an inner doc comment
//! For documenting the containing module
```

---

## Reserved Keywords

These cannot be used as variable or function names:

**Language Keywords:**
`fn`, `let`, `var`, `mut`, `const`, `type`, `struct`, `enum`, `trait`, `impl`,
`if`, `else`, `match`, `for`, `while`, `loop`, `break`, `continue`, `return`,
`in`, `as`, `where`, `pub`, `self`, `Self`

**Effect Keywords:**
`effect`, `handler`, `handle`, `with`, `perform`, `resume`

**Type Keywords:**
`linear`, `affine`, `move`, `copy`, `drop`

**GPU Keywords:**
`kernel`, `tile`, `device`, `shared`, `gpu`, `async`, `await`, `spawn`

**Scientific Keywords:**
`ode`, `pde`, `causal`, `nodes`, `edges`, `equations`,
`state`, `params`, `domain`, `boundary`, `initial`

**Probabilistic Keywords:**
`sample`, `observe`, `infer`, `proof`

**Autodiff Keywords (CANNOT be used as identifiers):**
`grad`, `jacobian`, `hessian`, `dual`

**Epistemic Keywords:**
`Knowledge`, `Quantity`, `Tensor`,
`do`, `counterfactual`, `query`

**Provenance Keywords:**
`Valid`, `ValidUntil`, `ValidWhile`,
`Derived`, `Source`, `Computed`, `Literature`, `Measured`, `Input`

---

## Common Workarounds

### Integer Literals Default to i64
```d
// Problem: type mismatch
let x: i32 = 5            // Error: expected i32, found i64

// Solution: use cast
let x: i32 = 5 as i32     // Works
let x = 5_i32             // Also works (suffix)
```

### Tuple Destructuring Not Working
```d
// Problem: variables not bound
let (a, b) = (1, 2)       // Error: Unknown variable: a

// Solution: use field access
let pair = (1, 2)
let a = pair.0
let b = pair.1
```

### Using `grad` as Variable Name
```d
// Problem: grad is a keyword
let grad = 0.0            // Error: Expected pattern, found Grad

// Solution: use alternative name
let deriv = 0.0           // Works
let gradient = 0.0        // Works
```

---

## Built-in Functions

These are available without import:

```d
// Math (f64 versions)
sqrt_f64(x: f64) -> f64
exp_f64(x: f64) -> f64
log_f64(x: f64) -> f64
sin_f64(x: f64) -> f64
cos_f64(x: f64) -> f64
abs_f64(x: f64) -> f64
pow_f64(base: f64, exp: f64) -> f64

// Math (generic)
min(a, b)
max(a, b)
```

---

## External Functions (FFI)

```d
extern "C" {
    fn printf(format: *const i8, ...) -> i32;
    fn malloc(size: u64) -> *mut u8;
    fn free(ptr: *mut u8);
}
```

---

Last verified: 2024-12-22
Compiler version: dc (latest main branch)
