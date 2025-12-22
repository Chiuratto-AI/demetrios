# Demetrios Syntax: Documentation vs Implementation

This document identifies discrepancies between what is documented (in CLAUDE.md and LLM_PROGRAMMING_GUIDE.md) versus what the compiler actually implements.

---

## Critical Discrepancies

### 1. Exclusive References: `&!` vs `&mut`

**DOCUMENTED** (LLM_PROGRAMMING_GUIDE.md line 149):
```d
// Exclusive reference (mutable) - uses &! NOT &mut
fn increment(x: &!i32) {
    *x = *x + 1
}
```

**IMPLEMENTED** (parser/mod.rs lines 2424-2437):
```d
// Only &mut is implemented, &! is NOT parsed
fn increment(x: &mut i32) {
    *x = *x + 1
}
```

**STATUS**: ⚠️ `&mut` works, `&!` fails with "Expected type, found Bang"

**RECOMMENDATION**: Either:
- Implement `&!` parsing in the lexer/parser, OR
- Update documentation to use `&mut`

---

### 2. Tuple Destructuring in Let Bindings

**DOCUMENTED** (LLM_PROGRAMMING_GUIDE.md implies pattern matching works):
```d
let (a, b) = (10, 20)  // Tuple destructuring
```

**IMPLEMENTED** (parser parses it, but type checker doesn't resolve variables):
```
error: Unknown variable: a
error: Unknown variable: b
```

**STATUS**: ⚠️ Parses correctly but fails at type checking - variables not bound

**RECOMMENDATION**: Fix the type checker to bind variables from tuple patterns

---

### 3. Import/Use Module Resolution

**DOCUMENTED** (LLM_PROGRAMMING_GUIDE.md lines 218-224):
```d
import std::io
use std::math       // alias for import
```

**IMPLEMENTED**: Parsing works, but resolution fails unless module exists
```
Error: Import not found: `std::math`
```

**STATUS**: ⚠️ Standard library modules not available for import

**RECOMMENDATION**: Document which modules actually exist, or implement stdlib

---

### 4. Function Pointer / Function Types

**DOCUMENTED** (implied by patterns):
```d
type BinaryOp = fn(i32, i32) -> i32
```

**IMPLEMENTED**: Arrow function types work differently:
```d
// This works:
type BinaryOp = (i32, i32) -> i32

// But this fails:
type BinaryOp = fn(i32, i32) -> i32  // "Expected Semi, found Fn"
```

**STATUS**: ⚠️ `fn(...)` syntax not supported in types, use `(...) -> T` instead

---

## Working Features (Verified)

### Variables ✅
```d
let x = 5           // immutable - WORKS
var y = 10          // mutable - WORKS
let mut z = 15      // also mutable - WORKS
```

### Reference Types ✅
```d
fn read(x: &i32) -> i32 { *x }           // shared ref - WORKS
fn write(x: &mut i32) { *x = 0 }         // mutable ref - WORKS
```

### For Loops ✅
```d
for i in 0..10 { }  // range loop - WORKS
```

### Doc Comments ✅
```d
/// This is a doc comment
fn documented() { }  // WORKS
```

### Array Concatenation ✅
```d
let c = a ++ b      // WORKS (with Panic effect warning)
```

### Effects System ✅
```d
fn io_op() with IO { }  // WORKS
```

### Tuple Types ✅
```d
let pair: (i32, i32) = (1, 2)  // WORKS (with i64 literal inference)
let x = pair.0                  // WORKS
let y = pair.1                  // WORKS
```

---

## Type Inference Issues

### Integer Literals
Integer literals default to `i64`:
```d
let pair: (i32, i32) = (1, 2)
// Error: expected (i32, i32), found (i64, i64)

// Fix: use explicit casts
let pair: (i32, i32) = (1 as i32, 2 as i32)
```

---

## Reserved Keywords That Cannot Be Identifiers

The following are keywords and cannot be used as variable/function names:
- `grad` - autodiff keyword
- `jacobian`, `hessian` - autodiff keywords
- `sample`, `observe`, `infer` - probabilistic keywords
- `query`, `do`, `counterfactual` - causal inference keywords
- Many others in tokens.rs

**Workaround**: Use alternative names like `deriv` instead of `grad`

---

## Summary Table

| Feature | Documented | Implemented | Status |
|---------|-----------|-------------|--------|
| `let` immutable | ✅ | ✅ | Working |
| `var` mutable | ✅ | ✅ | Working |
| `let mut` mutable | ✅ | ✅ | Working |
| `&T` shared ref | ✅ | ✅ | Working |
| `&mut T` mutable ref | ❌ | ✅ | Works but docs say use `&!` |
| `&!T` exclusive ref | ✅ | ❌ | Documented but not implemented |
| Tuple types | ✅ | ✅ | Working |
| Tuple destructuring | ✅ | ⚠️ | Parses but vars not bound |
| `for i in range` | ✅ | ✅ | Working |
| `use/import` | ✅ | ⚠️ | Parses but module resolution limited |
| `fn(T)->U` types | ✅ | ❌ | Use `T -> U` instead |
| `(T,U)->V` types | ❌ | ✅ | Works, not documented |
| `++` concat | ✅ | ✅ | Working |
| Effects (`with`) | ✅ | ✅ | Working |
| `///` doc comments | ✅ | ✅ | Working |
| `grad` as identifier | ✅ | ❌ | Reserved keyword |

---

## Recommended Documentation Updates

1. **Change `&!` to `&mut`** in all examples until `&!` is implemented
2. **Remove tuple destructuring examples** until type checker is fixed
3. **Document that `fn(T)->U` is `T -> U`** in type position
4. **List reserved keywords** that cannot be used as identifiers
5. **Document integer literal default** is `i64` (require explicit casts for other int types)
6. **List available stdlib modules** or note that stdlib is not yet implemented

---

Last updated: $(date +%Y-%m-%d)
