# Demetrios Compiler Error Index

This document lists all error codes produced by the Demetrios compiler, organized by category.

Use `dc explain <CODE>` to get detailed information about any error code.

---

## Quick Reference

| Category | Prefix | Range | Description |
|----------|--------|-------|-------------|
| Lexer | L | L0001-L0xxx | Tokenization errors |
| Parser | P | P0001-P0xxx | Syntax errors |
| Resolve | R | R0001-R0xxx | Name resolution errors |
| Type | T | T0001-T0xxx | Type checking errors |
| Effect | F | F0001-F0xxx | Effect system errors |
| Ownership | O | O0001-O0xxx | Ownership/linearity errors |
| Pattern | M | M0001-M0xxx | Pattern matching errors |
| Module | I | I0001-I0xxx | Import/module errors |
| Codegen | C | C0001-C0xxx | Code generation errors |
| Internal | E | E0001-E0xxx | Internal compiler errors |

---

## Lexer Errors

### L0001: Invalid character

The source file contains a character that is not valid in Demetrios source code.

```d
let x = @invalid;  // '@' is not a valid character
```

### L0002: Unterminated string literal

A string literal was started but never closed with a matching quote.

```d
let s = "hello;  // missing closing quote
```

### L0003: Invalid number literal

A number literal has an invalid format.

```d
let x = 0x;      // hex literal with no digits
let y = 1.2.3;   // multiple decimal points
```

### L0004: Unterminated block comment

A block comment `/*` was started but never closed with `*/`.

```d
/* This comment
   never ends
```

---

## Parser Errors

### P0001: Unexpected token

The parser encountered a token that was not expected at this position.

```d
fn foo( {  // expected parameter or ')', found '{'
```

### P0002: Expected expression

An expression was expected but not found.

```d
let x = ;  // missing expression after '='
```

### P0003: Expected type annotation

A type annotation was expected but not found.

```d
fn foo(x) {}  // parameter 'x' needs a type annotation
```

### P0004: Missing semicolon

A semicolon was expected to terminate a statement.

```d
let x = 1
let y = 2  // missing ';' after first statement
```

### P0005: Mismatched brackets

Opening and closing brackets do not match.

```d
let arr = [1, 2, 3);  // '[' closed with ')'
```

### P0006: Invalid pattern

The pattern syntax is invalid.

```d
match x {
    1 + 2 => {}  // patterns cannot contain operators
}
```

---

## Name Resolution Errors

### R0001: Undefined variable

The variable has not been declared in this scope or any enclosing scope.

```d
fn foo() {
    println(x);  // 'x' is not defined
}
```

### R0002: Undefined type

The type name does not refer to any known type.

```d
fn foo(x: Undefined) {}  // type 'Undefined' does not exist
```

### R0003: Undefined function

No function with this name exists in scope.

```d
fn main() {
    unknown_function();  // function not found
}
```

### R0004: Duplicate definition

An item with this name has already been defined in this scope.

```d
let x = 1;
let x = 2;  // 'x' already defined
```

### R0005: Import not found

The specified module or item could not be found.

```d
use nonexistent::module;  // module does not exist
```

### R0006: Private item

The item exists but is not accessible from this location.

```d
use other_module::private_fn;  // 'private_fn' is not public
```

---

## Type Errors

### T0001: Type mismatch

The expected type does not match the actual type of the expression.

```d
fn foo() -> i32 {
    return true;  // expected 'i32', found 'bool'
}
```

### T0002: Cannot infer type

The type of this expression cannot be determined. Add a type annotation.

```d
let x = [];  // cannot infer element type of empty array
```

### T0003: Invalid binary operation

The binary operator cannot be applied to these types.

```d
let x = "hello" - 1;  // cannot subtract i32 from string
```

### T0004: Invalid unary operation

The unary operator cannot be applied to this type.

```d
let x = -"hello";  // cannot negate a string
```

### T0005: Not callable

This expression cannot be called as a function.

```d
let x = 5;
x();  // 'i32' is not callable
```

### T0006: Wrong number of arguments

The function was called with the wrong number of arguments.

```d
fn foo(a: i32, b: i32) {}
foo(1);  // expected 2 arguments, got 1
```

### T0007: Not indexable

This type does not support indexing.

```d
let x = 5;
let y = x[0];  // 'i32' cannot be indexed
```

### T0008: Field not found

The struct or type does not have a field with this name.

```d
struct Point { x: i32, y: i32 }
let p = Point { x: 0, y: 0 };
let z = p.z;  // 'Point' has no field 'z'
```

### T0009: Method not found

No method with this name exists for this type.

```d
let x = 5;
x.unknown();  // 'i32' has no method 'unknown'
```

### T0010: Infinite type

Type inference resulted in an infinite type, which is not allowed.

```d
fn foo(x) { foo(foo) }  // type of 'foo' would be infinite
```

### T0011: Unit constraint violated

A units-of-measure constraint was violated.

```d
let distance: f64<m> = 5.0<kg>;  // expected meters, got kilograms
```

### T0012: Refinement type violation

The value does not satisfy the refinement type's predicate.

```d
type Positive = i32 where x => x > 0;
let x: Positive = -5;  // -5 does not satisfy x > 0
```

---

## Effect Errors

### F0001: Unhandled effect

The function performs an effect that is not declared in its signature.

```d
fn foo() {  // missing 'with IO'
    println("hello");
}
```

**Fix:** Add the effect to the function signature:

```d
fn foo() with IO {
    println("hello");
}
```

### F0002: Effect not available

The required effect is not available in the current context.

```d
fn pure_fn() {
    io_fn();  // cannot call IO function from pure context
}
```

### F0003: Effect handler not found

No handler for this effect was found in scope.

```d
fn main() {
    perform MyEffect;  // no handler for 'MyEffect'
}
```

### F0004: Invalid effect handler

The effect handler is not valid for this effect.

```d
handle IO with {  // incorrect handler signature
    read => {}
}
```

---

## Ownership Errors

### O0001: Use of moved value

The value has been moved and can no longer be used.

```d
let x = vec![1, 2, 3];
let y = x;      // x moved here
println(x);     // error: x has been moved
```

### O0002: Cannot borrow as mutable

The value cannot be borrowed mutably because it is not declared as mutable.

```d
let x = 5;
increment(&!x);  // cannot borrow 'x' as mutable
```

**Fix:** Declare the variable as mutable:

```d
var x = 5;
increment(&!x);  // OK
```

### O0003: Cannot borrow while already borrowed

The value is already borrowed and cannot be borrowed again in this way.

```d
let r1 = &x;
let r2 = &!x;  // cannot borrow mutably while immutably borrowed
```

### O0004: Linear value not used

A linear value must be used exactly once, but it was not used.

```d
fn foo() {
    let handle: linear FileHandle = open("file.txt");
}  // error: 'handle' must be used
```

**Fix:** Use the value or explicitly consume it:

```d
fn foo() {
    let handle: linear FileHandle = open("file.txt");
    close(handle);  // properly consumed
}
```

### O0005: Linear value used multiple times

A linear value can only be used once, but it was used multiple times.

```d
fn foo(x: linear Resource) {
    use(x);
    use(x);  // error: 'x' already used
}
```

### O0006: Reference outlives value

The reference would outlive the value it refers to.

```d
fn foo() -> &i32 {
    let x = 5;
    return &x;  // 'x' does not live long enough
}
```

### O0007: Cannot copy linear type

Linear types cannot be implicitly copied.

```d
linear struct Unique { value: i32 }
let a = Unique { value: 1 };
let b = a;  // move, not copy
let c = a;  // error: 'a' already moved
```

---

## Pattern Matching Errors

### M0001: Non-exhaustive patterns

The match expression does not cover all possible cases.

```d
enum Color { Red, Green, Blue }
match color {
    Color::Red => {}
    Color::Green => {}
    // missing Color::Blue
}
```

**Fix:** Add the missing case or use a wildcard:

```d
match color {
    Color::Red => {}
    Color::Green => {}
    Color::Blue => {}
}
// Or:
match color {
    Color::Red => {}
    _ => {}  // catch-all
}
```

### M0002: Unreachable pattern

This pattern will never be matched because previous patterns cover all cases.

```d
match x {
    _ => {}
    1 => {}  // unreachable: '_' matches everything
}
```

### M0003: Invalid pattern for type

This pattern cannot be used with this type.

```d
let x: i32 = 5;
match x {
    Some(n) => {}  // 'i32' is not an Option
}
```

---

## Module/Import Errors

### I0001: Circular import

There is a circular dependency between modules.

```d
// a.d
use b;
// b.d
use a;  // circular dependency
```

### I0002: Module not found

The specified module could not be found.

```d
use nonexistent_module;
```

### I0003: Ambiguous import

The import is ambiguous because multiple items match.

```d
use module_a::*;
use module_b::*;
foo();  // 'foo' exists in both modules
```

**Fix:** Use explicit imports:

```d
use module_a::foo as foo_a;
use module_b::foo as foo_b;
```

---

## Code Generation Errors

### C0001: FFI type not supported

This type cannot be used in FFI declarations.

```d
extern "C" {
    fn foo(x: String);  // 'String' is not FFI-safe
}
```

**Fix:** Use FFI-safe types:

```d
extern "C" {
    fn foo(x: *const i8);  // C string pointer
}
```

### C0002: Invalid inline assembly

The inline assembly syntax or constraints are invalid.

```d
asm!("invalid instruction");
```

---

## Internal Errors

### E0001: Internal compiler error

An unexpected internal error occurred. This is a bug in the compiler.

**What to do:**

1. Please report this error at https://github.com/anthropics/demetrios/issues
2. Include the full error message and stack trace
3. Provide a minimal reproduction case if possible

---

## Epistemic Errors (E01xx-E03xx)

These errors relate to the epistemic type system for confidence and provenance tracking.

### E0100: Low Confidence

An epistemic value has confidence below the required threshold.

```d
let estimate: Knowledge[f64, 0.95] = uncertain_value;  // confidence < 0.95
```

### E0200: High Heterogeneity

Data sources have too much variation to combine reliably.

### E0300: Missing Provenance

An epistemic operation requires provenance information that is not available.

---

## Common Error Patterns and Solutions

### "expected X, found Y"

This usually indicates a type mismatch. Check:
- Function return types
- Variable assignments
- Function argument types

### "cannot borrow as mutable"

You're trying to mutate something immutable. Solutions:
- Use `var` instead of `let` for mutable variables
- Use `&!` for mutable references in Demetrios (not `&mut`)

### "missing effect annotation"

Your function performs an effect (IO, Alloc, etc.) but doesn't declare it:

```d
// Wrong:
fn read_file(path: string) -> string { ... }

// Right:
fn read_file(path: string) -> string with IO { ... }
```

---

## See Also

- [CLI Reference](CLI_REFERENCE.md) - `dc explain` command
- [Language Specification](../spec/LANGUAGE_SPECIFICATION.md)
- [LLM Programming Guide](LLM_PROGRAMMING_GUIDE.md)
