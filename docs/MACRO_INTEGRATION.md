# Macro System Integration Guide

## Overview

This guide explains how to integrate the macro system into the Demetrios compiler pipeline.

## Architecture

```
Source Code
    ↓
Lexer (lexer::lex)
    ↓
Parser (parser::parse)
    ↓
Macro Expansion (macro_system::expand)
    ↓
Type Checker (check::check)
    ↓
HIR Generation (hir::lower)
    ↓
Code Generation (codegen::generate)
```

## Integration Points

### 1. Parser Integration

The parser should recognize macro invocations:

```rust
// In parser/mod.rs
fn parse_macro_invocation(&mut self) -> Result<Expr, ParseError> {
    let name = self.expect_ident()?;
    self.expect(TokenKind::Bang)?;
    
    let input = match self.current().kind {
        TokenKind::LParen => self.parse_delimited(Delimiter::Parenthesis)?,
        TokenKind::LBracket => self.parse_delimited(Delimiter::Bracket)?,
        TokenKind::LBrace => self.parse_delimited(Delimiter::Brace)?,
        _ => return Err(ParseError::expected("macro argument")),
    };
    
    Ok(Expr::MacroInvocation { name, input })
}
```

### 2. Type Checker Integration

The type checker should expand macros before type checking:

```rust
// In check/mod.rs
pub fn check(ast: &Ast) -> Result<Hir, CompileError> {
    let mut macro_ctx = MacroContext::new();
    
    // Load macro definitions from stdlib
    load_stdlib_macros(&mut macro_ctx)?;
    
    // Expand macros in AST
    let expanded_ast = expand_macros(ast, &mut macro_ctx)?;
    
    // Type check expanded AST
    let mut checker = TypeChecker::new();
    checker.check_module(&expanded_ast)
}

fn expand_macros(ast: &Ast, ctx: &mut MacroContext) -> Result<Ast, CompileError> {
    // Recursively expand all macros in AST
    // ...
}
```

### 3. Macro Definition Handling

Macros are defined at module level:

```d
// In D source code
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

The parser should recognize `macro_rules!` and create `MacroDef` entries:

```rust
// In parser/mod.rs
fn parse_macro_def(&mut self) -> Result<MacroDef, ParseError> {
    self.expect_ident("macro_rules")?;
    self.expect(TokenKind::Bang)?;
    
    let name = self.expect_ident()?;
    self.expect(TokenKind::LBrace)?;
    
    let mut arms = Vec::new();
    while !self.check(TokenKind::RBrace) {
        let pattern = self.parse_macro_pattern()?;
        self.expect(TokenKind::FatArrow)?;
        let template = self.parse_macro_template()?;
        
        arms.push(MacroArm { pattern, template, guard: None });
        
        if !self.check(TokenKind::RBrace) {
            self.expect(TokenKind::Semi)?;
        }
    }
    
    self.expect(TokenKind::RBrace)?;
    
    Ok(MacroDef {
        name,
        arms,
        is_pub: false,
        is_exported: false,
        doc: None,
        span: Span::default(),
    })
}
```

### 4. Procedural Macro Loading

Procedural macros are loaded from compiled libraries:

```rust
// In macro_system/proc_macro.rs
pub fn load_proc_macros(registry: &mut ProcMacroRegistry, lib_path: &Path) 
    -> Result<(), Box<dyn std::error::Error>> 
{
    // Load compiled proc macro library
    let lib = libloading::Library::new(lib_path)?;
    
    unsafe {
        // Get macro registration function
        let register: libloading::Symbol<fn(&mut ProcMacroRegistry)> = 
            lib.get(b"register_macros")?;
        register(registry);
    }
    
    Ok(())
}
```

### 5. CTFE Integration

Const functions are evaluated at compile time:

```rust
// In check/mod.rs
fn evaluate_const_expr(expr: &Expr, ctx: &mut CtfeContext) 
    -> Result<ConstValue, CompileError> 
{
    // Convert Expr to HIR
    let hir_expr = lower_expr(expr)?;
    
    // Evaluate in CTFE context
    ctx.eval(&hir_expr)
        .map_err(|e| CompileError::CtfeError(e))
}
```

## Usage Examples

### Declarative Macro

```d
macro_rules! assert {
    ($cond:expr) => {
        if !$cond {
            panic!("assertion failed");
        }
    };
}

fn main() {
    assert!(1 + 1 == 2);
}
```

### Procedural Macro (Derive)

```d
#[derive(Debug, Clone)]
struct Point {
    x: f64,
    y: f64,
}
```

### CTFE

```d
const fn factorial(n: i32) -> i32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

const FACT_5: i32 = factorial(5);  // Evaluated at compile time
```

### Scientific Macros

```d
use units::*;

let dose: mg = 500.0;
let volume: mL = 10.0;
let concentration: mg/mL = dose / volume;  // Type-checked
```

## Error Handling

Macro errors should be reported with source locations:

```rust
// In diagnostic/mod.rs
pub fn report_macro_error(error: &MacroError, source: &SourceFile) {
    match error {
        MacroError::PatternMismatch { expected, found, span } => {
            eprintln!("error: pattern mismatch");
            eprintln!("  expected: {}", expected);
            eprintln!("  found: {}", found);
            eprintln!("  at: {}", source.format_span(*span));
        }
        MacroError::RecursionLimit { depth, span } => {
            eprintln!("error: macro recursion limit ({}) exceeded", depth);
            eprintln!("  at: {}", source.format_span(*span));
        }
        // ... other errors
    }
}
```

## Performance Considerations

1. **Caching**: Cache expanded macros to avoid re-expansion
2. **Fuel Limits**: Set reasonable CTFE fuel limits (default: 1M steps)
3. **Recursion Limits**: Prevent infinite macro recursion (default: 128 depth)
4. **Lazy Expansion**: Only expand macros that are actually used

## Testing

Test macro expansion with:

```bash
# Run macro system tests
cargo test --lib macro_system

# Test specific macro
cargo test --lib macro_system::tests::test_macro_expansion_simple

# Test with verbose output
cargo test --lib macro_system -- --nocapture
```

## Debugging

Enable macro expansion traces:

```rust
// In macro_system/mod.rs
pub struct MacroContext {
    // ...
    pub debug_traces: bool,
}

impl MacroContext {
    pub fn with_debug(mut self) -> Self {
        self.debug_traces = true;
        self
    }
}
```

## Future Enhancements

1. **Macro Plugins**: Load macros from external crates
2. **Macro Debugging**: Step through macro expansion
3. **Macro Profiling**: Measure macro expansion time
4. **Macro Caching**: Cache expanded macros across builds
5. **Incremental Expansion**: Only re-expand changed macros

## References

- `src/macro_system/mod.rs` — Module root
- `src/macro_system/token_tree.rs` — Token tree implementation
- `src/macro_system/pattern.rs` — Pattern matching
- `src/macro_system/declarative.rs` — Declarative macros
- `src/macro_system/proc_macro.rs` — Procedural macros
- `src/macro_system/ctfe.rs` — CTFE engine
- `docs/MACRO_SYSTEM.md` — User guide
- `docs/api/MACRO_API.md` — API reference
