# Demetrios (D) Compiler — Day 3: First Working Pipeline

## Context

Day 1: Created scaffold structure
Day 2: Completed stub files, cargo build passes
Day 3: **Get a real D program through lexer → parser → AST → type check**

## Repository

```
/mnt/e/workspace/demetrios/
```

## Today's Mission

1. **Verify current state** - `cargo build` and `cargo test` pass
2. **Fix any broken imports/modules**
3. **Parse a minimal D program end-to-end**
4. **Print AST for debugging**
5. **Basic type inference working**

---

## PHASE 1: Health Check

```bash
cd /mnt/e/workspace/demetrios/compiler
cargo build 2>&1
cargo test 2>&1
```

If errors, fix them first. Common issues:
- Missing `mod` declarations in parent modules
- Circular dependencies
- Missing trait implementations

---

## PHASE 2: Integration Test

### 2.1 Create Test Program

Create `examples/minimal.d`:

```d
fn main() -> int {
    let x: int = 42
    return x
}
```

### 2.2 Create Integration Test

Create `tests/integration.rs`:

```rust
use demetrios::lexer::Lexer;
use demetrios::parser::Parser;

const MINIMAL: &str = r#"
fn main() -> int {
    let x: int = 42
    return x
}
"#;

#[test]
fn test_lex_minimal() {
    let mut lexer = Lexer::new(MINIMAL);
    let tokens: Vec<_> = lexer.collect();
    
    // Should have tokens: fn, main, (, ), ->, int, {, let, x, :, int, =, 42, return, x, }
    assert!(tokens.len() > 10, "Expected multiple tokens, got {}", tokens.len());
    
    // Check no errors
    for tok in &tokens {
        assert!(!matches!(tok.kind, demetrios::lexer::TokenKind::Error(_)));
    }
}

#[test]
fn test_parse_minimal() {
    let ast = demetrios::parser::parse(MINIMAL).expect("Parse failed");
    
    // Should have one item (the function)
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_function() {
    let src = "fn add(a: int, b: int) -> int { return a }";
    let ast = demetrios::parser::parse(src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_struct() {
    let src = "struct Point { x: f64, y: f64 }";
    let ast = demetrios::parser::parse(src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_linear_struct() {
    let src = "linear struct FileHandle { fd: int }";
    let ast = demetrios::parser::parse(src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_effect_annotation() {
    let src = "fn read_file(path: string) -> string with IO { return path }";
    let ast = demetrios::parser::parse(src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_unit_literal() {
    let src = "fn dose() -> f64 { let d: f64 = 500.0 return d }";
    let ast = demetrios::parser::parse(src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}
```

Run tests:
```bash
cargo test --test integration
```

---

## PHASE 3: AST Printer

### 3.1 Add Debug/Display to AST

Update `src/ast/mod.rs` - ensure all types derive Debug:

```rust
#[derive(Debug, Clone)]
pub struct Ast {
    pub items: Vec<Item>,
}

// ... ensure all AST nodes have #[derive(Debug, Clone)]
```

### 3.2 Create AST Printer

Create `src/ast/printer.rs`:

```rust
//! Pretty-printer for AST debugging

use super::*;
use std::fmt::Write;

pub struct AstPrinter {
    indent: usize,
    output: String,
}

impl AstPrinter {
    pub fn new() -> Self {
        Self {
            indent: 0,
            output: String::new(),
        }
    }

    pub fn print(ast: &Ast) -> String {
        let mut printer = Self::new();
        printer.print_ast(ast);
        printer.output
    }

    fn print_ast(&mut self, ast: &Ast) {
        for item in &ast.items {
            self.print_item(item);
            self.newline();
        }
    }

    fn print_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.print_function(f),
            Item::Struct(s) => self.print_struct(s),
            Item::Enum(e) => self.print_enum(e),
            Item::TypeAlias(t) => self.print_type_alias(t),
            Item::Const(c) => self.print_const(c),
            Item::Module(m) => self.print_module(m),
            Item::Import(i) => self.print_import(i),
            Item::Effect(e) => self.print_effect_def(e),
            Item::Kernel(k) => self.print_kernel(k),
            Item::Trait(t) => self.print_trait(t),
            Item::Impl(i) => self.print_impl(i),
            Item::Stmt(s) => self.print_stmt(s),
        }
    }

    fn print_function(&mut self, f: &FnDef) {
        self.write("fn ");
        self.write(&format!("#{}", f.id.0));
        self.write("(");
        for (i, param) in f.params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.print_param(param);
        }
        self.write(")");
        
        if let Some(ref ret) = f.return_type {
            self.write(" -> ");
            self.print_type(ret);
        }
        
        if !f.effects.is_empty() {
            self.write(" with ");
            for (i, eff) in f.effects.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                self.print_effect_ref(eff);
            }
        }
        
        if let Some(ref body) = f.body {
            self.write(" ");
            self.print_block(body);
        }
    }

    fn print_struct(&mut self, s: &StructDef) {
        if s.modifiers.linear {
            self.write("linear ");
        }
        if s.modifiers.affine {
            self.write("affine ");
        }
        self.write("struct ");
        self.write(&format!("#{}", s.id.0));
        self.write(" {");
        self.indent();
        for field in &s.fields {
            self.newline();
            self.print_field(field);
        }
        self.dedent();
        self.newline();
        self.write("}");
    }

    fn print_enum(&mut self, _e: &EnumDef) {
        self.write("enum { ... }");
    }

    fn print_type_alias(&mut self, _t: &TypeAlias) {
        self.write("type ... = ...");
    }

    fn print_const(&mut self, _c: &ConstDef) {
        self.write("const ...");
    }

    fn print_module(&mut self, _m: &ModuleDef) {
        self.write("module ...");
    }

    fn print_import(&mut self, _i: &ImportDef) {
        self.write("import ...");
    }

    fn print_effect_def(&mut self, _e: &EffectDef) {
        self.write("effect ...");
    }

    fn print_kernel(&mut self, _k: &KernelDef) {
        self.write("kernel fn ...");
    }

    fn print_trait(&mut self, _t: &TraitDef) {
        self.write("trait ...");
    }

    fn print_impl(&mut self, _i: &ImplDef) {
        self.write("impl ...");
    }

    fn print_param(&mut self, p: &Param) {
        self.write(&format!("#{}", p.id.0));
        self.write(": ");
        self.print_type(&p.ty);
    }

    fn print_field(&mut self, f: &Field) {
        self.write(&format!("#{}", f.id.0));
        self.write(": ");
        self.print_type(&f.ty);
    }

    fn print_type(&mut self, t: &Type) {
        match t {
            Type::Named(id) => self.write(&format!("#{}", id.0)),
            Type::Generic(id, args) => {
                self.write(&format!("#{}", id.0));
                self.write("<");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_type(arg);
                }
                self.write(">");
            }
            Type::Ref(kind, inner) => {
                match kind {
                    RefKind::Shared => self.write("&"),
                    RefKind::Exclusive => self.write("&!"),
                }
                self.print_type(inner);
            }
            Type::Own(inner) => {
                self.write("own ");
                self.print_type(inner);
            }
            Type::Linear(inner) => {
                self.write("linear ");
                self.print_type(inner);
            }
            Type::Affine(inner) => {
                self.write("affine ");
                self.print_type(inner);
            }
            Type::Tuple(types) => {
                self.write("(");
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_type(t);
                }
                self.write(")");
            }
            Type::Array(inner, size) => {
                self.write("[");
                self.print_type(inner);
                self.write(&format!("; {}]", size));
            }
            Type::Function(params, ret) => {
                self.write("fn(");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_type(p);
                }
                self.write(") -> ");
                self.print_type(ret);
            }
            Type::Unit => self.write("()"),
            Type::Infer => self.write("_"),
            Type::Error => self.write("!error"),
            Type::Refined(inner, pred) => {
                self.write("{ ");
                self.print_type(inner);
                self.write(" | ");
                self.write(&format!("{:?}", pred));
                self.write(" }");
            }
            Type::UnitAnnotated(inner, unit) => {
                self.print_type(inner);
                self.write(&format!("_{}", unit));
            }
        }
    }

    fn print_effect_ref(&mut self, e: &EffectRef) {
        match e {
            EffectRef::IO => self.write("IO"),
            EffectRef::Mut => self.write("Mut"),
            EffectRef::Alloc => self.write("Alloc"),
            EffectRef::Panic => self.write("Panic"),
            EffectRef::Async => self.write("Async"),
            EffectRef::GPU => self.write("GPU"),
            EffectRef::Prob => self.write("Prob"),
            EffectRef::Div => self.write("Div"),
            EffectRef::Named(n) => self.write(&format!("#{}", n.0)),
        }
    }

    fn print_block(&mut self, block: &Block) {
        self.write("{");
        self.indent();
        for stmt in &block.stmts {
            self.newline();
            self.print_stmt(stmt);
        }
        self.dedent();
        self.newline();
        self.write("}");
    }

    fn print_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => {
                self.write("let ");
                self.write(&format!("#{}", l.id.0));
                if let Some(ref ty) = l.ty {
                    self.write(": ");
                    self.print_type(ty);
                }
                if let Some(ref init) = l.init {
                    self.write(" = ");
                    self.print_expr(init);
                }
            }
            Stmt::Return(r) => {
                self.write("return");
                if let Some(ref val) = r.value {
                    self.write(" ");
                    self.print_expr(val);
                }
            }
            Stmt::Expr(e) => self.print_expr(&e.expr),
            Stmt::If(i) => {
                self.write("if ");
                self.print_expr(&i.condition);
                self.write(" ");
                self.print_block(&i.then_block);
                if let Some(ref else_block) = i.else_block {
                    self.write(" else ");
                    self.print_block(else_block);
                }
            }
            _ => self.write("/* stmt */"),
        }
    }

    fn print_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal(lit) => self.print_literal(lit),
            Expr::Path(p) => {
                for (i, seg) in p.segments.iter().enumerate() {
                    if i > 0 {
                        self.write("::");
                    }
                    self.write(&format!("#{}", seg.0));
                }
            }
            Expr::Binary(b) => {
                self.write("(");
                self.print_expr(&b.left);
                self.write(&format!(" {:?} ", b.op));
                self.print_expr(&b.right);
                self.write(")");
            }
            Expr::Unary(u) => {
                self.write(&format!("{:?}", u.op));
                self.print_expr(&u.operand);
            }
            Expr::Call(c) => {
                self.print_expr(&c.callee);
                self.write("(");
                for (i, arg) in c.args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_expr(arg);
                }
                self.write(")");
            }
            Expr::Block(b) => self.print_block(b),
            _ => self.write("/* expr */"),
        }
    }

    fn print_literal(&mut self, lit: &Literal) {
        match lit {
            Literal::Int(n) => self.write(&n.to_string()),
            Literal::Float(f) => self.write(&f.to_string()),
            Literal::Bool(b) => self.write(&b.to_string()),
            Literal::String(s) => self.write(&format!("{:?}", s)),
            Literal::Char(c) => self.write(&format!("{:?}", c)),
            Literal::Unit(val, unit) => self.write(&format!("{}_{}", val, unit)),
        }
    }

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn newline(&mut self) {
        self.output.push('\n');
        for _ in 0..self.indent {
            self.output.push_str("  ");
        }
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }
}

impl Default for AstPrinter {
    fn default() -> Self {
        Self::new()
    }
}
```

Update `src/ast/mod.rs`:
```rust
pub mod printer;
pub use printer::AstPrinter;
```

---

## PHASE 4: CLI Integration

### 4.1 Update main.rs for `check` command

Ensure `src/main.rs` has working `check` command:

```rust
fn cmd_check(file: &Path, show_ast: bool, show_types: bool) -> Result<()> {
    let source = std::fs::read_to_string(file)
        .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;
    
    // Parse
    let ast = demetrios::parser::parse(&source)?;
    
    if show_ast {
        println!("=== AST ===");
        println!("{}", demetrios::ast::AstPrinter::print(&ast));
    }
    
    // Type check
    let mut checker = demetrios::check::TypeChecker::new();
    let hir = checker.check_program(&ast)?;
    
    if show_types {
        println!("=== Types ===");
        // TODO: print inferred types
    }
    
    println!("✓ {} checked successfully", file.display());
    Ok(())
}
```

Add CLI flag:
```rust
#[derive(Parser)]
struct CheckArgs {
    /// Source file
    file: PathBuf,
    /// Show AST
    #[arg(long)]
    show_ast: bool,
    /// Show inferred types
    #[arg(long)]
    show_types: bool,
}
```

### 4.2 Test CLI

```bash
cargo run -- check examples/minimal.d --show-ast
```

Expected output:
```
=== AST ===
fn #0(#1: int) -> int {
  let #2: int = 42
  return #2
}

✓ examples/minimal.d checked successfully
```

---

## PHASE 5: Type Inference Basics

### 5.1 Implement Basic Type Checking

Update `src/check/mod.rs` to actually check expressions:

```rust
impl TypeChecker {
    fn infer_expr(&mut self, expr: &ast::Expr) -> Result<Type> {
        match expr {
            ast::Expr::Literal(lit) => self.infer_literal(lit),
            ast::Expr::Path(path) => self.infer_path(path),
            ast::Expr::Binary(bin) => self.infer_binary(bin),
            ast::Expr::Call(call) => self.infer_call(call),
            ast::Expr::Block(block) => self.infer_block(block),
            _ => Ok(Type::Unknown),
        }
    }

    fn infer_literal(&mut self, lit: &ast::Literal) -> Result<Type> {
        Ok(match lit {
            ast::Literal::Int(_) => Type::Int,
            ast::Literal::Float(_) => Type::F64,
            ast::Literal::Bool(_) => Type::Bool,
            ast::Literal::String(_) => Type::String,
            ast::Literal::Char(_) => Type::Char,
            ast::Literal::Unit(_, unit) => {
                // Look up unit type
                if let Some(unit_type) = self.units.lookup(unit) {
                    Type::WithUnit(Box::new(Type::F64), unit_type.clone())
                } else {
                    Type::F64
                }
            }
        })
    }

    fn infer_path(&mut self, path: &ast::PathExpr) -> Result<Type> {
        // Single segment = variable lookup
        if path.segments.len() == 1 {
            let name = &path.segments[0]; // NodeId - need to resolve
            // TODO: proper name resolution
            if let Some(binding) = self.env.lookup(&name.0.to_string()) {
                return Ok(binding.ty.clone());
            }
        }
        Ok(Type::Unknown)
    }

    fn infer_binary(&mut self, bin: &ast::BinaryExpr) -> Result<Type> {
        let left_ty = self.infer_expr(&bin.left)?;
        let right_ty = self.infer_expr(&bin.right)?;
        
        // Unify operand types
        let subst = crate::check::infer::unify(&left_ty, &right_ty)
            .map_err(|e| miette::miette!("Type error: {}", e))?;
        
        let result_ty = crate::check::infer::apply_subst(&subst, &left_ty);
        
        // Result type depends on operator
        Ok(match bin.op {
            ast::BinOp::Eq | ast::BinOp::Ne | ast::BinOp::Lt | 
            ast::BinOp::Le | ast::BinOp::Gt | ast::BinOp::Ge |
            ast::BinOp::And | ast::BinOp::Or => Type::Bool,
            _ => result_ty,
        })
    }

    fn infer_call(&mut self, _call: &ast::CallExpr) -> Result<Type> {
        // TODO: Look up function type and check args
        Ok(Type::Unknown)
    }

    fn infer_block(&mut self, block: &ast::Block) -> Result<Type> {
        self.env.push_scope();
        
        let mut last_ty = Type::Unit;
        for stmt in &block.stmts {
            last_ty = self.check_stmt(stmt)?;
        }
        
        self.env.pop_scope();
        Ok(last_ty)
    }

    fn check_stmt(&mut self, stmt: &ast::Stmt) -> Result<Type> {
        match stmt {
            ast::Stmt::Let(l) => {
                let init_ty = if let Some(ref init) = l.init {
                    self.infer_expr(init)?
                } else {
                    Type::Unknown
                };
                
                let declared_ty = if let Some(ref ty) = l.ty {
                    self.resolve_type(ty)?
                } else {
                    init_ty.clone()
                };
                
                // Unify declared and inferred
                if l.ty.is_some() && l.init.is_some() {
                    crate::check::infer::unify(&declared_ty, &init_ty)
                        .map_err(|e| miette::miette!("Type error in let: {}", e))?;
                }
                
                // Bind variable
                let name = l.id.0.to_string(); // TODO: proper name resolution
                self.env.bind(name, declared_ty, l.mutable, false);
                
                Ok(Type::Unit)
            }
            ast::Stmt::Return(r) => {
                if let Some(ref val) = r.value {
                    self.infer_expr(val)
                } else {
                    Ok(Type::Unit)
                }
            }
            ast::Stmt::Expr(e) => self.infer_expr(&e.expr),
            _ => Ok(Type::Unit),
        }
    }

    fn resolve_type(&mut self, ty: &ast::Type) -> Result<Type> {
        Ok(match ty {
            ast::Type::Named(id) => {
                // TODO: proper type resolution
                // For now, check built-in types
                match id.0 {
                    _ => Type::Named(crate::types::core::TypeId(id.0)),
                }
            }
            ast::Type::Unit => Type::Unit,
            ast::Type::Tuple(types) => {
                Type::Tuple(types.iter()
                    .map(|t| self.resolve_type(t))
                    .collect::<Result<Vec<_>>>()?)
            }
            ast::Type::Ref(kind, inner) => {
                let inner_ty = self.resolve_type(inner)?;
                Type::Ref(
                    match kind {
                        ast::RefKind::Shared => crate::types::core::RefKind::Shared,
                        ast::RefKind::Exclusive => crate::types::core::RefKind::Exclusive,
                    },
                    Box::new(inner_ty),
                )
            }
            ast::Type::Infer => self.fresh_tyvar(),
            _ => Type::Unknown,
        })
    }
}
```

---

## PHASE 6: Test End-to-End

### 6.1 Full Pipeline Test

```bash
# Parse and show AST
cargo run -- check examples/minimal.d --show-ast

# Should output clean AST and "checked successfully"
```

### 6.2 Test More Programs

Create `examples/arithmetic.d`:
```d
fn add(a: int, b: int) -> int {
    return a + b
}

fn main() -> int {
    let x: int = add(1, 2)
    return x
}
```

Create `examples/effects.d`:
```d
fn greet(name: string) -> string with IO {
    return name
}
```

Create `examples/units.d`:
```d
fn calculate_concentration() -> f64 {
    let dose: f64 = 500.0
    let volume: f64 = 10.0
    return dose / volume
}
```

Test all:
```bash
for f in examples/*.d; do
    echo "=== $f ==="
    cargo run -- check "$f" --show-ast
done
```

---

## Success Criteria

1. ✅ `cargo build` passes
2. ✅ `cargo test` passes (including integration tests)
3. ✅ `dc check examples/minimal.d` works
4. ✅ `--show-ast` prints readable AST
5. ✅ Basic type inference works (literals, variables, binary ops)
6. ✅ At least 3 example programs parse and check

---

## Troubleshooting

### "Module not found"
```rust
// In lib.rs, ensure all modules declared:
pub mod ast;
pub mod check;  // <-- missing?
pub mod codegen;
pub mod effects;
pub mod hir;
pub mod hlir;
pub mod lexer;
pub mod mlir;
pub mod parser;
pub mod types;
```

### "Type mismatch"
Check that AST types match what parser produces. Use `--show-ast` to debug.

### "Unresolved import"
Make sure `pub use` exports are correct in each `mod.rs`.

---

## Next Steps (Day 4)

- Name resolution (symbol table)
- Full bidirectional type checking
- Effect inference
- Better error messages with spans

---

**Day 3 Goal: Parse → AST → Type Check → Success**
