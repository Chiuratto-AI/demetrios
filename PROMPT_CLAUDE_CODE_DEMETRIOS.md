# Demetrios (D) Compiler — Claude Code Implementation Prompt

## Mission

Complete the scaffold for the Demetrios (D) programming language compiler. The project structure and core files have been started. Your job is to:

1. Copy the scaffold to the user's iCloud folder
2. Complete all stub files with working implementations
3. Ensure `cargo build` and `cargo test` pass
4. Create initial documentation

## Project Location

**Source (scaffold created by Claude):**
```
/home/claude/demetrios/
```

**Destination (user's iCloud):**
```
/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios/
```

## Step 1: Copy Scaffold

```bash
# Create destination structure
mkdir -p "/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios"

# Copy entire scaffold
cp -r /home/claude/demetrios/* "/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios/"

# Verify
ls -la "/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios/compiler/src/"
```

## Step 2: Complete Stub Files

The following files need completion:

### 2.1 `src/types/effects.rs` (CREATE)

```rust
//! Algebraic effect system
//! 
//! D has full algebraic effects with handlers, inspired by Koka/Eff.

use super::core::{Effect, EffectSet, TypeVar};

/// Effect definition
#[derive(Debug, Clone)]
pub struct EffectDef {
    pub name: String,
    pub operations: Vec<EffectOperation>,
}

/// Effect operation signature
#[derive(Debug, Clone)]
pub struct EffectOperation {
    pub name: String,
    pub params: Vec<super::Type>,
    pub return_type: super::Type,
}

/// Effect handler
#[derive(Debug, Clone)]
pub struct EffectHandler {
    pub effect: Effect,
    pub cases: Vec<HandlerCase>,
}

/// Handler case
#[derive(Debug, Clone)]
pub struct HandlerCase {
    pub operation: String,
    pub params: Vec<String>,
    pub body: (), // TODO: HIR expression
    pub resume: bool,
}

/// Effect inference context
pub struct EffectInference {
    /// Effect variables
    vars: Vec<TypeVar>,
    /// Constraints: effect1 ⊆ effect2
    constraints: Vec<(EffectSet, EffectSet)>,
}

impl EffectInference {
    pub fn new() -> Self {
        Self {
            vars: Vec::new(),
            constraints: Vec::new(),
        }
    }
    
    /// Create fresh effect variable
    pub fn fresh_var(&mut self) -> TypeVar {
        let v = TypeVar(self.vars.len() as u32);
        self.vars.push(v);
        v
    }
    
    /// Add constraint: e1 ⊆ e2
    pub fn add_constraint(&mut self, e1: EffectSet, e2: EffectSet) {
        self.constraints.push((e1, e2));
    }
    
    /// Solve constraints
    pub fn solve(&self) -> Result<Vec<(TypeVar, EffectSet)>, String> {
        // TODO: Implement effect unification
        Ok(Vec::new())
    }
}

impl Default for EffectInference {
    fn default() -> Self {
        Self::new()
    }
}
```

### 2.2 `src/types/refinement.rs` (CREATE)

```rust
//! Refinement types with SMT integration

use super::core::Type;

/// Refinement predicate
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Boolean literal
    Bool(bool),
    /// Integer literal
    Int(i64),
    /// Float literal
    Float(f64),
    /// Variable reference
    Var(String),
    /// Comparison
    Compare(CompareOp, Box<Predicate>, Box<Predicate>),
    /// Arithmetic
    Arith(ArithOp, Box<Predicate>, Box<Predicate>),
    /// Logical
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
    /// Implication
    Implies(Box<Predicate>, Box<Predicate>),
}

#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Eq, Ne, Lt, Le, Gt, Ge,
}

#[derive(Debug, Clone, Copy)]
pub enum ArithOp {
    Add, Sub, Mul, Div, Mod,
}

/// Refined type
#[derive(Debug, Clone)]
pub struct RefinedType {
    pub base: Type,
    pub var_name: String,
    pub predicate: Option<Predicate>,
}

impl RefinedType {
    pub fn new(base: Type, var_name: &str, predicate: Option<Predicate>) -> Self {
        Self {
            base,
            var_name: var_name.to_string(),
            predicate,
        }
    }
    
    pub fn unrefined(base: Type) -> Self {
        Self {
            base,
            var_name: "_".to_string(),
            predicate: None,
        }
    }
}

/// Pre-defined medical refinements
pub mod medical {
    use super::*;
    
    /// Positive dose: dose > 0
    pub fn positive_dose() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "dose",
            Some(Predicate::Compare(
                CompareOp::Gt,
                Box::new(Predicate::Var("dose".into())),
                Box::new(Predicate::Float(0.0)),
            )),
        )
    }
    
    /// Valid creatinine clearance: 0 < crcl < 200
    pub fn valid_crcl() -> RefinedType {
        RefinedType::new(
            Type::F64,
            "crcl",
            Some(Predicate::And(
                Box::new(Predicate::Compare(
                    CompareOp::Gt,
                    Box::new(Predicate::Var("crcl".into())),
                    Box::new(Predicate::Float(0.0)),
                )),
                Box::new(Predicate::Compare(
                    CompareOp::Lt,
                    Box::new(Predicate::Var("crcl".into())),
                    Box::new(Predicate::Float(200.0)),
                )),
            )),
        )
    }
}
```

### 2.3 `src/types/units.rs` (CREATE)

```rust
//! Units of measure with compile-time dimensional analysis

use std::collections::HashMap;

/// Unit expression
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// Base units with exponents: {"m": 1, "s": -2} = m/s²
    pub dimensions: HashMap<String, i32>,
    /// Scale factor (for conversions)
    pub scale: f64,
}

impl Unit {
    pub fn dimensionless() -> Self {
        Self {
            dimensions: HashMap::new(),
            scale: 1.0,
        }
    }
    
    pub fn base(name: &str) -> Self {
        let mut dims = HashMap::new();
        dims.insert(name.to_string(), 1);
        Self { dimensions: dims, scale: 1.0 }
    }
    
    pub fn multiply(&self, other: &Unit) -> Unit {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) += power;
        }
        dims.retain(|_, v| *v != 0);
        Unit {
            dimensions: dims,
            scale: self.scale * other.scale,
        }
    }
    
    pub fn divide(&self, other: &Unit) -> Unit {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) -= power;
        }
        dims.retain(|_, v| *v != 0);
        Unit {
            dimensions: dims,
            scale: self.scale / other.scale,
        }
    }
    
    pub fn power(&self, n: i32) -> Unit {
        let dims: HashMap<_, _> = self.dimensions
            .iter()
            .map(|(k, v)| (k.clone(), v * n))
            .filter(|(_, v)| *v != 0)
            .collect();
        Unit {
            dimensions: dims,
            scale: self.scale.powi(n),
        }
    }
    
    pub fn is_compatible(&self, other: &Unit) -> bool {
        self.dimensions == other.dimensions
    }
}

/// Pre-defined SI units
pub mod si {
    use super::*;
    
    pub fn meter() -> Unit { Unit::base("m") }
    pub fn kilogram() -> Unit { Unit::base("kg") }
    pub fn second() -> Unit { Unit::base("s") }
    pub fn ampere() -> Unit { Unit::base("A") }
    pub fn kelvin() -> Unit { Unit::base("K") }
    pub fn mole() -> Unit { Unit::base("mol") }
    pub fn candela() -> Unit { Unit::base("cd") }
}

/// Pre-defined medical units
pub mod medical {
    use super::*;
    
    pub fn milligram() -> Unit {
        Unit { dimensions: [("kg".into(), 1)].into(), scale: 1e-6 }
    }
    
    pub fn milliliter() -> Unit {
        Unit { dimensions: [("m".into(), 3)].into(), scale: 1e-9 }
    }
    
    pub fn liter() -> Unit {
        Unit { dimensions: [("m".into(), 3)].into(), scale: 1e-3 }
    }
    
    pub fn hour() -> Unit {
        Unit { dimensions: [("s".into(), 1)].into(), scale: 3600.0 }
    }
    
    pub fn mg_per_ml() -> Unit {
        milligram().divide(&milliliter())
    }
    
    pub fn ml_per_min() -> Unit {
        let min = Unit { dimensions: [("s".into(), 1)].into(), scale: 60.0 };
        milliliter().divide(&min)
    }
}

/// Unit checker
pub struct UnitChecker {
    /// Known unit aliases
    aliases: HashMap<String, Unit>,
}

impl UnitChecker {
    pub fn new() -> Self {
        let mut aliases = HashMap::new();
        
        // Register common aliases
        aliases.insert("mg".into(), medical::milligram());
        aliases.insert("mL".into(), medical::milliliter());
        aliases.insert("L".into(), medical::liter());
        aliases.insert("h".into(), medical::hour());
        aliases.insert("hours".into(), medical::hour());
        
        Self { aliases }
    }
    
    pub fn lookup(&self, name: &str) -> Option<&Unit> {
        self.aliases.get(name)
    }
    
    pub fn check_compatible(&self, u1: &Unit, u2: &Unit) -> bool {
        u1.is_compatible(u2)
    }
}

impl Default for UnitChecker {
    fn default() -> Self {
        Self::new()
    }
}
```

### 2.4 `src/check/mod.rs` (CREATE)

```rust
//! Type checker for Demetrios

use crate::ast::*;
use crate::hir::*;
use crate::types::*;
use miette::Result;

/// Type check an AST and produce HIR
pub fn check(ast: &Ast) -> Result<Hir> {
    let mut checker = TypeChecker::new();
    checker.check_program(ast)
}

/// Type checker state
pub struct TypeChecker {
    /// Type environment
    env: TypeEnv,
    /// Effect inference
    effects: effects::EffectInference,
    /// Unit checker
    units: units::UnitChecker,
    /// Errors
    errors: Vec<TypeError>,
}

/// Type environment
#[derive(Default)]
pub struct TypeEnv {
    scopes: Vec<Scope>,
}

#[derive(Default)]
struct Scope {
    bindings: std::collections::HashMap<String, Type>,
}

/// Type error
#[derive(Debug)]
pub struct TypeError {
    pub message: String,
    pub span: crate::common::Span,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::default(),
            effects: effects::EffectInference::new(),
            units: units::UnitChecker::new(),
            errors: Vec::new(),
        }
    }
    
    pub fn check_program(&mut self, ast: &Ast) -> Result<Hir> {
        let mut items = Vec::new();
        
        for item in &ast.items {
            if let Some(hir_item) = self.check_item(item)? {
                items.push(hir_item);
            }
        }
        
        if !self.errors.is_empty() {
            return Err(miette::miette!("Type errors found"));
        }
        
        Ok(Hir { items })
    }
    
    fn check_item(&mut self, item: &Item) -> Result<Option<HirItem>> {
        match item {
            Item::Function(f) => {
                let hir_fn = self.check_function(f)?;
                Ok(Some(HirItem::Function(hir_fn)))
            }
            Item::Struct(s) => {
                let hir_struct = self.check_struct(s)?;
                Ok(Some(HirItem::Struct(hir_struct)))
            }
            // TODO: Other items
            _ => Ok(None),
        }
    }
    
    fn check_function(&mut self, f: &FnDef) -> Result<HirFn> {
        self.env.push_scope();
        
        // TODO: Proper implementation
        let body = HirBlock {
            stmts: Vec::new(),
            ty: HirType::Unit,
        };
        
        self.env.pop_scope();
        
        Ok(HirFn {
            id: f.id,
            name: "todo".to_string(), // TODO: resolve name
            ty: HirFnType {
                params: Vec::new(),
                return_type: Box::new(HirType::Unit),
                effects: Vec::new(),
            },
            body,
        })
    }
    
    fn check_struct(&mut self, s: &StructDef) -> Result<HirStruct> {
        Ok(HirStruct {
            id: s.id,
            name: "todo".to_string(),
            fields: Vec::new(),
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
        })
    }
}

impl TypeEnv {
    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }
    
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    
    fn bind(&mut self, name: String, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(name, ty);
        }
    }
    
    fn lookup(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.bindings.get(name) {
                return Some(ty);
            }
        }
        None
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
```

### 2.5 `src/effects/mod.rs` (CREATE)

```rust
//! Effect system implementation

pub use crate::types::effects::*;
```

### 2.6 `src/mlir/mod.rs` (CREATE)

```rust
//! MLIR integration (stub)
//!
//! D uses MLIR for optimization and code generation.
//! This module handles lowering from HLIR to MLIR dialects.

/// MLIR context (placeholder)
pub struct MlirContext {
    // Will hold actual MLIR context when melior is integrated
}

impl MlirContext {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MlirContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower HLIR to MLIR (stub)
pub fn lower_to_mlir(_hlir: &crate::hlir::HlirModule) -> Result<(), String> {
    // TODO: Implement when MLIR bindings are available
    Err("MLIR integration not yet implemented".to_string())
}
```

### 2.7 `src/codegen/mod.rs` (CREATE)

```rust
//! Code generation backends

pub mod llvm;
pub mod cranelift;
pub mod gpu;

/// Backend selection
pub enum Backend {
    LLVM,
    Cranelift,
    GPU,
}
```

### 2.8 `src/codegen/llvm.rs` (CREATE)

```rust
//! LLVM backend (stub)

use crate::hlir::HlirModule;

pub struct LlvmCodegen {
    // Will hold inkwell context
}

impl LlvmCodegen {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn compile(&self, _module: &HlirModule) -> Result<Vec<u8>, String> {
        Err("LLVM backend not yet implemented".to_string())
    }
}

impl Default for LlvmCodegen {
    fn default() -> Self {
        Self::new()
    }
}
```

### 2.9 `src/codegen/cranelift.rs` (CREATE)

```rust
//! Cranelift JIT backend (stub)

use crate::hlir::HlirModule;

pub struct CraneliftJit {
    // Will hold cranelift context
}

impl CraneliftJit {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn compile_and_run(&self, _module: &HlirModule) -> Result<i64, String> {
        Err("Cranelift JIT not yet implemented".to_string())
    }
}

impl Default for CraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}
```

### 2.10 `src/codegen/gpu.rs` (CREATE)

```rust
//! GPU code generation (stub)

use crate::hlir::HlirModule;

pub struct GpuCodegen {
    target: GpuTarget,
}

pub enum GpuTarget {
    CUDA,   // NVIDIA PTX
    SPIRV,  // Vulkan/OpenCL
}

impl GpuCodegen {
    pub fn new(target: GpuTarget) -> Self {
        Self { target }
    }
    
    pub fn compile_kernels(&self, _module: &HlirModule) -> Result<Vec<u8>, String> {
        Err("GPU codegen not yet implemented".to_string())
    }
}
```

## Step 3: Create Documentation

### 3.1 `README.md`

```markdown
# Demetrios (D)

A novel L0 systems + scientific programming language.

## Features

- **Novel Syntax**: Designed for scientific and medical computing
- **Full Algebraic Effects**: IO, Mut, Alloc, GPU, Prob with handlers
- **Linear/Affine Types**: Safe resource management
- **Units of Measure**: Compile-time dimensional analysis
- **Refinement Types**: SMT-backed constraint verification
- **GPU-Native**: First-class GPU memory and kernels

## Building

```bash
cd compiler
cargo build --release
```

## Usage

```bash
# Compile
dc compile program.d -o program

# Type check only
dc check program.d

# Run with JIT
dc run program.d

# REPL
dc repl
```

## Example

```d
module example

let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume

fn simulate(params: PKParams) -> Vec<f64> with Prob, Alloc {
    let eta = sample(Normal(0.0, 0.3))
    // ...
}

kernel fn matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = gpu.thread_id.x
    // ...
}
```

## License

MIT OR Apache-2.0
```

### 3.2 `docs/ARCHITECTURE.md`

```markdown
# Demetrios Compiler Architecture

## Pipeline

```
Source → Lexer → Parser → AST → Type Checker → HIR → HLIR → MLIR → LLVM/GPU
```

## Directory Structure

```
compiler/src/
├── main.rs          # CLI
├── lib.rs           # Library root
├── lexer/           # Tokenization
├── parser/          # Recursive descent parser
├── ast/             # Abstract syntax tree
├── hir/             # High-level IR (typed)
├── hlir/            # SSA-based IR
├── types/           # Type system
│   ├── core.rs      # Core types
│   ├── ownership.rs # Ownership rules
│   ├── effects.rs   # Effect system
│   ├── refinement.rs# Refinement types
│   └── units.rs     # Units of measure
├── check/           # Type checker
├── effects/         # Effect handlers
├── mlir/            # MLIR integration
└── codegen/         # Backends
    ├── llvm.rs
    ├── cranelift.rs
    └── gpu.rs
```
```

## Step 4: Verify Build

```bash
cd "/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios/compiler"
cargo build
cargo test
cargo clippy
```

## Step 5: Initialize Git

```bash
cd "/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Demetrios"
git init
git add .
git commit -m "Initial scaffold for Demetrios (D) programming language"
```

## Summary of Files to Create/Complete

| File | Status | Lines |
|------|--------|-------|
| `Cargo.toml` | ✅ Done | ~80 |
| `src/main.rs` | ✅ Done | ~150 |
| `src/lib.rs` | ✅ Done | ~120 |
| `src/lexer/mod.rs` | ✅ Done | ~100 |
| `src/lexer/tokens.rs` | ✅ Done | ~400 |
| `src/parser/mod.rs` | ✅ Done | ~800 |
| `src/ast/mod.rs` | ✅ Done | ~500 |
| `src/hir/mod.rs` | ✅ Done | ~150 |
| `src/hlir/mod.rs` | ✅ Done | ~200 |
| `src/types/mod.rs` | ✅ Done | ~20 |
| `src/types/core.rs` | ✅ Done | ~200 |
| `src/types/ownership.rs` | ✅ Done | ~80 |
| `src/types/effects.rs` | ❌ Create | ~100 |
| `src/types/refinement.rs` | ❌ Create | ~120 |
| `src/types/units.rs` | ❌ Create | ~150 |
| `src/check/mod.rs` | ❌ Create | ~150 |
| `src/effects/mod.rs` | ❌ Create | ~10 |
| `src/mlir/mod.rs` | ❌ Create | ~30 |
| `src/codegen/mod.rs` | ❌ Create | ~20 |
| `src/codegen/llvm.rs` | ❌ Create | ~30 |
| `src/codegen/cranelift.rs` | ❌ Create | ~30 |
| `src/codegen/gpu.rs` | ❌ Create | ~30 |
| `README.md` | ❌ Create | ~80 |
| `docs/ARCHITECTURE.md` | ❌ Create | ~50 |

**Total: ~3,600 lines**

## Success Criteria

1. `cargo build` completes without errors
2. `cargo test` passes (lexer tests)
3. `dc --version` prints version info
4. `dc check examples/hello.d` runs (even if incomplete)
5. Git repository initialized with initial commit

---

**You are building a novel programming language. This is Day 0. Make it compile.**
