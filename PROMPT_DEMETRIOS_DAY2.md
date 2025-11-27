# Demetrios (D) Compiler — Day 2: Complete Scaffold & First Build

## Context

You are continuing development of **Demetrios (D)**, a novel L0 systems + scientific programming language. Day 1 bootstrap created the core scaffold. Your mission today is to:

1. **Complete all stub files** so `cargo build` passes
2. **Implement the type checker foundation**
3. **Achieve first successful compilation** of a minimal D program

## Repository Location

```
/mnt/e/workspace/demetrios/
```

## Current State (Day 1 Complete)

```
compiler/src/
├── main.rs          ✅ CLI with 6 commands
├── lib.rs           ✅ Library root + Session/Span/Symbol
├── lexer/
│   ├── mod.rs       ✅ Lexer framework + tests
│   └── tokens.rs    ✅ ~100 token types (Logos)
├── parser/
│   └── mod.rs       ✅ Recursive descent + Pratt (~800 lines)
├── ast/
│   └── mod.rs       ✅ Complete AST (~500 lines)
├── hir/
│   └── mod.rs       ✅ Typed IR stubs
├── hlir/
│   └── mod.rs       ✅ SSA IR definitions
├── types/
│   ├── mod.rs       ✅ Module exports
│   ├── core.rs      ✅ Core type definitions
│   └── ownership.rs ✅ Ownership rules stub
├── check/           ❌ EMPTY - needs mod.rs
├── effects/         ❌ EMPTY - needs mod.rs  
├── mlir/            ❌ EMPTY - needs mod.rs
└── codegen/         ❌ EMPTY - needs mod.rs + backends
```

## PHASE 1: Make It Compile

### Task 1.1: Create Missing Module Files

First, check what exists and create missing files:

```bash
cd /mnt/e/workspace/demetrios/compiler
find src -name "*.rs" | head -50
cargo check 2>&1 | head -100
```

### Task 1.2: Create `src/types/effects.rs`

```rust
//! Algebraic effect system for Demetrios
//!
//! D has full algebraic effects with handlers (Koka/Eff-style).
//! Effects track computational side-effects at the type level.

use std::collections::HashSet;
use std::fmt;

/// Effect identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Effect {
    /// File, network, console I/O
    IO,
    /// Mutable state
    Mut,
    /// Heap allocation
    Alloc,
    /// Recoverable failure
    Panic,
    /// Asynchronous operations
    Async,
    /// GPU operations
    GPU,
    /// Probabilistic computation
    Prob,
    /// Potential divergence
    Div,
    /// User-defined effect
    Named(String),
    /// Effect variable (for polymorphism)
    Var(u32),
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::IO => write!(f, "IO"),
            Effect::Mut => write!(f, "Mut"),
            Effect::Alloc => write!(f, "Alloc"),
            Effect::Panic => write!(f, "Panic"),
            Effect::Async => write!(f, "Async"),
            Effect::GPU => write!(f, "GPU"),
            Effect::Prob => write!(f, "Prob"),
            Effect::Div => write!(f, "Div"),
            Effect::Named(name) => write!(f, "{}", name),
            Effect::Var(id) => write!(f, "?E{}", id),
        }
    }
}

/// Set of effects
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectSet {
    effects: HashSet<Effect>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self { effects: HashSet::new() }
    }

    pub fn pure() -> Self {
        Self::new()
    }

    pub fn single(effect: Effect) -> Self {
        let mut set = Self::new();
        set.effects.insert(effect);
        set
    }

    pub fn insert(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    pub fn union(&self, other: &EffectSet) -> EffectSet {
        EffectSet {
            effects: self.effects.union(&other.effects).cloned().collect(),
        }
    }

    pub fn is_subset(&self, other: &EffectSet) -> bool {
        self.effects.is_subset(&other.effects)
    }

    pub fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }

    pub fn contains(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.effects.is_empty() {
            write!(f, "Pure")
        } else {
            let effects: Vec<_> = self.effects.iter().map(|e| e.to_string()).collect();
            write!(f, "{}", effects.join(", "))
        }
    }
}

/// Effect operation definition
#[derive(Debug, Clone)]
pub struct EffectOp {
    pub name: String,
    pub params: Vec<super::Type>,
    pub return_type: super::Type,
}

/// Effect definition
#[derive(Debug, Clone)]
pub struct EffectDef {
    pub name: String,
    pub operations: Vec<EffectOp>,
}

/// Effect handler case
#[derive(Debug, Clone)]
pub struct HandlerCase {
    pub op_name: String,
    pub params: Vec<String>,
    pub resume_param: Option<String>,
    // Body will be HIR expression
}

/// Effect handler
#[derive(Debug, Clone)]
pub struct Handler {
    pub effect: Effect,
    pub cases: Vec<HandlerCase>,
    pub return_case: Option<String>, // Variable for return handler
}

/// Effect inference state
pub struct EffectInference {
    next_var: u32,
    /// Constraints: (subset, superset)
    constraints: Vec<(EffectSet, EffectSet)>,
}

impl EffectInference {
    pub fn new() -> Self {
        Self {
            next_var: 0,
            constraints: Vec::new(),
        }
    }

    /// Create fresh effect variable
    pub fn fresh_var(&mut self) -> Effect {
        let var = Effect::Var(self.next_var);
        self.next_var += 1;
        var
    }

    /// Add constraint: effects1 ⊆ effects2
    pub fn add_subset_constraint(&mut self, subset: EffectSet, superset: EffectSet) {
        self.constraints.push((subset, superset));
    }

    /// Solve constraints (simple implementation)
    pub fn solve(&self) -> Result<(), String> {
        for (subset, superset) in &self.constraints {
            // Skip constraints with variables for now
            let has_vars = subset.iter().any(|e| matches!(e, Effect::Var(_)))
                || superset.iter().any(|e| matches!(e, Effect::Var(_)));
            
            if !has_vars && !subset.is_subset(superset) {
                return Err(format!(
                    "Effect mismatch: {} is not a subset of {}",
                    subset, superset
                ));
            }
        }
        Ok(())
    }
}

impl Default for EffectInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_set_operations() {
        let mut set1 = EffectSet::new();
        set1.insert(Effect::IO);
        set1.insert(Effect::Mut);

        let mut set2 = EffectSet::new();
        set2.insert(Effect::IO);

        assert!(set2.is_subset(&set1));
        assert!(!set1.is_subset(&set2));
    }

    #[test]
    fn test_pure_effects() {
        let pure = EffectSet::pure();
        assert!(pure.is_pure());
        
        let io = EffectSet::single(Effect::IO);
        assert!(!io.is_pure());
    }
}
```

### Task 1.3: Create `src/types/units.rs`

```rust
//! Units of measure with compile-time dimensional analysis
//!
//! D supports units natively: `let dose: mg = 500.0`

use std::collections::HashMap;
use std::fmt;

/// Unit type representing dimensional analysis
#[derive(Debug, Clone, PartialEq)]
pub struct UnitType {
    /// Base dimensions with exponents: {"m": 1, "s": -2} = m·s⁻²
    pub dimensions: HashMap<String, i32>,
    /// Scale factor relative to SI base
    pub scale: f64,
}

impl UnitType {
    /// Dimensionless unit (scalar)
    pub fn dimensionless() -> Self {
        Self {
            dimensions: HashMap::new(),
            scale: 1.0,
        }
    }

    /// Base SI unit
    pub fn base(name: &str) -> Self {
        let mut dims = HashMap::new();
        dims.insert(name.to_string(), 1);
        Self { dimensions: dims, scale: 1.0 }
    }

    /// Create unit with scale
    pub fn scaled(name: &str, scale: f64) -> Self {
        let mut dims = HashMap::new();
        dims.insert(name.to_string(), 1);
        Self { dimensions: dims, scale }
    }

    /// Multiply two units
    pub fn multiply(&self, other: &UnitType) -> UnitType {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) += power;
        }
        // Remove zero exponents
        dims.retain(|_, v| *v != 0);
        UnitType {
            dimensions: dims,
            scale: self.scale * other.scale,
        }
    }

    /// Divide two units
    pub fn divide(&self, other: &UnitType) -> UnitType {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) -= power;
        }
        dims.retain(|_, v| *v != 0);
        UnitType {
            dimensions: dims,
            scale: self.scale / other.scale,
        }
    }

    /// Raise to integer power
    pub fn power(&self, n: i32) -> UnitType {
        let dims: HashMap<_, _> = self.dimensions
            .iter()
            .map(|(k, v)| (k.clone(), v * n))
            .filter(|(_, v)| *v != 0)
            .collect();
        UnitType {
            dimensions: dims,
            scale: self.scale.powi(n),
        }
    }

    /// Check if units are dimensionally compatible
    pub fn is_compatible(&self, other: &UnitType) -> bool {
        self.dimensions == other.dimensions
    }

    /// Check if dimensionless
    pub fn is_dimensionless(&self) -> bool {
        self.dimensions.is_empty()
    }
}

impl fmt::Display for UnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dimensions.is_empty() {
            return write!(f, "1");
        }
        
        let mut parts: Vec<String> = Vec::new();
        for (unit, power) in &self.dimensions {
            if *power == 1 {
                parts.push(unit.clone());
            } else {
                parts.push(format!("{}^{}", unit, power));
            }
        }
        write!(f, "{}", parts.join("·"))
    }
}

/// Pre-defined SI base units
pub mod si {
    use super::*;

    pub fn meter() -> UnitType { UnitType::base("m") }
    pub fn kilogram() -> UnitType { UnitType::base("kg") }
    pub fn second() -> UnitType { UnitType::base("s") }
    pub fn ampere() -> UnitType { UnitType::base("A") }
    pub fn kelvin() -> UnitType { UnitType::base("K") }
    pub fn mole() -> UnitType { UnitType::base("mol") }
    pub fn candela() -> UnitType { UnitType::base("cd") }
}

/// Pre-defined medical/pharmacological units
pub mod medical {
    use super::*;

    pub fn milligram() -> UnitType {
        UnitType::scaled("kg", 1e-6)
    }

    pub fn microgram() -> UnitType {
        UnitType::scaled("kg", 1e-9)
    }

    pub fn milliliter() -> UnitType {
        UnitType {
            dimensions: [("m".into(), 3)].into_iter().collect(),
            scale: 1e-6,
        }
    }

    pub fn liter() -> UnitType {
        UnitType {
            dimensions: [("m".into(), 3)].into_iter().collect(),
            scale: 1e-3,
        }
    }

    pub fn hour() -> UnitType {
        UnitType::scaled("s", 3600.0)
    }

    pub fn minute() -> UnitType {
        UnitType::scaled("s", 60.0)
    }

    /// mg/mL concentration
    pub fn mg_per_ml() -> UnitType {
        milligram().divide(&milliliter())
    }

    /// mL/min (e.g., creatinine clearance)
    pub fn ml_per_min() -> UnitType {
        milliliter().divide(&minute())
    }

    /// mg/kg/day dosing
    pub fn mg_per_kg_per_day() -> UnitType {
        let kg = UnitType::base("kg");
        let day = UnitType::scaled("s", 86400.0);
        milligram().divide(&kg).divide(&day)
    }
}

/// Unit registry for lookup
pub struct UnitRegistry {
    units: HashMap<String, UnitType>,
}

impl UnitRegistry {
    pub fn new() -> Self {
        let mut units = HashMap::new();
        
        // SI base
        units.insert("m".into(), si::meter());
        units.insert("kg".into(), si::kilogram());
        units.insert("s".into(), si::second());
        units.insert("A".into(), si::ampere());
        units.insert("K".into(), si::kelvin());
        units.insert("mol".into(), si::mole());
        units.insert("cd".into(), si::candela());
        
        // Medical
        units.insert("mg".into(), medical::milligram());
        units.insert("mcg".into(), medical::microgram());
        units.insert("μg".into(), medical::microgram());
        units.insert("mL".into(), medical::milliliter());
        units.insert("L".into(), medical::liter());
        units.insert("h".into(), medical::hour());
        units.insert("hour".into(), medical::hour());
        units.insert("hours".into(), medical::hour());
        units.insert("min".into(), medical::minute());
        
        Self { units }
    }

    pub fn lookup(&self, name: &str) -> Option<&UnitType> {
        self.units.get(name)
    }

    pub fn register(&mut self, name: &str, unit: UnitType) {
        self.units.insert(name.to_string(), unit);
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_multiplication() {
        let m = si::meter();
        let s = si::second();
        let velocity = m.divide(&s); // m/s
        
        assert_eq!(velocity.dimensions.get("m"), Some(&1));
        assert_eq!(velocity.dimensions.get("s"), Some(&-1));
    }

    #[test]
    fn test_concentration_units() {
        let conc = medical::mg_per_ml();
        // mg/mL = kg·m⁻³ (with scaling)
        assert!(conc.dimensions.contains_key("kg"));
        assert!(conc.dimensions.contains_key("m"));
    }

    #[test]
    fn test_unit_compatibility() {
        let mg = medical::milligram();
        let kg = si::kilogram();
        
        // mg and kg are both mass, compatible
        assert!(mg.is_compatible(&kg));
    }
}
```

### Task 1.4: Create `src/types/refinement.rs`

```rust
//! Refinement types with predicate constraints
//!
//! Example: `type SafeDose = { dose: mg | dose > 0.0 && dose <= max_dose }`

use std::fmt;

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
    /// Comparison: x < y, x == y, etc.
    Compare(CompareOp, Box<Predicate>, Box<Predicate>),
    /// Arithmetic: x + y, x * y, etc.
    Arith(ArithOp, Box<Predicate>, Box<Predicate>),
    /// Logical AND
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR
    Or(Box<Predicate>, Box<Predicate>),
    /// Logical NOT
    Not(Box<Predicate>),
    /// Implication: p => q
    Implies(Box<Predicate>, Box<Predicate>),
    /// Function application
    App(String, Vec<Predicate>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,  // ==
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::Bool(b) => write!(f, "{}", b),
            Predicate::Int(n) => write!(f, "{}", n),
            Predicate::Float(n) => write!(f, "{}", n),
            Predicate::Var(v) => write!(f, "{}", v),
            Predicate::Compare(op, l, r) => {
                let op_str = match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ne => "!=",
                    CompareOp::Lt => "<",
                    CompareOp::Le => "<=",
                    CompareOp::Gt => ">",
                    CompareOp::Ge => ">=",
                };
                write!(f, "({} {} {})", l, op_str, r)
            }
            Predicate::Arith(op, l, r) => {
                let op_str = match op {
                    ArithOp::Add => "+",
                    ArithOp::Sub => "-",
                    ArithOp::Mul => "*",
                    ArithOp::Div => "/",
                    ArithOp::Mod => "%",
                };
                write!(f, "({} {} {})", l, op_str, r)
            }
            Predicate::And(l, r) => write!(f, "({} && {})", l, r),
            Predicate::Or(l, r) => write!(f, "({} || {})", l, r),
            Predicate::Not(p) => write!(f, "!{}", p),
            Predicate::Implies(l, r) => write!(f, "({} => {})", l, r),
            Predicate::App(name, args) => {
                let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
        }
    }
}

impl Predicate {
    /// Create x > 0
    pub fn positive(var: &str) -> Self {
        Predicate::Compare(
            CompareOp::Gt,
            Box::new(Predicate::Var(var.into())),
            Box::new(Predicate::Float(0.0)),
        )
    }

    /// Create x >= lo && x <= hi
    pub fn in_range(var: &str, lo: f64, hi: f64) -> Self {
        Predicate::And(
            Box::new(Predicate::Compare(
                CompareOp::Ge,
                Box::new(Predicate::Var(var.into())),
                Box::new(Predicate::Float(lo)),
            )),
            Box::new(Predicate::Compare(
                CompareOp::Le,
                Box::new(Predicate::Var(var.into())),
                Box::new(Predicate::Float(hi)),
            )),
        )
    }

    /// Substitute variable with value
    pub fn substitute(&self, var: &str, value: &Predicate) -> Predicate {
        match self {
            Predicate::Var(v) if v == var => value.clone(),
            Predicate::Compare(op, l, r) => Predicate::Compare(
                *op,
                Box::new(l.substitute(var, value)),
                Box::new(r.substitute(var, value)),
            ),
            Predicate::Arith(op, l, r) => Predicate::Arith(
                *op,
                Box::new(l.substitute(var, value)),
                Box::new(r.substitute(var, value)),
            ),
            Predicate::And(l, r) => Predicate::And(
                Box::new(l.substitute(var, value)),
                Box::new(r.substitute(var, value)),
            ),
            Predicate::Or(l, r) => Predicate::Or(
                Box::new(l.substitute(var, value)),
                Box::new(r.substitute(var, value)),
            ),
            Predicate::Not(p) => Predicate::Not(Box::new(p.substitute(var, value))),
            Predicate::Implies(l, r) => Predicate::Implies(
                Box::new(l.substitute(var, value)),
                Box::new(r.substitute(var, value)),
            ),
            Predicate::App(name, args) => Predicate::App(
                name.clone(),
                args.iter().map(|a| a.substitute(var, value)).collect(),
            ),
            other => other.clone(),
        }
    }

    /// Simple evaluation (no variables)
    pub fn eval_const(&self) -> Option<bool> {
        match self {
            Predicate::Bool(b) => Some(*b),
            Predicate::Compare(op, l, r) => {
                let l_val = l.eval_float()?;
                let r_val = r.eval_float()?;
                Some(match op {
                    CompareOp::Eq => (l_val - r_val).abs() < f64::EPSILON,
                    CompareOp::Ne => (l_val - r_val).abs() >= f64::EPSILON,
                    CompareOp::Lt => l_val < r_val,
                    CompareOp::Le => l_val <= r_val,
                    CompareOp::Gt => l_val > r_val,
                    CompareOp::Ge => l_val >= r_val,
                })
            }
            Predicate::And(l, r) => Some(l.eval_const()? && r.eval_const()?),
            Predicate::Or(l, r) => Some(l.eval_const()? || r.eval_const()?),
            Predicate::Not(p) => Some(!p.eval_const()?),
            _ => None,
        }
    }

    fn eval_float(&self) -> Option<f64> {
        match self {
            Predicate::Int(n) => Some(*n as f64),
            Predicate::Float(f) => Some(*f),
            Predicate::Arith(op, l, r) => {
                let l_val = l.eval_float()?;
                let r_val = r.eval_float()?;
                Some(match op {
                    ArithOp::Add => l_val + r_val,
                    ArithOp::Sub => l_val - r_val,
                    ArithOp::Mul => l_val * r_val,
                    ArithOp::Div => l_val / r_val,
                    ArithOp::Mod => l_val % r_val,
                })
            }
            _ => None,
        }
    }
}

/// Pre-defined medical refinements
pub mod medical {
    use super::*;

    /// Positive dose: dose > 0
    pub fn positive_dose() -> Predicate {
        Predicate::positive("dose")
    }

    /// Safe dose range: 0 < dose <= max
    pub fn safe_dose(max: f64) -> Predicate {
        Predicate::And(
            Box::new(Predicate::positive("dose")),
            Box::new(Predicate::Compare(
                CompareOp::Le,
                Box::new(Predicate::Var("dose".into())),
                Box::new(Predicate::Float(max)),
            )),
        )
    }

    /// Valid creatinine clearance: 0 < crcl < 200
    pub fn valid_crcl() -> Predicate {
        Predicate::And(
            Box::new(Predicate::positive("crcl")),
            Box::new(Predicate::Compare(
                CompareOp::Lt,
                Box::new(Predicate::Var("crcl".into())),
                Box::new(Predicate::Float(200.0)),
            )),
        )
    }

    /// Valid age: 0 < age < 150
    pub fn valid_age() -> Predicate {
        Predicate::in_range("age", 0.0, 150.0)
    }

    /// Valid weight: 0 < weight < 500 kg
    pub fn valid_weight() -> Predicate {
        Predicate::in_range("weight", 0.0, 500.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_display() {
        let p = Predicate::positive("x");
        assert_eq!(p.to_string(), "(x > 0)");
    }

    #[test]
    fn test_const_eval() {
        let p = Predicate::Compare(
            CompareOp::Gt,
            Box::new(Predicate::Float(5.0)),
            Box::new(Predicate::Float(3.0)),
        );
        assert_eq!(p.eval_const(), Some(true));
    }

    #[test]
    fn test_substitution() {
        let p = Predicate::positive("x");
        let p2 = p.substitute("x", &Predicate::Float(5.0));
        assert_eq!(p2.eval_const(), Some(true));
    }
}
```

### Task 1.5: Update `src/types/mod.rs`

```rust
//! Type system for Demetrios

pub mod core;
pub mod effects;
pub mod ownership;
pub mod refinement;
pub mod units;

pub use core::*;
pub use effects::{Effect, EffectSet, EffectInference};
pub use ownership::{OwnershipChecker, OwnershipState};
pub use refinement::Predicate;
pub use units::{UnitType, UnitRegistry};
```

### Task 1.6: Create `src/check/mod.rs`

```rust
//! Type checker for Demetrios

mod env;
mod infer;

use crate::ast;
use crate::hir;
use crate::types::{Effect, EffectSet, Type, UnitRegistry};
use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

pub use env::TypeEnv;

/// Type error
#[derive(Error, Debug, Diagnostic)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected}, found {found}")]
    Mismatch {
        expected: String,
        found: String,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Undefined variable: {name}")]
    UndefinedVar {
        name: String,
        #[label("not found")]
        span: SourceSpan,
    },

    #[error("Effect not handled: {effect}")]
    UnhandledEffect {
        effect: String,
        #[label("effect escapes")]
        span: SourceSpan,
    },

    #[error("Unit mismatch: {expected} vs {found}")]
    UnitMismatch {
        expected: String,
        found: String,
        #[label("incompatible units")]
        span: SourceSpan,
    },

    #[error("Linear value used more than once")]
    LinearityViolation {
        #[label("second use here")]
        span: SourceSpan,
    },

    #[error("Linear value not consumed")]
    LinearNotConsumed {
        #[label("value dropped here")]
        span: SourceSpan,
    },
}

/// Type checker state
pub struct TypeChecker {
    /// Type environment
    env: TypeEnv,
    /// Effect inference
    effects: crate::types::EffectInference,
    /// Unit registry
    units: UnitRegistry,
    /// Accumulated errors
    errors: Vec<TypeError>,
    /// Next type variable ID
    next_tyvar: u32,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            effects: crate::types::EffectInference::new(),
            units: UnitRegistry::new(),
            errors: Vec::new(),
            next_tyvar: 0,
        }
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &ast::Ast) -> Result<hir::Hir> {
        // First pass: collect declarations
        for item in &program.items {
            self.collect_item(item);
        }

        // Second pass: check bodies
        let mut hir_items = Vec::new();
        for item in &program.items {
            if let Some(hir_item) = self.check_item(item)? {
                hir_items.push(hir_item);
            }
        }

        // Report errors
        if !self.errors.is_empty() {
            // For now, just return first error
            return Err(miette::miette!("Type errors found"));
        }

        Ok(hir::Hir { items: hir_items })
    }

    fn collect_item(&mut self, item: &ast::Item) {
        // Collect type signatures
        match item {
            ast::Item::Function(f) => {
                // TODO: Build function type from signature
            }
            ast::Item::Struct(s) => {
                // TODO: Register struct type
            }
            _ => {}
        }
    }

    fn check_item(&mut self, item: &ast::Item) -> Result<Option<hir::HirItem>> {
        match item {
            ast::Item::Function(f) => {
                let hir_fn = self.check_function(f)?;
                Ok(Some(hir::HirItem::Function(hir_fn)))
            }
            ast::Item::Struct(s) => {
                let hir_struct = self.check_struct(s)?;
                Ok(Some(hir::HirItem::Struct(hir_struct)))
            }
            _ => Ok(None),
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) -> Result<hir::HirFn> {
        self.env.push_scope();

        // TODO: Add parameters to scope
        // TODO: Check body
        // TODO: Verify return type

        self.env.pop_scope();

        Ok(hir::HirFn {
            id: f.id,
            name: String::new(), // TODO
            ty: hir::HirFnType {
                params: Vec::new(),
                return_type: Box::new(hir::HirType::Unit),
                effects: Vec::new(),
            },
            body: hir::HirBlock {
                stmts: Vec::new(),
                ty: hir::HirType::Unit,
            },
        })
    }

    fn check_struct(&mut self, s: &ast::StructDef) -> Result<hir::HirStruct> {
        Ok(hir::HirStruct {
            id: s.id,
            name: String::new(), // TODO
            fields: Vec::new(),
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
        })
    }

    /// Generate fresh type variable
    fn fresh_tyvar(&mut self) -> Type {
        let id = self.next_tyvar;
        self.next_tyvar += 1;
        Type::Var(crate::types::core::TypeVar(id))
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
```

### Task 1.7: Create `src/check/env.rs`

```rust
//! Type environment (scoped symbol table)

use crate::types::Type;
use std::collections::HashMap;

/// Scoped type environment
pub struct TypeEnv {
    scopes: Vec<Scope>,
}

struct Scope {
    bindings: HashMap<String, Binding>,
}

#[derive(Clone)]
pub struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub linear: bool,
    pub used: bool,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    pub fn pop_scope(&mut self) -> Vec<(String, Binding)> {
        self.scopes
            .pop()
            .map(|s| s.bindings.into_iter().collect())
            .unwrap_or_default()
    }

    pub fn bind(&mut self, name: String, ty: Type, mutable: bool, linear: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(
                name,
                Binding {
                    ty,
                    mutable,
                    linear,
                    used: false,
                },
            );
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                return Some(binding);
            }
        }
        None
    }

    pub fn lookup_mut(&mut self, name: &str) -> Option<&mut Binding> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.bindings.get_mut(name) {
                return Some(binding);
            }
        }
        None
    }

    pub fn mark_used(&mut self, name: &str) -> bool {
        if let Some(binding) = self.lookup_mut(name) {
            binding.used = true;
            true
        } else {
            false
        }
    }
}

impl Scope {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}
```

### Task 1.8: Create `src/check/infer.rs`

```rust
//! Type inference (Hindley-Milner with extensions)

use crate::types::{Type, TypeVar};
use std::collections::HashMap;

/// Substitution from type variables to types
pub type Subst = HashMap<TypeVar, Type>;

/// Unification result
pub type UnifyResult = Result<Subst, String>;

/// Unify two types
pub fn unify(t1: &Type, t2: &Type) -> UnifyResult {
    match (t1, t2) {
        // Same type
        (a, b) if a == b => Ok(HashMap::new()),

        // Type variable
        (Type::Var(v), t) | (t, Type::Var(v)) => {
            if occurs_check(*v, t) {
                Err(format!("Infinite type: {:?} ~ {:?}", v, t))
            } else {
                let mut subst = HashMap::new();
                subst.insert(*v, t.clone());
                Ok(subst)
            }
        }

        // Function types
        (Type::Fn(f1), Type::Fn(f2)) => {
            if f1.params.len() != f2.params.len() {
                return Err("Arity mismatch".to_string());
            }
            let mut subst = HashMap::new();
            for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
                let s = unify(&apply_subst(&subst, p1), &apply_subst(&subst, p2))?;
                subst = compose_subst(subst, s);
            }
            let s = unify(
                &apply_subst(&subst, &f1.return_type),
                &apply_subst(&subst, &f2.return_type),
            )?;
            Ok(compose_subst(subst, s))
        }

        // References
        (Type::Ref(k1, inner1), Type::Ref(k2, inner2)) if k1 == k2 => {
            unify(inner1, inner2)
        }

        // Tuples
        (Type::Tuple(ts1), Type::Tuple(ts2)) if ts1.len() == ts2.len() => {
            let mut subst = HashMap::new();
            for (t1, t2) in ts1.iter().zip(ts2.iter()) {
                let s = unify(&apply_subst(&subst, t1), &apply_subst(&subst, t2))?;
                subst = compose_subst(subst, s);
            }
            Ok(subst)
        }

        // Arrays
        (Type::Array(inner1, n1), Type::Array(inner2, n2)) if n1 == n2 => {
            unify(inner1, inner2)
        }

        // Type constructors with arguments
        (Type::App(id1, args1), Type::App(id2, args2)) 
            if id1 == id2 && args1.len() == args2.len() => {
            let mut subst = HashMap::new();
            for (a1, a2) in args1.iter().zip(args2.iter()) {
                let s = unify(&apply_subst(&subst, a1), &apply_subst(&subst, a2))?;
                subst = compose_subst(subst, s);
            }
            Ok(subst)
        }

        // Mismatch
        _ => Err(format!("Cannot unify {:?} with {:?}", t1, t2)),
    }
}

/// Occurs check: does variable v occur in type t?
fn occurs_check(v: TypeVar, t: &Type) -> bool {
    match t {
        Type::Var(v2) => v == *v2,
        Type::Fn(f) => {
            f.params.iter().any(|p| occurs_check(v, p)) 
                || occurs_check(v, &f.return_type)
        }
        Type::Ref(_, inner) | Type::Own(inner) | Type::Linear(inner) | Type::Affine(inner) => {
            occurs_check(v, inner)
        }
        Type::Tuple(ts) => ts.iter().any(|t| occurs_check(v, t)),
        Type::Array(inner, _) | Type::Slice(inner) => occurs_check(v, inner),
        Type::App(_, args) => args.iter().any(|a| occurs_check(v, a)),
        _ => false,
    }
}

/// Apply substitution to a type
pub fn apply_subst(subst: &Subst, t: &Type) -> Type {
    match t {
        Type::Var(v) => subst.get(v).cloned().unwrap_or_else(|| t.clone()),
        Type::Fn(f) => Type::Fn(crate::types::core::FnType {
            params: f.params.iter().map(|p| apply_subst(subst, p)).collect(),
            return_type: Box::new(apply_subst(subst, &f.return_type)),
            effects: f.effects.clone(),
        }),
        Type::Ref(k, inner) => Type::Ref(*k, Box::new(apply_subst(subst, inner))),
        Type::Own(inner) => Type::Own(Box::new(apply_subst(subst, inner))),
        Type::Linear(inner) => Type::Linear(Box::new(apply_subst(subst, inner))),
        Type::Affine(inner) => Type::Affine(Box::new(apply_subst(subst, inner))),
        Type::Tuple(ts) => Type::Tuple(ts.iter().map(|t| apply_subst(subst, t)).collect()),
        Type::Array(inner, n) => Type::Array(Box::new(apply_subst(subst, inner)), *n),
        Type::Slice(inner) => Type::Slice(Box::new(apply_subst(subst, inner))),
        Type::App(id, args) => {
            Type::App(*id, args.iter().map(|a| apply_subst(subst, a)).collect())
        }
        _ => t.clone(),
    }
}

/// Compose two substitutions
pub fn compose_subst(s1: Subst, s2: Subst) -> Subst {
    let mut result: Subst = s2
        .into_iter()
        .map(|(k, v)| (k, apply_subst(&s1, &v)))
        .collect();
    for (k, v) in s1 {
        result.entry(k).or_insert(v);
    }
    result
}
```

### Task 1.9: Create `src/effects/mod.rs`

```rust
//! Effect system re-exports

pub use crate::types::effects::*;
```

### Task 1.10: Create `src/mlir/mod.rs`

```rust
//! MLIR integration (placeholder)
//!
//! D uses MLIR for optimization and multi-target codegen.

/// MLIR context (placeholder)
pub struct MlirContext;

impl MlirContext {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MlirContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower HLIR to MLIR
pub fn lower(_hlir: &crate::hlir::HlirModule) -> Result<(), String> {
    Err("MLIR integration not yet implemented".into())
}
```

### Task 1.11: Create `src/codegen/mod.rs`

```rust
//! Code generation backends

pub mod cranelift;
pub mod gpu;
pub mod llvm;

/// Target backend
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// LLVM for AOT compilation
    LLVM,
    /// Cranelift for JIT
    Cranelift,
    /// GPU (CUDA/Vulkan)
    GPU,
}
```

### Task 1.12: Create `src/codegen/llvm.rs`

```rust
//! LLVM backend (stub)

use crate::hlir::HlirModule;

pub struct LlvmCodegen;

impl LlvmCodegen {
    pub fn new() -> Self {
        Self
    }

    pub fn compile(&self, _module: &HlirModule) -> Result<Vec<u8>, String> {
        Err("LLVM backend not yet implemented".into())
    }
}

impl Default for LlvmCodegen {
    fn default() -> Self {
        Self::new()
    }
}
```

### Task 1.13: Create `src/codegen/cranelift.rs`

```rust
//! Cranelift JIT backend (stub)

use crate::hlir::HlirModule;

pub struct CraneliftJit;

impl CraneliftJit {
    pub fn new() -> Self {
        Self
    }

    pub fn compile_and_run(&self, _module: &HlirModule) -> Result<i64, String> {
        Err("Cranelift JIT not yet implemented".into())
    }
}

impl Default for CraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}
```

### Task 1.14: Create `src/codegen/gpu.rs`

```rust
//! GPU code generation (stub)

use crate::hlir::HlirModule;

#[derive(Debug, Clone, Copy)]
pub enum GpuTarget {
    /// NVIDIA PTX
    Cuda,
    /// Vulkan/OpenCL SPIR-V
    Spirv,
}

pub struct GpuCodegen {
    pub target: GpuTarget,
}

impl GpuCodegen {
    pub fn new(target: GpuTarget) -> Self {
        Self { target }
    }

    pub fn compile_kernels(&self, _module: &HlirModule) -> Result<Vec<u8>, String> {
        Err("GPU codegen not yet implemented".into())
    }
}
```

## PHASE 2: Verify Build

After creating all files:

```bash
cd /mnt/e/workspace/demetrios/compiler
cargo check
cargo build
cargo test
```

Fix any compilation errors. Common issues:
- Missing imports
- Module not declared in parent
- Type mismatches

## PHASE 3: First Program

Create `examples/hello.d`:

```d
// First Demetrios program

fn main() -> int {
    return 0
}
```

Test parsing:
```bash
cargo run -- check examples/hello.d
```

## Success Criteria

1. ✅ `cargo build` completes without errors
2. ✅ `cargo test` passes lexer tests
3. ✅ `dc --version` prints version
4. ✅ Directory structure complete
5. ✅ All modules compile

## Notes

- Don't worry about full implementation yet
- Stubs are fine for Phase 1
- Focus on making it compile
- We'll fill in real logic in Phase 2

---

**Day 2 Goal: Make. It. Compile.**
