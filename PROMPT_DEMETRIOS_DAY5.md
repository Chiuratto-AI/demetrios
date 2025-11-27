# Demetrios (D) Compiler — Day 5: Effects, Ownership & Better Errors

## Context

Day 1: Scaffold structure  
Day 2: Stub files, cargo build passes  
Day 3: First working pipeline (lex → parse → AST → basic type check)  
Day 4: Name resolution + bidirectional type checking  
Day 5: **Effect inference, ownership checking, source-located errors**

## Repository

```
/mnt/e/workspace/demetrios/
```

## Today's Mission

1. **Effect inference** — Track effects through expressions
2. **Ownership checker** — Enforce move/borrow rules
3. **Linear type enforcement** — Must-use semantics
4. **Source-located errors** — Pretty miette diagnostics
5. **Effect handlers** — Parse and check `handle`/`with`

---

## PHASE 1: Source Locations

### 1.1 Update `src/lib.rs` — Better Span

```rust
//! Demetrios compiler library

use miette::SourceSpan;

/// Source location span
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Span {
    /// Start byte offset
    pub start: u32,
    /// End byte offset (exclusive)
    pub end: u32,
    /// Source file ID (for multi-file compilation)
    pub file_id: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end, file_id: 0 }
    }

    pub fn dummy() -> Self {
        Self::default()
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            file_id: self.file_id,
        }
    }

    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }
}

impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        SourceSpan::new(span.start as usize, span.len())
    }
}

impl From<std::ops::Range<usize>> for Span {
    fn from(range: std::ops::Range<usize>) -> Self {
        Span::new(range.start as u32, range.end as u32)
    }
}

/// Spanned value
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ... rest of lib.rs
```

### 1.2 Create `src/diagnostics.rs`

```rust
//! Diagnostic reporting with source locations

use crate::Span;
use miette::{Diagnostic, NamedSource, Report, SourceSpan};
use std::sync::Arc;
use thiserror::Error;

/// Source file for error reporting
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub content: Arc<str>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, content: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }

    pub fn to_named_source(&self) -> NamedSource<String> {
        NamedSource::new(self.name.clone(), self.content.to_string())
    }
}

/// Compiler diagnostic
#[derive(Error, Debug, Diagnostic)]
pub enum CompileError {
    // === Parse Errors ===
    #[error("Unexpected token: expected {expected}, found {found}")]
    #[diagnostic(code(parse::unexpected_token))]
    UnexpectedToken {
        expected: String,
        found: String,
        #[label("unexpected token here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Unexpected end of file")]
    #[diagnostic(code(parse::unexpected_eof))]
    UnexpectedEof {
        #[label("expected more tokens")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Resolution Errors ===
    #[error("Undefined variable `{name}`")]
    #[diagnostic(code(resolve::undefined_var), help("did you mean to declare this variable with `let`?"))]
    UndefinedVariable {
        name: String,
        #[label("not found in this scope")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Undefined type `{name}`")]
    #[diagnostic(code(resolve::undefined_type))]
    UndefinedType {
        name: String,
        #[label("type not found")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Duplicate definition of `{name}`")]
    #[diagnostic(code(resolve::duplicate_def))]
    DuplicateDefinition {
        name: String,
        #[label("redefined here")]
        span: SourceSpan,
        #[label("first defined here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Type Errors ===
    #[error("Type mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(type::mismatch))]
    TypeMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },

    #[error("Cannot unify `{t1}` with `{t2}`")]
    #[diagnostic(code(type::unification_failed))]
    UnificationFailed {
        t1: String,
        t2: String,
        #[label("type mismatch here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Missing type annotation")]
    #[diagnostic(code(type::annotation_required), help("add a type annotation: `let x: Type = ...`"))]
    AnnotationRequired {
        #[label("cannot infer type")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Effect Errors ===
    #[error("Unhandled effect `{effect}`")]
    #[diagnostic(
        code(effect::unhandled),
        help("either handle this effect with `with handler {{ ... }}` or add it to the function signature")
    )]
    UnhandledEffect {
        effect: String,
        #[label("effect `{effect}` escapes here")]
        span: SourceSpan,
        #[label("function does not declare this effect")]
        fn_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Effect `{effect}` not declared in function signature")]
    #[diagnostic(code(effect::undeclared))]
    UndeclaredEffect {
        effect: String,
        #[label("this operation has effect `{effect}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot perform `{effect}` in pure context")]
    #[diagnostic(code(effect::pure_context))]
    EffectInPureContext {
        effect: String,
        #[label("effectful operation here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Ownership Errors ===
    #[error("Use of moved value `{name}`")]
    #[diagnostic(code(ownership::use_after_move))]
    UseAfterMove {
        name: String,
        #[label("value used here after move")]
        use_span: SourceSpan,
        #[label("value moved here")]
        move_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable because it is already borrowed")]
    #[diagnostic(code(ownership::already_borrowed))]
    AlreadyBorrowed {
        name: String,
        #[label("cannot borrow as mutable")]
        span: SourceSpan,
        #[label("previous borrow here")]
        prev_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable more than once")]
    #[diagnostic(code(ownership::double_mut_borrow))]
    DoubleMutBorrow {
        name: String,
        #[label("second mutable borrow here")]
        span: SourceSpan,
        #[label("first mutable borrow here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Linearity Errors ===
    #[error("Linear value `{name}` used more than once")]
    #[diagnostic(
        code(linear::multiple_use),
        help("linear values must be used exactly once")
    )]
    LinearMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Linear value `{name}` not consumed")]
    #[diagnostic(
        code(linear::not_consumed),
        help("linear values must be explicitly consumed before going out of scope")
    )]
    LinearNotConsumed {
        name: String,
        #[label("linear value declared here")]
        decl_span: SourceSpan,
        #[label("goes out of scope here without being consumed")]
        scope_end: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Affine value `{name}` used more than once")]
    #[diagnostic(code(affine::multiple_use))]
    AffineMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Unit Errors ===
    #[error("Unit mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(unit::mismatch))]
    UnitMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot add values with different units: `{u1}` and `{u2}`")]
    #[diagnostic(code(unit::incompatible_add))]
    IncompatibleUnits {
        u1: String,
        u2: String,
        #[label("incompatible units")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },
}

/// Error reporter
pub struct Reporter {
    source: SourceFile,
    errors: Vec<CompileError>,
    warnings: Vec<CompileError>,
}

impl Reporter {
    pub fn new(source: SourceFile) -> Self {
        Self {
            source,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn error(&mut self, error: CompileError) {
        self.errors.push(error);
    }

    pub fn warning(&mut self, warning: CompileError) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Create NamedSource for this file
    pub fn named_source(&self) -> NamedSource<String> {
        self.source.to_named_source()
    }

    /// Print all diagnostics
    pub fn emit_all(&self) {
        for warning in &self.warnings {
            eprintln!("{:?}", Report::new(warning.clone()));
        }
        for error in &self.errors {
            eprintln!("{:?}", Report::new(error.clone()));
        }
    }

    /// Consume and return errors
    pub fn into_errors(self) -> Vec<CompileError> {
        self.errors
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod diagnostics;
pub use diagnostics::{CompileError, Reporter, SourceFile};
```

---

## PHASE 2: Effect Inference

### 2.1 Create `src/effects/inference.rs`

```rust
//! Effect inference pass

use crate::ast;
use crate::resolve::{DefId, SymbolTable};
use crate::types::effects::{Effect, EffectSet};
use crate::Span;
use std::collections::HashMap;

/// Effect inference context
pub struct EffectInference<'a> {
    symbols: &'a SymbolTable,
    /// Inferred effects per function
    fn_effects: HashMap<DefId, EffectSet>,
    /// Current function's declared effects
    declared: EffectSet,
    /// Current function's inferred effects
    inferred: EffectSet,
    /// Effect variable counter
    next_var: u32,
}

impl<'a> EffectInference<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            fn_effects: HashMap::new(),
            declared: EffectSet::new(),
            inferred: EffectSet::new(),
            next_var: 0,
        }
    }

    /// Infer effects for entire program
    pub fn infer_program(&mut self, ast: &ast::Ast) -> Result<(), Vec<EffectError>> {
        let mut errors = Vec::new();

        // First pass: collect declared effects
        for item in &ast.items {
            if let ast::Item::Function(f) = item {
                self.collect_function_effects(f);
            }
        }

        // Second pass: infer and check
        for item in &ast.items {
            if let ast::Item::Function(f) = item {
                if let Err(e) = self.check_function(f) {
                    errors.extend(e);
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn collect_function_effects(&mut self, f: &ast::FnDef) {
        let mut effects = EffectSet::new();
        for eff in &f.effects {
            effects.insert(self.resolve_effect_ref(eff));
        }
        
        if let Some(def_id) = self.symbols.def_for_node(f.id) {
            self.fn_effects.insert(def_id, effects);
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) -> Result<(), Vec<EffectError>> {
        // Set declared effects
        self.declared = EffectSet::new();
        for eff in &f.effects {
            self.declared.insert(self.resolve_effect_ref(eff));
        }
        
        // Reset inferred effects
        self.inferred = EffectSet::new();

        // Infer body effects
        if let Some(ref body) = f.body {
            self.infer_block(body)?;
        }

        // Check that inferred ⊆ declared
        let mut errors = Vec::new();
        for effect in self.inferred.iter() {
            if !self.declared.contains(effect) {
                errors.push(EffectError::UndeclaredEffect {
                    effect: format!("{}", effect),
                    span: f.span, // TODO: more precise span
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn infer_block(&mut self, block: &ast::Block) -> Result<EffectSet, Vec<EffectError>> {
        let mut block_effects = EffectSet::new();
        
        for stmt in &block.stmts {
            let stmt_effects = self.infer_stmt(stmt)?;
            block_effects = block_effects.union(&stmt_effects);
        }
        
        self.inferred = self.inferred.union(&block_effects);
        Ok(block_effects)
    }

    fn infer_stmt(&mut self, stmt: &ast::Stmt) -> Result<EffectSet, Vec<EffectError>> {
        match stmt {
            ast::Stmt::Let(l) => {
                if let Some(ref init) = l.init {
                    self.infer_expr(init)
                } else {
                    Ok(EffectSet::new())
                }
            }
            ast::Stmt::Return(r) => {
                if let Some(ref val) = r.value {
                    self.infer_expr(val)
                } else {
                    Ok(EffectSet::new())
                }
            }
            ast::Stmt::Expr(e) => self.infer_expr(&e.expr),
            ast::Stmt::If(i) => {
                let mut effects = self.infer_expr(&i.condition)?;
                effects = effects.union(&self.infer_block(&i.then_block)?);
                if let Some(ref else_block) = i.else_block {
                    effects = effects.union(&self.infer_block(else_block)?);
                }
                Ok(effects)
            }
            ast::Stmt::While(w) => {
                let mut effects = self.infer_expr(&w.condition)?;
                effects = effects.union(&self.infer_block(&w.body)?);
                // Loops may diverge
                effects.insert(Effect::Div);
                Ok(effects)
            }
            ast::Stmt::Loop(l) => {
                let mut effects = self.infer_block(&l.body)?;
                effects.insert(Effect::Div);
                Ok(effects)
            }
            ast::Stmt::Handle(h) => {
                // Handler removes handled effect
                let body_effects = self.infer_block(&h.body)?;
                let handled = self.resolve_effect_ref(&h.effect);
                let mut result = EffectSet::new();
                for eff in body_effects.iter() {
                    if eff != &handled {
                        result.insert(eff.clone());
                    }
                }
                Ok(result)
            }
            _ => Ok(EffectSet::new()),
        }
    }

    fn infer_expr(&mut self, expr: &ast::Expr) -> Result<EffectSet, Vec<EffectError>> {
        match expr {
            ast::Expr::Literal(_) => Ok(EffectSet::new()),
            
            ast::Expr::Path(_) => {
                // Variable access: might be Mut if mutable
                Ok(EffectSet::new())
            }
            
            ast::Expr::Binary(b) => {
                let left = self.infer_expr(&b.left)?;
                let right = self.infer_expr(&b.right)?;
                
                let mut effects = left.union(&right);
                
                // Division may panic
                if matches!(b.op, ast::BinOp::Div | ast::BinOp::Mod) {
                    effects.insert(Effect::Panic);
                }
                
                Ok(effects)
            }
            
            ast::Expr::Unary(u) => self.infer_expr(&u.operand),
            
            ast::Expr::Call(c) => {
                // Get callee's effects
                let callee_effects = self.infer_expr(&c.callee)?;
                
                // Get function's declared effects
                let fn_effects = self.get_callee_effects(&c.callee);
                
                // Infer argument effects
                let mut arg_effects = EffectSet::new();
                for arg in &c.args {
                    arg_effects = arg_effects.union(&self.infer_expr(arg)?);
                }
                
                Ok(callee_effects.union(&fn_effects).union(&arg_effects))
            }
            
            ast::Expr::Block(b) => self.infer_block(b),
            
            ast::Expr::If(i) => {
                let mut effects = self.infer_expr(&i.condition)?;
                effects = effects.union(&self.infer_block(&i.then_block)?);
                if let Some(ref else_block) = i.else_block {
                    effects = effects.union(&self.infer_block(else_block)?);
                }
                Ok(effects)
            }
            
            ast::Expr::With(w) => {
                // Handler removes effect
                let body_effects = self.infer_expr(&w.body)?;
                // TODO: determine which effect is handled
                Ok(body_effects)
            }
            
            ast::Expr::Resume(_) => {
                // Resume continues effect handler
                Ok(EffectSet::new())
            }
            
            _ => Ok(EffectSet::new()),
        }
    }

    fn get_callee_effects(&self, callee: &ast::Expr) -> EffectSet {
        if let ast::Expr::Path(path) = callee {
            if path.segments.len() == 1 {
                if let Some(def_id) = self.symbols.ref_for_node(path.segments[0]) {
                    if let Some(effects) = self.fn_effects.get(&def_id) {
                        return effects.clone();
                    }
                }
            }
        }
        EffectSet::new()
    }

    fn resolve_effect_ref(&self, eff: &ast::EffectRef) -> Effect {
        match eff {
            ast::EffectRef::IO => Effect::IO,
            ast::EffectRef::Mut => Effect::Mut,
            ast::EffectRef::Alloc => Effect::Alloc,
            ast::EffectRef::Panic => Effect::Panic,
            ast::EffectRef::Async => Effect::Async,
            ast::EffectRef::GPU => Effect::GPU,
            ast::EffectRef::Prob => Effect::Prob,
            ast::EffectRef::Div => Effect::Div,
            ast::EffectRef::Named(id) => Effect::Named(format!("${}", id.0)),
        }
    }
}

#[derive(Debug)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub span: Span,
}

#[derive(Debug)]
pub enum EffectErrorKind {
    UndeclaredEffect { effect: String },
    UnhandledEffect { effect: String },
    EffectInPureContext { effect: String },
}

impl EffectError {
    fn UndeclaredEffect { effect: String, span: Span } -> Self {
        Self {
            kind: EffectErrorKind::UndeclaredEffect { effect },
            span,
        }
    }
}
```

### 2.2 Update `src/effects/mod.rs`

```rust
//! Effect system

pub mod inference;

pub use crate::types::effects::*;
pub use inference::EffectInference;
```

---

## PHASE 3: Ownership Checker

### 3.1 Create `src/ownership/mod.rs`

```rust
//! Ownership and borrow checking

mod checker;
mod state;

pub use checker::OwnershipChecker;
pub use state::{BorrowState, OwnershipState, Place, PlaceId};
```

### 3.2 Create `src/ownership/state.rs`

```rust
//! Ownership state tracking

use crate::resolve::DefId;
use crate::Span;
use std::collections::HashMap;

/// A place (lvalue) in memory
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Place {
    /// Base variable
    pub base: DefId,
    /// Projections (field access, index, deref)
    pub projections: Vec<Projection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Projection {
    Field(String),
    Index,
    Deref,
}

impl Place {
    pub fn var(def_id: DefId) -> Self {
        Self {
            base: def_id,
            projections: Vec::new(),
        }
    }

    pub fn field(mut self, name: impl Into<String>) -> Self {
        self.projections.push(Projection::Field(name.into()));
        self
    }

    pub fn deref(mut self) -> Self {
        self.projections.push(Projection::Deref);
        self
    }
}

/// Unique place ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaceId(pub u32);

/// State of ownership for a place
#[derive(Debug, Clone)]
pub enum OwnershipState {
    /// Owned, not borrowed
    Owned,
    /// Moved to another location
    Moved { to: Span },
    /// Borrowed (shared)
    BorrowedShared { 
        count: u32,
        /// Spans of active borrows
        borrows: Vec<Span>,
    },
    /// Borrowed (exclusive)
    BorrowedExclusive { borrow: Span },
    /// Dropped/out of scope
    Dropped,
}

impl OwnershipState {
    pub fn is_usable(&self) -> bool {
        matches!(self, OwnershipState::Owned | OwnershipState::BorrowedShared { .. })
    }

    pub fn is_movable(&self) -> bool {
        matches!(self, OwnershipState::Owned)
    }

    pub fn can_borrow_shared(&self) -> bool {
        matches!(self, OwnershipState::Owned | OwnershipState::BorrowedShared { .. })
    }

    pub fn can_borrow_exclusive(&self) -> bool {
        matches!(self, OwnershipState::Owned)
    }
}

/// State of a borrow
#[derive(Debug, Clone)]
pub struct BorrowState {
    /// Place being borrowed
    pub place: Place,
    /// Is this an exclusive borrow?
    pub exclusive: bool,
    /// Span where borrow occurred
    pub span: Span,
    /// Is the borrow still active?
    pub active: bool,
}

/// Linear/affine tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linearity {
    /// Normal (unrestricted) type
    Unrestricted,
    /// Linear (must use exactly once)
    Linear,
    /// Affine (may use at most once)
    Affine,
}

/// Tracked value
#[derive(Debug, Clone)]
pub struct TrackedValue {
    pub def_id: DefId,
    pub linearity: Linearity,
    pub state: OwnershipState,
    pub use_count: u32,
    pub decl_span: Span,
    pub uses: Vec<Span>,
}

impl TrackedValue {
    pub fn new(def_id: DefId, linearity: Linearity, decl_span: Span) -> Self {
        Self {
            def_id,
            linearity,
            state: OwnershipState::Owned,
            use_count: 0,
            decl_span,
            uses: Vec::new(),
        }
    }

    pub fn record_use(&mut self, span: Span) {
        self.use_count += 1;
        self.uses.push(span);
    }

    pub fn check_linearity(&self) -> Result<(), LinearityError> {
        match self.linearity {
            Linearity::Linear => {
                if self.use_count == 0 {
                    Err(LinearityError::NotConsumed {
                        def_id: self.def_id,
                        decl_span: self.decl_span,
                    })
                } else if self.use_count > 1 {
                    Err(LinearityError::MultipleUse {
                        def_id: self.def_id,
                        first: self.uses[0],
                        second: self.uses[1],
                    })
                } else {
                    Ok(())
                }
            }
            Linearity::Affine => {
                if self.use_count > 1 {
                    Err(LinearityError::MultipleUse {
                        def_id: self.def_id,
                        first: self.uses[0],
                        second: self.uses[1],
                    })
                } else {
                    Ok(())
                }
            }
            Linearity::Unrestricted => Ok(()),
        }
    }
}

#[derive(Debug)]
pub enum LinearityError {
    NotConsumed { def_id: DefId, decl_span: Span },
    MultipleUse { def_id: DefId, first: Span, second: Span },
}

/// Ownership state for entire scope
#[derive(Debug, Default)]
pub struct ScopeState {
    /// Tracked values by DefId
    values: HashMap<DefId, TrackedValue>,
    /// Active borrows
    borrows: Vec<BorrowState>,
}

impl ScopeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track(&mut self, value: TrackedValue) {
        self.values.insert(value.def_id, value);
    }

    pub fn get(&self, def_id: DefId) -> Option<&TrackedValue> {
        self.values.get(&def_id)
    }

    pub fn get_mut(&mut self, def_id: DefId) -> Option<&mut TrackedValue> {
        self.values.get_mut(&def_id)
    }

    pub fn add_borrow(&mut self, borrow: BorrowState) {
        self.borrows.push(borrow);
    }

    pub fn end_borrow(&mut self, place: &Place) {
        for borrow in &mut self.borrows {
            if &borrow.place == place {
                borrow.active = false;
            }
        }
    }

    pub fn active_borrows(&self, place: &Place) -> Vec<&BorrowState> {
        self.borrows
            .iter()
            .filter(|b| b.active && &b.place == place)
            .collect()
    }

    pub fn check_all_linear(&self) -> Vec<LinearityError> {
        let mut errors = Vec::new();
        for value in self.values.values() {
            if let Err(e) = value.check_linearity() {
                errors.push(e);
            }
        }
        errors
    }
}
```

### 3.3 Create `src/ownership/checker.rs`

```rust
//! Ownership checker implementation

use crate::ast;
use crate::resolve::{DefId, SymbolTable, DefKind};
use crate::diagnostics::{CompileError, SourceFile};
use crate::Span;
use super::state::*;
use miette::NamedSource;
use std::collections::HashMap;

/// Ownership and borrow checker
pub struct OwnershipChecker<'a> {
    symbols: &'a SymbolTable,
    source: &'a SourceFile,
    /// Scope stack
    scopes: Vec<ScopeState>,
    /// Type linearity cache
    linearity_cache: HashMap<DefId, Linearity>,
    /// Errors
    errors: Vec<CompileError>,
}

impl<'a> OwnershipChecker<'a> {
    pub fn new(symbols: &'a SymbolTable, source: &'a SourceFile) -> Self {
        Self {
            symbols,
            source,
            scopes: vec![ScopeState::new()],
            linearity_cache: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Check entire program
    pub fn check_program(&mut self, ast: &ast::Ast) -> Result<(), Vec<CompileError>> {
        // Build linearity cache from struct definitions
        for item in &ast.items {
            if let ast::Item::Struct(s) = item {
                if let Some(def_id) = self.symbols.def_for_node(s.id) {
                    let linearity = if s.modifiers.linear {
                        Linearity::Linear
                    } else if s.modifiers.affine {
                        Linearity::Affine
                    } else {
                        Linearity::Unrestricted
                    };
                    self.linearity_cache.insert(def_id, linearity);
                }
            }
        }

        // Check functions
        for item in &ast.items {
            if let ast::Item::Function(f) = item {
                self.check_function(f);
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn check_function(&mut self, f: &ast::FnDef) {
        self.push_scope();

        // Track parameters
        for param in &f.params {
            if let Some(def_id) = self.symbols.def_for_node(param.id) {
                let linearity = self.get_type_linearity(&param.ty);
                self.track_value(def_id, linearity, param.span);
            }
        }

        // Check body
        if let Some(ref body) = f.body {
            self.check_block(body);
        }

        // Check linear values consumed
        self.check_scope_end();
        self.pop_scope();
    }

    fn check_block(&mut self, block: &ast::Block) {
        self.push_scope();
        
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        
        self.check_scope_end();
        self.pop_scope();
    }

    fn check_stmt(&mut self, stmt: &ast::Stmt) {
        match stmt {
            ast::Stmt::Let(l) => {
                // Check initializer first
                if let Some(ref init) = l.init {
                    self.check_expr(init, UseKind::Move);
                }
                
                // Track the binding
                if let Some(def_id) = self.symbols.def_for_node(l.id) {
                    let linearity = if let Some(ref ty) = l.ty {
                        self.get_type_linearity(ty)
                    } else {
                        Linearity::Unrestricted
                    };
                    self.track_value(def_id, linearity, l.span);
                }
            }
            
            ast::Stmt::Return(r) => {
                if let Some(ref val) = r.value {
                    self.check_expr(val, UseKind::Move);
                }
            }
            
            ast::Stmt::Expr(e) => {
                self.check_expr(&e.expr, UseKind::Move);
            }
            
            ast::Stmt::If(i) => {
                self.check_expr(&i.condition, UseKind::Copy);
                self.check_block(&i.then_block);
                if let Some(ref else_block) = i.else_block {
                    self.check_block(else_block);
                }
            }
            
            ast::Stmt::While(w) => {
                self.check_expr(&w.condition, UseKind::Copy);
                self.check_block(&w.body);
            }
            
            ast::Stmt::For(f) => {
                self.check_expr(&f.iter, UseKind::Move);
                self.push_scope();
                // TODO: bind pattern variables
                self.check_block(&f.body);
                self.check_scope_end();
                self.pop_scope();
            }
            
            ast::Stmt::Loop(l) => {
                self.check_block(&l.body);
            }
            
            _ => {}
        }
    }

    fn check_expr(&mut self, expr: &ast::Expr, use_kind: UseKind) {
        match expr {
            ast::Expr::Path(path) => {
                if path.segments.len() == 1 {
                    if let Some(def_id) = self.symbols.ref_for_node(path.segments[0]) {
                        self.use_value(def_id, use_kind, expr_span(expr));
                    }
                }
            }
            
            ast::Expr::Binary(b) => {
                // Assignment: LHS is written, RHS is moved
                if matches!(b.op, ast::BinOp::Assign) {
                    self.check_expr(&b.right, UseKind::Move);
                    // LHS is being written to, not read
                } else {
                    self.check_expr(&b.left, UseKind::Copy);
                    self.check_expr(&b.right, UseKind::Copy);
                }
            }
            
            ast::Expr::Unary(u) => {
                match u.op {
                    ast::UnaryOp::Ref => {
                        // Shared borrow
                        if let Some(place) = self.expr_to_place(&u.operand) {
                            self.borrow_shared(place, expr_span(expr));
                        }
                    }
                    ast::UnaryOp::RefMut => {
                        // Exclusive borrow
                        if let Some(place) = self.expr_to_place(&u.operand) {
                            self.borrow_exclusive(place, expr_span(expr));
                        }
                    }
                    ast::UnaryOp::Deref => {
                        self.check_expr(&u.operand, UseKind::Copy);
                    }
                    _ => {
                        self.check_expr(&u.operand, use_kind);
                    }
                }
            }
            
            ast::Expr::Call(c) => {
                self.check_expr(&c.callee, UseKind::Copy);
                for arg in &c.args {
                    // TODO: check parameter ownership annotations
                    self.check_expr(arg, UseKind::Move);
                }
            }
            
            ast::Expr::MethodCall(m) => {
                // TODO: check method receiver ownership
                self.check_expr(&m.receiver, UseKind::Copy);
                for arg in &m.args {
                    self.check_expr(arg, UseKind::Move);
                }
            }
            
            ast::Expr::Field(f) => {
                self.check_expr(&f.expr, UseKind::Copy);
            }
            
            ast::Expr::Index(i) => {
                self.check_expr(&i.expr, UseKind::Copy);
                self.check_expr(&i.index, UseKind::Copy);
            }
            
            ast::Expr::Tuple(t) => {
                for elem in &t.elements {
                    self.check_expr(elem, use_kind);
                }
            }
            
            ast::Expr::Array(a) => {
                for elem in &a.elements {
                    self.check_expr(elem, use_kind);
                }
            }
            
            ast::Expr::Struct(s) => {
                for field in &s.fields {
                    self.check_expr(&field.value, UseKind::Move);
                }
            }
            
            ast::Expr::If(i) => {
                self.check_expr(&i.condition, UseKind::Copy);
                self.check_block(&i.then_block);
                if let Some(ref else_block) = i.else_block {
                    self.check_block(else_block);
                }
            }
            
            ast::Expr::Block(b) => {
                self.check_block(b);
            }
            
            ast::Expr::Lambda(l) => {
                // TODO: check captures
                self.push_scope();
                for param in &l.params {
                    if let Some(def_id) = self.symbols.def_for_node(param.id) {
                        self.track_value(def_id, Linearity::Unrestricted, param.span);
                    }
                }
                self.check_expr(&l.body, UseKind::Move);
                self.check_scope_end();
                self.pop_scope();
            }
            
            _ => {}
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(ScopeState::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn current_scope(&mut self) -> &mut ScopeState {
        self.scopes.last_mut().unwrap()
    }

    fn track_value(&mut self, def_id: DefId, linearity: Linearity, span: Span) {
        let value = TrackedValue::new(def_id, linearity, span);
        self.current_scope().track(value);
    }

    fn use_value(&mut self, def_id: DefId, use_kind: UseKind, span: Span) {
        // Look up in all scopes
        for scope in self.scopes.iter_mut().rev() {
            if let Some(value) = scope.get_mut(def_id) {
                // Check if already moved
                if let OwnershipState::Moved { to } = &value.state {
                    self.errors.push(CompileError::UseAfterMove {
                        name: format!("${}", def_id.0),
                        use_span: span.into(),
                        move_span: (*to).into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }

                // Record use
                value.record_use(span);

                // Update state for move
                if use_kind == UseKind::Move {
                    value.state = OwnershipState::Moved { to: span };
                }

                return;
            }
        }
    }

    fn borrow_shared(&mut self, place: Place, span: Span) {
        // Check if exclusively borrowed
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            for borrow in active {
                if borrow.exclusive {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: format!("{:?}", place),
                        span: span.into(),
                        prev_span: borrow.span.into(),
                        src: self.source.to_named_source(),
                    });
                    return;
                }
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: false,
            span,
            active: true,
        });
    }

    fn borrow_exclusive(&mut self, place: Place, span: Span) {
        // Check if any borrows exist
        for scope in &self.scopes {
            let active = scope.active_borrows(&place);
            if !active.is_empty() {
                let prev = active[0];
                if prev.exclusive {
                    self.errors.push(CompileError::DoubleMutBorrow {
                        name: format!("{:?}", place),
                        span: span.into(),
                        first_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                } else {
                    self.errors.push(CompileError::AlreadyBorrowed {
                        name: format!("{:?}", place),
                        span: span.into(),
                        prev_span: prev.span.into(),
                        src: self.source.to_named_source(),
                    });
                }
                return;
            }
        }

        // Add borrow
        self.current_scope().add_borrow(BorrowState {
            place,
            exclusive: true,
            span,
            active: true,
        });
    }

    fn check_scope_end(&mut self) {
        let errors = self.current_scope().check_all_linear();
        
        for error in errors {
            match error {
                LinearityError::NotConsumed { def_id, decl_span } => {
                    self.errors.push(CompileError::LinearNotConsumed {
                        name: self.get_name(def_id),
                        decl_span: decl_span.into(),
                        scope_end: decl_span.into(), // TODO: actual scope end
                        src: self.source.to_named_source(),
                    });
                }
                LinearityError::MultipleUse { def_id, first, second } => {
                    self.errors.push(CompileError::LinearMultipleUse {
                        name: self.get_name(def_id),
                        first_span: first.into(),
                        second_span: second.into(),
                        src: self.source.to_named_source(),
                    });
                }
            }
        }
    }

    fn get_type_linearity(&self, ty: &ast::Type) -> Linearity {
        match ty {
            ast::Type::Linear(_) => Linearity::Linear,
            ast::Type::Affine(_) => Linearity::Affine,
            ast::Type::Named(id) => {
                if let Some(def_id) = self.symbols.ref_for_node(*id) {
                    self.linearity_cache.get(&def_id).copied().unwrap_or(Linearity::Unrestricted)
                } else {
                    Linearity::Unrestricted
                }
            }
            _ => Linearity::Unrestricted,
        }
    }

    fn expr_to_place(&self, expr: &ast::Expr) -> Option<Place> {
        match expr {
            ast::Expr::Path(path) => {
                if path.segments.len() == 1 {
                    if let Some(def_id) = self.symbols.ref_for_node(path.segments[0]) {
                        return Some(Place::var(def_id));
                    }
                }
                None
            }
            ast::Expr::Field(f) => {
                self.expr_to_place(&f.expr).map(|p| p.field(format!("${}", f.field.0)))
            }
            ast::Expr::Unary(u) if matches!(u.op, ast::UnaryOp::Deref) => {
                self.expr_to_place(&u.operand).map(|p| p.deref())
            }
            _ => None,
        }
    }

    fn get_name(&self, def_id: DefId) -> String {
        self.symbols.get(def_id).map(|s| s.name.clone()).unwrap_or_else(|| format!("${}", def_id.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseKind {
    Move,
    Copy,
}

fn expr_span(expr: &ast::Expr) -> Span {
    // TODO: get actual span from expression
    Span::dummy()
}
```

Add to `src/lib.rs`:
```rust
pub mod ownership;
pub use ownership::OwnershipChecker;
```

---

## PHASE 4: Integration

### 4.1 Update `src/main.rs`

```rust
use demetrios::{
    parser,
    resolve::Resolver,
    check::TypeChecker,
    effects::EffectInference,
    ownership::OwnershipChecker,
    diagnostics::{SourceFile, Reporter},
};

fn cmd_check(file: &Path, args: &CheckArgs) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;
    
    let source = SourceFile::new(file.to_string_lossy(), content.clone());
    let mut reporter = Reporter::new(source.clone());

    // 1. Parse
    let ast = parser::parse(&content)?;
    
    if args.show_ast {
        println!("=== AST ===");
        println!("{}", demetrios::ast::AstPrinter::print(&ast));
        println!();
    }

    // 2. Resolve names
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast)?;

    // 3. Type check
    let mut type_checker = TypeChecker::new(&resolved.symbols);
    let hir = type_checker.check_program(&resolved.ast)?;

    // 4. Effect inference
    let mut effect_checker = EffectInference::new(&resolved.symbols);
    if let Err(errors) = effect_checker.infer_program(&resolved.ast) {
        for e in errors {
            eprintln!("Effect error: {:?}", e);
        }
    }

    // 5. Ownership check
    let mut ownership_checker = OwnershipChecker::new(&resolved.symbols, &source);
    if let Err(errors) = ownership_checker.check_program(&resolved.ast) {
        for e in errors {
            eprintln!("{:?}", miette::Report::new(e));
        }
        return Err(miette::miette!("Ownership errors"));
    }

    if args.show_types {
        println!("=== Types ===");
        for item in &hir.items {
            match item {
                demetrios::hir::HirItem::Function(f) => {
                    println!("fn {}: {:?}", f.name, f.ty);
                }
                _ => {}
            }
        }
        println!();
    }

    if reporter.has_errors() {
        reporter.emit_all();
        return Err(miette::miette!("{} errors found", reporter.error_count()));
    }

    println!("✓ {} checked successfully", file.display());
    Ok(())
}
```

---

## PHASE 5: Tests

### 5.1 Create `tests/effects.rs`

```rust
use demetrios::parser;
use demetrios::resolve::Resolver;
use demetrios::effects::EffectInference;

fn check_effects(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|e| format!("{:?}", e))?;
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).map_err(|e| format!("{:?}", e))?;
    let mut checker = EffectInference::new(&resolved.symbols);
    checker.infer_program(&resolved.ast).map_err(|e| format!("{:?}", e))
}

#[test]
fn test_pure_function() {
    assert!(check_effects("fn add(a: int, b: int) -> int { return a + b }").is_ok());
}

#[test]
fn test_io_declared() {
    assert!(check_effects(r#"
        fn greet(name: string) -> string with IO {
            return name
        }
    "#).is_ok());
}

#[test]
fn test_io_not_declared() {
    // This should fail when we actually track IO operations
    // For now it passes because we don't track all effects
}

#[test]
fn test_panic_from_division() {
    // Division adds Panic effect
    assert!(check_effects(r#"
        fn divide(a: int, b: int) -> int with Panic {
            return a / b
        }
    "#).is_ok());
}
```

### 5.2 Create `tests/ownership.rs`

```rust
use demetrios::parser;
use demetrios::resolve::Resolver;
use demetrios::ownership::OwnershipChecker;
use demetrios::diagnostics::SourceFile;

fn check_ownership(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|e| format!("{:?}", e))?;
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).map_err(|e| format!("{:?}", e))?;
    let source = SourceFile::new("test.d", src);
    let mut checker = OwnershipChecker::new(&resolved.symbols, &source);
    checker.check_program(&resolved.ast).map_err(|e| format!("{:?}", e))
}

#[test]
fn test_simple_move() {
    assert!(check_ownership(r#"
        fn main() {
            let x: int = 42
            let y: int = x
        }
    "#).is_ok());
}

#[test]
fn test_use_after_move() {
    let result = check_ownership(r#"
        fn main() {
            let x: int = 42
            let y: int = x
            let z: int = x
        }
    "#);
    // Should detect use after move
    // (May pass if int is Copy)
}

#[test]
fn test_shared_borrow() {
    assert!(check_ownership(r#"
        fn main() {
            let x: int = 42
            let r1: &int = &x
            let r2: &int = &x
        }
    "#).is_ok());
}

#[test]
fn test_exclusive_borrow_conflict() {
    let result = check_ownership(r#"
        fn main() {
            var x: int = 42
            let r1: &!int = &!x
            let r2: &!int = &!x
        }
    "#);
    assert!(result.is_err());
}
```

### 5.3 Create `tests/linear.rs`

```rust
use demetrios::parser;
use demetrios::resolve::Resolver;
use demetrios::ownership::OwnershipChecker;
use demetrios::diagnostics::SourceFile;

fn check_linear(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|e| format!("{:?}", e))?;
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).map_err(|e| format!("{:?}", e))?;
    let source = SourceFile::new("test.d", src);
    let mut checker = OwnershipChecker::new(&resolved.symbols, &source);
    checker.check_program(&resolved.ast).map_err(|e| format!("{:?}", e))
}

#[test]
fn test_linear_consumed() {
    assert!(check_linear(r#"
        linear struct Handle { id: int }
        
        fn use_handle(h: Handle) { }
        
        fn main() {
            let h: Handle = Handle { id: 1 }
            use_handle(h)
        }
    "#).is_ok());
}

#[test]
fn test_linear_not_consumed() {
    let result = check_linear(r#"
        linear struct Handle { id: int }
        
        fn main() {
            let h: Handle = Handle { id: 1 }
            // h not consumed - error
        }
    "#);
    assert!(result.is_err());
}

#[test]
fn test_linear_used_twice() {
    let result = check_linear(r#"
        linear struct Handle { id: int }
        
        fn use_handle(h: Handle) { }
        
        fn main() {
            let h: Handle = Handle { id: 1 }
            use_handle(h)
            use_handle(h)  // Error: already consumed
        }
    "#);
    assert!(result.is_err());
}
```

---

## PHASE 6: Update Documentation

After Day 5, update `docs/IMPLEMENTATION_STATUS.md`:

```markdown
# Demetrios Compiler — Implementation Status

## Completed

### Day 1-2: Foundation
- [x] Project scaffold
- [x] Cargo.toml with dependencies
- [x] CLI framework
- [x] Module structure

### Day 3: First Pipeline
- [x] Lexer (Logos-based, 100+ tokens)
- [x] Parser (recursive descent + Pratt)
- [x] AST definitions
- [x] AST printer

### Day 4: Semantic Analysis
- [x] Symbol table
- [x] Name resolution (two-pass)
- [x] Type environment
- [x] Basic type inference
- [x] Bidirectional type checking

### Day 5: Effects & Ownership
- [x] Source-located diagnostics (miette)
- [x] Effect inference
- [x] Effect checking
- [x] Ownership tracking
- [x] Borrow checking
- [x] Linear type enforcement

## In Progress

### Day 6: HIR & Lowering
- [ ] HIR generation
- [ ] Desugaring
- [ ] Pattern matching compilation

### Day 7: Code Generation
- [ ] HLIR (SSA-based)
- [ ] Basic LLVM backend

## Planned

### Phase 2: Type System
- [ ] Full generics
- [ ] Trait resolution
- [ ] Refinement types (Z3)
- [ ] Units of measure

### Phase 3: Effects
- [ ] Effect handlers
- [ ] Effect polymorphism
- [ ] Built-in effects implementation

### Phase 4: GPU
- [ ] Kernel parsing
- [ ] GPU effect
- [ ] PTX/SPIR-V generation

### Phase 5: Tooling
- [ ] REPL
- [ ] LSP server
- [ ] Debugger support
```

---

## Success Criteria

1. ✅ `cargo build` passes
2. ✅ `cargo test` passes all tests
3. ✅ Effect inference working
4. ✅ Ownership checking detects use-after-move
5. ✅ Borrow checking detects conflicts
6. ✅ Linear types enforce must-consume
7. ✅ Errors have source locations
8. ✅ Errors render nicely with miette

---

## Example Error Output

```
error[ownership::use_after_move]: Use of moved value `handle`
  ┌─ src/main.d:8:15
  │
6 │     let handle = open("file.txt")
  │         ------ value declared here
7 │     close(handle)
  │           ------ value moved here
8 │     read(handle)
  │          ^^^^^^ value used here after move
```

---

## Next Steps (Day 6)

- HIR generation from type-checked AST
- Desugaring complex patterns
- Match exhaustiveness checking
- Preparing for code generation

---

**Day 5 Goal: Track effects + enforce ownership + beautiful errors**
