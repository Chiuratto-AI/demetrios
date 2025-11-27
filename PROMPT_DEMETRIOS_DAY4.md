# Demetrios (D) Compiler — Day 4: Name Resolution & Bidirectional Type Checking

## Context

Day 1: Scaffold structure
Day 2: Stub files, cargo build passes
Day 3: First working pipeline (lex → parse → AST → basic type check)
Day 4: **Proper name resolution + bidirectional type checking**

## Repository

```
/mnt/e/workspace/demetrios/
```

## Today's Mission

1. **Symbol table** with proper scoping
2. **Name resolution pass** (AST → Resolved AST)
3. **Bidirectional type checking** (check + infer modes)
4. **Better error messages** with source spans
5. **Effect tracking** in function signatures

---

## PHASE 1: Symbol Table

### 1.1 Create `src/resolve/mod.rs`

```rust
//! Name resolution pass
//!
//! Resolves all names in the AST to their definitions.
//! Produces a resolved AST where every name has a unique DefId.

mod symbols;
mod resolver;

pub use symbols::{Symbol, SymbolTable, DefId, DefKind, Scope};
pub use resolver::{Resolver, ResolvedAst};
```

### 1.2 Create `src/resolve/symbols.rs`

```rust
//! Symbol table implementation

use std::collections::HashMap;
use crate::ast::NodeId;

/// Unique definition ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

impl DefId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Kind of definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    /// Function definition
    Function,
    /// Variable (let/var binding)
    Variable { mutable: bool },
    /// Function parameter
    Parameter,
    /// Struct type
    Struct { is_linear: bool, is_affine: bool },
    /// Enum type
    Enum,
    /// Enum variant
    Variant,
    /// Type alias
    TypeAlias,
    /// Type parameter (generic)
    TypeParam,
    /// Effect definition
    Effect,
    /// Effect operation
    EffectOp,
    /// Constant
    Const,
    /// Module
    Module,
    /// Trait
    Trait,
    /// Field (struct field)
    Field,
    /// Kernel function
    Kernel,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Unique ID
    pub def_id: DefId,
    /// Name as string
    pub name: String,
    /// Kind of definition
    pub kind: DefKind,
    /// Original AST node
    pub node_id: NodeId,
    /// Span in source
    pub span: crate::Span,
    /// Parent scope (for nested items)
    pub parent: Option<DefId>,
}

/// Scope level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Module/file level
    Module,
    /// Function body
    Function,
    /// Block (if, loop, etc.)
    Block,
    /// Struct/enum definition
    TypeDef,
    /// Impl block
    Impl,
}

/// A single scope
#[derive(Debug)]
pub struct Scope {
    pub kind: ScopeKind,
    /// Names defined in this scope
    pub names: HashMap<String, DefId>,
    /// Type names (separate namespace)
    pub types: HashMap<String, DefId>,
    /// Parent scope DefId (for functions/methods)
    pub parent_def: Option<DefId>,
}

impl Scope {
    pub fn new(kind: ScopeKind, parent_def: Option<DefId>) -> Self {
        Self {
            kind,
            names: HashMap::new(),
            types: HashMap::new(),
            parent_def,
        }
    }
}

/// Symbol table with scoped lookups
pub struct SymbolTable {
    /// All symbols by DefId
    symbols: HashMap<DefId, Symbol>,
    /// Scope stack
    scopes: Vec<Scope>,
    /// Next DefId
    next_id: u32,
    /// NodeId → DefId mapping (for definitions)
    node_to_def: HashMap<NodeId, DefId>,
    /// NodeId → DefId mapping (for references)
    node_to_ref: HashMap<NodeId, DefId>,
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut table = Self {
            symbols: HashMap::new(),
            scopes: Vec::new(),
            next_id: 0,
            node_to_def: HashMap::new(),
            node_to_ref: HashMap::new(),
        };
        // Start with module scope
        table.push_scope(ScopeKind::Module, None);
        // Register built-in types
        table.register_builtins();
        table
    }

    fn register_builtins(&mut self) {
        // Built-in types
        let builtins = [
            "int", "i8", "i16", "i32", "i64", "i128",
            "uint", "u8", "u16", "u32", "u64", "u128",
            "f32", "f64", "bool", "char", "string",
        ];
        
        for name in builtins {
            let def_id = self.fresh_def_id();
            self.define_type(name.to_string(), def_id);
            self.symbols.insert(def_id, Symbol {
                def_id,
                name: name.to_string(),
                kind: DefKind::TypeAlias, // Built-in as pseudo-alias
                node_id: NodeId(0),
                span: crate::Span::default(),
                parent: None,
            });
        }
    }

    /// Generate fresh DefId
    pub fn fresh_def_id(&mut self) -> DefId {
        let id = DefId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Push new scope
    pub fn push_scope(&mut self, kind: ScopeKind, parent_def: Option<DefId>) {
        self.scopes.push(Scope::new(kind, parent_def));
    }

    /// Pop scope, returns undefined linear variables (for checking)
    pub fn pop_scope(&mut self) -> Option<Scope> {
        self.scopes.pop()
    }

    /// Define a value name in current scope
    pub fn define(&mut self, name: String, def_id: DefId) -> Result<(), String> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.names.contains_key(&name) {
                return Err(format!("Duplicate definition: {}", name));
            }
            scope.names.insert(name, def_id);
            Ok(())
        } else {
            Err("No scope".to_string())
        }
    }

    /// Define a type name in current scope
    pub fn define_type(&mut self, name: String, def_id: DefId) -> Result<(), String> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.types.contains_key(&name) {
                return Err(format!("Duplicate type: {}", name));
            }
            scope.types.insert(name, def_id);
            Ok(())
        } else {
            Err("No scope".to_string())
        }
    }

    /// Look up a value name
    pub fn lookup(&self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&def_id) = scope.names.get(name) {
                return Some(def_id);
            }
        }
        None
    }

    /// Look up a type name
    pub fn lookup_type(&self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&def_id) = scope.types.get(name) {
                return Some(def_id);
            }
        }
        None
    }

    /// Get symbol by DefId
    pub fn get(&self, def_id: DefId) -> Option<&Symbol> {
        self.symbols.get(&def_id)
    }

    /// Insert symbol
    pub fn insert(&mut self, symbol: Symbol) {
        let def_id = symbol.def_id;
        let node_id = symbol.node_id;
        self.node_to_def.insert(node_id, def_id);
        self.symbols.insert(def_id, symbol);
    }

    /// Record a reference from NodeId to DefId
    pub fn record_ref(&mut self, node_id: NodeId, def_id: DefId) {
        self.node_to_ref.insert(node_id, def_id);
    }

    /// Get DefId for a definition node
    pub fn def_for_node(&self, node_id: NodeId) -> Option<DefId> {
        self.node_to_def.get(&node_id).copied()
    }

    /// Get DefId for a reference node
    pub fn ref_for_node(&self, node_id: NodeId) -> Option<DefId> {
        self.node_to_ref.get(&node_id).copied()
    }

    /// Current scope depth
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Check if we're at module level
    pub fn at_module_level(&self) -> bool {
        self.scopes.len() == 1
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_nesting() {
        let mut table = SymbolTable::new();
        
        let def1 = table.fresh_def_id();
        table.define("x".into(), def1).unwrap();
        
        table.push_scope(ScopeKind::Block, None);
        let def2 = table.fresh_def_id();
        table.define("y".into(), def2).unwrap();
        
        // Both visible
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_some());
        
        table.pop_scope();
        
        // Only x visible
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_none());
    }

    #[test]
    fn test_shadowing() {
        let mut table = SymbolTable::new();
        
        let def1 = table.fresh_def_id();
        table.define("x".into(), def1).unwrap();
        
        table.push_scope(ScopeKind::Block, None);
        let def2 = table.fresh_def_id();
        table.define("x".into(), def2).unwrap(); // Shadow
        
        assert_eq!(table.lookup("x"), Some(def2));
        
        table.pop_scope();
        assert_eq!(table.lookup("x"), Some(def1));
    }
}
```

### 1.3 Create `src/resolve/resolver.rs`

```rust
//! Name resolution pass

use crate::ast::*;
use crate::Span;
use super::symbols::*;
use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

/// Resolution error
#[derive(Error, Debug, Diagnostic)]
pub enum ResolveError {
    #[error("Undefined variable: {name}")]
    UndefinedVar {
        name: String,
        #[label("not found in scope")]
        span: SourceSpan,
    },

    #[error("Undefined type: {name}")]
    UndefinedType {
        name: String,
        #[label("type not found")]
        span: SourceSpan,
    },

    #[error("Duplicate definition: {name}")]
    DuplicateDef {
        name: String,
        #[label("already defined")]
        span: SourceSpan,
    },

    #[error("Cannot use {name} as a value")]
    NotAValue {
        name: String,
        #[label("this is a type, not a value")]
        span: SourceSpan,
    },

    #[error("Cannot use {name} as a type")]
    NotAType {
        name: String,
        #[label("this is a value, not a type")]
        span: SourceSpan,
    },
}

/// Resolved AST (AST + symbol table)
pub struct ResolvedAst {
    pub ast: Ast,
    pub symbols: SymbolTable,
}

/// Name resolver
pub struct Resolver {
    symbols: SymbolTable,
    errors: Vec<ResolveError>,
    /// String interner (temporary - use real interner later)
    names: std::collections::HashMap<NodeId, String>,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            errors: Vec::new(),
            names: std::collections::HashMap::new(),
        }
    }

    /// Resolve all names in the AST
    pub fn resolve(mut self, ast: Ast) -> Result<ResolvedAst> {
        // First pass: collect all top-level definitions
        for item in &ast.items {
            self.collect_item(item);
        }

        // Second pass: resolve bodies
        for item in &ast.items {
            self.resolve_item(item);
        }

        if !self.errors.is_empty() {
            // Return first error for now
            return Err(miette::miette!("Resolution errors: {:?}", self.errors));
        }

        Ok(ResolvedAst {
            ast,
            symbols: self.symbols,
        })
    }

    /// First pass: collect definitions
    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => {
                self.define_function(f);
            }
            Item::Struct(s) => {
                self.define_struct(s);
            }
            Item::Enum(e) => {
                self.define_enum(e);
            }
            Item::TypeAlias(t) => {
                self.define_type_alias(t);
            }
            Item::Const(c) => {
                self.define_const(c);
            }
            Item::Effect(e) => {
                self.define_effect(e);
            }
            Item::Kernel(k) => {
                self.define_kernel(k);
            }
            Item::Trait(t) => {
                self.define_trait(t);
            }
            _ => {}
        }
    }

    fn define_function(&mut self, f: &FnDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(f.name);
        
        if let Err(e) = self.symbols.define(name.clone(), def_id) {
            self.errors.push(ResolveError::DuplicateDef {
                name: name.clone(),
                span: self.span_to_source(f.span),
            });
            return;
        }

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Function,
            node_id: f.id,
            span: f.span,
            parent: None,
        });
    }

    fn define_struct(&mut self, s: &StructDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(s.name);

        if let Err(_) = self.symbols.define_type(name.clone(), def_id) {
            self.errors.push(ResolveError::DuplicateDef {
                name: name.clone(),
                span: self.span_to_source(s.span),
            });
            return;
        }

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Struct {
                is_linear: s.modifiers.linear,
                is_affine: s.modifiers.affine,
            },
            node_id: s.id,
            span: s.span,
            parent: None,
        });
    }

    fn define_enum(&mut self, e: &EnumDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(e.name);

        let _ = self.symbols.define_type(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Enum,
            node_id: e.id,
            span: e.span,
            parent: None,
        });
    }

    fn define_type_alias(&mut self, t: &TypeAlias) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(t.name);

        let _ = self.symbols.define_type(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::TypeAlias,
            node_id: t.id,
            span: t.span,
            parent: None,
        });
    }

    fn define_const(&mut self, c: &ConstDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(c.name);

        let _ = self.symbols.define(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Const,
            node_id: c.id,
            span: c.span,
            parent: None,
        });
    }

    fn define_effect(&mut self, e: &EffectDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(e.name);

        // Effects go in type namespace
        let _ = self.symbols.define_type(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Effect,
            node_id: e.id,
            span: e.span,
            parent: None,
        });
    }

    fn define_kernel(&mut self, k: &KernelDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(k.name);

        let _ = self.symbols.define(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Kernel,
            node_id: k.id,
            span: k.span,
            parent: None,
        });
    }

    fn define_trait(&mut self, t: &TraitDef) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(t.name);

        let _ = self.symbols.define_type(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Trait,
            node_id: t.id,
            span: t.span,
            parent: None,
        });
    }

    /// Second pass: resolve bodies
    fn resolve_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.resolve_function(f),
            Item::Struct(s) => self.resolve_struct(s),
            Item::Kernel(k) => self.resolve_kernel(k),
            _ => {}
        }
    }

    fn resolve_function(&mut self, f: &FnDef) {
        let fn_def_id = self.symbols.def_for_node(f.id);
        self.symbols.push_scope(ScopeKind::Function, fn_def_id);

        // Resolve parameters
        for param in &f.params {
            self.resolve_param(param);
        }

        // Resolve return type
        if let Some(ref ret_ty) = f.return_type {
            self.resolve_type(ret_ty);
        }

        // Resolve body
        if let Some(ref body) = f.body {
            self.resolve_block(body);
        }

        self.symbols.pop_scope();
    }

    fn resolve_struct(&mut self, s: &StructDef) {
        self.symbols.push_scope(ScopeKind::TypeDef, None);

        // Resolve type parameters
        for tp in &s.type_params {
            let def_id = self.symbols.fresh_def_id();
            let name = self.get_name(tp.name);
            let _ = self.symbols.define_type(name.clone(), def_id);
            self.symbols.insert(Symbol {
                def_id,
                name,
                kind: DefKind::TypeParam,
                node_id: tp.id,
                span: tp.span,
                parent: None,
            });
        }

        // Resolve field types
        for field in &s.fields {
            self.resolve_type(&field.ty);
        }

        self.symbols.pop_scope();
    }

    fn resolve_kernel(&mut self, k: &KernelDef) {
        let kern_def_id = self.symbols.def_for_node(k.id);
        self.symbols.push_scope(ScopeKind::Function, kern_def_id);

        for param in &k.params {
            self.resolve_param(param);
        }

        if let Some(ref ret_ty) = k.return_type {
            self.resolve_type(ret_ty);
        }

        self.resolve_block(&k.body);

        self.symbols.pop_scope();
    }

    fn resolve_param(&mut self, param: &Param) {
        let def_id = self.symbols.fresh_def_id();
        let name = self.get_name(param.name);

        let _ = self.symbols.define(name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name,
            kind: DefKind::Parameter,
            node_id: param.id,
            span: param.span,
            parent: None,
        });

        self.resolve_type(&param.ty);
    }

    fn resolve_type(&mut self, ty: &Type) {
        match ty {
            Type::Named(id) => {
                let name = self.get_name(*id);
                if let Some(def_id) = self.symbols.lookup_type(&name) {
                    self.symbols.record_ref(*id, def_id);
                } else {
                    self.errors.push(ResolveError::UndefinedType {
                        name,
                        span: SourceSpan::from(0..1), // TODO: proper span
                    });
                }
            }
            Type::Generic(id, args) => {
                let name = self.get_name(*id);
                if let Some(def_id) = self.symbols.lookup_type(&name) {
                    self.symbols.record_ref(*id, def_id);
                }
                for arg in args {
                    self.resolve_type(arg);
                }
            }
            Type::Ref(_, inner) | Type::Own(inner) | Type::Linear(inner) | Type::Affine(inner) => {
                self.resolve_type(inner);
            }
            Type::Tuple(types) => {
                for t in types {
                    self.resolve_type(t);
                }
            }
            Type::Array(inner, _) => {
                self.resolve_type(inner);
            }
            Type::Function(params, ret) => {
                for p in params {
                    self.resolve_type(p);
                }
                self.resolve_type(ret);
            }
            Type::Refined(inner, _) => {
                self.resolve_type(inner);
                // TODO: resolve predicate variables
            }
            Type::UnitAnnotated(inner, _) => {
                self.resolve_type(inner);
            }
            _ => {}
        }
    }

    fn resolve_block(&mut self, block: &Block) {
        self.symbols.push_scope(ScopeKind::Block, None);

        for stmt in &block.stmts {
            self.resolve_stmt(stmt);
        }

        self.symbols.pop_scope();
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(l) => {
                // Resolve initializer first (before binding)
                if let Some(ref init) = l.init {
                    self.resolve_expr(init);
                }
                if let Some(ref ty) = l.ty {
                    self.resolve_type(ty);
                }

                // Now bind the variable
                let def_id = self.symbols.fresh_def_id();
                let name = self.get_name(l.name);

                let _ = self.symbols.define(name.clone(), def_id);

                self.symbols.insert(Symbol {
                    def_id,
                    name,
                    kind: DefKind::Variable { mutable: l.mutable },
                    node_id: l.id,
                    span: l.span,
                    parent: None,
                });
            }
            Stmt::Return(r) => {
                if let Some(ref val) = r.value {
                    self.resolve_expr(val);
                }
            }
            Stmt::Expr(e) => {
                self.resolve_expr(&e.expr);
            }
            Stmt::If(i) => {
                self.resolve_expr(&i.condition);
                self.resolve_block(&i.then_block);
                if let Some(ref else_block) = i.else_block {
                    self.resolve_block(else_block);
                }
            }
            Stmt::While(w) => {
                self.resolve_expr(&w.condition);
                self.resolve_block(&w.body);
            }
            Stmt::For(f) => {
                self.symbols.push_scope(ScopeKind::Block, None);
                self.resolve_pattern(&f.pattern);
                self.resolve_expr(&f.iter);
                self.resolve_block(&f.body);
                self.symbols.pop_scope();
            }
            Stmt::Match(m) => {
                self.resolve_expr(&m.scrutinee);
                for arm in &m.arms {
                    self.symbols.push_scope(ScopeKind::Block, None);
                    self.resolve_pattern(&arm.pattern);
                    self.resolve_expr(&arm.body);
                    self.symbols.pop_scope();
                }
            }
            Stmt::Loop(l) => {
                self.resolve_block(&l.body);
            }
            Stmt::Handle(h) => {
                // Resolve handler cases
                self.resolve_block(&h.body);
            }
            _ => {}
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Path(p) => {
                // Single-segment path = variable lookup
                if p.segments.len() == 1 {
                    let name = self.get_name(p.segments[0]);
                    if let Some(def_id) = self.symbols.lookup(&name) {
                        self.symbols.record_ref(p.segments[0], def_id);
                    } else {
                        self.errors.push(ResolveError::UndefinedVar {
                            name,
                            span: SourceSpan::from(0..1), // TODO
                        });
                    }
                }
                // TODO: multi-segment paths (module::item)
            }
            Expr::Binary(b) => {
                self.resolve_expr(&b.left);
                self.resolve_expr(&b.right);
            }
            Expr::Unary(u) => {
                self.resolve_expr(&u.operand);
            }
            Expr::Call(c) => {
                self.resolve_expr(&c.callee);
                for arg in &c.args {
                    self.resolve_expr(arg);
                }
            }
            Expr::MethodCall(m) => {
                self.resolve_expr(&m.receiver);
                for arg in &m.args {
                    self.resolve_expr(arg);
                }
            }
            Expr::Index(i) => {
                self.resolve_expr(&i.expr);
                self.resolve_expr(&i.index);
            }
            Expr::Field(f) => {
                self.resolve_expr(&f.expr);
                // Field name resolved during type checking
            }
            Expr::Cast(c) => {
                self.resolve_expr(&c.expr);
                self.resolve_type(&c.ty);
            }
            Expr::Tuple(t) => {
                for e in &t.elements {
                    self.resolve_expr(e);
                }
            }
            Expr::Array(a) => {
                for e in &a.elements {
                    self.resolve_expr(e);
                }
            }
            Expr::Struct(s) => {
                // Resolve struct type
                let name = self.get_name(s.name);
                if let Some(def_id) = self.symbols.lookup_type(&name) {
                    self.symbols.record_ref(s.name, def_id);
                }
                for field in &s.fields {
                    self.resolve_expr(&field.value);
                }
            }
            Expr::If(i) => {
                self.resolve_expr(&i.condition);
                self.resolve_block(&i.then_block);
                if let Some(ref else_block) = i.else_block {
                    self.resolve_block(else_block);
                }
            }
            Expr::Match(m) => {
                self.resolve_expr(&m.scrutinee);
                for arm in &m.arms {
                    self.symbols.push_scope(ScopeKind::Block, None);
                    self.resolve_pattern(&arm.pattern);
                    self.resolve_expr(&arm.body);
                    self.symbols.pop_scope();
                }
            }
            Expr::Block(b) => {
                self.resolve_block(b);
            }
            Expr::Lambda(l) => {
                self.symbols.push_scope(ScopeKind::Function, None);
                for param in &l.params {
                    self.resolve_param(param);
                }
                self.resolve_expr(&l.body);
                self.symbols.pop_scope();
            }
            Expr::With(w) => {
                self.resolve_expr(&w.handler);
                self.resolve_expr(&w.body);
            }
            Expr::Resume(r) => {
                if let Some(ref val) = r.value {
                    self.resolve_expr(val);
                }
            }
            _ => {}
        }
    }

    fn resolve_pattern(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Binding(b) => {
                let def_id = self.symbols.fresh_def_id();
                let name = self.get_name(b.name);
                
                let _ = self.symbols.define(name.clone(), def_id);
                
                self.symbols.insert(Symbol {
                    def_id,
                    name,
                    kind: DefKind::Variable { mutable: b.mutable },
                    node_id: b.id,
                    span: b.span,
                    parent: None,
                });
            }
            Pattern::Tuple(t) => {
                for p in &t.patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Struct(s) => {
                // Resolve struct type
                let name = self.get_name(s.name);
                if let Some(def_id) = self.symbols.lookup_type(&name) {
                    self.symbols.record_ref(s.name, def_id);
                }
                for field in &s.fields {
                    self.resolve_pattern(&field.pattern);
                }
            }
            Pattern::Variant(v) => {
                // TODO: resolve enum variant
                if let Some(ref inner) = v.inner {
                    self.resolve_pattern(inner);
                }
            }
            Pattern::Array(a) => {
                for p in &a.patterns {
                    self.resolve_pattern(p);
                }
            }
            Pattern::Or(o) => {
                for p in &o.patterns {
                    self.resolve_pattern(p);
                }
            }
            _ => {}
        }
    }

    /// Get name string from NodeId (temporary until proper interner)
    fn get_name(&self, id: NodeId) -> String {
        self.names.get(&id).cloned().unwrap_or_else(|| format!("${}", id.0))
    }

    fn span_to_source(&self, span: Span) -> SourceSpan {
        SourceSpan::from(span.start as usize..span.end as usize)
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}
```

### 1.4 Update `src/lib.rs`

Add the resolve module:

```rust
pub mod resolve;

// ... other modules
```

---

## PHASE 2: String Interner

### 2.1 Create `src/intern.rs`

```rust
//! String interner for efficient name handling

use std::collections::HashMap;

/// Interned string ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedStr(pub u32);

/// String interner
pub struct Interner {
    map: HashMap<String, InternedStr>,
    strings: Vec<String>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Intern a string
    pub fn intern(&mut self, s: &str) -> InternedStr {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = InternedStr(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }

    /// Get string by ID
    pub fn get(&self, id: InternedStr) -> Option<&str> {
        self.strings.get(id.0 as usize).map(|s| s.as_str())
    }

    /// Get or empty string
    pub fn get_or_empty(&self, id: InternedStr) -> &str {
        self.get(id).unwrap_or("")
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern() {
        let mut interner = Interner::new();
        
        let a1 = interner.intern("hello");
        let a2 = interner.intern("hello");
        let b = interner.intern("world");
        
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_eq!(interner.get(a1), Some("hello"));
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod intern;
pub use intern::{Interner, InternedStr};
```

---

## PHASE 3: Bidirectional Type Checking

### 3.1 Update `src/check/mod.rs`

```rust
//! Bidirectional type checker
//!
//! Uses "check" mode (checking against known type) and "infer" mode (synthesizing type).

mod env;
mod infer;

use crate::ast;
use crate::hir;
use crate::resolve::{ResolvedAst, SymbolTable, DefId};
use crate::types::{Type, Effect, EffectSet, TypeVar, FnType, RefKind};
use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

pub use env::TypeEnv;
pub use infer::{unify, apply_subst, Subst};

/// Type error
#[derive(Error, Debug, Diagnostic)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected}, found {found}")]
    Mismatch {
        expected: String,
        found: String,
        #[label("expected {expected}")]
        span: SourceSpan,
    },

    #[error("Cannot unify {t1} with {t2}")]
    UnificationError {
        t1: String,
        t2: String,
        #[label("type mismatch")]
        span: SourceSpan,
    },

    #[error("Effect {effect} not handled")]
    UnhandledEffect {
        effect: String,
        #[label("effect escapes here")]
        span: SourceSpan,
    },

    #[error("Linear value {name} used more than once")]
    LinearUsedTwice {
        name: String,
        #[label("second use")]
        span: SourceSpan,
    },

    #[error("Linear value {name} not consumed")]
    LinearNotConsumed {
        name: String,
        #[label("goes out of scope")]
        span: SourceSpan,
    },

    #[error("Unit mismatch: expected {expected}, found {found}")]
    UnitMismatch {
        expected: String,
        found: String,
        #[label("incompatible units")]
        span: SourceSpan,
    },
}

/// Bidirectional type checker
pub struct TypeChecker<'a> {
    /// Symbol table from resolution
    symbols: &'a SymbolTable,
    /// Type environment
    env: TypeEnv,
    /// Current substitution
    subst: Subst,
    /// Current effect set
    effects: EffectSet,
    /// Expected effects (from function signature)
    expected_effects: EffectSet,
    /// Next type variable
    next_tyvar: u32,
    /// Errors
    errors: Vec<TypeError>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(symbols: &'a SymbolTable) -> Self {
        Self {
            symbols,
            env: TypeEnv::new(),
            subst: Subst::new(),
            effects: EffectSet::new(),
            expected_effects: EffectSet::new(),
            next_tyvar: 0,
            errors: Vec::new(),
        }
    }

    /// Check entire program
    pub fn check_program(&mut self, ast: &ast::Ast) -> Result<hir::Hir> {
        let mut items = Vec::new();

        for item in &ast.items {
            if let Some(hir_item) = self.check_item(item)? {
                items.push(hir_item);
            }
        }

        if !self.errors.is_empty() {
            return Err(miette::miette!("Type errors found"));
        }

        Ok(hir::Hir { items })
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

        // Build expected return type
        let return_ty = if let Some(ref ty) = f.return_type {
            self.resolve_ast_type(ty)?
        } else {
            Type::Unit
        };

        // Collect expected effects
        self.expected_effects = EffectSet::new();
        for eff in &f.effects {
            self.expected_effects.insert(self.resolve_effect_ref(eff));
        }

        // Add parameters to environment
        let mut param_types = Vec::new();
        for param in &f.params {
            let param_ty = self.resolve_ast_type(&param.ty)?;
            param_types.push(param_ty.clone());
            
            if let Some(def_id) = self.symbols.def_for_node(param.id) {
                self.env.bind_def(def_id, param_ty, false);
            }
        }

        // Check body
        let body_ty = if let Some(ref body) = f.body {
            self.check_block(body, &return_ty)?
        } else {
            Type::Unit
        };

        // Unify body type with return type
        if let Err(e) = unify(&body_ty, &return_ty) {
            self.errors.push(TypeError::Mismatch {
                expected: format!("{:?}", return_ty),
                found: format!("{:?}", body_ty),
                span: SourceSpan::from(0..1),
            });
        }

        // Check that all effects are declared
        for eff in self.effects.iter() {
            if !self.expected_effects.contains(eff) {
                self.errors.push(TypeError::UnhandledEffect {
                    effect: format!("{:?}", eff),
                    span: SourceSpan::from(0..1),
                });
            }
        }

        self.env.pop_scope();

        Ok(hir::HirFn {
            id: f.id,
            name: String::new(),
            ty: hir::HirFnType {
                params: param_types.into_iter().map(|t| self.type_to_hir(&t)).collect(),
                return_type: Box::new(self.type_to_hir(&return_ty)),
                effects: self.expected_effects.iter().map(|e| self.effect_to_hir(e)).collect(),
            },
            body: hir::HirBlock {
                stmts: Vec::new(),
                ty: self.type_to_hir(&return_ty),
            },
        })
    }

    fn check_struct(&mut self, s: &ast::StructDef) -> Result<hir::HirStruct> {
        let mut fields = Vec::new();

        for field in &s.fields {
            let ty = self.resolve_ast_type(&field.ty)?;
            fields.push(hir::HirField {
                name: String::new(), // TODO
                ty: self.type_to_hir(&ty),
            });
        }

        Ok(hir::HirStruct {
            id: s.id,
            name: String::new(),
            fields,
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
        })
    }

    /// Check block against expected type
    fn check_block(&mut self, block: &ast::Block, expected: &Type) -> Result<Type> {
        self.env.push_scope();

        let mut last_ty = Type::Unit;
        for stmt in &block.stmts {
            last_ty = self.check_stmt(stmt)?;
        }

        // Check linear values consumed
        let scope = self.env.pop_scope();
        for (name, binding) in scope {
            if binding.linear && !binding.used {
                self.errors.push(TypeError::LinearNotConsumed {
                    name,
                    span: SourceSpan::from(0..1),
                });
            }
        }

        Ok(last_ty)
    }

    fn check_stmt(&mut self, stmt: &ast::Stmt) -> Result<Type> {
        match stmt {
            ast::Stmt::Let(l) => {
                // Infer initializer type
                let init_ty = if let Some(ref init) = l.init {
                    self.infer_expr(init)?
                } else {
                    self.fresh_tyvar()
                };

                // Check against declared type if present
                let final_ty = if let Some(ref ty) = l.ty {
                    let declared = self.resolve_ast_type(ty)?;
                    let s = unify(&init_ty, &declared)
                        .map_err(|e| miette::miette!("Type error in let: {}", e))?;
                    self.apply_subst_to_env(&s);
                    apply_subst(&s, &declared)
                } else {
                    init_ty
                };

                // Bind variable
                if let Some(def_id) = self.symbols.def_for_node(l.id) {
                    let is_linear = matches!(&final_ty, Type::Linear(_));
                    self.env.bind_def(def_id, final_ty, is_linear);
                }

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

            ast::Stmt::If(i) => {
                let cond_ty = self.infer_expr(&i.condition)?;
                let _ = unify(&cond_ty, &Type::Bool)
                    .map_err(|_| miette::miette!("Condition must be bool"))?;

                let then_ty = self.check_block(&i.then_block, &Type::Unknown)?;
                
                if let Some(ref else_block) = i.else_block {
                    let else_ty = self.check_block(else_block, &then_ty)?;
                    let _ = unify(&then_ty, &else_ty)?;
                }

                Ok(then_ty)
            }

            _ => Ok(Type::Unit),
        }
    }

    /// Infer expression type (synthesis mode)
    fn infer_expr(&mut self, expr: &ast::Expr) -> Result<Type> {
        match expr {
            ast::Expr::Literal(lit) => self.infer_literal(lit),
            
            ast::Expr::Path(path) => {
                if path.segments.len() == 1 {
                    let node_id = path.segments[0];
                    if let Some(def_id) = self.symbols.ref_for_node(node_id) {
                        if let Some(ty) = self.env.lookup_def(def_id) {
                            // Mark as used for linearity
                            self.env.mark_used(def_id);
                            return Ok(ty.clone());
                        }
                    }
                }
                Ok(Type::Unknown)
            }

            ast::Expr::Binary(bin) => {
                let left_ty = self.infer_expr(&bin.left)?;
                let right_ty = self.infer_expr(&bin.right)?;

                let s = unify(&left_ty, &right_ty)
                    .map_err(|e| miette::miette!("Binary op type error: {}", e))?;
                self.apply_subst_to_env(&s);

                let operand_ty = apply_subst(&s, &left_ty);

                Ok(match bin.op {
                    ast::BinOp::Eq | ast::BinOp::Ne |
                    ast::BinOp::Lt | ast::BinOp::Le |
                    ast::BinOp::Gt | ast::BinOp::Ge |
                    ast::BinOp::And | ast::BinOp::Or => Type::Bool,
                    _ => operand_ty,
                })
            }

            ast::Expr::Unary(un) => {
                let inner_ty = self.infer_expr(&un.operand)?;
                Ok(match un.op {
                    ast::UnaryOp::Not => {
                        let _ = unify(&inner_ty, &Type::Bool)?;
                        Type::Bool
                    }
                    ast::UnaryOp::Neg => inner_ty,
                    ast::UnaryOp::Ref => Type::Ref(RefKind::Shared, Box::new(inner_ty)),
                    ast::UnaryOp::RefMut => Type::Ref(RefKind::Exclusive, Box::new(inner_ty)),
                    ast::UnaryOp::Deref => {
                        match inner_ty {
                            Type::Ref(_, inner) => *inner,
                            Type::Own(inner) => *inner,
                            _ => Type::Unknown,
                        }
                    }
                })
            }

            ast::Expr::Call(call) => {
                let callee_ty = self.infer_expr(&call.callee)?;
                
                match callee_ty {
                    Type::Fn(fn_ty) => {
                        // Check argument count
                        if call.args.len() != fn_ty.params.len() {
                            return Err(miette::miette!(
                                "Expected {} arguments, got {}",
                                fn_ty.params.len(),
                                call.args.len()
                            ));
                        }

                        // Check each argument
                        for (arg, param_ty) in call.args.iter().zip(&fn_ty.params) {
                            let arg_ty = self.infer_expr(arg)?;
                            let _ = unify(&arg_ty, param_ty)?;
                        }

                        // Add effects
                        self.effects = self.effects.union(&fn_ty.effects);

                        Ok(*fn_ty.return_type)
                    }
                    _ => Ok(Type::Unknown),
                }
            }

            ast::Expr::Block(block) => {
                self.check_block(block, &Type::Unknown)
            }

            ast::Expr::If(if_expr) => {
                let cond_ty = self.infer_expr(&if_expr.condition)?;
                let _ = unify(&cond_ty, &Type::Bool)?;

                let then_ty = self.check_block(&if_expr.then_block, &Type::Unknown)?;
                
                if let Some(ref else_block) = if_expr.else_block {
                    let else_ty = self.check_block(else_block, &then_ty)?;
                    let s = unify(&then_ty, &else_ty)?;
                    Ok(apply_subst(&s, &then_ty))
                } else {
                    Ok(Type::Unit)
                }
            }

            ast::Expr::Tuple(tup) => {
                let mut types = Vec::new();
                for elem in &tup.elements {
                    types.push(self.infer_expr(elem)?);
                }
                Ok(Type::Tuple(types))
            }

            ast::Expr::Array(arr) => {
                if arr.elements.is_empty() {
                    Ok(Type::Array(Box::new(self.fresh_tyvar()), 0))
                } else {
                    let first_ty = self.infer_expr(&arr.elements[0])?;
                    for elem in &arr.elements[1..] {
                        let elem_ty = self.infer_expr(elem)?;
                        let _ = unify(&first_ty, &elem_ty)?;
                    }
                    Ok(Type::Array(Box::new(first_ty), arr.elements.len()))
                }
            }

            _ => Ok(Type::Unknown),
        }
    }

    /// Check expression against expected type (checking mode)
    fn check_expr(&mut self, expr: &ast::Expr, expected: &Type) -> Result<()> {
        let inferred = self.infer_expr(expr)?;
        let _ = unify(&inferred, expected)
            .map_err(|e| miette::miette!("Type mismatch: {}", e))?;
        Ok(())
    }

    fn infer_literal(&self, lit: &ast::Literal) -> Result<Type> {
        Ok(match lit {
            ast::Literal::Int(_) => Type::Int,
            ast::Literal::Float(_) => Type::F64,
            ast::Literal::Bool(_) => Type::Bool,
            ast::Literal::String(_) => Type::String,
            ast::Literal::Char(_) => Type::Char,
            ast::Literal::Unit(_, _unit) => Type::F64, // TODO: with unit
        })
    }

    fn fresh_tyvar(&mut self) -> Type {
        let id = self.next_tyvar;
        self.next_tyvar += 1;
        Type::Var(TypeVar(id))
    }

    fn apply_subst_to_env(&mut self, s: &Subst) {
        self.env.apply_subst(s);
        self.subst = infer::compose_subst(self.subst.clone(), s.clone());
    }

    fn resolve_ast_type(&mut self, ty: &ast::Type) -> Result<Type> {
        Ok(match ty {
            ast::Type::Named(id) => {
                // Look up in symbol table
                if let Some(def_id) = self.symbols.ref_for_node(*id) {
                    Type::Named(crate::types::core::TypeId(def_id.0))
                } else {
                    // Built-in type check
                    Type::Unknown
                }
            }
            ast::Type::Unit => Type::Unit,
            ast::Type::Tuple(types) => {
                Type::Tuple(types.iter()
                    .map(|t| self.resolve_ast_type(t))
                    .collect::<Result<Vec<_>>>()?)
            }
            ast::Type::Ref(kind, inner) => {
                let inner_ty = self.resolve_ast_type(inner)?;
                Type::Ref(
                    match kind {
                        ast::RefKind::Shared => RefKind::Shared,
                        ast::RefKind::Exclusive => RefKind::Exclusive,
                    },
                    Box::new(inner_ty),
                )
            }
            ast::Type::Own(inner) => {
                Type::Own(Box::new(self.resolve_ast_type(inner)?))
            }
            ast::Type::Linear(inner) => {
                Type::Linear(Box::new(self.resolve_ast_type(inner)?))
            }
            ast::Type::Affine(inner) => {
                Type::Affine(Box::new(self.resolve_ast_type(inner)?))
            }
            ast::Type::Array(inner, size) => {
                Type::Array(Box::new(self.resolve_ast_type(inner)?), *size)
            }
            ast::Type::Infer => self.fresh_tyvar(),
            _ => Type::Unknown,
        })
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

    fn type_to_hir(&self, ty: &Type) -> hir::HirType {
        match ty {
            Type::Unit => hir::HirType::Unit,
            Type::Bool => hir::HirType::Primitive(hir::PrimitiveType::Bool),
            Type::Int => hir::HirType::Primitive(hir::PrimitiveType::Int),
            Type::I8 => hir::HirType::Primitive(hir::PrimitiveType::I8),
            Type::I16 => hir::HirType::Primitive(hir::PrimitiveType::I16),
            Type::I32 => hir::HirType::Primitive(hir::PrimitiveType::I32),
            Type::I64 => hir::HirType::Primitive(hir::PrimitiveType::I64),
            Type::F32 => hir::HirType::Primitive(hir::PrimitiveType::F32),
            Type::F64 => hir::HirType::Primitive(hir::PrimitiveType::F64),
            Type::String => hir::HirType::Primitive(hir::PrimitiveType::String),
            Type::Char => hir::HirType::Primitive(hir::PrimitiveType::Char),
            _ => hir::HirType::Error,
        }
    }

    fn effect_to_hir(&self, eff: &Effect) -> hir::HirEffect {
        match eff {
            Effect::IO => hir::HirEffect::IO,
            Effect::Mut => hir::HirEffect::Mut,
            Effect::Alloc => hir::HirEffect::Alloc,
            Effect::Panic => hir::HirEffect::Panic,
            Effect::Async => hir::HirEffect::Async,
            Effect::GPU => hir::HirEffect::GPU,
            Effect::Prob => hir::HirEffect::Prob,
            Effect::Div => hir::HirEffect::Div,
            Effect::Named(n) => hir::HirEffect::Named(n.clone()),
            Effect::Var(_) => hir::HirEffect::Named("?".into()),
        }
    }
}
```

### 3.2 Update `src/check/env.rs`

```rust
//! Type environment with DefId-based lookups

use crate::resolve::DefId;
use crate::types::Type;
use crate::check::infer::{apply_subst, Subst};
use std::collections::HashMap;

/// Binding information
#[derive(Clone)]
pub struct Binding {
    pub ty: Type,
    pub linear: bool,
    pub used: bool,
}

/// Type environment
pub struct TypeEnv {
    /// Scopes: DefId -> Binding
    scopes: Vec<HashMap<DefId, Binding>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) -> Vec<(String, Binding)> {
        // Return bindings for linearity checking
        self.scopes
            .pop()
            .map(|s| s.into_iter().map(|(id, b)| (format!("{:?}", id), b)).collect())
            .unwrap_or_default()
    }

    pub fn bind_def(&mut self, def_id: DefId, ty: Type, linear: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(def_id, Binding { ty, linear, used: false });
        }
    }

    pub fn lookup_def(&self, def_id: DefId) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.get(&def_id) {
                return Some(&binding.ty);
            }
        }
        None
    }

    pub fn mark_used(&mut self, def_id: DefId) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.get_mut(&def_id) {
                binding.used = true;
                return;
            }
        }
    }

    pub fn apply_subst(&mut self, subst: &Subst) {
        for scope in &mut self.scopes {
            for binding in scope.values_mut() {
                binding.ty = apply_subst(subst, &binding.ty);
            }
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## PHASE 4: Integration

### 4.1 Update `src/main.rs` for Full Pipeline

```rust
fn cmd_check(file: &Path, show_ast: bool, show_types: bool, show_resolved: bool) -> Result<()> {
    let source = std::fs::read_to_string(file)
        .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;
    
    // 1. Parse
    let ast = demetrios::parser::parse(&source)?;
    
    if show_ast {
        println!("=== AST ===");
        println!("{}", demetrios::ast::AstPrinter::print(&ast));
        println!();
    }
    
    // 2. Resolve names
    let resolver = demetrios::resolve::Resolver::new();
    let resolved = resolver.resolve(ast)?;
    
    if show_resolved {
        println!("=== Symbols ===");
        // Print symbol table summary
        println!("(symbol table resolved)");
        println!();
    }
    
    // 3. Type check
    let mut checker = demetrios::check::TypeChecker::new(&resolved.symbols);
    let hir = checker.check_program(&resolved.ast)?;
    
    if show_types {
        println!("=== Types ===");
        for item in &hir.items {
            match item {
                demetrios::hir::HirItem::Function(f) => {
                    println!("fn {}: {:?}", f.name, f.ty);
                }
                demetrios::hir::HirItem::Struct(s) => {
                    println!("struct {}: {} fields", s.name, s.fields.len());
                }
                _ => {}
            }
        }
        println!();
    }
    
    println!("✓ {} checked successfully", file.display());
    Ok(())
}
```

### 4.2 Add CLI Flags

```rust
#[derive(Parser)]
struct CheckArgs {
    file: PathBuf,
    #[arg(long)]
    show_ast: bool,
    #[arg(long)]
    show_types: bool,
    #[arg(long)]
    show_resolved: bool,
}
```

---

## PHASE 5: Tests

### 5.1 Create `tests/resolve.rs`

```rust
use demetrios::parser;
use demetrios::resolve::Resolver;

#[test]
fn test_resolve_function() {
    let src = "fn foo(x: int) -> int { return x }";
    let ast = parser::parse(src).unwrap();
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).unwrap();
    
    // Check that 'foo' is defined
    assert!(resolved.symbols.lookup("foo").is_some());
}

#[test]
fn test_resolve_variable() {
    let src = r#"
        fn main() -> int {
            let x: int = 42
            return x
        }
    "#;
    let ast = parser::parse(src).unwrap();
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).unwrap();
    
    assert!(resolved.symbols.lookup("main").is_some());
}

#[test]
fn test_undefined_variable() {
    let src = r#"
        fn main() -> int {
            return y
        }
    "#;
    let ast = parser::parse(src).unwrap();
    let resolver = Resolver::new();
    let result = resolver.resolve(ast);
    
    assert!(result.is_err());
}

#[test]
fn test_shadowing() {
    let src = r#"
        fn main() -> int {
            let x: int = 1
            let x: int = 2
            return x
        }
    "#;
    let ast = parser::parse(src).unwrap();
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).unwrap();
    
    // Should succeed (shadowing is allowed)
    assert!(resolved.symbols.lookup("main").is_some());
}
```

### 5.2 Create `tests/typecheck.rs`

```rust
use demetrios::parser;
use demetrios::resolve::Resolver;
use demetrios::check::TypeChecker;

fn check(src: &str) -> Result<(), String> {
    let ast = parser::parse(src).map_err(|e| format!("{:?}", e))?;
    let resolver = Resolver::new();
    let resolved = resolver.resolve(ast).map_err(|e| format!("{:?}", e))?;
    let mut checker = TypeChecker::new(&resolved.symbols);
    checker.check_program(&resolved.ast).map_err(|e| format!("{:?}", e))?;
    Ok(())
}

#[test]
fn test_simple_function() {
    assert!(check("fn main() -> int { return 42 }").is_ok());
}

#[test]
fn test_type_mismatch() {
    let result = check(r#"
        fn main() -> int {
            return true
        }
    "#);
    // Should fail: returning bool when int expected
    assert!(result.is_err());
}

#[test]
fn test_binary_ops() {
    assert!(check(r#"
        fn add(a: int, b: int) -> int {
            return a + b
        }
    "#).is_ok());
}

#[test]
fn test_comparison() {
    assert!(check(r#"
        fn is_positive(x: int) -> bool {
            return x > 0
        }
    "#).is_ok());
}

#[test]
fn test_if_expression() {
    assert!(check(r#"
        fn max(a: int, b: int) -> int {
            if a > b {
                return a
            } else {
                return b
            }
        }
    "#).is_ok());
}
```

---

## Success Criteria

1. ✅ `cargo build` passes
2. ✅ `cargo test` passes all tests
3. ✅ Name resolution working (undefined variable detected)
4. ✅ Shadowing works correctly
5. ✅ Type checking detects mismatches
6. ✅ Effects tracked on functions
7. ✅ `dc check file.d --show-resolved --show-types` works

---

## Example Session

```bash
$ cargo run -- check examples/minimal.d --show-ast --show-resolved --show-types

=== AST ===
fn #0() -> int {
  let #1: int = 42
  return #1
}

=== Symbols ===
(symbol table resolved)

=== Types ===
fn main: HirFnType { params: [], return_type: Int, effects: [] }

✓ examples/minimal.d checked successfully
```

---

## Next Steps (Day 5)

- Effect handlers and `with` expressions
- Ownership and borrow checking
- Error recovery in parser
- Source locations in error messages
- Pretty error printing with miette

---

**Day 4 Goal: Real name resolution + bidirectional type checking = solid foundation**
