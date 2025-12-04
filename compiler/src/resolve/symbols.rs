//! Symbol table implementation

use crate::common::{NodeId, Span};
use std::collections::HashMap;

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
    /// Variable (let binding)
    Variable { mutable: bool },
    /// Function parameter
    Parameter { mutable: bool },
    /// Struct type
    Struct { is_linear: bool, is_affine: bool },
    /// Enum type
    Enum { is_linear: bool, is_affine: bool },
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
    /// Built-in type
    BuiltinType,
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
    pub span: Span,
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
    /// Names defined in this scope (values: variables, functions, etc.)
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
#[derive(Debug)]
pub struct SymbolTable {
    /// All symbols by DefId
    symbols: HashMap<DefId, Symbol>,
    /// Scope stack
    scopes: Vec<Scope>,
    /// Next DefId
    next_id: u32,
    /// NodeId -> DefId mapping (for definitions)
    node_to_def: HashMap<NodeId, DefId>,
    /// NodeId -> DefId mapping (for references)
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
            "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
            "f32", "f64", "bool", "char", "String", "str",
        ];

        for name in builtins {
            let def_id = self.fresh_def_id();
            let _ = self.define_type(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::BuiltinType,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
        }

        // Built-in effects
        let builtin_effects = [
            "IO",    // File, network, console I/O
            "Mut",   // Mutable state
            "Alloc", // Heap allocation
            "Panic", // Recoverable failure
            "Async", // Asynchronous operations
            "GPU",   // GPU kernel launch, device memory
            "Prob",  // Probabilistic computation
            "Div",   // Potential divergence
        ];

        for name in builtin_effects {
            let def_id = self.fresh_def_id();
            let _ = self.define_type(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::Effect,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
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

    /// Pop scope
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

    /// Get all symbols
    pub fn all_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.values()
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

    #[test]
    fn test_builtin_types() {
        let table = SymbolTable::new();

        // Built-in types should be available
        assert!(table.lookup_type("i32").is_some());
        assert!(table.lookup_type("bool").is_some());
        assert!(table.lookup_type("String").is_some());

        // Unknown type should not exist
        assert!(table.lookup_type("FooBar").is_none());
    }
}
