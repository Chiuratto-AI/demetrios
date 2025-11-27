//! Core type definitions

use std::collections::HashSet;

/// Type variable for polymorphism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);

/// Core type representation
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitives
    Unit,
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F32,
    F64,
    Char,
    Str,
    String,

    // Compound types
    /// Reference: &T or &mut T
    Ref {
        mutable: bool,
        lifetime: Option<Lifetime>,
        inner: Box<Type>,
    },
    /// Array: [T; N] or slice [T]
    Array {
        element: Box<Type>,
        size: Option<usize>,
    },
    /// Tuple: (T1, T2, ...)
    Tuple(Vec<Type>),
    /// Function type: fn(A, B) -> C
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
        effects: EffectSet,
    },
    /// Named type (struct, enum, type alias)
    Named {
        name: String,
        args: Vec<Type>,
    },

    // Polymorphism
    /// Type variable
    Var(TypeVar),
    /// Forall quantifier: forall a. T
    Forall {
        vars: Vec<TypeVar>,
        inner: Box<Type>,
    },

    // Special types
    /// Never type (!)
    Never,
    /// Unknown type (for inference)
    Unknown,
    /// Error type (for error recovery)
    Error,
    /// Self type (within impl blocks)
    SelfType,
}

impl Type {
    /// Check if this type is a primitive
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Type::Unit
                | Type::Bool
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
                | Type::F32
                | Type::F64
                | Type::Char
        )
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
                | Type::F32
                | Type::F64
        )
    }

    /// Check if this type is an integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
        )
    }

    /// Check if this type is a floating point
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Check if this type is signed
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 | Type::Isize
        )
    }

    /// Get all free type variables in this type
    pub fn free_vars(&self) -> HashSet<TypeVar> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut HashSet<TypeVar>) {
        match self {
            Type::Var(v) => {
                vars.insert(*v);
            }
            Type::Ref { inner, .. } => inner.collect_free_vars(vars),
            Type::Array { element, .. } => element.collect_free_vars(vars),
            Type::Tuple(elems) => {
                for elem in elems {
                    elem.collect_free_vars(vars);
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    param.collect_free_vars(vars);
                }
                return_type.collect_free_vars(vars);
            }
            Type::Named { args, .. } => {
                for arg in args {
                    arg.collect_free_vars(vars);
                }
            }
            Type::Forall { vars: bound, inner } => {
                let mut inner_vars = HashSet::new();
                inner.collect_free_vars(&mut inner_vars);
                for v in inner_vars {
                    if !bound.contains(&v) {
                        vars.insert(v);
                    }
                }
            }
            _ => {}
        }
    }

    /// Substitute type variables
    pub fn substitute(&self, subst: &std::collections::HashMap<TypeVar, Type>) -> Type {
        match self {
            Type::Var(v) => subst.get(v).cloned().unwrap_or_else(|| self.clone()),
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(inner.substitute(subst)),
            },
            Type::Array { element, size } => Type::Array {
                element: Box::new(element.substitute(subst)),
                size: *size,
            },
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| e.substitute(subst)).collect()),
            Type::Function {
                params,
                return_type,
                effects,
            } => Type::Function {
                params: params.iter().map(|p| p.substitute(subst)).collect(),
                return_type: Box::new(return_type.substitute(subst)),
                effects: effects.clone(),
            },
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| a.substitute(subst)).collect(),
            },
            Type::Forall { vars, inner } => {
                // Avoid capturing bound variables
                let mut new_subst = subst.clone();
                for v in vars {
                    new_subst.remove(v);
                }
                Type::Forall {
                    vars: vars.clone(),
                    inner: Box::new(inner.substitute(&new_subst)),
                }
            }
            _ => self.clone(),
        }
    }
}

/// Lifetime for references
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub name: String,
}

impl Lifetime {
    pub fn static_lifetime() -> Self {
        Self {
            name: "'static".to_string(),
        }
    }

    pub fn anonymous() -> Self {
        Self {
            name: "'_".to_string(),
        }
    }
}

/// Effect in a type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub name: String,
    pub args: Vec<Type>,
}

impl Effect {
    pub fn io() -> Self {
        Self {
            name: "IO".to_string(),
            args: Vec::new(),
        }
    }

    pub fn mut_effect() -> Self {
        Self {
            name: "Mut".to_string(),
            args: Vec::new(),
        }
    }

    pub fn alloc() -> Self {
        Self {
            name: "Alloc".to_string(),
            args: Vec::new(),
        }
    }

    pub fn prob() -> Self {
        Self {
            name: "Prob".to_string(),
            args: Vec::new(),
        }
    }

    pub fn gpu() -> Self {
        Self {
            name: "GPU".to_string(),
            args: Vec::new(),
        }
    }
}

/// Set of effects
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EffectSet {
    pub effects: HashSet<String>,
    pub vars: HashSet<TypeVar>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pure() -> Self {
        Self::new()
    }

    pub fn single(effect: Effect) -> Self {
        let mut set = Self::new();
        set.effects.insert(effect.name);
        set
    }

    pub fn add(&mut self, effect: Effect) {
        self.effects.insert(effect.name);
    }

    pub fn union(&self, other: &EffectSet) -> EffectSet {
        EffectSet {
            effects: self.effects.union(&other.effects).cloned().collect(),
            vars: self.vars.union(&other.vars).cloned().collect(),
        }
    }

    pub fn is_pure(&self) -> bool {
        self.effects.is_empty() && self.vars.is_empty()
    }

    pub fn contains(&self, effect: &str) -> bool {
        self.effects.contains(effect)
    }
}

/// Type scheme (polymorphic type)
#[derive(Debug, Clone)]
pub struct TypeScheme {
    pub vars: Vec<TypeVar>,
    pub ty: Type,
}

impl TypeScheme {
    pub fn mono(ty: Type) -> Self {
        Self {
            vars: Vec::new(),
            ty,
        }
    }

    pub fn instantiate(&self, fresh_vars: &[Type]) -> Type {
        let mut subst = std::collections::HashMap::new();
        for (var, ty) in self.vars.iter().zip(fresh_vars.iter()) {
            subst.insert(*var, ty.clone());
        }
        self.ty.substitute(&subst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_is_numeric() {
        assert!(Type::I32.is_numeric());
        assert!(Type::F64.is_numeric());
        assert!(!Type::Bool.is_numeric());
        assert!(!Type::String.is_numeric());
    }

    #[test]
    fn test_free_vars() {
        let v1 = TypeVar(1);
        let v2 = TypeVar(2);
        let ty = Type::Function {
            params: vec![Type::Var(v1)],
            return_type: Box::new(Type::Var(v2)),
            effects: EffectSet::new(),
        };
        let vars = ty.free_vars();
        assert!(vars.contains(&v1));
        assert!(vars.contains(&v2));
    }

    #[test]
    fn test_effect_set() {
        let mut effects = EffectSet::new();
        assert!(effects.is_pure());

        effects.add(Effect::io());
        assert!(!effects.is_pure());
        assert!(effects.contains("IO"));
    }
}
