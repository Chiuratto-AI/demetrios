//! Runtime values for the interpreter

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::hir::HirFn;

/// Runtime value
#[derive(Clone)]
pub enum Value {
    /// Unit value `()`
    Unit,
    /// Boolean
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// String
    String(String),
    /// Array (mutable interior)
    Array(Rc<RefCell<Vec<Value>>>),
    /// Tuple
    Tuple(Vec<Value>),
    /// Struct instance
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    /// Enum variant
    Variant {
        enum_name: String,
        variant_name: String,
        fields: Vec<Value>,
    },
    /// Function closure
    Function {
        func: Rc<HirFn>,
        /// Captured environment for closures
        captures: HashMap<String, Value>,
    },
    /// Reference to a value
    Ref(Rc<RefCell<Value>>),
    /// Option::None
    None,
    /// Option::Some(value)
    Some(Box<Value>),
    /// Result::Ok(value)
    Ok(Box<Value>),
    /// Result::Err(value)
    Err(Box<Value>),
}

impl Value {
    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "unit",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tuple(_) => "tuple",
            Value::Struct { .. } => "struct",
            Value::Variant { .. } => "variant",
            Value::Function { .. } => "function",
            Value::Ref(_) => "ref",
            Value::None => "None",
            Value::Some(_) => "Some",
            Value::Ok(_) => "Ok",
            Value::Err(_) => "Err",
        }
    }

    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Unit => false,
            Value::None => false,
            _ => true,
        }
    }

    /// Try to get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to get as float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{:?}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                let arr = arr.borrow();
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant {
                enum_name,
                variant_name,
                fields,
            } => {
                write!(f, "{}::{}", enum_name, variant_name)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function { func, .. } => write!(f, "<fn {}>", func.name),
            Value::Ref(r) => write!(f, "&{:?}", r.borrow()),
            Value::None => write!(f, "None"),
            Value::Some(v) => write!(f, "Some({:?})", v),
            Value::Ok(v) => write!(f, "Ok({:?})", v),
            Value::Err(v) => write!(f, "Err({:?})", v),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                let arr = arr.borrow();
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vals) => {
                write!(f, "(")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::Variant {
                variant_name,
                fields,
                ..
            } => {
                write!(f, "{}", variant_name)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::Function { func, .. } => write!(f, "<fn {}>", func.name),
            Value::Ref(r) => write!(f, "{}", r.borrow()),
            Value::None => write!(f, "None"),
            Value::Some(v) => write!(f, "Some({})", v),
            Value::Ok(v) => write!(f, "Ok({})", v),
            Value::Err(v) => write!(f, "Err({})", v),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::Some(a), Value::Some(b)) => a == b,
            (Value::Ok(a), Value::Ok(b)) => a == b,
            (Value::Err(a), Value::Err(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (
                Value::Struct {
                    name: n1,
                    fields: f1,
                },
                Value::Struct {
                    name: n2,
                    fields: f2,
                },
            ) => n1 == n2 && f1 == f2,
            (
                Value::Variant {
                    enum_name: e1,
                    variant_name: v1,
                    fields: f1,
                },
                Value::Variant {
                    enum_name: e2,
                    variant_name: v2,
                    fields: f2,
                },
            ) => e1 == e2 && v1 == v2 && f1 == f2,
            _ => false,
        }
    }
}

/// Control flow signal (not an error, just flow control)
#[derive(Debug, Clone)]
pub enum ControlFlow {
    /// Return from function with value
    Return(Value),
    /// Break from loop with optional value
    Break(Option<Value>),
    /// Continue to next iteration
    Continue,
}
