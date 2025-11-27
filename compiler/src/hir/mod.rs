//! High-level Intermediate Representation (HIR)
//!
//! HIR is the typed AST produced by the type checker. It contains:
//! - Resolved types for all expressions
//! - Resolved names (no more path resolution needed)
//! - Desugared constructs
//! - Ownership and borrowing information

use crate::common::NodeId;

/// HIR root
#[derive(Debug, Clone)]
pub struct Hir {
    pub items: Vec<HirItem>,
}

/// HIR item
#[derive(Debug, Clone)]
pub enum HirItem {
    Function(HirFn),
    Struct(HirStruct),
    Enum(HirEnum),
    Trait(HirTrait),
    Impl(HirImpl),
    TypeAlias(HirTypeAlias),
    Effect(HirEffect),
    Handler(HirHandler),
    Global(HirGlobal),
}

// ==================== FUNCTIONS ====================

/// HIR function
#[derive(Debug, Clone)]
pub struct HirFn {
    pub id: NodeId,
    pub name: String,
    pub ty: HirFnType,
    pub body: HirBlock,
}

/// Function type in HIR
#[derive(Debug, Clone)]
pub struct HirFnType {
    pub params: Vec<HirParam>,
    pub return_type: Box<HirType>,
    pub effects: Vec<HirEffect>,
}

/// HIR parameter
#[derive(Debug, Clone)]
pub struct HirParam {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
    pub is_mut: bool,
}

// ==================== TYPES ====================

/// HIR struct
#[derive(Debug, Clone)]
pub struct HirStruct {
    pub id: NodeId,
    pub name: String,
    pub fields: Vec<HirField>,
    pub is_linear: bool,
    pub is_affine: bool,
}

/// HIR field
#[derive(Debug, Clone)]
pub struct HirField {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
}

/// HIR enum
#[derive(Debug, Clone)]
pub struct HirEnum {
    pub id: NodeId,
    pub name: String,
    pub variants: Vec<HirVariant>,
    pub is_linear: bool,
    pub is_affine: bool,
}

/// HIR variant
#[derive(Debug, Clone)]
pub struct HirVariant {
    pub id: NodeId,
    pub name: String,
    pub fields: Vec<HirType>,
}

/// HIR trait
#[derive(Debug, Clone)]
pub struct HirTrait {
    pub id: NodeId,
    pub name: String,
    pub methods: Vec<HirTraitMethod>,
}

/// HIR trait method
#[derive(Debug, Clone)]
pub struct HirTraitMethod {
    pub id: NodeId,
    pub name: String,
    pub ty: HirFnType,
    pub has_default: bool,
}

/// HIR impl
#[derive(Debug, Clone)]
pub struct HirImpl {
    pub id: NodeId,
    pub trait_ref: Option<String>,
    pub self_ty: HirType,
    pub methods: Vec<HirFn>,
}

/// HIR type alias
#[derive(Debug, Clone)]
pub struct HirTypeAlias {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
}

/// HIR global
#[derive(Debug, Clone)]
pub struct HirGlobal {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
    pub value: HirExpr,
    pub is_const: bool,
}

// ==================== EFFECTS ====================

/// HIR effect
#[derive(Debug, Clone)]
pub struct HirEffect {
    pub id: NodeId,
    pub name: String,
    pub operations: Vec<HirEffectOp>,
}

/// HIR effect operation
#[derive(Debug, Clone)]
pub struct HirEffectOp {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<HirType>,
    pub return_type: HirType,
}

/// HIR handler
#[derive(Debug, Clone)]
pub struct HirHandler {
    pub id: NodeId,
    pub name: String,
    pub effect: String,
    pub cases: Vec<HirHandlerCase>,
}

/// HIR handler case
#[derive(Debug, Clone)]
pub struct HirHandlerCase {
    pub id: NodeId,
    pub op_name: String,
    pub params: Vec<String>,
    pub body: HirExpr,
}

// ==================== TYPES ====================

/// HIR type (fully resolved)
#[derive(Debug, Clone, PartialEq)]
pub enum HirType {
    /// Unit type
    Unit,
    /// Boolean
    Bool,
    /// Signed integers
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    /// Unsigned integers
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    /// Floating point
    F32,
    F64,
    /// Character
    Char,
    /// String (owned)
    String,
    /// Reference
    Ref {
        mutable: bool,
        inner: Box<HirType>,
    },
    /// Array
    Array {
        element: Box<HirType>,
        size: Option<usize>,
    },
    /// Tuple
    Tuple(Vec<HirType>),
    /// Named type (struct/enum/type alias)
    Named {
        name: String,
        args: Vec<HirType>,
    },
    /// Function type
    Fn {
        params: Vec<HirType>,
        return_type: Box<HirType>,
    },
    /// Type variable (for generics)
    Var(u32),
    /// Never type (for diverging expressions)
    Never,
    /// Error type (for error recovery)
    Error,
}

impl HirType {
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            HirType::Unit
                | HirType::Bool
                | HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
                | HirType::F32
                | HirType::F64
                | HirType::Char
        )
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
                | HirType::F32
                | HirType::F64
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, HirType::F32 | HirType::F64)
    }
}

// ==================== EXPRESSIONS ====================

/// HIR expression (with type information)
#[derive(Debug, Clone)]
pub struct HirExpr {
    pub id: NodeId,
    pub kind: HirExprKind,
    pub ty: HirType,
}

/// HIR expression kind
#[derive(Debug, Clone)]
pub enum HirExprKind {
    /// Literal value
    Literal(HirLiteral),
    /// Local variable
    Local(String),
    /// Global variable
    Global(String),
    /// Binary operation
    Binary {
        op: HirBinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
    /// Unary operation
    Unary { op: HirUnaryOp, expr: Box<HirExpr> },
    /// Function call
    Call {
        func: Box<HirExpr>,
        args: Vec<HirExpr>,
    },
    /// Method call
    MethodCall {
        receiver: Box<HirExpr>,
        method: String,
        args: Vec<HirExpr>,
    },
    /// Field access
    Field { base: Box<HirExpr>, field: String },
    /// Tuple field access
    TupleField { base: Box<HirExpr>, index: usize },
    /// Index operation
    Index {
        base: Box<HirExpr>,
        index: Box<HirExpr>,
    },
    /// Type cast
    Cast { expr: Box<HirExpr>, target: HirType },
    /// Block
    Block(HirBlock),
    /// If expression
    If {
        condition: Box<HirExpr>,
        then_branch: HirBlock,
        else_branch: Option<Box<HirExpr>>,
    },
    /// Match expression
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },
    /// Loop
    Loop(HirBlock),
    /// Return
    Return(Option<Box<HirExpr>>),
    /// Break
    Break(Option<Box<HirExpr>>),
    /// Continue
    Continue,
    /// Closure
    Closure {
        params: Vec<HirParam>,
        body: Box<HirExpr>,
    },
    /// Tuple
    Tuple(Vec<HirExpr>),
    /// Array
    Array(Vec<HirExpr>),
    /// Struct literal
    Struct {
        name: String,
        fields: Vec<(String, HirExpr)>,
    },
    /// Enum variant constructor
    Variant {
        enum_name: String,
        variant: String,
        fields: Vec<HirExpr>,
    },
    /// Reference
    Ref { mutable: bool, expr: Box<HirExpr> },
    /// Dereference
    Deref(Box<HirExpr>),
    /// Effect operation
    Perform {
        effect: String,
        op: String,
        args: Vec<HirExpr>,
    },
    /// Handle effect
    Handle { expr: Box<HirExpr>, handler: String },
    /// Sample from distribution
    Sample(Box<HirExpr>),
}

/// HIR literal
#[derive(Debug, Clone)]
pub enum HirLiteral {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
}

/// HIR binary operator
#[derive(Debug, Clone, Copy)]
pub enum HirBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// HIR unary operator
#[derive(Debug, Clone, Copy)]
pub enum HirUnaryOp {
    Neg,
    Not,
    Ref,
    RefMut,
    Deref,
}

/// HIR match arm
#[derive(Debug, Clone)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<Box<HirExpr>>,
    pub body: HirExpr,
}

// ==================== PATTERNS ====================

/// HIR pattern
#[derive(Debug, Clone)]
pub enum HirPattern {
    Wildcard,
    Literal(HirLiteral),
    Binding {
        name: String,
        mutable: bool,
    },
    Tuple(Vec<HirPattern>),
    Struct {
        name: String,
        fields: Vec<(String, HirPattern)>,
    },
    Variant {
        enum_name: String,
        variant: String,
        patterns: Vec<HirPattern>,
    },
    Or(Vec<HirPattern>),
}

// ==================== BLOCKS & STATEMENTS ====================

/// HIR block
#[derive(Debug, Clone)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub ty: HirType,
}

/// HIR statement
#[derive(Debug, Clone)]
pub enum HirStmt {
    /// Let binding
    Let {
        name: String,
        ty: HirType,
        value: Option<HirExpr>,
        is_mut: bool,
    },
    /// Expression
    Expr(HirExpr),
    /// Assignment
    Assign { target: HirExpr, value: HirExpr },
}
