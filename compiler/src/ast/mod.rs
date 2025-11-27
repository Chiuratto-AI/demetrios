//! Abstract Syntax Tree for the Demetrios language
//!
//! This module defines the AST types produced by the parser.

use crate::common::{NodeId, Span};
use serde::{Deserialize, Serialize};

/// Top-level AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
}

/// Item visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
}

/// Common modifiers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Modifiers {
    pub linear: bool,
    pub affine: bool,
    pub is_async: bool,
    pub is_unsafe: bool,
}

/// Type modifiers (linear/affine)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeModifiers {
    pub linear: bool,
    pub affine: bool,
}

/// Function modifiers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FnModifiers {
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_kernel: bool,
}

/// Top-level item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    Function(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplDef),
    TypeAlias(TypeAliasDef),
    Effect(EffectDef),
    Handler(HandlerDef),
    Import(ImportDef),
    Extern(ExternBlock),
    Global(GlobalDef),
}

// ==================== FUNCTIONS ====================

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: FnModifiers,
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effects: Vec<EffectRef>,
    pub where_clause: Vec<WherePredicate>,
    pub body: Block,
    pub span: Span,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub id: NodeId,
    pub is_mut: bool,
    pub pattern: Pattern,
    pub ty: TypeExpr,
}

// ==================== STRUCTS ====================

/// Struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub fields: Vec<FieldDef>,
    pub span: Span,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub ty: TypeExpr,
}

// ==================== ENUMS ====================

/// Enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub variants: Vec<VariantDef>,
    pub span: Span,
}

/// Enum variant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantDef {
    pub id: NodeId,
    pub name: String,
    pub data: VariantData,
}

/// Variant data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantData {
    Unit,
    Tuple(Vec<TypeExpr>),
    Struct(Vec<FieldDef>),
}

// ==================== TRAITS ====================

/// Trait definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub supertraits: Vec<Path>,
    pub where_clause: Vec<WherePredicate>,
    pub items: Vec<TraitItem>,
    pub span: Span,
}

/// Trait item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraitItem {
    Fn(TraitFnDef),
    Type(TraitTypeDef),
}

/// Trait function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitFnDef {
    pub id: NodeId,
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effects: Vec<EffectRef>,
    pub where_clause: Vec<WherePredicate>,
    pub default_body: Option<Block>,
}

/// Trait associated type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitTypeDef {
    pub id: NodeId,
    pub name: String,
    pub bounds: Vec<Path>,
    pub default: Option<TypeExpr>,
}

// ==================== IMPL ====================

/// Impl block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplDef {
    pub id: NodeId,
    pub generics: Generics,
    pub trait_ref: Option<Path>,
    pub target_type: TypeExpr,
    pub where_clause: Vec<WherePredicate>,
    pub items: Vec<ImplItem>,
    pub span: Span,
}

/// Impl item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplItem {
    Fn(FnDef),
    Type(ImplTypeDef),
}

/// Impl associated type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplTypeDef {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
}

// ==================== TYPE ALIAS ====================

/// Type alias definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAliasDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub ty: TypeExpr,
    pub span: Span,
}

// ==================== EFFECTS ====================

/// Effect definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub operations: Vec<EffectOpDef>,
    pub span: Span,
}

/// Effect operation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectOpDef {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
}

/// Effect reference in function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRef {
    pub id: NodeId,
    pub name: Path,
    pub args: Vec<TypeExpr>,
}

/// Handler definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub effect: Path,
    pub cases: Vec<HandlerCase>,
    pub span: Span,
}

/// Handler case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerCase {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub body: Expr,
}

// ==================== IMPORTS & EXTERN ====================

/// Import definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDef {
    pub id: NodeId,
    pub path: Path,
    pub span: Span,
}

/// Extern block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternBlock {
    pub id: NodeId,
    pub abi: String,
    pub items: Vec<ExternFn>,
    pub span: Span,
}

/// Extern function declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternFn {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
}

// ==================== GLOBALS ====================

/// Global variable/constant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub is_const: bool,
    pub is_mut: bool,
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

// ==================== GENERICS ====================

/// Generic parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Generics {
    pub params: Vec<GenericParam>,
}

/// Generic parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenericParam {
    Type {
        name: String,
        bounds: Vec<Path>,
        default: Option<TypeExpr>,
    },
    Const {
        name: String,
        ty: TypeExpr,
    },
}

/// Where predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WherePredicate {
    pub ty: TypeExpr,
    pub bounds: Vec<Path>,
}

// ==================== TYPES ====================

/// Type expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeExpr {
    /// Unit type ()
    Unit,
    /// Self type (in traits and impls)
    SelfType,
    /// Named type: Path<Args>
    Named {
        path: Path,
        args: Vec<TypeExpr>,
        unit: Option<String>,
    },
    /// Reference type: &T or &mut T
    Reference { mutable: bool, inner: Box<TypeExpr> },
    /// Array type: [T] or [T; N]
    Array {
        element: Box<TypeExpr>,
        size: Option<Box<Expr>>,
    },
    /// Tuple type: (T1, T2, ...)
    Tuple(Vec<TypeExpr>),
    /// Function type: Fn(A) -> B
    Function {
        params: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
        effects: Vec<EffectRef>,
    },
    /// Infer type: _
    Infer,
}

// ==================== EXPRESSIONS ====================

/// Expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Literal value
    Literal { id: NodeId, value: Literal },
    /// Path reference
    Path { id: NodeId, path: Path },
    /// Binary operation
    Binary {
        id: NodeId,
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation
    Unary {
        id: NodeId,
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// Function call
    Call {
        id: NodeId,
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Method call
    MethodCall {
        id: NodeId,
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    /// Field access
    Field {
        id: NodeId,
        base: Box<Expr>,
        field: String,
    },
    /// Tuple field access
    TupleField {
        id: NodeId,
        base: Box<Expr>,
        index: usize,
    },
    /// Index operation
    Index {
        id: NodeId,
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// Type cast
    Cast {
        id: NodeId,
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    /// Block expression
    Block { id: NodeId, block: Block },
    /// If expression
    If {
        id: NodeId,
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },
    /// Match expression
    Match {
        id: NodeId,
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Loop expression
    Loop { id: NodeId, body: Block },
    /// While loop
    While {
        id: NodeId,
        condition: Box<Expr>,
        body: Block,
    },
    /// For loop
    For {
        id: NodeId,
        pattern: Pattern,
        iter: Box<Expr>,
        body: Block,
    },
    /// Return expression
    Return {
        id: NodeId,
        value: Option<Box<Expr>>,
    },
    /// Break expression
    Break {
        id: NodeId,
        value: Option<Box<Expr>>,
    },
    /// Continue expression
    Continue { id: NodeId },
    /// Closure expression
    Closure {
        id: NodeId,
        params: Vec<(String, Option<TypeExpr>)>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },
    /// Tuple expression
    Tuple { id: NodeId, elements: Vec<Expr> },
    /// Array expression
    Array { id: NodeId, elements: Vec<Expr> },
    /// Struct literal
    StructLit {
        id: NodeId,
        path: Path,
        fields: Vec<(String, Expr)>,
    },
    /// Try expression (?)
    Try { id: NodeId, expr: Box<Expr> },
    /// Perform effect operation
    Perform {
        id: NodeId,
        effect: Path,
        op: String,
        args: Vec<Expr>,
    },
    /// Handle effect
    Handle {
        id: NodeId,
        expr: Box<Expr>,
        handler: Path,
    },
    /// Sample from distribution
    Sample { id: NodeId, distribution: Box<Expr> },
    /// Await async expression
    Await { id: NodeId, expr: Box<Expr> },
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
    /// Integer with unit of measure (e.g., 500_mg)
    IntUnit(i64, String),
    /// Float with unit of measure (e.g., 10.5_mL)
    FloatUnit(f64, String),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    Ref,
    RefMut,
    Deref,
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Expr,
}

// ==================== STATEMENTS ====================

/// Statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    /// Let binding
    Let {
        is_mut: bool,
        pattern: Pattern,
        ty: Option<TypeExpr>,
        value: Option<Expr>,
    },
    /// Expression statement
    Expr { expr: Expr, has_semi: bool },
    /// Assignment
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// Empty statement (;)
    Empty,
}

/// Assignment operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

/// Block of statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

// ==================== PATTERNS ====================

/// Pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Wildcard pattern: _
    Wildcard,
    /// Literal pattern
    Literal(Literal),
    /// Variable binding
    Binding { name: String, mutable: bool },
    /// Tuple pattern: (p1, p2, ...)
    Tuple(Vec<Pattern>),
    /// Struct pattern: S { field: pattern, ... }
    Struct {
        path: Path,
        fields: Vec<(String, Pattern)>,
    },
    /// Enum variant pattern: E::V(p1, p2, ...)
    Enum {
        path: Path,
        patterns: Option<Vec<Pattern>>,
    },
    /// Or pattern: p1 | p2
    Or(Vec<Pattern>),
}

// ==================== PATHS ====================

/// Path (e.g., std::io::Write)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    pub segments: Vec<String>,
}

impl Path {
    pub fn simple(name: &str) -> Self {
        Path {
            segments: vec![name.to_string()],
        }
    }

    pub fn is_simple(&self) -> bool {
        self.segments.len() == 1
    }

    pub fn name(&self) -> Option<&str> {
        self.segments.last().map(|s| s.as_str())
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.join("::"))
    }
}
