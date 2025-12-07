//! Abstract Syntax Tree for the Demetrios language
//!
//! This module defines the AST types produced by the parser.

use crate::common::{NodeId, Span};
use serde::{Deserialize, Serialize};

/// Top-level AST
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    MacroInvocation(MacroInvocation),
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

/// ABI specification for FFI
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Abi {
    /// C ABI (default for extern)
    C,
    /// C ABI with unwind support
    CUnwind,
    /// Rust ABI (default for normal functions)
    Rust,
    /// System ABI (stdcall on Windows, C elsewhere)
    System,
    /// System ABI with unwind support
    SystemUnwind,
    /// x86 stdcall
    Stdcall,
    /// x86 stdcall with unwind support
    StdcallUnwind,
    /// x86 fastcall
    Fastcall,
    /// x86 fastcall with unwind support
    FastcallUnwind,
    /// x86 cdecl
    Cdecl,
    /// Arm AAPCS
    Aapcs,
    /// Win64 ABI
    Win64,
    /// SysV64 ABI
    SysV64,
    /// Platform intrinsic
    PlatformIntrinsic,
    /// Unknown ABI (for forward compatibility)
    Unknown(String),
}

impl Default for Abi {
    fn default() -> Self {
        Abi::Rust
    }
}

impl Abi {
    /// Parse an ABI string
    pub fn from_str(s: &str) -> Self {
        match s {
            "C" => Abi::C,
            "C-unwind" => Abi::CUnwind,
            "Rust" => Abi::Rust,
            "system" => Abi::System,
            "system-unwind" => Abi::SystemUnwind,
            "stdcall" => Abi::Stdcall,
            "stdcall-unwind" => Abi::StdcallUnwind,
            "fastcall" => Abi::Fastcall,
            "fastcall-unwind" => Abi::FastcallUnwind,
            "cdecl" => Abi::Cdecl,
            "aapcs" => Abi::Aapcs,
            "win64" => Abi::Win64,
            "sysv64" => Abi::SysV64,
            "platform-intrinsic" => Abi::PlatformIntrinsic,
            other => Abi::Unknown(other.to_string()),
        }
    }

    /// Get the ABI as a string
    pub fn as_str(&self) -> &str {
        match self {
            Abi::C => "C",
            Abi::CUnwind => "C-unwind",
            Abi::Rust => "Rust",
            Abi::System => "system",
            Abi::SystemUnwind => "system-unwind",
            Abi::Stdcall => "stdcall",
            Abi::StdcallUnwind => "stdcall-unwind",
            Abi::Fastcall => "fastcall",
            Abi::FastcallUnwind => "fastcall-unwind",
            Abi::Cdecl => "cdecl",
            Abi::Aapcs => "aapcs",
            Abi::Win64 => "win64",
            Abi::SysV64 => "sysv64",
            Abi::PlatformIntrinsic => "platform-intrinsic",
            Abi::Unknown(s) => s,
        }
    }

    /// Check if this ABI supports unwinding
    pub fn supports_unwind(&self) -> bool {
        matches!(
            self,
            Abi::CUnwind | Abi::SystemUnwind | Abi::StdcallUnwind | Abi::FastcallUnwind | Abi::Rust
        )
    }
}

impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Extern block containing foreign declarations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternBlock {
    pub id: NodeId,
    pub abi: Abi,
    pub items: Vec<ExternItem>,
    pub span: Span,
}

/// Item in an extern block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternItem {
    /// Foreign function declaration
    Fn(ExternFn),
    /// Foreign static variable
    Static(ExternStatic),
    /// Foreign type (opaque)
    Type(ExternType),
}

/// Extern function declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternFn {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub is_variadic: bool,
    pub link_name: Option<String>,
    pub span: Span,
}

/// Extern static variable declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternStatic {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
    pub is_mut: bool,
    pub link_name: Option<String>,
    pub span: Span,
}

/// Extern opaque type declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternType {
    pub id: NodeId,
    pub name: String,
    pub span: Span,
}

/// Representation attribute for FFI types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Repr {
    /// Default D representation
    D,
    /// C-compatible representation
    C,
    /// Transparent (single-field newtype)
    Transparent,
    /// Packed representation (no padding)
    Packed,
    /// Specific alignment
    Align(usize),
    /// Integer representation for enums
    Int(IntRepr),
}

/// Integer representation for enums
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntRepr {
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
}

/// Calling convention for function pointers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallingConvention {
    /// Default D calling convention
    D,
    /// C calling convention
    C,
    /// System calling convention
    System,
    /// stdcall (Windows)
    Stdcall,
    /// fastcall (Windows)
    Fastcall,
    /// cdecl
    Cdecl,
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

    // ==================== DEMETRIOS EPISTEMIC TYPES ====================
    /// Knowledge type: Knowledge[T, ε < 0.05, Valid(duration), Derived]
    /// The core epistemic type tracking uncertainty, validity, and provenance
    Knowledge {
        /// The underlying value type (τ)
        value_type: Box<TypeExpr>,
        /// Uncertainty bound (ε)
        epsilon: Option<EpsilonBound>,
        /// Temporal validity condition (δ)
        validity: Option<ValidityCondition>,
        /// Data provenance marker (Φ)
        provenance: Option<ProvenanceMarker>,
    },

    /// Quantity type: Quantity[f64, meters] or f64@kg
    /// Value with physical units for dimensional analysis
    Quantity {
        /// The numeric type (f32, f64, i32, etc.)
        numeric_type: Box<TypeExpr>,
        /// The physical unit expression
        unit: UnitExpr,
    },

    /// Tensor type: Tensor[f32, (batch, channels, height, width)]
    /// Multi-dimensional array with named dimensions
    Tensor {
        /// Element type
        element_type: Box<TypeExpr>,
        /// Shape dimensions (can be expressions or named dimensions)
        shape: Vec<TensorDim>,
    },

    /// Ontology type: OntologyTerm[SNOMED:12345]
    /// Reference to an ontology term for semantic interoperability
    Ontology {
        /// Ontology namespace (SNOMED, ICD10, NIDM, etc.)
        ontology: String,
        /// Optional specific term within the ontology
        term: Option<String>,
    },

    /// Linear/affine type annotation: T @ linear
    /// For GPU memory safety and resource tracking
    Linear {
        inner: Box<TypeExpr>,
        linearity: LinearityKind,
    },

    /// Effect row type: T ! {IO, GPU, Random}
    /// Type annotated with computational effects
    Effected {
        inner: Box<TypeExpr>,
        effects: EffectRow,
    },
}

// ==================== EPISTEMIC TYPE COMPONENTS ====================

/// Uncertainty bound: ε < 0.05, ε = σ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsilonBound {
    /// Comparison operator: "<", "<=", "=", ">", ">="
    pub operator: ComparisonOp,
    /// The bound value (can be a literal or expression)
    pub value: Box<Expr>,
}

/// Comparison operators for epsilon bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

/// Validity condition: Valid(duration), ValidUntil(date), ValidWhile(condition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityCondition {
    /// Kind of validity: "Valid", "ValidUntil", "ValidWhile"
    pub kind: ValidityKind,
    /// The condition expression
    pub condition: Box<Expr>,
}

/// Kinds of temporal validity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidityKind {
    /// Valid for a duration
    Valid,
    /// Valid until a specific time
    ValidUntil,
    /// Valid while a condition holds
    ValidWhile,
}

/// Provenance marker: Derived, Source(name), Computed, Literature(citation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceMarker {
    /// Kind of provenance
    pub kind: ProvenanceKind,
    /// Optional source reference
    pub source: Option<Box<Expr>>,
}

/// Kinds of data provenance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    /// Derived from other data
    Derived,
    /// Primary source data
    Source,
    /// Computed/calculated value
    Computed,
    /// From published literature
    Literature,
    /// Experimentally measured
    Measured,
    /// User-provided input
    Input,
}

/// Physical unit expression: meters, kg*m/s^2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitExpr {
    /// Base units with exponents: [(unit_name, exponent), ...]
    /// e.g., kg*m/s^2 = [("kg", 1), ("m", 1), ("s", -2)]
    pub base_units: Vec<(String, i32)>,
}

impl UnitExpr {
    /// Create a simple unit
    pub fn simple(name: &str) -> Self {
        Self {
            base_units: vec![(name.to_string(), 1)],
        }
    }

    /// Create a dimensionless unit
    pub fn dimensionless() -> Self {
        Self { base_units: vec![] }
    }
}

/// Tensor dimension (can be named or sized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDim {
    /// Named dimension: batch, channels, height, width
    Named(String),
    /// Fixed size dimension
    Fixed(usize),
    /// Dynamic/inferred dimension
    Dynamic,
    /// Expression-based dimension
    Expr(Box<Expr>),
}

/// Linearity kind for resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearityKind {
    /// Normal unrestricted type
    Unrestricted,
    /// Must be used exactly once
    Linear,
    /// Must be used at most once
    Affine,
    /// Must be used at least once
    Relevant,
}

/// Effect row: {IO, GPU, Random, ...}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRow {
    /// List of effects
    pub effects: Vec<String>,
    /// Whether this is an open row (can have more effects)
    pub is_open: bool,
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
    /// Async block: async { ... }
    AsyncBlock { id: NodeId, block: Block },
    /// Async closure: async |x| { ... }
    AsyncClosure {
        id: NodeId,
        params: Vec<(String, Option<TypeExpr>)>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },
    /// Spawn async task: spawn { ... }
    Spawn { id: NodeId, expr: Box<Expr> },
    /// Select expression for waiting on multiple futures
    Select { id: NodeId, arms: Vec<SelectArm> },
    /// Join expression for concurrent execution
    Join { id: NodeId, futures: Vec<Expr> },
    /// Macro invocation
    MacroInvocation(MacroInvocation),

    // ==================== DEMETRIOS EPISTEMIC EXPRESSIONS ====================
    /// Causal do expression: do(X = 1)
    /// Represents intervention in causal inference (Pearl's do-calculus)
    Do {
        id: NodeId,
        /// List of interventions: [(variable, value), ...]
        interventions: Vec<(String, Box<Expr>)>,
    },

    /// Counterfactual expression: counterfactual { factual; do(X=1); outcome }
    /// Three-step counterfactual computation (abduction, action, prediction)
    Counterfactual {
        id: NodeId,
        /// The factual observation
        factual: Box<Expr>,
        /// The intervention to apply
        intervention: Box<Expr>,
        /// The outcome query
        outcome: Box<Expr>,
    },

    /// Knowledge construction: Knowledge::new(value, epsilon, validity, provenance)
    KnowledgeExpr {
        id: NodeId,
        /// The underlying value
        value: Box<Expr>,
        /// Optional uncertainty bound
        epsilon: Option<Box<Expr>>,
        /// Optional validity condition
        validity: Option<Box<Expr>>,
        /// Optional provenance marker
        provenance: Option<Box<Expr>>,
    },

    /// Uncertainty propagation: x ± σ or x.with_uncertainty(σ)
    Uncertain {
        id: NodeId,
        /// The central value
        value: Box<Expr>,
        /// The uncertainty/standard deviation
        uncertainty: Box<Expr>,
    },

    /// GPU-annotated expression: expr @ gpu.epistemic
    GpuAnnotated {
        id: NodeId,
        /// The inner expression
        expr: Box<Expr>,
        /// GPU annotation kind
        annotation: GpuAnnotation,
    },

    /// Observe expression for probabilistic programming: observe(data ~ distribution)
    Observe {
        id: NodeId,
        /// The observed data
        data: Box<Expr>,
        /// The distribution it's drawn from
        distribution: Box<Expr>,
    },

    /// Query expression: P(Y | X, do(Z))
    /// Probabilistic query with optional conditioning and intervention
    Query {
        id: NodeId,
        /// The target variable/expression
        target: Box<Expr>,
        /// Conditioning variables
        given: Vec<Expr>,
        /// Interventions (do-expressions)
        interventions: Vec<(String, Box<Expr>)>,
    },
}

// ==================== GPU ANNOTATIONS ====================

/// GPU annotation: @gpu, @gpu.epistemic, @gpu.reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAnnotation {
    /// Annotation kind
    pub kind: GpuAnnotationKind,
    /// Optional parameters
    pub params: Vec<(String, Expr)>,
}

/// Kinds of GPU annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAnnotationKind {
    /// Basic GPU execution
    Gpu,
    /// Epistemic-aware GPU (uncertainty propagation)
    GpuEpistemic,
    /// GPU reduction operation
    GpuReduction,
    /// GPU parallel execution
    GpuParallel,
    /// GPU shared memory
    GpuShared,
    /// GPU device memory
    GpuDevice,
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

/// Select arm for async select expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectArm {
    /// The future expression to wait on
    pub future: Expr,
    /// Pattern to bind the result
    pub pattern: Pattern,
    /// Optional guard condition
    pub guard: Option<Box<Expr>>,
    /// Body expression to execute when this arm matches
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
    /// Macro invocation
    MacroInvocation(MacroInvocation),
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

// ==================== MACROS ====================

/// Macro invocation (e.g., vec![1, 2, 3])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInvocation {
    pub id: NodeId,
    pub name: String,
    pub args: Vec<crate::macro_system::token_tree::TokenTree>,
    pub span: Span,
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
