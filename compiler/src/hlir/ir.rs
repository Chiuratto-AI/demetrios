//! HLIR IR definitions - SSA-based intermediate representation
//!
//! This module defines the core IR types for HLIR, which uses SSA form
//! with explicit basic blocks and control flow.

use crate::hir::HirType;
use std::collections::HashMap;

/// HLIR module - top-level compilation unit
#[derive(Debug, Clone)]
pub struct HlirModule {
    pub name: String,
    pub functions: Vec<HlirFunction>,
    pub globals: Vec<HlirGlobal>,
    pub types: Vec<HlirTypeDef>,
}

impl HlirModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            globals: Vec::new(),
            types: Vec::new(),
        }
    }

    pub fn find_function(&self, name: &str) -> Option<&HlirFunction> {
        self.functions.iter().find(|f| f.name == name)
    }
}

/// HLIR function
#[derive(Debug, Clone)]
pub struct HlirFunction {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<HlirParam>,
    pub return_type: HlirType,
    pub effects: Vec<String>,
    pub blocks: Vec<HlirBlock>,
    pub is_kernel: bool,
    /// Local variable types (for stack allocation)
    pub locals: HashMap<ValueId, HlirType>,
}

impl HlirFunction {
    pub fn entry_block(&self) -> Option<&HlirBlock> {
        self.blocks.first()
    }

    pub fn get_block(&self, id: BlockId) -> Option<&HlirBlock> {
        self.blocks.iter().find(|b| b.id == id)
    }

    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut HlirBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }
}

/// Function identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

/// Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub const ENTRY: BlockId = BlockId(0);
}

/// Value identifier (SSA value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl ValueId {
    pub const UNIT: ValueId = ValueId(u32::MAX);
}

/// HLIR parameter
#[derive(Debug, Clone)]
pub struct HlirParam {
    pub value: ValueId,
    pub name: String,
    pub ty: HlirType,
}

/// HLIR global variable
#[derive(Debug, Clone)]
pub struct HlirGlobal {
    pub id: ValueId,
    pub name: String,
    pub ty: HlirType,
    pub init: Option<HlirConstant>,
    pub is_const: bool,
}

/// HLIR type definition
#[derive(Debug, Clone)]
pub struct HlirTypeDef {
    pub name: String,
    pub kind: HlirTypeDefKind,
}

/// Type definition kind
#[derive(Debug, Clone)]
pub enum HlirTypeDefKind {
    Struct(Vec<(String, HlirType)>),
    Enum(Vec<(String, Vec<HlirType>)>),
}

/// HLIR type
#[derive(Debug, Clone, PartialEq)]
pub enum HlirType {
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Ptr(Box<HlirType>),
    Array(Box<HlirType>, usize),
    Struct(String),
    Tuple(Vec<HlirType>),
    Function {
        params: Vec<HlirType>,
        return_type: Box<HlirType>,
    },
}

impl HlirType {
    pub fn from_hir(ty: &HirType) -> Self {
        match ty {
            HirType::Unit => HlirType::Void,
            HirType::Bool => HlirType::Bool,
            HirType::I8 => HlirType::I8,
            HirType::I16 => HlirType::I16,
            HirType::I32 => HlirType::I32,
            HirType::I64 => HlirType::I64,
            HirType::I128 => HlirType::I128,
            HirType::Isize => HlirType::I64,
            HirType::U8 => HlirType::U8,
            HirType::U16 => HlirType::U16,
            HirType::U32 => HlirType::U32,
            HirType::U64 => HlirType::U64,
            HirType::U128 => HlirType::U128,
            HirType::Usize => HlirType::U64,
            HirType::F32 => HlirType::F32,
            HirType::F64 => HlirType::F64,
            HirType::Char => HlirType::U32,
            HirType::String => HlirType::Ptr(Box::new(HlirType::U8)),
            HirType::Ref { inner, .. } => HlirType::Ptr(Box::new(Self::from_hir(inner))),
            HirType::Array { element, size } => {
                let elem = Self::from_hir(element);
                HlirType::Array(Box::new(elem), size.unwrap_or(0))
            }
            HirType::Tuple(elems) if elems.is_empty() => HlirType::Void,
            HirType::Tuple(elems) => HlirType::Tuple(elems.iter().map(Self::from_hir).collect()),
            HirType::Named { name, .. } => HlirType::Struct(name.clone()),
            HirType::Fn {
                params,
                return_type,
            } => HlirType::Function {
                params: params.iter().map(Self::from_hir).collect(),
                return_type: Box::new(Self::from_hir(return_type)),
            },
            HirType::Var(_) | HirType::Error | HirType::Never => HlirType::Void,
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            HlirType::I8
                | HlirType::I16
                | HlirType::I32
                | HlirType::I64
                | HlirType::I128
                | HlirType::U8
                | HlirType::U16
                | HlirType::U32
                | HlirType::U64
                | HlirType::U128
        )
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            HlirType::I8 | HlirType::I16 | HlirType::I32 | HlirType::I64 | HlirType::I128
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, HlirType::F32 | HlirType::F64)
    }

    pub fn size_bits(&self) -> usize {
        match self {
            HlirType::Void => 0,
            HlirType::Bool => 8,
            HlirType::I8 | HlirType::U8 => 8,
            HlirType::I16 | HlirType::U16 => 16,
            HlirType::I32 | HlirType::U32 | HlirType::F32 => 32,
            HlirType::I64 | HlirType::U64 | HlirType::F64 => 64,
            HlirType::I128 | HlirType::U128 => 128,
            HlirType::Ptr(_) => 64,
            HlirType::Array(elem, size) => elem.size_bits() * size,
            HlirType::Struct(_) => 64, // Conservative estimate
            HlirType::Tuple(elems) => elems.iter().map(|e| e.size_bits()).sum(),
            HlirType::Function { .. } => 64, // Function pointer
        }
    }
}

/// HLIR basic block
#[derive(Debug, Clone)]
pub struct HlirBlock {
    pub id: BlockId,
    pub label: String,
    pub params: Vec<(ValueId, HlirType)>,
    pub instructions: Vec<HlirInstr>,
    pub terminator: HlirTerminator,
}

impl HlirBlock {
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: HlirTerminator::Unreachable,
        }
    }
}

/// HLIR instruction
#[derive(Debug, Clone)]
pub struct HlirInstr {
    pub result: Option<ValueId>,
    pub op: Op,
    pub ty: HlirType,
}

/// HLIR operation
#[derive(Debug, Clone)]
pub enum Op {
    /// Constant value
    Const(HlirConstant),
    /// Copy a value
    Copy(ValueId),
    /// Binary operation
    Binary {
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
    },
    /// Unary operation
    Unary { op: UnaryOp, operand: ValueId },
    /// Function call
    Call { func: ValueId, args: Vec<ValueId> },
    /// Direct function call by name
    CallDirect { name: String, args: Vec<ValueId> },
    /// Load from memory
    Load { ptr: ValueId },
    /// Store to memory
    Store { ptr: ValueId, value: ValueId },
    /// Get pointer to struct field
    GetFieldPtr { base: ValueId, field: usize },
    /// Get pointer to array element
    GetElementPtr { base: ValueId, index: ValueId },
    /// Allocate stack memory
    Alloca { ty: HlirType },
    /// Type cast
    Cast { value: ValueId, target: HlirType },
    /// Phi node (SSA)
    Phi { incoming: Vec<(BlockId, ValueId)> },
    /// Extract value from aggregate
    ExtractValue { base: ValueId, index: usize },
    /// Insert value into aggregate
    InsertValue {
        base: ValueId,
        value: ValueId,
        index: usize,
    },
    /// Construct tuple
    Tuple(Vec<ValueId>),
    /// Construct array
    Array(Vec<ValueId>),
    /// Construct struct
    Struct {
        name: String,
        fields: Vec<(String, ValueId)>,
    },
    /// Perform effect operation
    PerformEffect {
        effect: String,
        op: String,
        args: Vec<ValueId>,
    },
}

/// HLIR constant
#[derive(Debug, Clone)]
pub enum HlirConstant {
    Unit,
    Bool(bool),
    Int(i64, HlirType),
    Float(f64, HlirType),
    String(String),
    Array(Vec<HlirConstant>),
    Struct(Vec<HlirConstant>),
    Null(HlirType),
    Undef(HlirType),
    FunctionRef(String),
    GlobalRef(String),
}

/// Binary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Integer arithmetic
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    // Float arithmetic
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    AShr,
    LShr,
    // Integer comparison
    Eq,
    Ne,
    SLt,
    SLe,
    SGt,
    SGe,
    ULt,
    ULe,
    UGt,
    UGe,
    // Float comparison
    FOEq,
    FONe,
    FOLt,
    FOLe,
    FOGt,
    FOGe,
}

/// Unary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    FNeg,
    Not,
}

/// Block terminator
#[derive(Debug, Clone)]
pub enum HlirTerminator {
    /// Return from function
    Return(Option<ValueId>),
    /// Unconditional branch
    Branch(BlockId),
    /// Conditional branch
    CondBranch {
        condition: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    },
    /// Switch on integer value
    Switch {
        value: ValueId,
        default: BlockId,
        cases: Vec<(i64, BlockId)>,
    },
    /// Unreachable code
    Unreachable,
}
