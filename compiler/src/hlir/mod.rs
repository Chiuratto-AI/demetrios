//! High-Level IR (HLIR) - SSA-based IR
//!
//! HLIR is an SSA-based intermediate representation suitable for optimization
//! and lowering to MLIR/LLVM. It features:
//! - Static Single Assignment (SSA) form
//! - Basic blocks with explicit control flow
//! - Explicit memory operations
//! - Effect annotations

use crate::hir::{Hir, HirItem, HirType};

/// Lower HIR to HLIR
pub fn lower(hir: &Hir) -> HlirModule {
    let mut lowering = HlirLowering::new();
    lowering.lower_hir(hir)
}

/// HLIR module
#[derive(Debug, Clone)]
pub struct HlirModule {
    pub name: String,
    pub functions: Vec<HlirFunction>,
    pub globals: Vec<HlirGlobal>,
    pub types: Vec<HlirTypeDef>,
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
}

/// Function identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

/// Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// Value identifier (SSA value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

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
            HirType::Tuple(_) => {
                // Tuples become anonymous structs
                HlirType::Struct("tuple".to_string())
            }
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

/// HLIR instruction
#[derive(Debug, Clone)]
pub struct HlirInstr {
    pub result: Option<ValueId>,
    pub kind: HlirInstrKind,
}

/// HLIR instruction kind
#[derive(Debug, Clone)]
pub enum HlirInstrKind {
    /// Constant value
    Const(HlirConstant),
    /// Binary operation
    BinOp {
        op: HlirBinOp,
        left: ValueId,
        right: ValueId,
    },
    /// Unary operation
    UnaryOp { op: HlirUnaryOp, operand: ValueId },
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

/// HLIR binary operator
#[derive(Debug, Clone, Copy)]
pub enum HlirBinOp {
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
    // Comparison
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

/// HLIR unary operator
#[derive(Debug, Clone, Copy)]
pub enum HlirUnaryOp {
    Neg,
    FNeg,
    Not,
}

/// HLIR block terminator
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

/// HLIR lowering context
struct HlirLowering {
    next_func_id: u32,
    next_block_id: u32,
    next_value_id: u32,
    functions: Vec<HlirFunction>,
    globals: Vec<HlirGlobal>,
    types: Vec<HlirTypeDef>,
}

impl HlirLowering {
    fn new() -> Self {
        Self {
            next_func_id: 0,
            next_block_id: 0,
            next_value_id: 0,
            functions: Vec::new(),
            globals: Vec::new(),
            types: Vec::new(),
        }
    }

    fn fresh_func_id(&mut self) -> FunctionId {
        let id = FunctionId(self.next_func_id);
        self.next_func_id += 1;
        id
    }

    fn fresh_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    fn fresh_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    fn lower_hir(&mut self, hir: &Hir) -> HlirModule {
        for item in &hir.items {
            match item {
                HirItem::Function(f) => {
                    self.lower_function(f);
                }
                HirItem::Struct(s) => {
                    let fields: Vec<_> = s
                        .fields
                        .iter()
                        .map(|f| (f.name.clone(), HlirType::from_hir(&f.ty)))
                        .collect();
                    self.types.push(HlirTypeDef {
                        name: s.name.clone(),
                        kind: HlirTypeDefKind::Struct(fields),
                    });
                }
                HirItem::Enum(e) => {
                    let variants: Vec<_> = e
                        .variants
                        .iter()
                        .map(|v| {
                            (
                                v.name.clone(),
                                v.fields.iter().map(HlirType::from_hir).collect(),
                            )
                        })
                        .collect();
                    self.types.push(HlirTypeDef {
                        name: e.name.clone(),
                        kind: HlirTypeDefKind::Enum(variants),
                    });
                }
                HirItem::Global(g) => {
                    let global = HlirGlobal {
                        id: self.fresh_value_id(),
                        name: g.name.clone(),
                        ty: HlirType::from_hir(&g.ty),
                        init: None, // TODO: Lower initializer
                        is_const: g.is_const,
                    };
                    self.globals.push(global);
                }
                _ => {
                    // TODO: Handle other items
                }
            }
        }

        HlirModule {
            name: "main".to_string(),
            functions: std::mem::take(&mut self.functions),
            globals: std::mem::take(&mut self.globals),
            types: std::mem::take(&mut self.types),
        }
    }

    fn lower_function(&mut self, f: &crate::hir::HirFn) {
        let func_id = self.fresh_func_id();

        // Create parameters
        let params: Vec<_> =
            f.ty.params
                .iter()
                .map(|p| HlirParam {
                    value: self.fresh_value_id(),
                    name: p.name.clone(),
                    ty: HlirType::from_hir(&p.ty),
                })
                .collect();

        // Create entry block
        let entry_block = HlirBlock {
            id: self.fresh_block_id(),
            label: "entry".to_string(),
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: HlirTerminator::Return(None),
        };

        let func = HlirFunction {
            id: func_id,
            name: f.name.clone(),
            params,
            return_type: HlirType::from_hir(&f.ty.return_type),
            effects: Vec::new(),
            blocks: vec![entry_block],
            is_kernel: false,
        };

        self.functions.push(func);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlir_type_conversion() {
        assert_eq!(HlirType::from_hir(&HirType::I32), HlirType::I32);
        assert_eq!(HlirType::from_hir(&HirType::F64), HlirType::F64);
        assert_eq!(HlirType::from_hir(&HirType::Unit), HlirType::Void);
    }
}
