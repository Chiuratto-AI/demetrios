//! GPU IR Instructions
//!
//! Defines the instruction set for GPU intermediate representation,
//! mapping closely to PTX/SPIR-V operations.

use super::types::{AddressSpace, GpuType, ScalarType};
use std::fmt;

/// Value identifier in the IR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Block identifier for control flow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Comparison operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
}

impl CmpOp {
    pub fn ptx_suffix(self, signed: bool) -> &'static str {
        match (self, signed) {
            (CmpOp::Eq, _) => "eq",
            (CmpOp::Ne, _) => "ne",
            (CmpOp::Lt, true) => "lt",
            (CmpOp::Lt, false) => "lo",
            (CmpOp::Le, true) => "le",
            (CmpOp::Le, false) => "ls",
            (CmpOp::Gt, true) => "gt",
            (CmpOp::Gt, false) => "hi",
            (CmpOp::Ge, true) => "ge",
            (CmpOp::Ge, false) => "hs",
        }
    }
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Eq => write!(f, "eq"),
            CmpOp::Ne => write!(f, "ne"),
            CmpOp::Lt => write!(f, "lt"),
            CmpOp::Le => write!(f, "le"),
            CmpOp::Gt => write!(f, "gt"),
            CmpOp::Ge => write!(f, "ge"),
        }
    }
}

/// Binary arithmetic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    // Integer operations
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,

    // Floating-point operations
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,

    // Min/Max
    Min,
    Max,
    FMin,
    FMax,
}

impl BinOp {
    pub fn is_float(self) -> bool {
        matches!(
            self,
            BinOp::FAdd
                | BinOp::FSub
                | BinOp::FMul
                | BinOp::FDiv
                | BinOp::FRem
                | BinOp::FMin
                | BinOp::FMax
        )
    }

    pub fn ptx_name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Rem => "rem",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::Shr => "shr",
            BinOp::FAdd => "add",
            BinOp::FSub => "sub",
            BinOp::FMul => "mul",
            BinOp::FDiv => "div",
            BinOp::FRem => "rem",
            BinOp::Min => "min",
            BinOp::Max => "max",
            BinOp::FMin => "min",
            BinOp::FMax => "max",
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ptx_name())
    }
}

/// Unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    // Integer
    Neg,
    Not,

    // Floating-point
    FNeg,
    FAbs,
    FSqrt,
    FRsqrt, // Reciprocal square root (1/sqrt(x))
    FRcp,   // Reciprocal (1/x)
    FSin,
    FCos,
    FExp2,
    FLog2,
    FFloor,
    FCeil,
    FTrunc,
    FRound,
}

impl UnaryOp {
    pub fn is_float(self) -> bool {
        !matches!(self, UnaryOp::Neg | UnaryOp::Not)
    }

    pub fn ptx_name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Not => "not",
            UnaryOp::FNeg => "neg",
            UnaryOp::FAbs => "abs",
            UnaryOp::FSqrt => "sqrt",
            UnaryOp::FRsqrt => "rsqrt",
            UnaryOp::FRcp => "rcp",
            UnaryOp::FSin => "sin",
            UnaryOp::FCos => "cos",
            UnaryOp::FExp2 => "ex2",
            UnaryOp::FLog2 => "lg2",
            UnaryOp::FFloor => "floor",
            UnaryOp::FCeil => "ceil",
            UnaryOp::FTrunc => "trunc",
            UnaryOp::FRound => "round",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ptx_name())
    }
}

/// Atomic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Min,
    Max,
    Exch,
    CAS, // Compare-and-swap
}

impl AtomicOp {
    pub fn ptx_name(self) -> &'static str {
        match self {
            AtomicOp::Add => "add",
            AtomicOp::Sub => "sub",
            AtomicOp::And => "and",
            AtomicOp::Or => "or",
            AtomicOp::Xor => "xor",
            AtomicOp::Min => "min",
            AtomicOp::Max => "max",
            AtomicOp::Exch => "exch",
            AtomicOp::CAS => "cas",
        }
    }
}

impl fmt::Display for AtomicOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ptx_name())
    }
}

/// Memory ordering for synchronization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryOrder {
    /// Relaxed ordering (no synchronization)
    #[default]
    Relaxed,
    /// Acquire semantics
    Acquire,
    /// Release semantics
    Release,
    /// Acquire-release semantics
    AcqRel,
    /// Sequential consistency
    SeqCst,
}

impl fmt::Display for MemoryOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryOrder::Relaxed => write!(f, "relaxed"),
            MemoryOrder::Acquire => write!(f, "acquire"),
            MemoryOrder::Release => write!(f, "release"),
            MemoryOrder::AcqRel => write!(f, "acq_rel"),
            MemoryOrder::SeqCst => write!(f, "seq_cst"),
        }
    }
}

/// Barrier scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BarrierScope {
    /// Thread block (CTA)
    Block,
    /// GPU (device)
    Device,
    /// System (including host)
    System,
}

impl fmt::Display for BarrierScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BarrierScope::Block => write!(f, "cta"),
            BarrierScope::Device => write!(f, "gpu"),
            BarrierScope::System => write!(f, "sys"),
        }
    }
}

/// Constant value
#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Constant {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Constant::Bool(_) => ScalarType::Bool,
            Constant::I8(_) => ScalarType::I8,
            Constant::I16(_) => ScalarType::I16,
            Constant::I32(_) => ScalarType::I32,
            Constant::I64(_) => ScalarType::I64,
            Constant::U8(_) => ScalarType::U8,
            Constant::U16(_) => ScalarType::U16,
            Constant::U32(_) => ScalarType::U32,
            Constant::U64(_) => ScalarType::U64,
            Constant::F32(_) => ScalarType::F32,
            Constant::F64(_) => ScalarType::F64,
        }
    }

    pub fn gpu_type(&self) -> GpuType {
        GpuType::Scalar(self.scalar_type())
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constant::Bool(v) => write!(f, "{}", if *v { "1" } else { "0" }),
            Constant::I8(v) => write!(f, "{}", v),
            Constant::I16(v) => write!(f, "{}", v),
            Constant::I32(v) => write!(f, "{}", v),
            Constant::I64(v) => write!(f, "{}", v),
            Constant::U8(v) => write!(f, "{}", v),
            Constant::U16(v) => write!(f, "{}", v),
            Constant::U32(v) => write!(f, "{}", v),
            Constant::U64(v) => write!(f, "{}", v),
            Constant::F32(v) => write!(f, "0F{:08X}", v.to_bits()),
            Constant::F64(v) => write!(f, "0D{:016X}", v.to_bits()),
        }
    }
}

/// GPU IR Instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    // Constants
    Const {
        dst: ValueId,
        value: Constant,
    },

    // Binary operations
    BinOp {
        dst: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: ScalarType,
    },

    // Unary operations
    UnaryOp {
        dst: ValueId,
        op: UnaryOp,
        src: ValueId,
        ty: ScalarType,
    },

    // Comparison
    Cmp {
        dst: ValueId,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: ScalarType,
    },

    // Type conversion
    Convert {
        dst: ValueId,
        src: ValueId,
        from: ScalarType,
        to: ScalarType,
    },

    // Bitcast (reinterpret bits)
    Bitcast {
        dst: ValueId,
        src: ValueId,
        to: GpuType,
    },

    // Memory operations
    Load {
        dst: ValueId,
        ptr: ValueId,
        ty: GpuType,
        space: AddressSpace,
        volatile: bool,
    },

    Store {
        ptr: ValueId,
        value: ValueId,
        ty: GpuType,
        space: AddressSpace,
        volatile: bool,
    },

    // Atomic operations
    Atomic {
        dst: ValueId,
        op: AtomicOp,
        ptr: ValueId,
        value: ValueId,
        ty: ScalarType,
        space: AddressSpace,
        order: MemoryOrder,
    },

    // Compare-and-swap (separate due to extra operand)
    AtomicCAS {
        dst: ValueId,
        ptr: ValueId,
        expected: ValueId,
        desired: ValueId,
        ty: ScalarType,
        space: AddressSpace,
    },

    // Address arithmetic
    GetElementPtr {
        dst: ValueId,
        base: ValueId,
        indices: Vec<ValueId>,
        pointee_ty: GpuType,
    },

    // Control flow
    Branch {
        target: BlockId,
    },

    CondBranch {
        cond: ValueId,
        true_target: BlockId,
        false_target: BlockId,
    },

    Return {
        value: Option<ValueId>,
    },

    // Synchronization
    Barrier {
        scope: BarrierScope,
    },

    MemFence {
        scope: BarrierScope,
        order: MemoryOrder,
    },

    // Thread identification
    ThreadIdx {
        dst: ValueId,
        dim: Dimension,
    },

    BlockIdx {
        dst: ValueId,
        dim: Dimension,
    },

    BlockDim {
        dst: ValueId,
        dim: Dimension,
    },

    GridDim {
        dst: ValueId,
        dim: Dimension,
    },

    // Warp-level operations
    WarpId {
        dst: ValueId,
    },

    LaneId {
        dst: ValueId,
    },

    WarpShuffle {
        dst: ValueId,
        src: ValueId,
        lane: ValueId,
        ty: ScalarType,
    },

    WarpVote {
        dst: ValueId,
        pred: ValueId,
        kind: WarpVoteKind,
    },

    WarpReduce {
        dst: ValueId,
        src: ValueId,
        op: BinOp,
        ty: ScalarType,
    },

    // Function call
    Call {
        dst: Option<ValueId>,
        func: String,
        args: Vec<ValueId>,
    },

    // Select (ternary)
    Select {
        dst: ValueId,
        cond: ValueId,
        true_val: ValueId,
        false_val: ValueId,
    },

    // Phi node (SSA)
    Phi {
        dst: ValueId,
        incoming: Vec<(BlockId, ValueId)>,
    },

    // Fused multiply-add (a * b + c)
    FMA {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
        c: ValueId,
        ty: ScalarType,
    },

    // Shared memory allocation
    SharedAlloc {
        dst: ValueId,
        ty: GpuType,
        size: usize,
    },
}

/// Dimension for thread/block indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    X,
    Y,
    Z,
}

impl Dimension {
    pub fn ptx_suffix(self) -> &'static str {
        match self {
            Dimension::X => "x",
            Dimension::Y => "y",
            Dimension::Z => "z",
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ptx_suffix())
    }
}

/// Warp vote operation kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarpVoteKind {
    All, // All threads have true predicate
    Any, // Any thread has true predicate
    Uni, // All threads have same predicate
}

impl Instruction {
    /// Get the destination value ID if this instruction produces a value
    pub fn dst(&self) -> Option<ValueId> {
        match self {
            Instruction::Const { dst, .. }
            | Instruction::BinOp { dst, .. }
            | Instruction::UnaryOp { dst, .. }
            | Instruction::Cmp { dst, .. }
            | Instruction::Convert { dst, .. }
            | Instruction::Bitcast { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Atomic { dst, .. }
            | Instruction::AtomicCAS { dst, .. }
            | Instruction::GetElementPtr { dst, .. }
            | Instruction::ThreadIdx { dst, .. }
            | Instruction::BlockIdx { dst, .. }
            | Instruction::BlockDim { dst, .. }
            | Instruction::GridDim { dst, .. }
            | Instruction::WarpId { dst, .. }
            | Instruction::LaneId { dst, .. }
            | Instruction::WarpShuffle { dst, .. }
            | Instruction::WarpVote { dst, .. }
            | Instruction::WarpReduce { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Phi { dst, .. }
            | Instruction::FMA { dst, .. }
            | Instruction::SharedAlloc { dst, .. } => Some(*dst),

            Instruction::Call { dst, .. } => *dst,

            Instruction::Store { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Return { .. }
            | Instruction::Barrier { .. }
            | Instruction::MemFence { .. } => None,
        }
    }

    /// Check if this instruction is a terminator
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            Instruction::Branch { .. }
                | Instruction::CondBranch { .. }
                | Instruction::Return { .. }
        )
    }

    /// Check if this instruction has side effects
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Instruction::Store { .. }
                | Instruction::Atomic { .. }
                | Instruction::AtomicCAS { .. }
                | Instruction::Call { .. }
                | Instruction::Barrier { .. }
                | Instruction::MemFence { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_display() {
        assert_eq!(Constant::I32(42).to_string(), "42");
        assert_eq!(Constant::Bool(true).to_string(), "1");
        assert_eq!(Constant::F32(1.0).to_string(), "0F3F800000");
    }

    #[test]
    fn test_instruction_dst() {
        let inst = Instruction::Const {
            dst: ValueId(0),
            value: Constant::I32(42),
        };
        assert_eq!(inst.dst(), Some(ValueId(0)));

        let inst = Instruction::Barrier {
            scope: BarrierScope::Block,
        };
        assert_eq!(inst.dst(), None);
    }

    #[test]
    fn test_instruction_terminator() {
        let inst = Instruction::Branch { target: BlockId(0) };
        assert!(inst.is_terminator());

        let inst = Instruction::Const {
            dst: ValueId(0),
            value: Constant::I32(42),
        };
        assert!(!inst.is_terminator());
    }
}
