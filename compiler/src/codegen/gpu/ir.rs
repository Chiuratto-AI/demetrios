//! GPU Intermediate Representation
//!
//! A specialized IR for GPU code that captures:
//! - Thread hierarchy (grid, block, thread)
//! - Memory spaces (global, shared, local, constant)
//! - Synchronization primitives
//! - GPU-specific operations

use rustc_hash::FxHashMap;
use std::fmt;

/// GPU module containing kernels
#[derive(Debug, Clone)]
pub struct GpuModule {
    /// Module name
    pub name: String,

    /// Kernel functions
    pub kernels: FxHashMap<String, GpuKernel>,

    /// Device functions (callable from kernels)
    pub device_functions: FxHashMap<String, GpuFunction>,

    /// Global constants
    pub constants: Vec<GpuConstant>,

    /// Target architecture
    pub target: GpuTarget,
}

/// GPU target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTarget {
    /// NVIDIA CUDA (PTX)
    Cuda { compute_capability: (u32, u32) },

    /// Vulkan SPIR-V
    Vulkan { version: (u32, u32) },

    /// OpenCL SPIR-V
    OpenCL { version: (u32, u32) },

    /// AMD ROCm (future)
    Rocm,

    /// Intel oneAPI (future)
    OneApi,
}

impl Default for GpuTarget {
    fn default() -> Self {
        GpuTarget::Cuda {
            compute_capability: (7, 5),
        }
    }
}

impl fmt::Display for GpuTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuTarget::Cuda { compute_capability } => {
                write!(
                    f,
                    "CUDA sm_{}{}",
                    compute_capability.0, compute_capability.1
                )
            }
            GpuTarget::Vulkan { version } => {
                write!(f, "Vulkan {}.{}", version.0, version.1)
            }
            GpuTarget::OpenCL { version } => {
                write!(f, "OpenCL {}.{}", version.0, version.1)
            }
            GpuTarget::Rocm => write!(f, "ROCm"),
            GpuTarget::OneApi => write!(f, "oneAPI"),
        }
    }
}

/// GPU kernel function
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name
    pub name: String,

    /// Parameters
    pub params: Vec<GpuParam>,

    /// Shared memory declarations
    pub shared_memory: Vec<SharedMemDecl>,

    /// Basic blocks
    pub blocks: Vec<GpuBlock>,

    /// Entry block
    pub entry: BlockId,

    /// Maximum threads per block (optional hint)
    pub max_threads: Option<u32>,

    /// Required shared memory (bytes)
    pub shared_mem_size: u32,
}

/// GPU device function (non-kernel, callable from GPU)
#[derive(Debug, Clone)]
pub struct GpuFunction {
    /// Function name
    pub name: String,

    /// Parameters
    pub params: Vec<GpuParam>,

    /// Return type
    pub return_type: GpuType,

    /// Basic blocks
    pub blocks: Vec<GpuBlock>,

    /// Entry block
    pub entry: BlockId,

    /// Is inline hint
    pub inline: bool,
}

/// GPU parameter
#[derive(Debug, Clone)]
pub struct GpuParam {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub ty: GpuType,

    /// Memory space
    pub space: MemorySpace,

    /// Is restrict (no aliasing)
    pub restrict: bool,
}

/// GPU type
#[derive(Debug, Clone, PartialEq)]
pub enum GpuType {
    // Scalar types
    Void,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,

    // Vector types
    Vec2(Box<GpuType>),
    Vec3(Box<GpuType>),
    Vec4(Box<GpuType>),

    // Pointer types
    Ptr(Box<GpuType>, MemorySpace),

    // Array types
    Array(Box<GpuType>, u32),

    // Struct types
    Struct(String, Vec<(String, GpuType)>),
}

impl GpuType {
    pub fn size_bytes(&self) -> u32 {
        match self {
            GpuType::Void => 0,
            GpuType::Bool | GpuType::I8 | GpuType::U8 => 1,
            GpuType::I16 | GpuType::U16 | GpuType::F16 => 2,
            GpuType::I32 | GpuType::U32 | GpuType::F32 => 4,
            GpuType::I64 | GpuType::U64 | GpuType::F64 => 8,
            GpuType::Vec2(t) => t.size_bytes() * 2,
            GpuType::Vec3(t) => t.size_bytes() * 3,
            GpuType::Vec4(t) => t.size_bytes() * 4,
            GpuType::Ptr(_, _) => 8,
            GpuType::Array(t, n) => t.size_bytes() * n,
            GpuType::Struct(_, fields) => fields.iter().map(|(_, t)| t.size_bytes()).sum(),
        }
    }

    pub fn alignment(&self) -> u32 {
        match self {
            GpuType::Vec2(t) | GpuType::Vec3(t) | GpuType::Vec4(t) => t.alignment() * 2,
            GpuType::Array(t, _) => t.alignment(),
            GpuType::Struct(_, fields) => {
                fields.iter().map(|(_, t)| t.alignment()).max().unwrap_or(1)
            }
            _ => self.size_bytes().max(1),
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, GpuType::F16 | GpuType::F32 | GpuType::F64)
    }

    /// Check if this is a signed integer type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            GpuType::I8 | GpuType::I16 | GpuType::I32 | GpuType::I64
        )
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            GpuType::U8 | GpuType::U16 | GpuType::U32 | GpuType::U64
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        self.is_signed() || self.is_unsigned()
    }
}

impl fmt::Display for GpuType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuType::Void => write!(f, "void"),
            GpuType::Bool => write!(f, "bool"),
            GpuType::I8 => write!(f, "i8"),
            GpuType::I16 => write!(f, "i16"),
            GpuType::I32 => write!(f, "i32"),
            GpuType::I64 => write!(f, "i64"),
            GpuType::U8 => write!(f, "u8"),
            GpuType::U16 => write!(f, "u16"),
            GpuType::U32 => write!(f, "u32"),
            GpuType::U64 => write!(f, "u64"),
            GpuType::F16 => write!(f, "f16"),
            GpuType::F32 => write!(f, "f32"),
            GpuType::F64 => write!(f, "f64"),
            GpuType::Vec2(t) => write!(f, "vec2<{}>", t),
            GpuType::Vec3(t) => write!(f, "vec3<{}>", t),
            GpuType::Vec4(t) => write!(f, "vec4<{}>", t),
            GpuType::Ptr(t, space) => write!(f, "*{:?} {}", space, t),
            GpuType::Array(t, n) => write!(f, "[{}; {}]", t, n),
            GpuType::Struct(name, _) => write!(f, "struct {}", name),
        }
    }
}

/// Memory space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    /// Global device memory (DRAM)
    Global,

    /// Shared memory (on-chip, per block)
    Shared,

    /// Local memory (per thread, register spill)
    Local,

    /// Constant memory (cached, read-only)
    Constant,

    /// Texture memory (cached, 2D locality)
    Texture,

    /// Generic (resolved at runtime)
    Generic,
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemorySpace::Global => write!(f, "global"),
            MemorySpace::Shared => write!(f, "shared"),
            MemorySpace::Local => write!(f, "local"),
            MemorySpace::Constant => write!(f, "constant"),
            MemorySpace::Texture => write!(f, "texture"),
            MemorySpace::Generic => write!(f, "generic"),
        }
    }
}

/// Shared memory declaration
#[derive(Debug, Clone)]
pub struct SharedMemDecl {
    /// Variable name
    pub name: String,

    /// Element type
    pub elem_type: GpuType,

    /// Number of elements
    pub size: u32,

    /// Alignment
    pub align: u32,
}

/// Global constant
#[derive(Debug, Clone)]
pub struct GpuConstant {
    /// Constant name
    pub name: String,

    /// Type
    pub ty: GpuType,

    /// Value
    pub value: GpuConstValue,
}

/// Constant value
#[derive(Debug, Clone)]
pub enum GpuConstValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<GpuConstValue>),
    Struct(Vec<GpuConstValue>),
}

/// Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BB{}", self.0)
    }
}

/// Value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// GPU basic block
#[derive(Debug, Clone)]
pub struct GpuBlock {
    /// Block ID
    pub id: BlockId,

    /// Block label
    pub label: String,

    /// Instructions
    pub instructions: Vec<(ValueId, GpuOp)>,

    /// Terminator
    pub terminator: GpuTerminator,
}

/// GPU operations
#[derive(Debug, Clone)]
pub enum GpuOp {
    // === Constants ===
    ConstInt(i64, GpuType),
    ConstFloat(f64, GpuType),
    ConstBool(bool),

    // === Arithmetic ===
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    Rem(ValueId, ValueId),
    Neg(ValueId),

    // === Floating-point ===
    FAdd(ValueId, ValueId),
    FSub(ValueId, ValueId),
    FMul(ValueId, ValueId),
    FDiv(ValueId, ValueId),
    FNeg(ValueId),

    // === Fast math (relaxed precision) ===
    FMulAdd(ValueId, ValueId, ValueId), // a * b + c
    FastSin(ValueId),
    FastCos(ValueId),
    FastExp(ValueId),
    FastLog(ValueId),
    FastSqrt(ValueId),
    FastRsqrt(ValueId), // 1/sqrt(x)

    // === Comparisons ===
    Eq(ValueId, ValueId),
    Ne(ValueId, ValueId),
    Lt(ValueId, ValueId),
    Le(ValueId, ValueId),
    Gt(ValueId, ValueId),
    Ge(ValueId, ValueId),

    // Float comparisons
    FEq(ValueId, ValueId),
    FNe(ValueId, ValueId),
    FLt(ValueId, ValueId),
    FLe(ValueId, ValueId),
    FGt(ValueId, ValueId),
    FGe(ValueId, ValueId),

    // === Logical ===
    And(ValueId, ValueId),
    Or(ValueId, ValueId),
    Xor(ValueId, ValueId),
    Not(ValueId),

    // === Bit operations ===
    Shl(ValueId, ValueId),
    Shr(ValueId, ValueId),  // Arithmetic
    LShr(ValueId, ValueId), // Logical
    BitAnd(ValueId, ValueId),
    BitOr(ValueId, ValueId),
    BitXor(ValueId, ValueId),
    BitNot(ValueId),
    PopCount(ValueId),
    Clz(ValueId), // Count leading zeros
    Ctz(ValueId), // Count trailing zeros

    // === Conversions ===
    Trunc(ValueId, GpuType),
    ZExt(ValueId, GpuType),
    SExt(ValueId, GpuType),
    FpTrunc(ValueId, GpuType),
    FpExt(ValueId, GpuType),
    FpToSi(ValueId, GpuType),
    FpToUi(ValueId, GpuType),
    SiToFp(ValueId, GpuType),
    UiToFp(ValueId, GpuType),
    Bitcast(ValueId, GpuType),

    // === Memory ===
    Load(ValueId, MemorySpace),
    Store(ValueId, ValueId, MemorySpace), // ptr, value

    // Atomic operations
    AtomicAdd(ValueId, ValueId),
    AtomicSub(ValueId, ValueId),
    AtomicMin(ValueId, ValueId),
    AtomicMax(ValueId, ValueId),
    AtomicAnd(ValueId, ValueId),
    AtomicOr(ValueId, ValueId),
    AtomicXor(ValueId, ValueId),
    AtomicExch(ValueId, ValueId),
    AtomicCas(ValueId, ValueId, ValueId), // ptr, compare, value

    // === Address computation ===
    GetElementPtr(ValueId, Vec<ValueId>),
    PtrToInt(ValueId),
    IntToPtr(ValueId, GpuType),

    // === GPU Intrinsics ===
    ThreadIdX,
    ThreadIdY,
    ThreadIdZ,
    BlockIdX,
    BlockIdY,
    BlockIdZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
    GridDimX,
    GridDimY,
    GridDimZ,

    WarpId,
    LaneId,
    WarpSize,

    // === Synchronization ===
    SyncThreads,   // Block-level barrier
    SyncWarp(u32), // Warp-level sync (mask)
    MemoryFence(MemorySpace),

    // === Warp operations ===
    WarpShuffle(ValueId, ValueId),     // value, lane
    WarpShuffleUp(ValueId, ValueId),   // value, delta
    WarpShuffleDown(ValueId, ValueId), // value, delta
    WarpShuffleXor(ValueId, ValueId),  // value, mask
    WarpVote(WarpVoteOp, ValueId),     // all, any, ballot
    WarpReduce(WarpReduceOp, ValueId), // sum, min, max
    WarpMatch(ValueId),                // Find matching lanes

    // === Texture/Surface ===
    TexFetch(ValueId, ValueId),            // texture, coord
    TexFetch2D(ValueId, ValueId, ValueId), // texture, x, y
    SurfRead(ValueId, ValueId),            // surface, coord
    SurfWrite(ValueId, ValueId, ValueId),  // surface, coord, value

    // === Control flow ===
    Phi(Vec<(BlockId, ValueId)>),
    Select(ValueId, ValueId, ValueId), // cond, true, false

    // === Function call ===
    Call(String, Vec<ValueId>),

    // === Parameter ===
    Param(u32),

    // === Shared memory ===
    SharedAddr(String), // Get address of shared memory variable
}

/// Warp vote operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpVoteOp {
    All,    // All lanes true?
    Any,    // Any lane true?
    Ballot, // Bitmask of true lanes
    Eq,     // All lanes same value?
}

/// Warp reduce operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReduceOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

/// GPU terminator
#[derive(Debug, Clone)]
pub enum GpuTerminator {
    /// Unconditional branch
    Br(BlockId),

    /// Conditional branch
    CondBr(ValueId, BlockId, BlockId),

    /// Return from kernel (void)
    ReturnVoid,

    /// Return value (device function only)
    Return(ValueId),

    /// Unreachable (after divergent exit)
    Unreachable,
}

impl GpuModule {
    pub fn new(name: impl Into<String>, target: GpuTarget) -> Self {
        Self {
            name: name.into(),
            kernels: FxHashMap::default(),
            device_functions: FxHashMap::default(),
            constants: Vec::new(),
            target,
        }
    }

    pub fn add_kernel(&mut self, kernel: GpuKernel) {
        self.kernels.insert(kernel.name.clone(), kernel);
    }

    pub fn add_device_function(&mut self, func: GpuFunction) {
        self.device_functions.insert(func.name.clone(), func);
    }

    pub fn add_constant(&mut self, constant: GpuConstant) {
        self.constants.push(constant);
    }

    /// Get the total number of functions (kernels + device functions)
    pub fn function_count(&self) -> usize {
        self.kernels.len() + self.device_functions.len()
    }
}

impl GpuKernel {
    /// Create a new empty kernel
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            shared_memory: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
            max_threads: None,
            shared_mem_size: 0,
        }
    }

    /// Add a parameter to the kernel
    pub fn add_param(&mut self, param: GpuParam) {
        self.params.push(param);
    }

    /// Add a shared memory declaration
    pub fn add_shared_memory(&mut self, decl: SharedMemDecl) {
        self.shared_mem_size += decl.elem_type.size_bytes() * decl.size;
        self.shared_memory.push(decl);
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: GpuBlock) {
        self.blocks.push(block);
    }

    /// Get the number of parameters
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

impl GpuFunction {
    /// Create a new empty device function
    pub fn new(name: impl Into<String>, return_type: GpuType) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            return_type,
            blocks: Vec::new(),
            entry: BlockId(0),
            inline: false,
        }
    }

    /// Add a parameter to the function
    pub fn add_param(&mut self, param: GpuParam) {
        self.params.push(param);
    }

    /// Add a basic block
    pub fn add_block(&mut self, block: GpuBlock) {
        self.blocks.push(block);
    }
}

impl GpuBlock {
    /// Create a new empty block
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            instructions: Vec::new(),
            terminator: GpuTerminator::Unreachable,
        }
    }

    /// Add an instruction to the block
    pub fn add_instruction(&mut self, value_id: ValueId, op: GpuOp) {
        self.instructions.push((value_id, op));
    }

    /// Set the terminator for the block
    pub fn set_terminator(&mut self, terminator: GpuTerminator) {
        self.terminator = terminator;
    }
}

/// Builder for GPU modules
pub struct GpuModuleBuilder {
    module: GpuModule,
    next_value_id: u32,
    next_block_id: u32,
}

impl GpuModuleBuilder {
    pub fn new(name: impl Into<String>, target: GpuTarget) -> Self {
        Self {
            module: GpuModule::new(name, target),
            next_value_id: 0,
            next_block_id: 0,
        }
    }

    /// Get the next value ID
    pub fn next_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Get the next block ID
    pub fn next_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Add a kernel to the module
    pub fn add_kernel(&mut self, kernel: GpuKernel) {
        self.module.add_kernel(kernel);
    }

    /// Add a device function to the module
    pub fn add_device_function(&mut self, func: GpuFunction) {
        self.module.add_device_function(func);
    }

    /// Build the module
    pub fn build(self) -> GpuModule {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_type_size() {
        assert_eq!(GpuType::Void.size_bytes(), 0);
        assert_eq!(GpuType::Bool.size_bytes(), 1);
        assert_eq!(GpuType::I32.size_bytes(), 4);
        assert_eq!(GpuType::I64.size_bytes(), 8);
        assert_eq!(GpuType::F32.size_bytes(), 4);
        assert_eq!(GpuType::F64.size_bytes(), 8);
        assert_eq!(GpuType::Vec2(Box::new(GpuType::F32)).size_bytes(), 8);
        assert_eq!(GpuType::Vec3(Box::new(GpuType::F32)).size_bytes(), 12);
        assert_eq!(GpuType::Vec4(Box::new(GpuType::F32)).size_bytes(), 16);
        assert_eq!(GpuType::Array(Box::new(GpuType::F32), 10).size_bytes(), 40);
    }

    #[test]
    fn test_gpu_type_properties() {
        assert!(GpuType::F32.is_float());
        assert!(GpuType::F64.is_float());
        assert!(!GpuType::I32.is_float());

        assert!(GpuType::I32.is_signed());
        assert!(!GpuType::U32.is_signed());

        assert!(GpuType::U32.is_unsigned());
        assert!(!GpuType::I32.is_unsigned());

        assert!(GpuType::I32.is_integer());
        assert!(GpuType::U64.is_integer());
        assert!(!GpuType::F32.is_integer());
    }

    #[test]
    fn test_gpu_module_creation() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let kernel = GpuKernel::new("my_kernel");
        module.add_kernel(kernel);

        assert_eq!(module.kernels.len(), 1);
        assert!(module.kernels.contains_key("my_kernel"));
    }

    #[test]
    fn test_gpu_kernel_building() {
        let mut kernel = GpuKernel::new("add_one");

        kernel.add_param(GpuParam {
            name: "data".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        kernel.add_shared_memory(SharedMemDecl {
            name: "cache".to_string(),
            elem_type: GpuType::F32,
            size: 256,
            align: 4,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        assert_eq!(kernel.param_count(), 1);
        assert_eq!(kernel.shared_mem_size, 256 * 4);
        assert_eq!(kernel.blocks.len(), 1);
    }

    #[test]
    fn test_gpu_target_display() {
        let cuda = GpuTarget::Cuda {
            compute_capability: (8, 6),
        };
        assert_eq!(format!("{}", cuda), "CUDA sm_86");

        let vulkan = GpuTarget::Vulkan { version: (1, 2) };
        assert_eq!(format!("{}", vulkan), "Vulkan 1.2");
    }
}
