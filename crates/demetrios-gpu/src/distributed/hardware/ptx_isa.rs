//! Complete PTX ISA reference for distributed computing primitives
//!
//! PTX (Parallel Thread Execution) is NVIDIA's virtual ISA.
//! It compiles to SASS (actual hardware instructions) via ptxas.
//!
//! Key insight: PTX is NOT 1:1 with hardware. It's an abstraction.
//! Understanding both PTX semantics AND hardware mapping is essential.

/// PTX version and target architecture
#[derive(Debug, Clone, Copy)]
pub struct PtxTarget {
    /// PTX version (e.g., 7.5)
    pub version: (u8, u8),
    /// SM target (e.g., sm_80 for A100, sm_89 for L4)
    pub sm: u8,
    /// Address size (32 or 64)
    pub address_size: u8,
}

impl PtxTarget {
    /// SM 8.0 (A100)
    pub fn sm80() -> Self {
        Self {
            version: (7, 5),
            sm: 80,
            address_size: 64,
        }
    }

    /// SM 8.9 (L4/L40)
    pub fn sm89() -> Self {
        Self {
            version: (8, 0),
            sm: 89,
            address_size: 64,
        }
    }

    /// SM 9.0 (H100)
    pub fn sm90() -> Self {
        Self {
            version: (8, 3),
            sm: 90,
            address_size: 64,
        }
    }

    /// Generate PTX header
    pub fn header(&self) -> String {
        format!(
            ".version {}.{}\n.target sm_{}\n.address_size {}",
            self.version.0, self.version.1, self.sm, self.address_size
        )
    }
}

// ============================================================================
// REGISTER TYPES
// ============================================================================

/// PTX register types with exact bit widths
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegType {
    /// Predicate register (1 bit, but stored as 32-bit)
    Pred,
    /// 8-bit signed/unsigned
    B8,
    S8,
    U8,
    /// 16-bit signed/unsigned/float
    B16,
    S16,
    U16,
    F16,
    BF16,
    /// 32-bit signed/unsigned/float
    B32,
    S32,
    U32,
    F32,
    /// 64-bit signed/unsigned/float
    B64,
    S64,
    U64,
    F64,
    /// 128-bit (for vector loads)
    B128,
}

impl RegType {
    /// Get bit width of register type
    pub fn bits(&self) -> u32 {
        match self {
            Self::Pred => 1,
            Self::B8 | Self::S8 | Self::U8 => 8,
            Self::B16 | Self::S16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::B32 | Self::S32 | Self::U32 | Self::F32 => 32,
            Self::B64 | Self::S64 | Self::U64 | Self::F64 => 64,
            Self::B128 => 128,
        }
    }

    /// Get byte size
    pub fn bytes(&self) -> usize {
        (self.bits() as usize + 7) / 8
    }

    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Pred => "pred",
            Self::B8 => "b8",
            Self::S8 => "s8",
            Self::U8 => "u8",
            Self::B16 => "b16",
            Self::S16 => "s16",
            Self::U16 => "u16",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::B32 => "b32",
            Self::S32 => "s32",
            Self::U32 => "u32",
            Self::F32 => "f32",
            Self::B64 => "b64",
            Self::S64 => "s64",
            Self::U64 => "u64",
            Self::F64 => "f64",
            Self::B128 => "b128",
        }
    }

    /// Check if this is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Check if this is a signed integer type
    pub fn is_signed(&self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }
}

// ============================================================================
// MEMORY SPACES
// ============================================================================

/// PTX state spaces (memory regions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpace {
    /// Registers (fastest, limited)
    Reg,
    /// Special registers (read-only: %tid, %ctaid, etc.)
    Sreg,
    /// Constant memory (cached, read-only, 64KB)
    Const,
    /// Global memory (DRAM, ~400 cycles latency)
    Global,
    /// Local memory (per-thread, spills to DRAM)
    Local,
    /// Shared memory (per-block, ~20 cycles, bank conflicts)
    Shared,
    /// Texture memory (cached, read-only, spatial locality)
    Tex,
    /// Surface memory (read-write textures)
    Surf,
    /// Parameter space (kernel arguments)
    Param,
}

impl StateSpace {
    /// Typical latency in cycles
    pub fn latency_cycles(&self) -> u32 {
        match self {
            Self::Reg | Self::Sreg => 0,   // Same cycle
            Self::Const => 4,              // L1 cached
            Self::Shared => 20,            // Shared memory
            Self::Local => 400,            // Spills to DRAM
            Self::Global => 400,           // DRAM
            Self::Tex | Self::Surf => 100, // Texture cache
            Self::Param => 4,              // Constant cache
        }
    }

    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Reg | Self::Sreg => "",
            Self::Const => ".const",
            Self::Global => ".global",
            Self::Local => ".local",
            Self::Shared => ".shared",
            Self::Tex => ".tex",
            Self::Surf => ".surf",
            Self::Param => ".param",
        }
    }

    /// Is this a cached memory space?
    pub fn is_cached(&self) -> bool {
        matches!(self, Self::Const | Self::Tex | Self::Global)
    }
}

// ============================================================================
// MEMORY ORDERING
// ============================================================================

/// Memory ordering semantics (PTX memory model)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryOrdering {
    /// No ordering constraints (default)
    Relaxed,
    /// Acquire semantics: subsequent ops see prior releases
    Acquire,
    /// Release semantics: prior ops visible to subsequent acquires
    Release,
    /// Full fence (both acquire and release)
    AcqRel,
    /// Sequential consistency (strongest, rarely needed)
    SeqCst,
    /// Weak ordering (weaker than relaxed, for specific patterns)
    Weak,
    /// Memory model "mmio" for device I/O
    Mmio,
    /// Volatile (no caching, immediate visibility)
    Volatile,
}

impl MemoryOrdering {
    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Relaxed => ".relaxed",
            Self::Acquire => ".acquire",
            Self::Release => ".release",
            Self::AcqRel => ".acq_rel",
            Self::SeqCst => ".sc",
            Self::Weak => ".weak",
            Self::Mmio => ".mmio",
            Self::Volatile => ".volatile",
        }
    }
}

/// Memory scope (visibility of memory operations)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryScope {
    /// CTA (Cooperative Thread Array = block)
    Cta,
    /// GPU (all CTAs on this GPU)
    Gpu,
    /// System (all GPUs and CPU)
    Sys,
}

impl MemoryScope {
    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Cta => ".cta",
            Self::Gpu => ".gpu",
            Self::Sys => ".sys",
        }
    }

    /// Check if this scope includes another
    pub fn includes(&self, other: &Self) -> bool {
        *self >= *other
    }
}

// ============================================================================
// CACHE OPERATORS
// ============================================================================

/// Cache operators for loads/stores
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOp {
    /// Cache at all levels (default)
    Ca,
    /// Cache at global level only (bypass L1)
    Cg,
    /// Cache streaming (evict first)
    Cs,
    /// Last use (hint for eviction)
    Lu,
    /// Cache volatile (bypass all caches)
    Cv,
    /// Write-back (default for stores)
    Wb,
    /// Write-through
    Wt,
}

impl CacheOp {
    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Ca => ".ca",
            Self::Cg => ".cg",
            Self::Cs => ".cs",
            Self::Lu => ".lu",
            Self::Cv => ".cv",
            Self::Wb => ".wb",
            Self::Wt => ".wt",
        }
    }
}

// ============================================================================
// LOAD/STORE INSTRUCTIONS
// ============================================================================

/// Load instruction with full options
#[derive(Debug, Clone)]
pub struct LdInstruction {
    /// Memory ordering
    pub ordering: Option<MemoryOrdering>,
    /// Scope
    pub scope: Option<MemoryScope>,
    /// State space
    pub space: StateSpace,
    /// Cache operator
    pub cache_op: Option<CacheOp>,
    /// Vector width (1, 2, or 4)
    pub vector: u8,
    /// Data type
    pub dtype: RegType,
}

impl LdInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, addr: &str) -> String {
        let mut instr = String::from("ld");

        if let Some(ord) = &self.ordering {
            instr.push_str(ord.suffix());
        }
        if let Some(scope) = &self.scope {
            instr.push_str(scope.suffix());
        }
        instr.push_str(self.space.suffix());
        if let Some(cache) = &self.cache_op {
            instr.push_str(cache.suffix());
        }
        if self.vector > 1 {
            instr.push_str(&format!(".v{}", self.vector));
        }
        instr.push_str(&format!(".{}", self.dtype.suffix()));

        format!("{} {}, [{}];", instr, dst, addr)
    }

    /// Global load with acquire semantics
    pub fn global_acquire(dtype: RegType) -> Self {
        Self {
            ordering: Some(MemoryOrdering::Acquire),
            scope: Some(MemoryScope::Gpu),
            space: StateSpace::Global,
            cache_op: None,
            vector: 1,
            dtype,
        }
    }

    /// Global load with cache bypass (for non-temporal data)
    pub fn global_streaming(dtype: RegType) -> Self {
        Self {
            ordering: None,
            scope: None,
            space: StateSpace::Global,
            cache_op: Some(CacheOp::Cs),
            vector: 1,
            dtype,
        }
    }

    /// Vectorized load (128-bit)
    pub fn global_vector4(dtype: RegType) -> Self {
        Self {
            ordering: None,
            scope: None,
            space: StateSpace::Global,
            cache_op: Some(CacheOp::Ca),
            vector: 4,
            dtype,
        }
    }

    /// Shared memory load
    pub fn shared(dtype: RegType) -> Self {
        Self {
            ordering: None,
            scope: None,
            space: StateSpace::Shared,
            cache_op: None,
            vector: 1,
            dtype,
        }
    }
}

/// Store instruction with full options
#[derive(Debug, Clone)]
pub struct StInstruction {
    /// Memory ordering
    pub ordering: Option<MemoryOrdering>,
    /// Scope
    pub scope: Option<MemoryScope>,
    /// State space
    pub space: StateSpace,
    /// Cache operator
    pub cache_op: Option<CacheOp>,
    /// Vector width (1, 2, or 4)
    pub vector: u8,
    /// Data type
    pub dtype: RegType,
}

impl StInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self, addr: &str, src: &str) -> String {
        let mut instr = String::from("st");

        if let Some(ord) = &self.ordering {
            instr.push_str(ord.suffix());
        }
        if let Some(scope) = &self.scope {
            instr.push_str(scope.suffix());
        }
        instr.push_str(self.space.suffix());
        if let Some(cache) = &self.cache_op {
            instr.push_str(cache.suffix());
        }
        if self.vector > 1 {
            instr.push_str(&format!(".v{}", self.vector));
        }
        instr.push_str(&format!(".{}", self.dtype.suffix()));

        format!("{} [{}], {};", instr, addr, src)
    }

    /// Global store with release semantics
    pub fn global_release(dtype: RegType) -> Self {
        Self {
            ordering: Some(MemoryOrdering::Release),
            scope: Some(MemoryScope::Gpu),
            space: StateSpace::Global,
            cache_op: None,
            vector: 1,
            dtype,
        }
    }

    /// Shared memory store
    pub fn shared(dtype: RegType) -> Self {
        Self {
            ordering: None,
            scope: None,
            space: StateSpace::Shared,
            cache_op: None,
            vector: 1,
            dtype,
        }
    }
}

// ============================================================================
// ATOMIC INSTRUCTIONS
// ============================================================================

/// Atomic operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    /// Exchange
    Exch,
    /// Compare and swap
    Cas,
    /// Add
    Add,
    /// Increment (mod n)
    Inc,
    /// Decrement (mod n)
    Dec,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
}

impl AtomicOp {
    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Exch => ".exch",
            Self::Cas => ".cas",
            Self::Add => ".add",
            Self::Inc => ".inc",
            Self::Dec => ".dec",
            Self::Min => ".min",
            Self::Max => ".max",
            Self::And => ".and",
            Self::Or => ".or",
            Self::Xor => ".xor",
        }
    }

    /// Throughput in operations per cycle per SM
    pub fn throughput_per_sm(&self) -> f64 {
        match self {
            // All atomics to same address serialize
            // But different addresses can be parallel
            Self::Add => 16.0, // 16 atomic units on A100
            Self::Cas => 8.0,
            _ => 8.0,
        }
    }
}

/// Complete atomic instruction
#[derive(Debug, Clone)]
pub struct AtomInstruction {
    /// Memory ordering
    pub ordering: MemoryOrdering,
    /// Scope
    pub scope: MemoryScope,
    /// State space
    pub space: StateSpace,
    /// Atomic operation
    pub op: AtomicOp,
    /// Data type
    pub dtype: RegType,
}

impl AtomInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, addr: &str, operands: &[&str]) -> String {
        let mut instr = String::from("atom");

        instr.push_str(self.ordering.suffix());
        instr.push_str(self.scope.suffix());
        instr.push_str(self.space.suffix());
        instr.push_str(self.op.suffix());
        instr.push_str(&format!(".{}", self.dtype.suffix()));

        match self.op {
            AtomicOp::Cas => {
                format!(
                    "{} {}, [{}], {}, {};",
                    instr, dst, addr, operands[0], operands[1]
                )
            }
            _ => {
                format!("{} {}, [{}], {};", instr, dst, addr, operands[0])
            }
        }
    }

    /// Atomic add for reductions
    pub fn global_add_relaxed(dtype: RegType) -> Self {
        Self {
            ordering: MemoryOrdering::Relaxed,
            scope: MemoryScope::Gpu,
            space: StateSpace::Global,
            op: AtomicOp::Add,
            dtype,
        }
    }

    /// Atomic CAS for lock-free algorithms
    pub fn global_cas_acqrel(dtype: RegType) -> Self {
        Self {
            ordering: MemoryOrdering::AcqRel,
            scope: MemoryScope::Gpu,
            space: StateSpace::Global,
            op: AtomicOp::Cas,
            dtype,
        }
    }

    /// System-wide atomic for CPU-GPU sync
    pub fn system_add(dtype: RegType) -> Self {
        Self {
            ordering: MemoryOrdering::AcqRel,
            scope: MemoryScope::Sys,
            space: StateSpace::Global,
            op: AtomicOp::Add,
            dtype,
        }
    }

    /// Shared memory atomic
    pub fn shared_add(dtype: RegType) -> Self {
        Self {
            ordering: MemoryOrdering::Relaxed,
            scope: MemoryScope::Cta,
            space: StateSpace::Shared,
            op: AtomicOp::Add,
            dtype,
        }
    }
}

// ============================================================================
// WARP-LEVEL PRIMITIVES
// ============================================================================

/// Warp shuffle modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleMode {
    /// Exchange with specific lane
    Idx,
    /// Exchange with lane + delta
    Up,
    /// Exchange with lane - delta
    Down,
    /// XOR with bitmask (butterfly)
    Bfly,
}

impl ShuffleMode {
    /// Get PTX suffix
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Idx => ".idx",
            Self::Up => ".up",
            Self::Down => ".down",
            Self::Bfly => ".bfly",
        }
    }
}

/// Warp shuffle instruction
#[derive(Debug, Clone)]
pub struct ShflInstruction {
    /// Shuffle mode
    pub mode: ShuffleMode,
    /// Data type
    pub dtype: RegType,
    /// Sync mask (0xffffffff for all lanes)
    pub membermask: u32,
}

impl ShflInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, src: &str, lane_or_delta: &str, clamp: &str) -> String {
        format!(
            "shfl.sync{}.{} {}, {}, {}, {}, 0x{:08x};",
            self.mode.suffix(),
            self.dtype.suffix(),
            dst,
            src,
            lane_or_delta,
            clamp,
            self.membermask
        )
    }

    /// Butterfly shuffle for reduction
    pub fn butterfly(dtype: RegType) -> Self {
        Self {
            mode: ShuffleMode::Bfly,
            dtype,
            membermask: 0xffffffff,
        }
    }

    /// Down shuffle for scan
    pub fn down(dtype: RegType) -> Self {
        Self {
            mode: ShuffleMode::Down,
            dtype,
            membermask: 0xffffffff,
        }
    }

    /// Up shuffle for reverse scan
    pub fn up(dtype: RegType) -> Self {
        Self {
            mode: ShuffleMode::Up,
            dtype,
            membermask: 0xffffffff,
        }
    }

    /// Indexed shuffle for arbitrary communication
    pub fn indexed(dtype: RegType) -> Self {
        Self {
            mode: ShuffleMode::Idx,
            dtype,
            membermask: 0xffffffff,
        }
    }
}

/// Warp vote instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteOp {
    /// All lanes have predicate true
    All,
    /// Any lane has predicate true
    Any,
    /// Ballot: bitmask of predicates
    Ballot,
}

impl VoteOp {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, src: &str, mask: u32) -> String {
        match self {
            Self::All => format!("vote.sync.all.pred {}, {}, 0x{:08x};", dst, src, mask),
            Self::Any => format!("vote.sync.any.pred {}, {}, 0x{:08x};", dst, src, mask),
            Self::Ballot => format!("vote.sync.ballot.b32 {}, {}, 0x{:08x};", dst, src, mask),
        }
    }
}

/// Warp match instruction (SM 7.0+)
#[derive(Debug, Clone)]
pub struct MatchInstruction {
    /// Data type
    pub dtype: RegType,
    /// Member mask
    pub membermask: u32,
}

impl MatchInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst_mask: &str, src: &str) -> String {
        format!(
            "match.sync.any.{} {}, {}, 0x{:08x};",
            self.dtype.suffix(),
            dst_mask,
            src,
            self.membermask
        )
    }
}

// ============================================================================
// SYNCHRONIZATION INSTRUCTIONS
// ============================================================================

/// Barrier types
#[derive(Debug, Clone)]
pub enum BarrierType {
    /// Block-level barrier
    Sync(u32),
    /// Block-level barrier with thread count
    SyncCount { id: u32, count: u32 },
    /// Arrive at barrier (don't wait)
    Arrive { id: u32, count: u32 },
    /// Wait at barrier
    Wait { id: u32 },
}

impl BarrierType {
    /// Generate PTX instruction
    pub fn to_ptx(&self) -> String {
        match self {
            Self::Sync(id) => format!("bar.sync {};", id),
            Self::SyncCount { id, count } => format!("bar.sync {}, {};", id, count),
            Self::Arrive { id, count } => format!("bar.arrive {}, {};", id, count),
            Self::Wait { id } => format!("bar.sync {};", id),
        }
    }
}

/// Memory fence instruction
#[derive(Debug, Clone)]
pub struct FenceInstruction {
    /// Scope
    pub scope: MemoryScope,
    /// Ordering
    pub ordering: MemoryOrdering,
}

impl FenceInstruction {
    /// Generate PTX instruction
    pub fn to_ptx(&self) -> String {
        let op = match self.ordering {
            MemoryOrdering::AcqRel => "fence",
            MemoryOrdering::SeqCst => "fence.sc",
            _ => "membar",
        };

        format!("{}{};", op, self.scope.suffix())
    }

    /// Full fence at GPU scope
    pub fn gpu_fence() -> Self {
        Self {
            scope: MemoryScope::Gpu,
            ordering: MemoryOrdering::AcqRel,
        }
    }

    /// System fence (for CPU-GPU sync)
    pub fn sys_fence() -> Self {
        Self {
            scope: MemoryScope::Sys,
            ordering: MemoryOrdering::AcqRel,
        }
    }

    /// CTA fence
    pub fn cta_fence() -> Self {
        Self {
            scope: MemoryScope::Cta,
            ordering: MemoryOrdering::AcqRel,
        }
    }
}

// ============================================================================
// ASYNCHRONOUS COPY (SM 8.0+)
// ============================================================================

/// Async copy from global to shared memory
#[derive(Debug, Clone)]
pub struct AsyncCopy {
    /// Bytes to copy (4, 8, or 16)
    pub size: u32,
    /// Cache hint
    pub cache: Option<CacheOp>,
}

impl AsyncCopy {
    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, src: &str) -> String {
        let cache = self
            .cache
            .map(|c| c.suffix().to_string())
            .unwrap_or_default();

        format!(
            "cp.async{}.shared.global [{}], [{}], {};",
            cache, dst, src, self.size
        )
    }

    /// Commit group of async copies
    pub fn commit_group() -> String {
        "cp.async.commit_group;".to_string()
    }

    /// Wait for N groups to complete
    pub fn wait_group(n: u32) -> String {
        format!("cp.async.wait_group {};", n)
    }

    /// Wait for all groups
    pub fn wait_all() -> String {
        "cp.async.wait_all;".to_string()
    }
}

// ============================================================================
// INSTRUCTION TIMING MODEL
// ============================================================================

/// Hardware execution unit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionUnit {
    /// Integer ALU
    Int,
    /// Floating-point ALU (FP32)
    Fp32,
    /// Double-precision ALU
    Fp64,
    /// Tensor cores
    Tensor,
    /// Special function unit (sin, cos, etc.)
    Sfu,
    /// Load/Store unit
    Ldst,
    /// Texture unit
    Tex,
    /// Shared memory
    Smem,
}

/// Hardware execution characteristics
#[derive(Debug, Clone)]
pub struct InstructionTiming {
    /// Latency in cycles (from issue to result available)
    pub latency: u32,
    /// Throughput in instructions per cycle per SM
    pub throughput: f64,
    /// Which execution unit
    pub unit: ExecutionUnit,
}

/// Get timing for common instructions
pub fn instruction_timing(instr: &str, _sm: u8) -> InstructionTiming {
    // SM 8.0 (A100) timings
    match instr {
        // Arithmetic
        "add.f32" | "mul.f32" | "fma.rn.f32" => InstructionTiming {
            latency: 4,
            throughput: 64.0, // 64 FP32 ops per cycle per SM
            unit: ExecutionUnit::Fp32,
        },
        "add.f64" | "mul.f64" | "fma.rn.f64" => InstructionTiming {
            latency: 8,
            throughput: 32.0, // 32 FP64 ops per cycle
            unit: ExecutionUnit::Fp64,
        },
        "add.s32" | "mul.lo.s32" | "mad.lo.s32" => InstructionTiming {
            latency: 4,
            throughput: 64.0,
            unit: ExecutionUnit::Int,
        },

        // Special functions
        "sin.approx.f32" | "cos.approx.f32" | "ex2.approx.f32" => InstructionTiming {
            latency: 8,
            throughput: 16.0, // 4 SFUs per SM, 4 cycles each
            unit: ExecutionUnit::Sfu,
        },
        "rcp.approx.f32" | "rsqrt.approx.f32" => InstructionTiming {
            latency: 8,
            throughput: 16.0,
            unit: ExecutionUnit::Sfu,
        },

        // Memory
        "ld.global.f32" | "ld.global.u32" => InstructionTiming {
            latency: 400, // DRAM latency
            throughput: 32.0,
            unit: ExecutionUnit::Ldst,
        },
        "ld.shared.f32" => InstructionTiming {
            latency: 20, // Shared memory
            throughput: 32.0,
            unit: ExecutionUnit::Smem,
        },
        "st.global.f32" => InstructionTiming {
            latency: 400,
            throughput: 32.0,
            unit: ExecutionUnit::Ldst,
        },

        // Atomics
        "atom.global.add.f32" => InstructionTiming {
            latency: 400, // Round trip to L2
            throughput: 16.0,
            unit: ExecutionUnit::Ldst,
        },

        // Shuffle
        "shfl.sync.bfly.b32" => InstructionTiming {
            latency: 2,
            throughput: 32.0, // One per warp per cycle
            unit: ExecutionUnit::Int,
        },

        // Default
        _ => InstructionTiming {
            latency: 4,
            throughput: 32.0,
            unit: ExecutionUnit::Int,
        },
    }
}

// ============================================================================
// KERNEL GENERATION
// ============================================================================

/// Generate optimized warp reduction kernel
pub fn generate_optimized_warp_reduce_kernel(
    target: PtxTarget,
    dtype: RegType,
    op: AtomicOp,
) -> String {
    let op_str = match op {
        AtomicOp::Add => "add",
        AtomicOp::Min => "min",
        AtomicOp::Max => "max",
        _ => "add",
    };

    let ty = dtype.suffix();

    format!(
        r#"
{}

.visible .entry warp_reduce_optimized(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {{
    // Register declarations
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .{ty} %f<8>;
    .reg .{ty} %f_val;
    .reg .{ty} %f_shfl;
    .reg .pred %p<4>;

    // Get thread identifiers
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %laneid;

    // Calculate global index
    mad.lo.u32 %r4, %r1, %r2, %r0;

    // Load parameters
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r5, [n];

    // Bounds check
    setp.ge.u32 %p0, %r4, %r5;
    @%p0 bra exit;

    // Load input value (coalesced)
    mul.wide.u32 %rd2, %r4, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.{ty} %f_val, [%rd3];

    // ===== WARP REDUCTION =====
    // 5 steps of butterfly shuffle

    // Step 1: XOR with 16
    shfl.sync.bfly.b32 %f_shfl, %f_val, 16, 0x1f, 0xffffffff;
    {op_str}.{ty} %f_val, %f_val, %f_shfl;

    // Step 2: XOR with 8
    shfl.sync.bfly.b32 %f_shfl, %f_val, 8, 0x1f, 0xffffffff;
    {op_str}.{ty} %f_val, %f_val, %f_shfl;

    // Step 3: XOR with 4
    shfl.sync.bfly.b32 %f_shfl, %f_val, 4, 0x1f, 0xffffffff;
    {op_str}.{ty} %f_val, %f_val, %f_shfl;

    // Step 4: XOR with 2
    shfl.sync.bfly.b32 %f_shfl, %f_val, 2, 0x1f, 0xffffffff;
    {op_str}.{ty} %f_val, %f_val, %f_shfl;

    // Step 5: XOR with 1
    shfl.sync.bfly.b32 %f_shfl, %f_val, 1, 0x1f, 0xffffffff;
    {op_str}.{ty} %f_val, %f_val, %f_shfl;

    // Lane 0 writes result
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra exit;

    // Atomic add to output (for block-level aggregation)
    atom.global.add.{ty} %f0, [%rd1], %f_val;

exit:
    ret;
}}
        "#,
        target.header(),
        ty = ty,
        op_str = op_str,
    )
}

/// Generate block-level reduction kernel
pub fn generate_block_reduce_kernel(target: PtxTarget, dtype: RegType, block_size: u32) -> String {
    let ty = dtype.suffix();
    let num_warps = block_size / 32;

    format!(
        r#"
{}

.visible .entry block_reduce(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {{
    // Shared memory for warp results
    .shared .{ty} warp_results[{num_warps}];

    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .{ty} %f<8>;
    .reg .pred %p<4>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;

    // Calculate warp and lane
    shr.u32 %r3, %r0, 5;  // warp_id = tid / 32
    and.b32 %r4, %r0, 31; // lane_id = tid % 32

    // Global index
    mad.lo.u32 %r5, %r1, %r2, %r0;

    // Load parameters
    ld.param.u64 %rd0, [input];
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r6, [n];

    // Load value (0 if out of bounds)
    setp.lt.u32 %p0, %r5, %r6;
    @!%p0 mov.{ty} %f0, 0;
    @%p0 mul.wide.u32 %rd2, %r5, 4;
    @%p0 add.u64 %rd3, %rd0, %rd2;
    @%p0 ld.global.{ty} %f0, [%rd3];

    // Warp reduction
    shfl.sync.bfly.b32 %f1, %f0, 16, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 8, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 4, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 2, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 1, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;

    // Lane 0 of each warp writes to shared memory
    setp.eq.u32 %p1, %r4, 0;
    @%p1 mul.wide.u32 %rd4, %r3, 4;
    @%p1 mov.u64 %rd5, warp_results;
    @%p1 add.u64 %rd6, %rd5, %rd4;
    @%p1 st.shared.{ty} [%rd6], %f0;

    // Barrier
    bar.sync 0;

    // Final reduction in warp 0
    setp.eq.u32 %p2, %r3, 0;
    @!%p2 bra done;

    setp.lt.u32 %p3, %r4, {num_warps};
    @!%p3 mov.{ty} %f0, 0;
    @%p3 mul.wide.u32 %rd4, %r4, 4;
    @%p3 mov.u64 %rd5, warp_results;
    @%p3 add.u64 %rd6, %rd5, %rd4;
    @%p3 ld.shared.{ty} %f0, [%rd6];

    // Final warp reduction
    shfl.sync.bfly.b32 %f1, %f0, 16, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 8, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 4, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 2, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 1, 0x1f, 0xffffffff;
    add.{ty} %f0, %f0, %f1;

    // Thread 0 writes final result
    setp.eq.u32 %p1, %r0, 0;
    @%p1 atom.global.add.{ty} %f2, [%rd1], %f0;

done:
    ret;
}}
        "#,
        target.header(),
        ty = ty,
        num_warps = num_warps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg_type_bits() {
        assert_eq!(RegType::F32.bits(), 32);
        assert_eq!(RegType::F64.bits(), 64);
        assert_eq!(RegType::B128.bits(), 128);
    }

    #[test]
    fn test_load_instruction() {
        let ld = LdInstruction::global_acquire(RegType::F32);
        let ptx = ld.to_ptx("%f0", "%rd0");
        assert!(ptx.contains("ld.acquire.gpu.global.f32"));
    }

    #[test]
    fn test_store_instruction() {
        let st = StInstruction::global_release(RegType::F32);
        let ptx = st.to_ptx("%rd0", "%f0");
        assert!(ptx.contains("st.release.gpu.global.f32"));
    }

    #[test]
    fn test_atomic_instruction() {
        let atom = AtomInstruction::global_add_relaxed(RegType::F32);
        let ptx = atom.to_ptx("%f0", "%rd0", &["%f1"]);
        assert!(ptx.contains("atom.relaxed.gpu.global.add.f32"));
    }

    #[test]
    fn test_shuffle_instruction() {
        let shfl = ShflInstruction::butterfly(RegType::B32);
        let ptx = shfl.to_ptx("%r0", "%r1", "16", "0x1f");
        assert!(ptx.contains("shfl.sync.bfly.b32"));
        assert!(ptx.contains("0xffffffff"));
    }

    #[test]
    fn test_fence_instruction() {
        let fence = FenceInstruction::gpu_fence();
        let ptx = fence.to_ptx();
        assert!(ptx.contains("fence.gpu"));
    }

    #[test]
    fn test_barrier_instruction() {
        let barrier = BarrierType::Sync(0);
        assert_eq!(barrier.to_ptx(), "bar.sync 0;");
    }

    #[test]
    fn test_warp_reduce_generation() {
        let target = PtxTarget::sm80();
        let kernel = generate_optimized_warp_reduce_kernel(target, RegType::F32, AtomicOp::Add);

        assert!(kernel.contains("shfl.sync.bfly"));
        assert!(kernel.contains("add.f32"));
        assert!(kernel.contains(".version 7.5"));
    }

    #[test]
    fn test_instruction_timing() {
        let timing = instruction_timing("add.f32", 80);
        assert_eq!(timing.latency, 4);
        assert_eq!(timing.throughput, 64.0);
        assert_eq!(timing.unit, ExecutionUnit::Fp32);

        let ld_timing = instruction_timing("ld.global.f32", 80);
        assert_eq!(ld_timing.latency, 400);
    }

    #[test]
    fn test_ptx_target_header() {
        let target = PtxTarget::sm90();
        let header = target.header();
        assert!(header.contains(".version 8.3"));
        assert!(header.contains("sm_90"));
    }
}
