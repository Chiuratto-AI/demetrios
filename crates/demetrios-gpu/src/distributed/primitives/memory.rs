//! Memory Consistency Model for Distributed GPU Computing
//!
//! GPUs have a relaxed memory model. Without explicit synchronization:
//! - Writes may not be visible to other threads
//! - Reads may return stale data
//! - Reordering is permitted
//!
//! We need explicit fences and atomics for correctness.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Memory fence types for GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Thread-local (register -> L1)
    Thread,
    /// Warp-level (implicit coherence within warp)
    Warp,
    /// Block-level (shared memory coherence)
    Block,
    /// Device-level (L2 coherence)
    Device,
    /// System-level (CPU-GPU coherence)
    System,
}

/// Memory ordering for atomic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// No ordering constraints
    Relaxed,
    /// Acquire semantics (reads after this see writes before release)
    Acquire,
    /// Release semantics (writes before this visible after acquire)
    Release,
    /// Full sequential consistency
    SeqCst,
}

/// PTX memory fence instruction
#[derive(Debug, Clone, Copy)]
pub struct MemFence {
    pub scope: MemoryScope,
    pub order: MemoryOrder,
}

impl MemFence {
    /// Create a new memory fence
    pub fn new(scope: MemoryScope, order: MemoryOrder) -> Self {
        Self { scope, order }
    }

    /// Device-level acquire fence
    pub fn device_acquire() -> Self {
        Self::new(MemoryScope::Device, MemoryOrder::Acquire)
    }

    /// Device-level release fence
    pub fn device_release() -> Self {
        Self::new(MemoryScope::Device, MemoryOrder::Release)
    }

    /// System-level full fence
    pub fn system_seqcst() -> Self {
        Self::new(MemoryScope::System, MemoryOrder::SeqCst)
    }

    /// Generate PTX instruction
    pub fn to_ptx(&self) -> String {
        let scope = match self.scope {
            MemoryScope::Thread => ".cta", // Actually no fence needed
            MemoryScope::Warp => ".cta",
            MemoryScope::Block => ".cta",
            MemoryScope::Device => ".gpu",
            MemoryScope::System => ".sys",
        };

        let order = match self.order {
            MemoryOrder::Relaxed => "",
            MemoryOrder::Acquire => ".acq",
            MemoryOrder::Release => ".rel",
            MemoryOrder::SeqCst => ".acq_rel",
        };

        format!("membar{}{};", scope, order)
    }
}

/// Atomic operation types
#[derive(Debug, Clone, Copy)]
pub enum AtomicOp {
    /// Atomic load
    Load,
    /// Atomic store
    Store,
    /// Atomic exchange
    Exchange,
    /// Compare and swap
    CompareExchange { expected: u64 },
    /// Atomic add
    Add,
    /// Atomic subtract (implemented as add with negation)
    Sub,
    /// Atomic minimum
    Min,
    /// Atomic maximum
    Max,
    /// Atomic AND
    And,
    /// Atomic OR
    Or,
    /// Atomic XOR
    Xor,
}

/// Atomic data types supported
#[derive(Debug, Clone, Copy)]
pub enum AtomicDataType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
}

impl AtomicDataType {
    /// Get PTX type suffix
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            Self::U32 => ".u32",
            Self::U64 => ".u64",
            Self::I32 => ".s32",
            Self::I64 => ".s64",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// PTX atomic instruction generator
pub struct AtomicInstruction {
    pub op: AtomicOp,
    pub scope: MemoryScope,
    pub order: MemoryOrder,
    pub data_type: AtomicDataType,
}

impl AtomicInstruction {
    pub fn new(
        op: AtomicOp,
        scope: MemoryScope,
        order: MemoryOrder,
        data_type: AtomicDataType,
    ) -> Self {
        Self {
            op,
            scope,
            order,
            data_type,
        }
    }

    /// Generate PTX instruction
    pub fn to_ptx(&self, dst: &str, addr: &str, operand: &str) -> String {
        let op = match self.op {
            AtomicOp::Load => "ld",
            AtomicOp::Store => "st",
            AtomicOp::Exchange => "exch",
            AtomicOp::CompareExchange { .. } => "cas",
            AtomicOp::Add => "add",
            AtomicOp::Sub => "add", // negate operand
            AtomicOp::Min => "min",
            AtomicOp::Max => "max",
            AtomicOp::And => "and",
            AtomicOp::Or => "or",
            AtomicOp::Xor => "xor",
        };

        let scope = match self.scope {
            MemoryScope::Block => ".cta",
            MemoryScope::Device => ".gpu",
            MemoryScope::System => ".sys",
            _ => ".gpu",
        };

        let order = match self.order {
            MemoryOrder::Relaxed => ".relaxed",
            MemoryOrder::Acquire => ".acquire",
            MemoryOrder::Release => ".release",
            MemoryOrder::SeqCst => ".acq_rel",
        };

        let ty = self.data_type.ptx_suffix();

        match self.op {
            AtomicOp::Load => {
                format!("ld{}{}.global{} {}, [{}];", order, scope, ty, dst, addr)
            }
            AtomicOp::Store => {
                format!("st{}{}.global{} [{}], {};", order, scope, ty, addr, operand)
            }
            AtomicOp::CompareExchange { expected } => {
                format!(
                    "atom{}{}.global.cas{} {}, [{}], {}, {};",
                    order, scope, ty, dst, addr, expected, operand
                )
            }
            _ => {
                format!(
                    "atom{}{}.global.{}{} {}, [{}], {};",
                    order, scope, op, ty, dst, addr, operand
                )
            }
        }
    }
}

/// Synchronization flag for producer-consumer pattern
///
/// Used for signaling between devices or between CPU and GPU
#[repr(C, align(64))] // Cache line aligned
pub struct SyncFlag {
    /// The flag value (0 = not ready, 1 = ready)
    flag: AtomicU64,
    /// Padding to prevent false sharing
    _pad: [u64; 7],
}

impl SyncFlag {
    pub const fn new() -> Self {
        Self {
            flag: AtomicU64::new(0),
            _pad: [0; 7],
        }
    }

    /// Signal (release semantics)
    pub fn signal(&self) {
        self.flag.store(1, Ordering::Release);
    }

    /// Wait with spinning (acquire semantics)
    pub fn wait(&self) {
        while self.flag.load(Ordering::Acquire) == 0 {
            std::hint::spin_loop();
        }
    }

    /// Reset for reuse
    pub fn reset(&self) {
        self.flag.store(0, Ordering::Relaxed);
    }

    /// Try to acquire (returns immediately)
    pub fn try_wait(&self) -> bool {
        self.flag.load(Ordering::Acquire) != 0
    }

    /// Get current value
    pub fn value(&self) -> u64 {
        self.flag.load(Ordering::Relaxed)
    }
}

impl Default for SyncFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-side synchronization primitive
///
/// For signaling between GPU kernels or between GPU and CPU
#[derive(Debug)]
pub struct GpuSyncFlag {
    /// Device pointer to flag (in GPU memory)
    pub device_ptr: u64,
    /// Host-mapped pointer (for CPU access)
    pub host_ptr: *mut u64,
    /// Whether this is in system memory (for CPU-GPU sync)
    pub is_system_memory: bool,
}

impl GpuSyncFlag {
    /// Create a new GPU sync flag
    pub fn new(device_ptr: u64, host_ptr: *mut u64, is_system_memory: bool) -> Self {
        Self {
            device_ptr,
            host_ptr,
            is_system_memory,
        }
    }

    /// Generate PTX code for GPU-side signal
    pub fn ptx_signal(&self) -> String {
        let scope = if self.is_system_memory {
            ".sys"
        } else {
            ".gpu"
        };
        format!(
            r#"
            // Signal flag
            mov.u64 %rd_flag, {};
            st.release{}.global.u64 [%rd_flag], 1;
            "#,
            self.device_ptr, scope
        )
    }

    /// Generate PTX code for GPU-side wait
    pub fn ptx_wait(&self) -> String {
        let scope = if self.is_system_memory {
            ".sys"
        } else {
            ".gpu"
        };
        format!(
            r#"
            // Wait for flag
            mov.u64 %rd_flag, {};
        wait_loop:
            ld.acquire{}.global.u64 %rd_val, [%rd_flag];
            setp.eq.u64 %p_wait, %rd_val, 0;
            @%p_wait bra wait_loop;
            "#,
            self.device_ptr, scope
        )
    }
}

/// Status values for sequence number
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SeqStatus {
    /// Not yet started
    Invalid = 0,
    /// Partial result available
    Partial = 1,
    /// Final result available
    Complete = 2,
}

/// Sequence number for ordering operations
///
/// Used in decoupled look-back algorithms
#[repr(C, align(8))]
pub struct SequenceNumber {
    /// Combined state: high 32 bits = sequence, low 32 bits = status
    state: AtomicU64,
}

impl SequenceNumber {
    pub const fn new() -> Self {
        Self {
            state: AtomicU64::new(0),
        }
    }

    /// Set partial value
    pub fn set_partial(&self, value: u32) {
        let state = ((value as u64) << 32) | (SeqStatus::Partial as u64);
        self.state.store(state, Ordering::Release);
    }

    /// Set complete value
    pub fn set_complete(&self, value: u32) {
        let state = ((value as u64) << 32) | (SeqStatus::Complete as u64);
        self.state.store(state, Ordering::Release);
    }

    /// Read status and value
    pub fn read(&self) -> (SeqStatus, u32) {
        let state = self.state.load(Ordering::Acquire);
        let status = match (state & 0xFFFFFFFF) as u32 {
            0 => SeqStatus::Invalid,
            1 => SeqStatus::Partial,
            _ => SeqStatus::Complete,
        };
        let value = (state >> 32) as u32;
        (status, value)
    }

    /// Generate PTX for reading
    pub fn ptx_read(addr_reg: &str, status_reg: &str, value_reg: &str) -> String {
        format!(
            r#"
            // Read sequence number
            ld.acquire.gpu.global.u64 %rd_seq, [{addr}];
            cvt.u32.u64 {status}, %rd_seq;      // Low 32 bits = status
            shr.u64 %rd_tmp, %rd_seq, 32;
            cvt.u32.u64 {value}, %rd_tmp;       // High 32 bits = value
            "#,
            addr = addr_reg,
            status = status_reg,
            value = value_reg
        )
    }
}

impl Default for SequenceNumber {
    fn default() -> Self {
        Self::new()
    }
}

/// Barrier implementation for multi-GPU synchronization
pub struct DistributedBarrier {
    /// Number of participants
    pub count: u32,
    /// Arrival counter
    arrivals: AtomicU32,
    /// Sense flag (toggles each barrier)
    sense: AtomicU32,
    /// Per-participant local sense
    local_sense: Vec<AtomicU32>,
}

impl DistributedBarrier {
    pub fn new(count: u32) -> Self {
        Self {
            count,
            arrivals: AtomicU32::new(0),
            sense: AtomicU32::new(0),
            local_sense: (0..count).map(|_| AtomicU32::new(1)).collect(),
        }
    }

    /// Wait at barrier (returns when all participants have arrived)
    pub fn wait(&self, participant_id: u32) {
        let my_sense = self.local_sense[participant_id as usize].load(Ordering::Relaxed);

        // Arrive
        let arrived = self.arrivals.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.count {
            // Last to arrive: reset and release
            self.arrivals.store(0, Ordering::Relaxed);
            self.sense.store(my_sense, Ordering::Release);
        } else {
            // Wait for last participant
            while self.sense.load(Ordering::Acquire) != my_sense {
                std::hint::spin_loop();
            }
        }

        // Toggle local sense for next barrier
        self.local_sense[participant_id as usize].store(1 - my_sense, Ordering::Relaxed);
    }

    /// Get number of participants currently at barrier
    pub fn arrivals(&self) -> u32 {
        self.arrivals.load(Ordering::Relaxed)
    }
}

/// Lock-free MPSC queue for work stealing
///
/// Multiple producers (work donors), single consumer (work stealer)
pub struct WorkStealQueue<T> {
    /// Ring buffer
    buffer: Box<[UnsafeCell<Option<T>>]>,
    /// Capacity (power of 2)
    capacity: usize,
    /// Mask for wrapping
    mask: usize,
    /// Head (dequeue point, owned by consumer)
    head: AtomicU64,
    /// Tail (enqueue point, shared by producers)
    tail: AtomicU64,
}

unsafe impl<T: Send> Send for WorkStealQueue<T> {}
unsafe impl<T: Send> Sync for WorkStealQueue<T> {}

impl<T> WorkStealQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());

        let buffer: Vec<_> = (0..capacity).map(|_| UnsafeCell::new(None)).collect();

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }

    /// Push item (producer side)
    pub fn push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail - head >= self.capacity as u64 {
            return Err(item); // Queue full
        }

        let slot = (tail & self.mask as u64) as usize;

        // Write item
        unsafe {
            *self.buffer[slot].get() = Some(item);
        }

        // Publish
        self.tail.store(tail + 1, Ordering::Release);

        Ok(())
    }

    /// Pop item (consumer side, LIFO for cache locality)
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);

        if tail == 0 {
            return None;
        }

        let new_tail = tail - 1;
        self.tail.store(new_tail, Ordering::Relaxed);

        std::sync::atomic::fence(Ordering::SeqCst);

        let head = self.head.load(Ordering::Relaxed);

        if head <= new_tail {
            // Safe to take
            let slot = (new_tail & self.mask as u64) as usize;
            let item = unsafe { (*self.buffer[slot].get()).take() };
            return item;
        }

        // Race with stealer, restore tail
        self.tail.store(tail, Ordering::Relaxed);

        if head == tail {
            // Queue empty
            return None;
        }

        // Try CAS to claim the last item
        let slot = (new_tail & self.mask as u64) as usize;
        if self
            .head
            .compare_exchange(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            self.tail.store(new_tail, Ordering::Relaxed);
            let item = unsafe { (*self.buffer[slot].get()).take() };
            return item;
        }

        // Lost race
        self.tail.store(tail, Ordering::Relaxed);
        None
    }

    /// Steal item (stealer side, FIFO)
    pub fn steal(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);

        std::sync::atomic::fence(Ordering::SeqCst);

        let tail = self.tail.load(Ordering::Acquire);

        if head >= tail {
            return None; // Queue empty
        }

        let slot = (head & self.mask as u64) as usize;
        let item = unsafe { (*self.buffer[slot].get()).take() };

        if self
            .head
            .compare_exchange(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            return item;
        }

        // Lost race, put item back
        unsafe {
            *self.buffer[slot].get() = item;
        }
        None
    }

    /// Current length
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        (tail - head) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_flag() {
        let flag = SyncFlag::new();

        assert!(!flag.try_wait());

        flag.signal();
        assert!(flag.try_wait());

        flag.reset();
        assert!(!flag.try_wait());
    }

    #[test]
    fn test_sequence_number() {
        let seq = SequenceNumber::new();

        let (status, _) = seq.read();
        assert_eq!(status, SeqStatus::Invalid);

        seq.set_partial(42);
        let (status, value) = seq.read();
        assert_eq!(status, SeqStatus::Partial);
        assert_eq!(value, 42);

        seq.set_complete(100);
        let (status, value) = seq.read();
        assert_eq!(status, SeqStatus::Complete);
        assert_eq!(value, 100);
    }

    #[test]
    fn test_work_steal_queue() {
        let queue = WorkStealQueue::new(16);

        // Push items
        for i in 0..10u32 {
            queue.push(i).unwrap();
        }

        assert_eq!(queue.len(), 10);

        // Pop (LIFO)
        assert_eq!(queue.pop(), Some(9));
        assert_eq!(queue.pop(), Some(8));

        // Steal (FIFO)
        assert_eq!(queue.steal(), Some(0));
        assert_eq!(queue.steal(), Some(1));
    }

    #[test]
    fn test_distributed_barrier() {
        use std::sync::Arc;
        use std::thread;

        let barrier = Arc::new(DistributedBarrier::new(4));
        let mut handles = Vec::new();

        for i in 0..4 {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait(i);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_mem_fence_ptx() {
        let fence = MemFence::device_release();
        let ptx = fence.to_ptx();
        assert!(ptx.contains("membar"));
        assert!(ptx.contains(".gpu"));
        assert!(ptx.contains(".rel"));
    }

    #[test]
    fn test_atomic_instruction_ptx() {
        let inst = AtomicInstruction::new(
            AtomicOp::Add,
            MemoryScope::Device,
            MemoryOrder::Relaxed,
            AtomicDataType::F32,
        );

        let ptx = inst.to_ptx("%f0", "%rd0", "%f1");
        assert!(ptx.contains("atom"));
        assert!(ptx.contains("add"));
        assert!(ptx.contains(".f32"));
    }
}
