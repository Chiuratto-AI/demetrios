//! Prefetching and Latency Hiding
//!
//! GPU can prefetch data to L2 cache asynchronously. This module provides:
//! - Prefetch hint generation
//! - Async copy operations (SM 8.0+)
//! - Double buffering for latency hiding
//! - Software pipelining utilities

use crate::ir::inst::ValueId;
use crate::ir::types::AddressSpace;

/// Prefetch target cache level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLevel {
    /// Prefetch to L2 cache
    L2,
    /// Prefetch to L1 cache (if available)
    L1,
    /// Prefetch for streaming (don't pollute cache)
    Streaming,
}

impl PrefetchLevel {
    /// Get PTX cache qualifier
    pub fn ptx_qualifier(&self) -> &'static str {
        match self {
            PrefetchLevel::L2 => "L2",
            PrefetchLevel::L1 => "L1",
            PrefetchLevel::Streaming => "L2::evict_last",
        }
    }
}

/// Prefetch instruction descriptor
#[derive(Debug, Clone)]
pub struct Prefetch {
    /// Address register to prefetch from
    pub addr: ValueId,
    /// Number of bytes to prefetch
    pub size: usize,
    /// Target cache level
    pub level: PrefetchLevel,
    /// Memory space
    pub space: AddressSpace,
}

impl Prefetch {
    /// Create a new prefetch instruction
    pub fn new(addr: ValueId, size: usize) -> Self {
        Self {
            addr,
            size,
            level: PrefetchLevel::L2,
            space: AddressSpace::Global,
        }
    }

    /// Set cache level
    pub fn with_level(mut self, level: PrefetchLevel) -> Self {
        self.level = level;
        self
    }

    /// Set memory space
    pub fn with_space(mut self, space: AddressSpace) -> Self {
        self.space = space;
        self
    }

    /// Generate PTX prefetch instruction
    pub fn to_ptx(&self, addr_reg: &str) -> String {
        format!(
            "prefetch.{}.{} [{}];",
            self.space.ptx_name(),
            self.level.ptx_qualifier(),
            addr_reg
        )
    }
}

/// Prefetch insertion pass
pub struct PrefetchInserter {
    /// Lookahead distance (in bytes)
    lookahead: usize,
    /// Minimum iteration count to benefit from prefetching
    min_iterations: usize,
    /// Minimum trip count for software pipelining
    min_trip_count: usize,
}

impl PrefetchInserter {
    /// Create a new prefetch inserter
    pub fn new() -> Self {
        Self {
            lookahead: 128 * 32, // ~4KB ahead
            min_iterations: 16,
            min_trip_count: 8,
        }
    }

    /// Set lookahead distance
    pub fn with_lookahead(mut self, bytes: usize) -> Self {
        self.lookahead = bytes;
        self
    }

    /// Set minimum iterations
    pub fn with_min_iterations(mut self, count: usize) -> Self {
        self.min_iterations = count;
        self
    }

    /// Get lookahead distance
    pub fn lookahead(&self) -> usize {
        self.lookahead
    }

    /// Check if a loop is worth prefetching
    pub fn should_prefetch(&self, estimated_iterations: usize, bytes_per_iteration: usize) -> bool {
        estimated_iterations >= self.min_iterations && bytes_per_iteration >= 128
    }

    /// Calculate optimal prefetch distance based on memory latency
    pub fn optimal_distance(
        &self,
        memory_latency_cycles: usize,
        cycles_per_iteration: usize,
    ) -> usize {
        if cycles_per_iteration == 0 {
            return 4;
        }
        // Distance = latency / cycles_per_iteration, rounded up
        (memory_latency_cycles + cycles_per_iteration - 1) / cycles_per_iteration
    }
}

impl Default for PrefetchInserter {
    fn default() -> Self {
        Self::new()
    }
}

/// Async memory copy (SM 8.0+)
#[derive(Debug, Clone)]
pub struct AsyncCopy {
    /// Source address (global memory)
    pub src: ValueId,
    /// Destination address (shared memory)
    pub dst: ValueId,
    /// Size in bytes (must be 4, 8, or 16)
    pub size: usize,
    /// Is this a commit barrier?
    pub commit: bool,
}

impl AsyncCopy {
    /// Create a new async copy
    pub fn new(src: ValueId, dst: ValueId, size: usize) -> Self {
        Self {
            src,
            dst,
            size,
            commit: false,
        }
    }

    /// Create a commit barrier
    pub fn commit() -> Self {
        Self {
            src: ValueId(0),
            dst: ValueId(0),
            size: 0,
            commit: true,
        }
    }

    /// Generate PTX async copy instruction
    pub fn to_ptx(&self, src_reg: &str, dst_reg: &str) -> String {
        if self.commit {
            "cp.async.commit_group;".to_string()
        } else {
            format!(
                "cp.async.ca.shared.global [{}], [{}], {};",
                dst_reg, src_reg, self.size
            )
        }
    }

    /// Generate PTX wait instruction
    pub fn wait_ptx(groups: u32) -> String {
        format!("cp.async.wait_group {};", groups)
    }

    /// Generate PTX wait_all instruction
    pub fn wait_all_ptx() -> &'static str {
        "cp.async.wait_all;"
    }
}

/// Double buffering for latency hiding
#[derive(Debug, Clone)]
pub struct DoubleBuffer {
    /// Buffer A offset in shared memory
    pub buffer_a: usize,
    /// Buffer B offset in shared memory
    pub buffer_b: usize,
    /// Buffer size in bytes
    pub buffer_size: usize,
    /// Current buffer index (0 = A, 1 = B)
    pub current: usize,
}

impl DoubleBuffer {
    /// Create a new double buffer
    pub fn new(buffer_size: usize, shared_base: usize) -> Self {
        Self {
            buffer_a: shared_base,
            buffer_b: shared_base + buffer_size,
            buffer_size,
            current: 0,
        }
    }

    /// Get current read buffer offset
    pub fn read_buffer(&self) -> usize {
        if self.current == 0 {
            self.buffer_a
        } else {
            self.buffer_b
        }
    }

    /// Get current write buffer offset (opposite of read)
    pub fn write_buffer(&self) -> usize {
        if self.current == 0 {
            self.buffer_b
        } else {
            self.buffer_a
        }
    }

    /// Swap buffers
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    /// Total shared memory needed
    pub fn total_shared(&self) -> usize {
        self.buffer_size * 2
    }

    /// Get buffer by index
    pub fn buffer(&self, index: usize) -> usize {
        if index == 0 {
            self.buffer_a
        } else {
            self.buffer_b
        }
    }
}

/// Triple buffering for higher throughput
#[derive(Debug, Clone)]
pub struct TripleBuffer {
    /// Buffer offsets
    pub buffers: [usize; 3],
    /// Buffer size
    pub buffer_size: usize,
    /// Current read index
    pub read_index: usize,
    /// Current write index
    pub write_index: usize,
}

impl TripleBuffer {
    /// Create a new triple buffer
    pub fn new(buffer_size: usize, shared_base: usize) -> Self {
        Self {
            buffers: [
                shared_base,
                shared_base + buffer_size,
                shared_base + 2 * buffer_size,
            ],
            buffer_size,
            read_index: 0,
            write_index: 2,
        }
    }

    /// Get current read buffer
    pub fn read_buffer(&self) -> usize {
        self.buffers[self.read_index]
    }

    /// Get current write buffer
    pub fn write_buffer(&self) -> usize {
        self.buffers[self.write_index]
    }

    /// Get compute buffer (middle one)
    pub fn compute_buffer(&self) -> usize {
        let compute_index = (self.read_index + 1) % 3;
        self.buffers[compute_index]
    }

    /// Advance to next iteration
    pub fn advance(&mut self) {
        self.read_index = (self.read_index + 1) % 3;
        self.write_index = (self.write_index + 1) % 3;
    }

    /// Total shared memory needed
    pub fn total_shared(&self) -> usize {
        self.buffer_size * 3
    }
}

/// Software pipelining configuration
#[derive(Debug, Clone)]
pub struct SoftwarePipeline {
    /// Number of pipeline stages
    pub stages: usize,
    /// Elements per stage
    pub elements_per_stage: usize,
    /// Total elements
    pub total_elements: usize,
}

impl SoftwarePipeline {
    /// Create a new software pipeline
    pub fn new(total_elements: usize, stages: usize) -> Self {
        let elements_per_stage = (total_elements + stages - 1) / stages;
        Self {
            stages,
            elements_per_stage,
            total_elements,
        }
    }

    /// Number of prologue iterations
    pub fn prologue_iterations(&self) -> usize {
        self.stages - 1
    }

    /// Number of main loop iterations
    pub fn main_iterations(&self) -> usize {
        self.total_elements
            .saturating_sub(self.prologue_iterations())
    }

    /// Number of epilogue iterations
    pub fn epilogue_iterations(&self) -> usize {
        self.stages - 1
    }

    /// Check if iteration i is in prologue
    pub fn is_prologue(&self, i: usize) -> bool {
        i < self.prologue_iterations()
    }

    /// Check if iteration i is in epilogue
    pub fn is_epilogue(&self, i: usize) -> bool {
        i >= self
            .total_elements
            .saturating_sub(self.epilogue_iterations())
    }

    /// Get stage for a given iteration
    pub fn stage_for_iteration(&self, i: usize) -> usize {
        i % self.stages
    }
}

/// Memory access pattern for prefetch analysis
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Base address register
    pub base: ValueId,
    /// Stride between accesses (bytes)
    pub stride: i64,
    /// Access size (bytes)
    pub access_size: usize,
    /// Number of accesses
    pub count: usize,
}

impl MemoryAccessPattern {
    /// Total bytes accessed
    pub fn total_bytes(&self) -> usize {
        self.access_size * self.count
    }

    /// Check if pattern is sequential
    pub fn is_sequential(&self) -> bool {
        self.stride == self.access_size as i64
    }

    /// Check if pattern can benefit from prefetching
    pub fn can_prefetch(&self) -> bool {
        // Sequential or strided with reasonable stride
        self.is_sequential() || (self.stride > 0 && self.stride <= 4096)
    }
}

/// Prefetch schedule for a loop
#[derive(Debug, Clone)]
pub struct PrefetchSchedule {
    /// Prefetches to insert at loop header
    pub header_prefetches: Vec<Prefetch>,
    /// Prefetches to insert in loop body
    pub body_prefetches: Vec<Prefetch>,
    /// Distance ahead (in iterations) to prefetch
    pub lookahead_iterations: usize,
}

impl PrefetchSchedule {
    /// Create an empty schedule
    pub fn empty() -> Self {
        Self {
            header_prefetches: Vec::new(),
            body_prefetches: Vec::new(),
            lookahead_iterations: 0,
        }
    }

    /// Check if schedule has any prefetches
    pub fn is_empty(&self) -> bool {
        self.header_prefetches.is_empty() && self.body_prefetches.is_empty()
    }

    /// Total number of prefetches
    pub fn total_prefetches(&self) -> usize {
        self.header_prefetches.len() + self.body_prefetches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer() {
        let mut buf = DoubleBuffer::new(1024, 0);

        assert_eq!(buf.read_buffer(), 0);
        assert_eq!(buf.write_buffer(), 1024);

        buf.swap();

        assert_eq!(buf.read_buffer(), 1024);
        assert_eq!(buf.write_buffer(), 0);

        assert_eq!(buf.total_shared(), 2048);
    }

    #[test]
    fn test_triple_buffer() {
        let mut buf = TripleBuffer::new(1024, 0);

        assert_eq!(buf.read_buffer(), 0);
        assert_eq!(buf.compute_buffer(), 1024);
        assert_eq!(buf.write_buffer(), 2048);

        buf.advance();

        assert_eq!(buf.read_buffer(), 1024);
        assert_eq!(buf.compute_buffer(), 2048);
        assert_eq!(buf.write_buffer(), 0);

        assert_eq!(buf.total_shared(), 3072);
    }

    #[test]
    fn test_prefetch_ptx() {
        let pf = Prefetch::new(ValueId(0), 128)
            .with_level(PrefetchLevel::L2)
            .with_space(AddressSpace::Global);

        let ptx = pf.to_ptx("%rd0");
        assert!(ptx.contains("prefetch"));
        assert!(ptx.contains("global"));
        assert!(ptx.contains("L2"));
    }

    #[test]
    fn test_async_copy_ptx() {
        let copy = AsyncCopy::new(ValueId(0), ValueId(1), 16);
        let ptx = copy.to_ptx("%rd0", "%rd1");

        assert!(ptx.contains("cp.async"));
        assert!(ptx.contains("shared.global"));
        assert!(ptx.contains("16"));
    }

    #[test]
    fn test_async_copy_commit() {
        let commit = AsyncCopy::commit();
        let ptx = commit.to_ptx("", "");

        assert_eq!(ptx, "cp.async.commit_group;");
    }

    #[test]
    fn test_software_pipeline() {
        let pipeline = SoftwarePipeline::new(100, 4);

        assert_eq!(pipeline.prologue_iterations(), 3);
        assert_eq!(pipeline.epilogue_iterations(), 3);

        assert!(pipeline.is_prologue(0));
        assert!(pipeline.is_prologue(2));
        assert!(!pipeline.is_prologue(3));

        assert!(!pipeline.is_epilogue(0));
        assert!(pipeline.is_epilogue(99));
    }

    #[test]
    fn test_memory_access_pattern() {
        let pattern = MemoryAccessPattern {
            base: ValueId(0),
            stride: 4,
            access_size: 4,
            count: 1024,
        };

        assert!(pattern.is_sequential());
        assert!(pattern.can_prefetch());
        assert_eq!(pattern.total_bytes(), 4096);
    }

    #[test]
    fn test_prefetch_inserter() {
        let inserter = PrefetchInserter::new();

        // Should prefetch for large loops with significant data
        assert!(inserter.should_prefetch(100, 256));

        // Should not prefetch for small loops
        assert!(!inserter.should_prefetch(4, 256));

        // Should not prefetch for tiny data
        assert!(!inserter.should_prefetch(100, 32));
    }

    #[test]
    fn test_optimal_distance() {
        let inserter = PrefetchInserter::new();

        // 500 cycle latency, 100 cycles per iteration = 5 iterations ahead
        let distance = inserter.optimal_distance(500, 100);
        assert_eq!(distance, 5);

        // 500 cycle latency, 200 cycles per iteration = 3 iterations ahead
        let distance = inserter.optimal_distance(500, 200);
        assert_eq!(distance, 3);
    }

    #[test]
    fn test_prefetch_levels() {
        assert_eq!(PrefetchLevel::L1.ptx_qualifier(), "L1");
        assert_eq!(PrefetchLevel::L2.ptx_qualifier(), "L2");
        assert_eq!(PrefetchLevel::Streaming.ptx_qualifier(), "L2::evict_last");
    }

    #[test]
    fn test_prefetch_schedule() {
        let mut schedule = PrefetchSchedule::empty();
        assert!(schedule.is_empty());

        schedule
            .body_prefetches
            .push(Prefetch::new(ValueId(0), 128));
        schedule.lookahead_iterations = 4;

        assert!(!schedule.is_empty());
        assert_eq!(schedule.total_prefetches(), 1);
    }
}
