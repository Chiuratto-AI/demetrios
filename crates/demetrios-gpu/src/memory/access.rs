//! Memory Access Patterns
//!
//! Provides abstractions for coalesced memory access and access pattern analysis.

use super::spaces::{Global, MemorySpace, Shared};
use std::marker::PhantomData;

/// Memory access pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential/contiguous access (optimal)
    Sequential,
    /// Strided access with a fixed stride
    Strided(usize),
    /// Random access (worst case)
    Random,
    /// Broadcast (all threads read same location)
    Broadcast,
    /// Scatter (each thread writes to different location)
    Scatter,
    /// Gather (each thread reads from different location)
    Gather,
}

impl AccessPattern {
    /// Check if this pattern is coalesced (efficient)
    pub fn is_coalesced(&self) -> bool {
        matches!(self, AccessPattern::Sequential | AccessPattern::Broadcast)
    }

    /// Get the expected memory efficiency (0.0 to 1.0)
    pub fn efficiency(&self) -> f32 {
        match self {
            AccessPattern::Sequential => 1.0,
            AccessPattern::Broadcast => 1.0,
            AccessPattern::Strided(stride) => {
                // Efficiency decreases with stride
                1.0 / (*stride as f32).max(1.0)
            }
            AccessPattern::Gather => 0.3,
            AccessPattern::Scatter => 0.3,
            AccessPattern::Random => 0.1,
        }
    }
}

/// Memory access mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    /// Read-only access
    Read,
    /// Write-only access
    Write,
    /// Read-write access
    ReadWrite,
    /// Atomic access
    Atomic,
}

impl AccessMode {
    pub fn can_read(&self) -> bool {
        matches!(
            self,
            AccessMode::Read | AccessMode::ReadWrite | AccessMode::Atomic
        )
    }

    pub fn can_write(&self) -> bool {
        matches!(
            self,
            AccessMode::Write | AccessMode::ReadWrite | AccessMode::Atomic
        )
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, AccessMode::Atomic)
    }
}

/// Memory access descriptor
#[derive(Debug, Clone)]
pub struct MemoryAccess<S: MemorySpace> {
    /// Access pattern
    pub pattern: AccessPattern,
    /// Access mode
    pub mode: AccessMode,
    /// Element size in bytes
    pub element_size: usize,
    /// Number of elements accessed per thread
    pub elements_per_thread: usize,
    /// Space marker
    _space: PhantomData<S>,
}

impl<S: MemorySpace> MemoryAccess<S> {
    /// Create a new memory access descriptor
    pub fn new(pattern: AccessPattern, mode: AccessMode, element_size: usize) -> Self {
        MemoryAccess {
            pattern,
            mode,
            element_size,
            elements_per_thread: 1,
            _space: PhantomData,
        }
    }

    /// Create a sequential read access
    pub fn sequential_read(element_size: usize) -> Self {
        Self::new(AccessPattern::Sequential, AccessMode::Read, element_size)
    }

    /// Create a sequential write access
    pub fn sequential_write(element_size: usize) -> Self {
        Self::new(AccessPattern::Sequential, AccessMode::Write, element_size)
    }

    /// Create a broadcast read access
    pub fn broadcast_read(element_size: usize) -> Self {
        Self::new(AccessPattern::Broadcast, AccessMode::Read, element_size)
    }

    /// Set elements per thread
    pub fn with_elements(mut self, count: usize) -> Self {
        self.elements_per_thread = count;
        self
    }

    /// Get the total bytes accessed per thread
    pub fn bytes_per_thread(&self) -> usize {
        self.element_size * self.elements_per_thread
    }

    /// Estimate bandwidth utilization
    pub fn bandwidth_utilization(&self) -> f32 {
        self.pattern.efficiency()
    }
}

/// Coalesced memory accessor for global memory
#[derive(Debug)]
pub struct CoalescedAccessor<T> {
    /// Base pointer
    base: *mut T,
    /// Total elements
    count: usize,
    /// Warp size
    warp_size: usize,
}

impl<T: Copy> CoalescedAccessor<T> {
    /// Create a new coalesced accessor
    pub fn new(base: *mut T, count: usize) -> Self {
        CoalescedAccessor {
            base,
            count,
            warp_size: 32,
        }
    }

    /// Set warp size
    pub fn with_warp_size(mut self, size: usize) -> Self {
        self.warp_size = size;
        self
    }

    /// Calculate coalesced index for a thread
    pub fn coalesced_index(&self, thread_id: usize, element_idx: usize) -> usize {
        let warp_id = thread_id / self.warp_size;
        let lane_id = thread_id % self.warp_size;

        // For coalesced access: consecutive threads access consecutive memory
        warp_id * self.warp_size + lane_id + element_idx * self.warp_size
    }

    /// Load a value with coalesced access pattern
    pub unsafe fn load(&self, thread_id: usize, element_idx: usize) -> Option<T> {
        let idx = self.coalesced_index(thread_id, element_idx);
        if idx < self.count {
            Some(*self.base.add(idx))
        } else {
            None
        }
    }

    /// Store a value with coalesced access pattern
    pub unsafe fn store(&self, thread_id: usize, element_idx: usize, value: T) -> bool {
        let idx = self.coalesced_index(thread_id, element_idx);
        if idx < self.count {
            *self.base.add(idx) = value;
            true
        } else {
            false
        }
    }
}

/// Shared memory bank conflict analyzer
#[derive(Debug)]
pub struct BankConflictAnalyzer {
    /// Number of banks
    num_banks: usize,
    /// Bank width in bytes
    bank_width: usize,
}

impl BankConflictAnalyzer {
    /// Create a new analyzer for typical GPU (32 banks, 4-byte width)
    pub fn new() -> Self {
        BankConflictAnalyzer {
            num_banks: 32,
            bank_width: 4,
        }
    }

    /// Create with custom configuration
    pub fn with_config(num_banks: usize, bank_width: usize) -> Self {
        BankConflictAnalyzer {
            num_banks,
            bank_width,
        }
    }

    /// Calculate which bank an address maps to
    pub fn bank_for_address(&self, byte_offset: usize) -> usize {
        (byte_offset / self.bank_width) % self.num_banks
    }

    /// Analyze bank conflicts for a set of addresses
    pub fn analyze(&self, addresses: &[usize]) -> BankConflictResult {
        let mut bank_access_count = vec![0usize; self.num_banks];

        for &addr in addresses {
            let bank = self.bank_for_address(addr);
            bank_access_count[bank] += 1;
        }

        let max_conflicts = *bank_access_count.iter().max().unwrap_or(&0);
        let banks_used = bank_access_count.iter().filter(|&&c| c > 0).count();

        BankConflictResult {
            max_way_conflict: max_conflicts,
            banks_used,
            total_banks: self.num_banks,
            conflict_free: max_conflicts <= 1,
        }
    }

    /// Suggest padding to avoid conflicts for strided access
    pub fn suggest_padding(&self, element_size: usize, stride: usize) -> usize {
        // If stride causes bank conflicts, add padding
        let stride_banks = (stride * element_size / self.bank_width) % self.num_banks;
        if stride_banks == 0 || self.num_banks % stride_banks == 0 {
            // Conflict pattern detected, suggest padding
            self.bank_width
        } else {
            0
        }
    }
}

impl Default for BankConflictAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of bank conflict analysis
#[derive(Debug, Clone)]
pub struct BankConflictResult {
    /// Maximum conflicts for any bank (1 = no conflict)
    pub max_way_conflict: usize,
    /// Number of banks actually used
    pub banks_used: usize,
    /// Total number of banks
    pub total_banks: usize,
    /// Whether the access is conflict-free
    pub conflict_free: bool,
}

impl BankConflictResult {
    /// Get the serialization factor (1.0 = no serialization)
    pub fn serialization_factor(&self) -> f32 {
        self.max_way_conflict as f32
    }

    /// Get efficiency (0.0 to 1.0)
    pub fn efficiency(&self) -> f32 {
        1.0 / self.serialization_factor()
    }
}

/// Memory coalescing analyzer for global memory
#[derive(Debug)]
pub struct CoalescingAnalyzer {
    /// Cache line size in bytes
    cache_line_size: usize,
    /// Warp size
    warp_size: usize,
}

impl CoalescingAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        CoalescingAnalyzer {
            cache_line_size: 128, // Typical L1 cache line
            warp_size: 32,
        }
    }

    /// Analyze coalescing for a set of addresses
    pub fn analyze(&self, addresses: &[usize], element_size: usize) -> CoalescingResult {
        if addresses.is_empty() {
            return CoalescingResult {
                transactions: 0,
                min_transactions: 0,
                efficiency: 1.0,
                is_coalesced: true,
            };
        }

        // Count unique cache lines accessed
        let mut cache_lines: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &addr in addresses {
            let line = addr / self.cache_line_size;
            cache_lines.insert(line);
        }

        let transactions = cache_lines.len();
        let total_bytes = addresses.len() * element_size;
        let min_transactions = (total_bytes + self.cache_line_size - 1) / self.cache_line_size;

        let efficiency = if transactions > 0 {
            min_transactions as f32 / transactions as f32
        } else {
            1.0
        };

        CoalescingResult {
            transactions,
            min_transactions,
            efficiency,
            is_coalesced: efficiency >= 0.75,
        }
    }
}

impl Default for CoalescingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of coalescing analysis
#[derive(Debug, Clone)]
pub struct CoalescingResult {
    /// Actual memory transactions
    pub transactions: usize,
    /// Minimum possible transactions (perfect coalescing)
    pub min_transactions: usize,
    /// Efficiency (0.0 to 1.0)
    pub efficiency: f32,
    /// Whether access is considered coalesced
    pub is_coalesced: bool,
}

/// Memory access optimization hints
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Use vectorized loads
    pub vectorize: bool,
    /// Suggested vector width
    pub vector_width: usize,
    /// Use cache hints
    pub cache_hint: CacheHint,
    /// Suggested padding for alignment
    pub padding: usize,
}

/// Cache hint for memory operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    /// Default caching
    Default,
    /// Cache at L1 level
    L1,
    /// Cache at L2 level only
    L2,
    /// Stream through cache (non-temporal)
    Streaming,
    /// Don't cache (volatile)
    NoCache,
}

impl CacheHint {
    pub fn ptx_modifier(&self) -> &'static str {
        match self {
            CacheHint::Default => "",
            CacheHint::L1 => ".ca",
            CacheHint::L2 => ".cg",
            CacheHint::Streaming => ".cs",
            CacheHint::NoCache => ".cv",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern() {
        assert!(AccessPattern::Sequential.is_coalesced());
        assert!(AccessPattern::Broadcast.is_coalesced());
        assert!(!AccessPattern::Random.is_coalesced());

        assert_eq!(AccessPattern::Sequential.efficiency(), 1.0);
        assert!(AccessPattern::Random.efficiency() < 0.5);
    }

    #[test]
    fn test_bank_conflict_analysis() {
        let analyzer = BankConflictAnalyzer::new();

        // Sequential access - no conflicts
        let addresses: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let result = analyzer.analyze(&addresses);
        assert!(result.conflict_free);
        assert_eq!(result.max_way_conflict, 1);

        // All same bank - maximum conflicts
        let addresses: Vec<usize> = (0..32).map(|i| i * 128).collect();
        let result = analyzer.analyze(&addresses);
        assert!(!result.conflict_free);
        assert_eq!(result.max_way_conflict, 32);
    }

    #[test]
    fn test_coalescing_analysis() {
        let analyzer = CoalescingAnalyzer::new();

        // Perfect coalescing
        let addresses: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let result = analyzer.analyze(&addresses, 4);
        assert!(result.is_coalesced);
        assert!(result.efficiency >= 0.75);

        // Strided access - poor coalescing
        let addresses: Vec<usize> = (0..32).map(|i| i * 256).collect();
        let result = analyzer.analyze(&addresses, 4);
        assert!(!result.is_coalesced);
    }

    #[test]
    fn test_cache_hint() {
        assert_eq!(CacheHint::Default.ptx_modifier(), "");
        assert_eq!(CacheHint::L1.ptx_modifier(), ".ca");
        assert_eq!(CacheHint::Streaming.ptx_modifier(), ".cs");
    }
}
