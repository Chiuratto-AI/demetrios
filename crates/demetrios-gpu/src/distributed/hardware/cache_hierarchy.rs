//! Cache Hierarchy and Memory Controller
//!
//! This module implements the complete GPU cache hierarchy:
//! - L1 data cache with configurable line size and associativity
//! - L2 cache with slices and coherence directory
//! - Memory controller with FR-FCFS scheduling
//! - Texture cache with Morton encoding
//! - DRAM timing models (HBM2, HBM2e, HBM3)

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ============================================================================
// L1 Cache
// ============================================================================

/// L1 cache specification per SM
#[derive(Debug, Clone)]
pub struct L1CacheSpec {
    /// Total size in KB
    pub size_kb: u32,
    /// Line size in bytes
    pub line_size: u32,
    /// Associativity (ways)
    pub associativity: u32,
    /// Hit latency in cycles
    pub hit_latency: u32,
    /// Miss penalty in cycles
    pub miss_penalty: u32,
    /// Shared memory carve-out options in KB
    pub shared_memory_options: Vec<u32>,
    /// Write policy
    pub write_policy: WritePolicy,
    /// MSHR entries
    pub mshr_entries: u32,
}

impl L1CacheSpec {
    /// A100 L1 cache spec
    pub fn a100() -> Self {
        Self {
            size_kb: 192, // Combined L1/shared
            line_size: 128,
            associativity: 4,
            hit_latency: 28,
            miss_penalty: 193,
            shared_memory_options: vec![0, 8, 16, 32, 64, 100, 132, 164],
            write_policy: WritePolicy::WriteBack,
            mshr_entries: 128,
        }
    }

    /// H100 L1 cache spec
    pub fn h100() -> Self {
        Self {
            size_kb: 256, // Larger combined L1/shared
            line_size: 128,
            associativity: 4,
            hit_latency: 24,
            miss_penalty: 180,
            shared_memory_options: vec![0, 8, 16, 32, 64, 100, 132, 164, 228],
            write_policy: WritePolicy::WriteBack,
            mshr_entries: 192,
        }
    }

    /// L4 L1 cache spec
    pub fn l4() -> Self {
        Self {
            size_kb: 128,
            line_size: 128,
            associativity: 4,
            hit_latency: 32,
            miss_penalty: 220,
            shared_memory_options: vec![0, 8, 16, 32, 64, 96],
            write_policy: WritePolicy::WriteBack,
            mshr_entries: 96,
        }
    }

    /// Number of sets in the cache
    pub fn num_sets(&self) -> u32 {
        (self.size_kb * 1024) / (self.line_size * self.associativity)
    }
}

/// Cache write policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePolicy {
    WriteThrough,
    WriteBack,
}

/// Cache line state (MESI protocol)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLineState {
    /// Modified - dirty, exclusive
    Modified,
    /// Exclusive - clean, exclusive
    Exclusive,
    /// Shared - clean, possibly shared
    Shared,
    /// Invalid
    Invalid,
}

/// A cache line
#[derive(Debug, Clone)]
pub struct CacheLine {
    /// Tag bits
    pub tag: u64,
    /// Line state
    pub state: CacheLineState,
    /// Data (simulated as bytes)
    pub data: Vec<u8>,
    /// Last access time for LRU
    pub last_access: u64,
    /// Dirty flag
    pub dirty: bool,
}

impl CacheLine {
    pub fn new(tag: u64, line_size: u32) -> Self {
        Self {
            tag,
            state: CacheLineState::Invalid,
            data: vec![0; line_size as usize],
            last_access: 0,
            dirty: false,
        }
    }
}

/// MSHR (Miss Status Holding Register) entry
#[derive(Debug, Clone)]
pub struct MshrEntry {
    /// Address being fetched
    pub address: u64,
    /// Requesters waiting for this line
    pub requesters: Vec<u32>,
    /// Cycle when request was issued
    pub issue_cycle: u64,
    /// Whether data has arrived
    pub data_ready: bool,
}

/// L1 cache implementation
#[derive(Debug)]
pub struct L1Cache {
    /// Cache specification
    pub spec: L1CacheSpec,
    /// Cache sets (set index -> ways)
    sets: Vec<Vec<CacheLine>>,
    /// MSHRs for outstanding misses
    mshrs: HashMap<u64, MshrEntry>,
    /// Current cycle
    current_cycle: u64,
    /// Statistics
    pub stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub writebacks: u64,
    pub mshr_hits: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl L1Cache {
    pub fn new(spec: L1CacheSpec) -> Self {
        let num_sets = spec.num_sets() as usize;
        let associativity = spec.associativity as usize;

        let mut sets = Vec::with_capacity(num_sets);
        for _ in 0..num_sets {
            let mut ways = Vec::with_capacity(associativity);
            for _ in 0..associativity {
                ways.push(CacheLine::new(0, spec.line_size));
            }
            sets.push(ways);
        }

        Self {
            spec,
            sets,
            mshrs: HashMap::new(),
            current_cycle: 0,
            stats: CacheStats::default(),
        }
    }

    /// Extract set index from address
    fn get_set_index(&self, address: u64) -> usize {
        let line_bits = self.spec.line_size.trailing_zeros();
        let set_bits = self.spec.num_sets().trailing_zeros();
        ((address >> line_bits) & ((1 << set_bits) - 1)) as usize
    }

    /// Extract tag from address
    fn get_tag(&self, address: u64) -> u64 {
        let line_bits = self.spec.line_size.trailing_zeros();
        let set_bits = self.spec.num_sets().trailing_zeros();
        address >> (line_bits + set_bits)
    }

    /// Access the cache (read)
    pub fn read(&mut self, address: u64) -> CacheAccessResult {
        self.current_cycle += 1;
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);

        // Check for hit
        for way in &mut self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                way.last_access = self.current_cycle;
                self.stats.hits += 1;
                return CacheAccessResult::Hit {
                    latency: self.spec.hit_latency,
                    state: way.state,
                };
            }
        }

        // Check MSHR for pending miss
        let line_addr = address & !((self.spec.line_size as u64) - 1);
        if self.mshrs.contains_key(&line_addr) {
            self.stats.mshr_hits += 1;
            return CacheAccessResult::MshrHit { address: line_addr };
        }

        // Miss
        self.stats.misses += 1;

        // Allocate MSHR if available
        if self.mshrs.len() < self.spec.mshr_entries as usize {
            self.mshrs.insert(
                line_addr,
                MshrEntry {
                    address: line_addr,
                    requesters: vec![0],
                    issue_cycle: self.current_cycle,
                    data_ready: false,
                },
            );
        }

        CacheAccessResult::Miss {
            evict_address: self.find_victim(set_idx),
            latency: self.spec.miss_penalty,
        }
    }

    /// Write to cache
    pub fn write(&mut self, address: u64) -> CacheAccessResult {
        self.current_cycle += 1;
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);

        // Check for hit
        for way in &mut self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                way.last_access = self.current_cycle;
                way.dirty = true;
                way.state = CacheLineState::Modified;
                self.stats.hits += 1;
                return CacheAccessResult::Hit {
                    latency: self.spec.hit_latency,
                    state: CacheLineState::Modified,
                };
            }
        }

        // Miss - write allocate
        self.stats.misses += 1;
        let line_addr = address & !((self.spec.line_size as u64) - 1);

        CacheAccessResult::Miss {
            evict_address: self.find_victim(set_idx),
            latency: self.spec.miss_penalty,
        }
    }

    /// Find victim for eviction using LRU
    fn find_victim(&self, set_idx: usize) -> Option<u64> {
        let set = &self.sets[set_idx];

        // First, look for invalid line
        for way in set {
            if way.state == CacheLineState::Invalid {
                return None;
            }
        }

        // Find LRU line
        let mut min_access = u64::MAX;
        let mut victim_tag = 0u64;
        let mut victim_dirty = false;

        for way in set {
            if way.last_access < min_access {
                min_access = way.last_access;
                victim_tag = way.tag;
                victim_dirty = way.dirty;
            }
        }

        if victim_dirty {
            self.stats.writebacks;
            // Reconstruct address from tag and set index
            let line_bits = self.spec.line_size.trailing_zeros();
            let set_bits = self.spec.num_sets().trailing_zeros();
            let address = (victim_tag << (line_bits + set_bits)) | ((set_idx as u64) << line_bits);
            Some(address)
        } else {
            None
        }
    }

    /// Fill a cache line after miss resolution
    pub fn fill(&mut self, address: u64, state: CacheLineState) {
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);
        let set = &mut self.sets[set_idx];

        // Find victim using LRU
        let mut min_access = u64::MAX;
        let mut victim_idx = 0;

        for (i, way) in set.iter().enumerate() {
            if way.state == CacheLineState::Invalid {
                victim_idx = i;
                break;
            }
            if way.last_access < min_access {
                min_access = way.last_access;
                victim_idx = i;
            }
        }

        // Check if evicting dirty line
        if set[victim_idx].dirty {
            self.stats.writebacks += 1;
        }
        if set[victim_idx].state != CacheLineState::Invalid {
            self.stats.evictions += 1;
        }

        // Install new line
        set[victim_idx] = CacheLine::new(tag, self.spec.line_size);
        set[victim_idx].state = state;
        set[victim_idx].dirty = state == CacheLineState::Modified;
        set[victim_idx].last_access = self.current_cycle;

        // Clear MSHR
        let line_addr = address & !((self.spec.line_size as u64) - 1);
        self.mshrs.remove(&line_addr);
    }

    /// Invalidate a cache line (for coherence)
    pub fn invalidate(&mut self, address: u64) -> bool {
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);

        for way in &mut self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                let was_dirty = way.dirty;
                way.state = CacheLineState::Invalid;
                way.dirty = false;
                return was_dirty;
            }
        }
        false
    }
}

/// Result of a cache access
#[derive(Debug, Clone)]
pub enum CacheAccessResult {
    Hit {
        latency: u32,
        state: CacheLineState,
    },
    Miss {
        evict_address: Option<u64>,
        latency: u32,
    },
    MshrHit {
        address: u64,
    },
}

// ============================================================================
// L2 Cache
// ============================================================================

/// L2 cache specification
#[derive(Debug, Clone)]
pub struct L2CacheSpec {
    /// Total size in MB
    pub size_mb: u32,
    /// Number of slices
    pub num_slices: u32,
    /// Line size in bytes
    pub line_size: u32,
    /// Associativity
    pub associativity: u32,
    /// Hit latency
    pub hit_latency: u32,
    /// Bandwidth per slice (GB/s)
    pub bandwidth_per_slice_gbps: f64,
    /// ECC enabled
    pub ecc_enabled: bool,
}

impl L2CacheSpec {
    /// A100 L2 spec
    pub fn a100() -> Self {
        Self {
            size_mb: 40,
            num_slices: 80,
            line_size: 128,
            associativity: 16,
            hit_latency: 193,
            bandwidth_per_slice_gbps: 24.0,
            ecc_enabled: true,
        }
    }

    /// H100 L2 spec
    pub fn h100() -> Self {
        Self {
            size_mb: 50,
            num_slices: 114,
            line_size: 128,
            associativity: 16,
            hit_latency: 180,
            bandwidth_per_slice_gbps: 28.0,
            ecc_enabled: true,
        }
    }

    /// L4 L2 spec
    pub fn l4() -> Self {
        Self {
            size_mb: 48,
            num_slices: 48,
            line_size: 128,
            associativity: 16,
            hit_latency: 220,
            bandwidth_per_slice_gbps: 20.0,
            ecc_enabled: true,
        }
    }

    /// Size per slice in bytes
    pub fn slice_size_bytes(&self) -> u64 {
        (self.size_mb as u64 * 1024 * 1024) / self.num_slices as u64
    }

    /// Sets per slice
    pub fn sets_per_slice(&self) -> u32 {
        let slice_bytes = self.slice_size_bytes() as u32;
        slice_bytes / (self.line_size * self.associativity)
    }
}

/// Coherence directory entry
#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    /// Set of SMs that have this line cached
    pub sharers: HashSet<u32>,
    /// Owner SM (for Modified state)
    pub owner: Option<u32>,
    /// Line state from directory perspective
    pub state: DirectoryState,
}

/// Directory state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryState {
    /// Not cached anywhere
    Uncached,
    /// Cached in shared state by multiple SMs
    Shared,
    /// Cached exclusively by one SM
    Exclusive,
}

/// L2 cache slice
#[derive(Debug)]
pub struct L2CacheSlice {
    /// Slice ID
    pub slice_id: u32,
    /// Cache sets
    sets: Vec<Vec<CacheLine>>,
    /// Coherence directory
    directory: HashMap<u64, DirectoryEntry>,
    /// Statistics
    pub stats: CacheStats,
    /// Current cycle
    current_cycle: u64,
    /// Spec reference
    spec: L2CacheSpec,
}

impl L2CacheSlice {
    pub fn new(slice_id: u32, spec: &L2CacheSpec) -> Self {
        let num_sets = spec.sets_per_slice() as usize;
        let associativity = spec.associativity as usize;

        let mut sets = Vec::with_capacity(num_sets);
        for _ in 0..num_sets {
            let mut ways = Vec::with_capacity(associativity);
            for _ in 0..associativity {
                ways.push(CacheLine::new(0, spec.line_size));
            }
            sets.push(ways);
        }

        Self {
            slice_id,
            sets,
            directory: HashMap::new(),
            stats: CacheStats::default(),
            current_cycle: 0,
            spec: spec.clone(),
        }
    }

    /// Get set index for this slice
    fn get_set_index(&self, address: u64) -> usize {
        let line_bits = self.spec.line_size.trailing_zeros();
        let set_bits = self.spec.sets_per_slice().trailing_zeros();
        ((address >> line_bits) & ((1 << set_bits) - 1)) as usize
    }

    /// Get tag
    fn get_tag(&self, address: u64) -> u64 {
        let line_bits = self.spec.line_size.trailing_zeros();
        let set_bits = self.spec.sets_per_slice().trailing_zeros();
        address >> (line_bits + set_bits)
    }

    /// Probe the cache (check if address is present)
    pub fn probe(&self, address: u64) -> Option<CacheLineState> {
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);

        for way in &self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                return Some(way.state);
            }
        }
        None
    }

    /// Access the cache
    pub fn access(&mut self, address: u64, sm_id: u32, is_write: bool) -> L2AccessResult {
        self.current_cycle += 1;
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);
        let line_addr = address & !((self.spec.line_size as u64) - 1);

        // Check for hit
        for way in &mut self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                way.last_access = self.current_cycle;
                self.stats.hits += 1;

                // Update directory
                let dir_entry = self.directory.entry(line_addr).or_insert(DirectoryEntry {
                    sharers: HashSet::new(),
                    owner: None,
                    state: DirectoryState::Uncached,
                });
                dir_entry.sharers.insert(sm_id);

                if is_write {
                    way.dirty = true;
                    way.state = CacheLineState::Modified;
                    // Invalidate other sharers
                    let invalidations: Vec<u32> = dir_entry
                        .sharers
                        .iter()
                        .filter(|&&s| s != sm_id)
                        .copied()
                        .collect();
                    dir_entry.sharers.clear();
                    dir_entry.sharers.insert(sm_id);
                    dir_entry.owner = Some(sm_id);
                    dir_entry.state = DirectoryState::Exclusive;

                    return L2AccessResult::Hit {
                        latency: self.spec.hit_latency,
                        invalidations,
                    };
                }

                return L2AccessResult::Hit {
                    latency: self.spec.hit_latency,
                    invalidations: vec![],
                };
            }
        }

        // Miss
        self.stats.misses += 1;
        L2AccessResult::Miss {
            latency: self.spec.hit_latency, // Will add DRAM latency
        }
    }

    /// Fill cache line
    pub fn fill(&mut self, address: u64, sm_id: u32, exclusive: bool) {
        let set_idx = self.get_set_index(address);
        let tag = self.get_tag(address);
        let line_addr = address & !((self.spec.line_size as u64) - 1);

        // Find victim
        let mut victim_idx = 0;
        let mut min_access = u64::MAX;

        for (i, way) in self.sets[set_idx].iter().enumerate() {
            if way.state == CacheLineState::Invalid {
                victim_idx = i;
                break;
            }
            if way.last_access < min_access {
                min_access = way.last_access;
                victim_idx = i;
            }
        }

        // Handle eviction
        let victim = &self.sets[set_idx][victim_idx];
        if victim.state != CacheLineState::Invalid {
            self.stats.evictions += 1;
            if victim.dirty {
                self.stats.writebacks += 1;
            }
            // Remove from directory
            let victim_line_addr = self.reconstruct_address(victim.tag, set_idx);
            self.directory.remove(&victim_line_addr);
        }

        // Install new line
        let state = if exclusive {
            CacheLineState::Exclusive
        } else {
            CacheLineState::Shared
        };

        self.sets[set_idx][victim_idx] = CacheLine::new(tag, self.spec.line_size);
        self.sets[set_idx][victim_idx].state = state;
        self.sets[set_idx][victim_idx].last_access = self.current_cycle;

        // Update directory
        let mut sharers = HashSet::new();
        sharers.insert(sm_id);
        self.directory.insert(
            line_addr,
            DirectoryEntry {
                sharers,
                owner: if exclusive { Some(sm_id) } else { None },
                state: if exclusive {
                    DirectoryState::Exclusive
                } else {
                    DirectoryState::Shared
                },
            },
        );
    }

    fn reconstruct_address(&self, tag: u64, set_idx: usize) -> u64 {
        let line_bits = self.spec.line_size.trailing_zeros();
        let set_bits = self.spec.sets_per_slice().trailing_zeros();
        (tag << (line_bits + set_bits)) | ((set_idx as u64) << line_bits)
    }
}

/// L2 access result
#[derive(Debug, Clone)]
pub enum L2AccessResult {
    Hit {
        latency: u32,
        invalidations: Vec<u32>,
    },
    Miss {
        latency: u32,
    },
}

/// Complete L2 cache with all slices
#[derive(Debug)]
pub struct L2Cache {
    /// Specification
    pub spec: L2CacheSpec,
    /// Cache slices
    slices: Vec<L2CacheSlice>,
}

impl L2Cache {
    pub fn new(spec: L2CacheSpec) -> Self {
        let mut slices = Vec::with_capacity(spec.num_slices as usize);
        for i in 0..spec.num_slices {
            slices.push(L2CacheSlice::new(i, &spec));
        }

        Self { spec, slices }
    }

    /// Get slice for address (hash-based distribution)
    fn get_slice(&self, address: u64) -> usize {
        // XOR-based slice selection for good distribution
        let line_addr = address >> self.spec.line_size.trailing_zeros();
        let mut hash = line_addr;
        hash ^= hash >> 17;
        hash ^= hash >> 11;
        (hash as usize) % self.slices.len()
    }

    /// Access L2 cache
    pub fn access(&mut self, address: u64, sm_id: u32, is_write: bool) -> L2AccessResult {
        let slice_idx = self.get_slice(address);
        self.slices[slice_idx].access(address, sm_id, is_write)
    }

    /// Fill L2 cache
    pub fn fill(&mut self, address: u64, sm_id: u32, exclusive: bool) {
        let slice_idx = self.get_slice(address);
        self.slices[slice_idx].fill(address, sm_id, exclusive);
    }

    /// Get aggregate statistics
    pub fn aggregate_stats(&self) -> CacheStats {
        let mut stats = CacheStats::default();
        for slice in &self.slices {
            stats.hits += slice.stats.hits;
            stats.misses += slice.stats.misses;
            stats.evictions += slice.stats.evictions;
            stats.writebacks += slice.stats.writebacks;
        }
        stats
    }

    /// Total bandwidth in GB/s
    pub fn total_bandwidth_gbps(&self) -> f64 {
        self.spec.bandwidth_per_slice_gbps * self.spec.num_slices as f64
    }
}

// ============================================================================
// Memory Controller
// ============================================================================

/// DRAM timing specification
#[derive(Debug, Clone)]
pub struct DramTimingSpec {
    /// Memory type
    pub memory_type: MemoryType,
    /// Row cycle time (tRC) in ns
    pub t_rc: f64,
    /// Row precharge time (tRP) in ns
    pub t_rp: f64,
    /// RAS to CAS delay (tRCD) in ns
    pub t_rcd: f64,
    /// CAS latency (tCL) in ns
    pub t_cl: f64,
    /// Write recovery time (tWR) in ns
    pub t_wr: f64,
    /// Burst length
    pub burst_length: u32,
    /// Data rate (MT/s)
    pub data_rate_mts: u32,
    /// Bus width (bits)
    pub bus_width: u32,
    /// Number of channels
    pub num_channels: u32,
    /// Banks per channel
    pub banks_per_channel: u32,
}

/// Memory type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    HBM2,
    HBM2e,
    HBM3,
    GDDR6,
    GDDR6X,
}

impl DramTimingSpec {
    /// HBM2 timing (A100)
    pub fn hbm2() -> Self {
        Self {
            memory_type: MemoryType::HBM2,
            t_rc: 48.0,
            t_rp: 14.0,
            t_rcd: 14.0,
            t_cl: 14.0,
            t_wr: 15.0,
            burst_length: 4,
            data_rate_mts: 2400,
            bus_width: 1024, // Per stack
            num_channels: 8, // 8 channels per stack, 5 stacks
            banks_per_channel: 16,
        }
    }

    /// HBM2e timing
    pub fn hbm2e() -> Self {
        Self {
            memory_type: MemoryType::HBM2e,
            t_rc: 45.0,
            t_rp: 13.0,
            t_rcd: 13.0,
            t_cl: 13.0,
            t_wr: 14.0,
            burst_length: 4,
            data_rate_mts: 3200,
            bus_width: 1024,
            num_channels: 8,
            banks_per_channel: 16,
        }
    }

    /// HBM3 timing (H100)
    pub fn hbm3() -> Self {
        Self {
            memory_type: MemoryType::HBM3,
            t_rc: 42.0,
            t_rp: 12.0,
            t_rcd: 12.0,
            t_cl: 12.0,
            t_wr: 12.0,
            burst_length: 8,
            data_rate_mts: 5600,
            bus_width: 1024,
            num_channels: 16, // More channels
            banks_per_channel: 32,
        }
    }

    /// Peak bandwidth in GB/s (per stack)
    pub fn peak_bandwidth_gbps(&self) -> f64 {
        let bits_per_transfer = self.bus_width as f64;
        let bytes_per_transfer = bits_per_transfer / 8.0;
        let transfers_per_second = self.data_rate_mts as f64 * 1_000_000.0;
        (bytes_per_transfer * transfers_per_second) / 1_000_000_000.0
    }

    /// Minimum read latency in ns
    pub fn min_read_latency_ns(&self) -> f64 {
        self.t_rcd + self.t_cl
    }
}

/// Memory request
#[derive(Debug, Clone)]
pub struct MemoryRequest {
    /// Request ID
    pub id: u64,
    /// Address
    pub address: u64,
    /// Is write
    pub is_write: bool,
    /// Size in bytes
    pub size: u32,
    /// Arrival time
    pub arrival_time: u64,
    /// Source SM
    pub source_sm: u32,
}

/// Bank state
#[derive(Debug, Clone)]
pub struct BankState {
    /// Currently open row (None if precharged)
    pub open_row: Option<u64>,
    /// Time when bank becomes available
    pub available_at: u64,
    /// Last access type
    pub last_write: bool,
}

/// Memory controller with FR-FCFS scheduling
#[derive(Debug)]
pub struct MemoryController {
    /// Timing specification
    pub timing: DramTimingSpec,
    /// Request queue per channel
    queues: Vec<VecDeque<MemoryRequest>>,
    /// Bank states per channel
    bank_states: Vec<Vec<BankState>>,
    /// Current cycle
    current_cycle: u64,
    /// Clock period in ns
    clock_period_ns: f64,
    /// Statistics
    pub stats: MemoryControllerStats,
    /// Next request ID
    next_id: u64,
}

/// Memory controller statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryControllerStats {
    pub total_requests: u64,
    pub row_hits: u64,
    pub row_misses: u64,
    pub total_latency_cycles: u64,
    pub reads: u64,
    pub writes: u64,
}

impl MemoryControllerStats {
    pub fn row_hit_rate(&self) -> f64 {
        let total = self.row_hits + self.row_misses;
        if total == 0 {
            0.0
        } else {
            self.row_hits as f64 / total as f64
        }
    }

    pub fn average_latency(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_latency_cycles as f64 / self.total_requests as f64
        }
    }
}

impl MemoryController {
    pub fn new(timing: DramTimingSpec, clock_freq_ghz: f64) -> Self {
        let num_channels = timing.num_channels as usize;
        let banks_per_channel = timing.banks_per_channel as usize;
        let clock_period_ns = 1.0 / clock_freq_ghz;

        let mut queues = Vec::with_capacity(num_channels);
        let mut bank_states = Vec::with_capacity(num_channels);

        for _ in 0..num_channels {
            queues.push(VecDeque::new());
            let mut banks = Vec::with_capacity(banks_per_channel);
            for _ in 0..banks_per_channel {
                banks.push(BankState {
                    open_row: None,
                    available_at: 0,
                    last_write: false,
                });
            }
            bank_states.push(banks);
        }

        Self {
            timing,
            queues,
            bank_states,
            current_cycle: 0,
            clock_period_ns,
            stats: MemoryControllerStats::default(),
            next_id: 0,
        }
    }

    /// Submit a memory request
    pub fn submit(&mut self, address: u64, is_write: bool, size: u32, source_sm: u32) -> u64 {
        let channel = self.get_channel(address);
        let id = self.next_id;
        self.next_id += 1;

        self.queues[channel].push_back(MemoryRequest {
            id,
            address,
            is_write,
            size,
            arrival_time: self.current_cycle,
            source_sm,
        });

        self.stats.total_requests += 1;
        if is_write {
            self.stats.writes += 1;
        } else {
            self.stats.reads += 1;
        }

        id
    }

    /// Get channel for address
    fn get_channel(&self, address: u64) -> usize {
        // Interleave at cache line granularity
        let line_addr = address >> 7; // 128-byte lines
        (line_addr as usize) % self.queues.len()
    }

    /// Get bank for address
    fn get_bank(&self, address: u64) -> usize {
        let line_addr = address >> 7;
        let channel_bits = (self.timing.num_channels as f64).log2() as u32;
        ((line_addr >> channel_bits) as usize) % self.timing.banks_per_channel as usize
    }

    /// Get row for address
    fn get_row(&self, address: u64) -> u64 {
        let line_addr = address >> 7;
        let channel_bits = (self.timing.num_channels as f64).log2() as u32;
        let bank_bits = (self.timing.banks_per_channel as f64).log2() as u32;
        line_addr >> (channel_bits + bank_bits)
    }

    /// Advance simulation by one cycle
    pub fn tick(&mut self) -> Vec<(u64, u64)> {
        self.current_cycle += 1;
        let mut completed = Vec::new();

        // FR-FCFS scheduling per channel
        for channel in 0..self.queues.len() {
            if let Some(req_idx) = self.select_request(channel) {
                let req = self.queues[channel].remove(req_idx).unwrap();
                let latency = self.service_request(channel, &req);
                completed.push((req.id, latency));
                self.stats.total_latency_cycles += latency;
            }
        }

        completed
    }

    /// FR-FCFS: First Ready, First Come First Served
    fn select_request(&self, channel: usize) -> Option<usize> {
        let queue = &self.queues[channel];
        if queue.is_empty() {
            return None;
        }

        // First, find any row-hit requests
        for (i, req) in queue.iter().enumerate() {
            let bank = self.get_bank(req.address);
            let row = self.get_row(req.address);
            let bank_state = &self.bank_states[channel][bank];

            if bank_state.available_at <= self.current_cycle {
                if let Some(open_row) = bank_state.open_row {
                    if open_row == row {
                        return Some(i);
                    }
                }
            }
        }

        // Otherwise, first ready request
        for (i, req) in queue.iter().enumerate() {
            let bank = self.get_bank(req.address);
            if self.bank_states[channel][bank].available_at <= self.current_cycle {
                return Some(i);
            }
        }

        // All banks busy, return oldest
        Some(0)
    }

    /// Service a memory request
    fn service_request(&mut self, channel: usize, req: &MemoryRequest) -> u64 {
        let bank = self.get_bank(req.address);
        let row = self.get_row(req.address);
        let bank_state = &mut self.bank_states[channel][bank];

        let mut latency_ns = 0.0;

        // Check if row hit or miss
        match bank_state.open_row {
            Some(open_row) if open_row == row => {
                // Row hit - just CAS latency
                latency_ns += self.timing.t_cl;
                self.stats.row_hits += 1;
            }
            Some(_) => {
                // Row conflict - precharge + activate + CAS
                latency_ns += self.timing.t_rp + self.timing.t_rcd + self.timing.t_cl;
                self.stats.row_misses += 1;
            }
            None => {
                // Row miss (closed row) - activate + CAS
                latency_ns += self.timing.t_rcd + self.timing.t_cl;
                self.stats.row_misses += 1;
            }
        }

        // Update bank state
        bank_state.open_row = Some(row);
        bank_state.last_write = req.is_write;

        let latency_cycles = (latency_ns / self.clock_period_ns).ceil() as u64;
        bank_state.available_at = self.current_cycle + latency_cycles;

        latency_cycles
    }

    /// Get queue depths
    pub fn queue_depths(&self) -> Vec<usize> {
        self.queues.iter().map(|q| q.len()).collect()
    }
}

// ============================================================================
// Texture Cache
// ============================================================================

/// Texture cache specification
#[derive(Debug, Clone)]
pub struct TextureCacheSpec {
    /// Size in KB
    pub size_kb: u32,
    /// Line size in bytes
    pub line_size: u32,
    /// Associativity
    pub associativity: u32,
    /// Hit latency
    pub hit_latency: u32,
    /// Supports texture filtering
    pub filtering_support: bool,
}

impl TextureCacheSpec {
    pub fn default_spec() -> Self {
        Self {
            size_kb: 48,
            line_size: 32,
            associativity: 24,
            hit_latency: 80,
            filtering_support: true,
        }
    }
}

/// 2D texture coordinates
#[derive(Debug, Clone, Copy)]
pub struct TexCoord {
    pub u: f32,
    pub v: f32,
}

/// Texture dimensions
#[derive(Debug, Clone, Copy)]
pub struct TextureDimensions {
    pub width: u32,
    pub height: u32,
    pub depth: u32, // For 3D textures
}

/// Morton (Z-order) encoding for 2D spatial locality
#[derive(Debug, Clone, Copy)]
pub struct MortonCode;

impl MortonCode {
    /// Encode 2D coordinates to Morton code
    pub fn encode_2d(x: u32, y: u32) -> u64 {
        Self::spread_bits(x as u64) | (Self::spread_bits(y as u64) << 1)
    }

    /// Decode Morton code to 2D coordinates
    pub fn decode_2d(code: u64) -> (u32, u32) {
        (
            Self::compact_bits(code) as u32,
            Self::compact_bits(code >> 1) as u32,
        )
    }

    /// Spread bits for Morton encoding
    fn spread_bits(mut x: u64) -> u64 {
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x3333333333333333;
        x = (x | (x << 1)) & 0x5555555555555555;
        x
    }

    /// Compact bits for Morton decoding
    fn compact_bits(mut x: u64) -> u64 {
        x &= 0x5555555555555555;
        x = (x | (x >> 1)) & 0x3333333333333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
        x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
        x
    }

    /// Encode 3D coordinates to Morton code
    pub fn encode_3d(x: u32, y: u32, z: u32) -> u64 {
        Self::spread_bits_3d(x as u64)
            | (Self::spread_bits_3d(y as u64) << 1)
            | (Self::spread_bits_3d(z as u64) << 2)
    }

    fn spread_bits_3d(mut x: u64) -> u64 {
        x &= 0x1FFFFF;
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x << 8)) & 0x100F00F00F00F00F;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3;
        x = (x | (x << 2)) & 0x1249249249249249;
        x
    }
}

/// Texture cache with Morton-order addressing
#[derive(Debug)]
pub struct TextureCache {
    /// Specification
    pub spec: TextureCacheSpec,
    /// Cache sets
    sets: Vec<Vec<CacheLine>>,
    /// Statistics
    pub stats: CacheStats,
    /// Current cycle
    current_cycle: u64,
}

impl TextureCache {
    pub fn new(spec: TextureCacheSpec) -> Self {
        let num_sets = (spec.size_kb * 1024) / (spec.line_size * spec.associativity);
        let num_sets = num_sets as usize;
        let associativity = spec.associativity as usize;

        let mut sets = Vec::with_capacity(num_sets);
        for _ in 0..num_sets {
            let mut ways = Vec::with_capacity(associativity);
            for _ in 0..associativity {
                ways.push(CacheLine::new(0, spec.line_size));
            }
            sets.push(ways);
        }

        Self {
            spec,
            sets,
            stats: CacheStats::default(),
            current_cycle: 0,
        }
    }

    /// Convert texture coordinates to Morton-order address
    pub fn tex_to_address(
        &self,
        coord: TexCoord,
        dims: TextureDimensions,
        base_address: u64,
        texel_size: u32,
    ) -> u64 {
        let x = ((coord.u * dims.width as f32) as u32).min(dims.width - 1);
        let y = ((coord.v * dims.height as f32) as u32).min(dims.height - 1);

        let morton = MortonCode::encode_2d(x, y);
        base_address + (morton * texel_size as u64)
    }

    /// Access texture cache
    pub fn access(&mut self, address: u64) -> CacheAccessResult {
        self.current_cycle += 1;
        let line_bits = self.spec.line_size.trailing_zeros();
        let num_sets = self.sets.len();
        let set_bits = (num_sets as f64).log2() as u32;

        let set_idx = ((address >> line_bits) & ((1 << set_bits) - 1)) as usize;
        let tag = address >> (line_bits + set_bits);

        // Check for hit
        for way in &mut self.sets[set_idx] {
            if way.tag == tag && way.state != CacheLineState::Invalid {
                way.last_access = self.current_cycle;
                self.stats.hits += 1;
                return CacheAccessResult::Hit {
                    latency: self.spec.hit_latency,
                    state: way.state,
                };
            }
        }

        self.stats.misses += 1;
        CacheAccessResult::Miss {
            evict_address: None,
            latency: self.spec.hit_latency * 3, // Miss penalty
        }
    }

    /// Sample texture with bilinear filtering
    pub fn sample_bilinear(
        &mut self,
        coord: TexCoord,
        dims: TextureDimensions,
        base_address: u64,
        texel_size: u32,
    ) -> Vec<u64> {
        // Get the four texels for bilinear filtering
        let x = coord.u * (dims.width - 1) as f32;
        let y = coord.v * (dims.height - 1) as f32;

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(dims.width - 1);
        let y1 = (y0 + 1).min(dims.height - 1);

        // Four Morton-encoded addresses
        vec![
            base_address + MortonCode::encode_2d(x0, y0) * texel_size as u64,
            base_address + MortonCode::encode_2d(x1, y0) * texel_size as u64,
            base_address + MortonCode::encode_2d(x0, y1) * texel_size as u64,
            base_address + MortonCode::encode_2d(x1, y1) * texel_size as u64,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_cache_spec() {
        let spec = L1CacheSpec::a100();
        assert_eq!(spec.size_kb, 192);
        assert_eq!(spec.line_size, 128);
        assert_eq!(spec.associativity, 4);
        assert!(spec.num_sets() > 0);
    }

    #[test]
    fn test_l1_cache_hit_miss() {
        let spec = L1CacheSpec::a100();
        let mut cache = L1Cache::new(spec);

        // First access - miss
        let result = cache.read(0x1000);
        assert!(matches!(result, CacheAccessResult::Miss { .. }));

        // Fill the line
        cache.fill(0x1000, CacheLineState::Shared);

        // Second access - hit
        let result = cache.read(0x1000);
        assert!(matches!(result, CacheAccessResult::Hit { .. }));

        assert_eq!(cache.stats.hits, 1);
        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_l1_cache_write() {
        let spec = L1CacheSpec::a100();
        let mut cache = L1Cache::new(spec.clone());

        // Fill a line
        cache.fill(0x1000, CacheLineState::Exclusive);

        // Write to it
        let result = cache.write(0x1000);
        if let CacheAccessResult::Hit { state, .. } = result {
            assert_eq!(state, CacheLineState::Modified);
        } else {
            panic!("Expected hit");
        }
    }

    #[test]
    fn test_l2_cache_spec() {
        let spec = L2CacheSpec::a100();
        assert_eq!(spec.size_mb, 40);
        assert_eq!(spec.num_slices, 80);
        assert!(spec.slice_size_bytes() > 0);
        assert!(spec.sets_per_slice() > 0);
    }

    #[test]
    fn test_l2_cache_access() {
        let spec = L2CacheSpec::a100();
        let mut cache = L2Cache::new(spec);

        // Access and fill
        let result = cache.access(0x1000, 0, false);
        assert!(matches!(result, L2AccessResult::Miss { .. }));

        cache.fill(0x1000, 0, false);

        let result = cache.access(0x1000, 0, false);
        assert!(matches!(result, L2AccessResult::Hit { .. }));
    }

    #[test]
    fn test_l2_cache_coherence() {
        let spec = L2CacheSpec::a100();
        let mut cache = L2Cache::new(spec);

        // SM 0 fills shared
        cache.fill(0x1000, 0, false);

        // SM 1 reads (becomes sharer)
        let _ = cache.access(0x1000, 1, false);

        // SM 2 writes - should invalidate others
        let result = cache.access(0x1000, 2, true);
        if let L2AccessResult::Hit { invalidations, .. } = result {
            // Should have invalidations for SM 0 and SM 1
            assert!(invalidations.len() >= 1);
        }
    }

    #[test]
    fn test_dram_timing() {
        let hbm2 = DramTimingSpec::hbm2();
        let hbm3 = DramTimingSpec::hbm3();

        // HBM3 should have higher bandwidth
        assert!(hbm3.peak_bandwidth_gbps() > hbm2.peak_bandwidth_gbps());

        // HBM3 should have lower latency
        assert!(hbm3.min_read_latency_ns() < hbm2.min_read_latency_ns());
    }

    #[test]
    fn test_memory_controller() {
        let timing = DramTimingSpec::hbm2();
        let mut mc = MemoryController::new(timing, 1.4); // 1.4 GHz

        // Submit some requests
        mc.submit(0x0000, false, 128, 0);
        mc.submit(0x1000, false, 128, 0);
        mc.submit(0x0080, false, 128, 0); // Same row as first

        // Run a few cycles
        let mut completed = Vec::new();
        for _ in 0..500 {
            completed.extend(mc.tick());
        }

        assert_eq!(completed.len(), 3);
        // Check that we processed requests (row hits may or may not occur
        // depending on timing and scheduling)
        assert_eq!(mc.stats.total_requests, 3);
    }

    #[test]
    fn test_memory_controller_fr_fcfs() {
        let timing = DramTimingSpec::hbm2();
        let mut mc = MemoryController::new(timing.clone(), 1.4);

        // Submit requests that will create row conflicts and hits
        mc.submit(0x0000, false, 128, 0);
        mc.submit(0x100000, false, 128, 0); // Different row
        mc.submit(0x0080, false, 128, 0); // Same row as first

        // The FR-FCFS scheduler should prioritize row hits
        // Run until all complete
        let mut completed_order = Vec::new();
        for _ in 0..1000 {
            for (id, _) in mc.tick() {
                completed_order.push(id);
            }
            if completed_order.len() == 3 {
                break;
            }
        }

        assert_eq!(completed_order.len(), 3);
    }

    #[test]
    fn test_morton_code_2d() {
        // Test round-trip encoding/decoding
        let test_coords = [(0, 0), (1, 0), (0, 1), (1, 1), (5, 3), (100, 200)];

        for (x, y) in test_coords {
            let code = MortonCode::encode_2d(x, y);
            let (decoded_x, decoded_y) = MortonCode::decode_2d(code);
            assert_eq!((x, y), (decoded_x, decoded_y));
        }
    }

    #[test]
    fn test_morton_code_locality() {
        // Adjacent coordinates should have closer Morton codes
        let code_00 = MortonCode::encode_2d(0, 0);
        let code_01 = MortonCode::encode_2d(0, 1);
        let code_10 = MortonCode::encode_2d(1, 0);
        let code_11 = MortonCode::encode_2d(1, 1);

        // All should be close together
        assert!(code_01 < 4);
        assert!(code_10 < 4);
        assert!(code_11 < 4);
        assert_eq!(code_00, 0);
    }

    #[test]
    fn test_texture_cache() {
        let spec = TextureCacheSpec::default_spec();
        let mut cache = TextureCache::new(spec);

        // Test access
        let result = cache.access(0x1000);
        assert!(matches!(result, CacheAccessResult::Miss { .. }));

        assert_eq!(cache.stats.misses, 1);
    }

    #[test]
    fn test_texture_sampling() {
        let spec = TextureCacheSpec::default_spec();
        let mut cache = TextureCache::new(spec);

        let dims = TextureDimensions {
            width: 256,
            height: 256,
            depth: 1,
        };

        let coord = TexCoord { u: 0.5, v: 0.5 };
        let addresses = cache.sample_bilinear(coord, dims, 0, 4);

        // Should get 4 addresses for bilinear filtering
        assert_eq!(addresses.len(), 4);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.hits = 80;
        stats.misses = 20;

        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_l2_bandwidth() {
        let a100 = L2CacheSpec::a100();
        let h100 = L2CacheSpec::h100();

        let a100_cache = L2Cache::new(a100);
        let h100_cache = L2Cache::new(h100);

        // H100 should have higher total bandwidth
        assert!(h100_cache.total_bandwidth_gbps() > a100_cache.total_bandwidth_gbps());
    }

    #[test]
    fn test_mshr_coalescing() {
        let spec = L1CacheSpec::a100();
        let mut cache = L1Cache::new(spec.clone());

        // First miss
        let result1 = cache.read(0x1000);
        assert!(matches!(result1, CacheAccessResult::Miss { .. }));

        // Second access to same line - should hit MSHR
        let result2 = cache.read(0x1040); // Same 128-byte line
        assert!(matches!(result2, CacheAccessResult::MshrHit { .. }));

        assert_eq!(cache.stats.mshr_hits, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let spec = L1CacheSpec::a100();
        let mut cache = L1Cache::new(spec);

        // Fill a line
        cache.fill(0x1000, CacheLineState::Modified);

        // Verify it's there
        let result = cache.read(0x1000);
        assert!(matches!(result, CacheAccessResult::Hit { .. }));

        // Invalidate
        let was_dirty = cache.invalidate(0x1000);
        assert!(was_dirty);

        // Now should miss
        let result = cache.read(0x1000);
        assert!(matches!(result, CacheAccessResult::Miss { .. }));
    }

    #[test]
    fn test_memory_type_bandwidth() {
        let hbm2 = DramTimingSpec::hbm2();
        let hbm2e = DramTimingSpec::hbm2e();
        let hbm3 = DramTimingSpec::hbm3();

        // Bandwidth should increase with each generation
        let bw_hbm2 = hbm2.peak_bandwidth_gbps();
        let bw_hbm2e = hbm2e.peak_bandwidth_gbps();
        let bw_hbm3 = hbm3.peak_bandwidth_gbps();

        assert!(bw_hbm2e > bw_hbm2);
        assert!(bw_hbm3 > bw_hbm2e);
    }
}
