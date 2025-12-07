//! Unified Virtual Memory (UVM) Implementation
//!
//! This module implements CUDA Unified Memory:
//! - Page table management for CPU-GPU address translation
//! - Page fault handling and migration
//! - Access pattern detection for prefetching
//! - Memory migration policies and heuristics

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ============================================================================
// Page Table
// ============================================================================

/// Page size options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageSize {
    /// 4KB small page
    Small4K,
    /// 64KB large page
    Large64K,
    /// 2MB huge page
    Huge2M,
}

impl PageSize {
    /// Size in bytes
    pub fn bytes(&self) -> u64 {
        match self {
            PageSize::Small4K => 4 * 1024,
            PageSize::Large64K => 64 * 1024,
            PageSize::Huge2M => 2 * 1024 * 1024,
        }
    }

    /// Number of bits for offset
    pub fn offset_bits(&self) -> u32 {
        match self {
            PageSize::Small4K => 12,
            PageSize::Large64K => 16,
            PageSize::Huge2M => 21,
        }
    }
}

/// Memory location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// CPU/Host memory
    Cpu,
    /// GPU device memory
    Gpu(u32), // GPU ID
    /// In migration (transient state)
    Migrating { from: u32, to: u32 },
    /// Not allocated
    NotMapped,
}

/// Page table entry
#[derive(Debug, Clone)]
pub struct PageTableEntry {
    /// Virtual page number
    pub vpn: u64,
    /// Physical page number
    pub ppn: u64,
    /// Page size
    pub page_size: PageSize,
    /// Current location
    pub location: MemoryLocation,
    /// Valid bit
    pub valid: bool,
    /// Read permission
    pub read: bool,
    /// Write permission
    pub write: bool,
    /// Accessed bit
    pub accessed: bool,
    /// Dirty bit
    pub dirty: bool,
    /// Access counter for migration decisions
    pub access_count: u64,
    /// Last access cycle
    pub last_access: u64,
}

impl PageTableEntry {
    pub fn new(vpn: u64, ppn: u64, page_size: PageSize) -> Self {
        Self {
            vpn,
            ppn,
            page_size,
            location: MemoryLocation::NotMapped,
            valid: false,
            read: true,
            write: true,
            accessed: false,
            dirty: false,
            access_count: 0,
            last_access: 0,
        }
    }
}

/// GPU page table
#[derive(Debug)]
pub struct GpuPageTable {
    /// Page table entries (VPN -> PTE)
    entries: HashMap<u64, PageTableEntry>,
    /// Default page size
    default_page_size: PageSize,
    /// TLB (simplified)
    tlb: HashMap<u64, PageTableEntry>,
    /// TLB capacity
    tlb_capacity: usize,
    /// Statistics
    pub stats: PageTableStats,
}

/// Page table statistics
#[derive(Debug, Clone, Default)]
pub struct PageTableStats {
    pub tlb_hits: u64,
    pub tlb_misses: u64,
    pub page_faults: u64,
    pub migrations: u64,
}

impl PageTableStats {
    pub fn tlb_hit_rate(&self) -> f64 {
        let total = self.tlb_hits + self.tlb_misses;
        if total == 0 {
            0.0
        } else {
            self.tlb_hits as f64 / total as f64
        }
    }
}

impl GpuPageTable {
    pub fn new(default_page_size: PageSize, tlb_capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            default_page_size,
            tlb: HashMap::new(),
            tlb_capacity,
            stats: PageTableStats::default(),
        }
    }

    /// Get VPN from virtual address
    fn get_vpn(&self, va: u64, page_size: PageSize) -> u64 {
        va >> page_size.offset_bits()
    }

    /// Translate virtual address to physical
    pub fn translate(&mut self, va: u64) -> Result<u64, PageFault> {
        let vpn = self.get_vpn(va, self.default_page_size);
        let offset = va & ((1u64 << self.default_page_size.offset_bits()) - 1);

        // Check TLB first
        if let Some(pte) = self.tlb.get(&vpn) {
            self.stats.tlb_hits += 1;
            if pte.valid {
                return Ok((pte.ppn << self.default_page_size.offset_bits()) | offset);
            }
        }

        self.stats.tlb_misses += 1;

        // Check page table
        if let Some(pte) = self.entries.get(&vpn).cloned() {
            if pte.valid {
                let pa = (pte.ppn << self.default_page_size.offset_bits()) | offset;
                // Update TLB
                self.update_tlb(vpn, pte);
                return Ok(pa);
            }
        }

        // Page fault
        self.stats.page_faults += 1;
        Err(PageFault {
            virtual_address: va,
            vpn,
            fault_type: PageFaultType::NotPresent,
        })
    }

    /// Update TLB with eviction if needed
    fn update_tlb(&mut self, vpn: u64, pte: PageTableEntry) {
        if self.tlb.len() >= self.tlb_capacity {
            // Simple eviction: remove first entry
            if let Some(&key) = self.tlb.keys().next() {
                self.tlb.remove(&key);
            }
        }
        self.tlb.insert(vpn, pte);
    }

    /// Map a page
    pub fn map_page(&mut self, vpn: u64, ppn: u64, location: MemoryLocation) {
        let pte = PageTableEntry {
            vpn,
            ppn,
            page_size: self.default_page_size,
            location,
            valid: true,
            read: true,
            write: true,
            accessed: false,
            dirty: false,
            access_count: 0,
            last_access: 0,
        };
        self.entries.insert(vpn, pte);
    }

    /// Unmap a page
    pub fn unmap_page(&mut self, vpn: u64) {
        self.entries.remove(&vpn);
        self.tlb.remove(&vpn);
    }

    /// Update page location (for migration)
    pub fn update_location(&mut self, vpn: u64, new_ppn: u64, new_location: MemoryLocation) {
        if let Some(pte) = self.entries.get_mut(&vpn) {
            pte.ppn = new_ppn;
            pte.location = new_location;
            self.tlb.remove(&vpn); // Invalidate TLB entry
        }
    }

    /// Mark page as accessed
    pub fn mark_accessed(&mut self, vpn: u64, cycle: u64) {
        if let Some(pte) = self.entries.get_mut(&vpn) {
            pte.accessed = true;
            pte.access_count += 1;
            pte.last_access = cycle;
        }
    }

    /// Mark page as dirty
    pub fn mark_dirty(&mut self, vpn: u64) {
        if let Some(pte) = self.entries.get_mut(&vpn) {
            pte.dirty = true;
        }
    }

    /// Get page info
    pub fn get_page(&self, vpn: u64) -> Option<&PageTableEntry> {
        self.entries.get(&vpn)
    }

    /// Flush TLB
    pub fn flush_tlb(&mut self) {
        self.tlb.clear();
    }
}

// ============================================================================
// Page Faults
// ============================================================================

/// Page fault types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageFaultType {
    /// Page not present in memory
    NotPresent,
    /// Page is on different device
    WrongDevice,
    /// Permission violation
    PermissionViolation,
    /// Page is being migrated
    MigrationInProgress,
}

/// Page fault information
#[derive(Debug, Clone)]
pub struct PageFault {
    /// Faulting virtual address
    pub virtual_address: u64,
    /// Virtual page number
    pub vpn: u64,
    /// Fault type
    pub fault_type: PageFaultType,
}

/// Page fault handler
#[derive(Debug)]
pub struct PageFaultHandler {
    /// Pending faults queue
    pending: VecDeque<PageFault>,
    /// Faults being handled
    in_progress: HashMap<u64, PageFaultResolution>,
    /// Resolution timeout in cycles
    timeout_cycles: u64,
    /// Statistics
    pub stats: FaultHandlerStats,
}

/// Page fault resolution
#[derive(Debug, Clone)]
pub struct PageFaultResolution {
    /// Fault info
    pub fault: PageFault,
    /// Resolution action
    pub action: ResolutionAction,
    /// Start cycle
    pub start_cycle: u64,
    /// Expected completion cycle
    pub expected_complete: u64,
}

/// Actions to resolve page faults
#[derive(Debug, Clone)]
pub enum ResolutionAction {
    /// Allocate new page on GPU
    AllocateOnGpu(u32),
    /// Migrate page from CPU
    MigrateFromCpu,
    /// Migrate from another GPU
    MigrateFromGpu(u32),
    /// Retry after migration completes
    WaitForMigration,
}

/// Fault handler statistics
#[derive(Debug, Clone, Default)]
pub struct FaultHandlerStats {
    pub total_faults: u64,
    pub allocations: u64,
    pub migrations_from_cpu: u64,
    pub migrations_from_gpu: u64,
    pub avg_resolution_cycles: f64,
}

impl PageFaultHandler {
    pub fn new(timeout_cycles: u64) -> Self {
        Self {
            pending: VecDeque::new(),
            in_progress: HashMap::new(),
            timeout_cycles,
            stats: FaultHandlerStats::default(),
        }
    }

    /// Report a page fault
    pub fn report_fault(&mut self, fault: PageFault) {
        self.pending.push_back(fault);
        self.stats.total_faults += 1;
    }

    /// Begin handling pending faults
    pub fn handle_faults(
        &mut self,
        page_table: &GpuPageTable,
        current_cycle: u64,
        target_gpu: u32,
    ) -> Vec<PageFaultResolution> {
        let mut resolutions = Vec::new();

        while let Some(fault) = self.pending.pop_front() {
            // Determine resolution action based on current location
            let action = if let Some(pte) = page_table.get_page(fault.vpn) {
                match pte.location {
                    MemoryLocation::Cpu => {
                        self.stats.migrations_from_cpu += 1;
                        ResolutionAction::MigrateFromCpu
                    }
                    MemoryLocation::Gpu(other_gpu) if other_gpu != target_gpu => {
                        self.stats.migrations_from_gpu += 1;
                        ResolutionAction::MigrateFromGpu(other_gpu)
                    }
                    MemoryLocation::Migrating { .. } => ResolutionAction::WaitForMigration,
                    _ => {
                        self.stats.allocations += 1;
                        ResolutionAction::AllocateOnGpu(target_gpu)
                    }
                }
            } else {
                self.stats.allocations += 1;
                ResolutionAction::AllocateOnGpu(target_gpu)
            };

            let resolution = PageFaultResolution {
                fault: fault.clone(),
                action: action.clone(),
                start_cycle: current_cycle,
                expected_complete: current_cycle + self.resolution_latency(&action),
            };

            self.in_progress.insert(fault.vpn, resolution.clone());
            resolutions.push(resolution);
        }

        resolutions
    }

    /// Latency for different resolution actions
    fn resolution_latency(&self, action: &ResolutionAction) -> u64 {
        match action {
            ResolutionAction::AllocateOnGpu(_) => 1000, // ~1us at 1GHz
            ResolutionAction::MigrateFromCpu => 50000,  // ~50us for page migration
            ResolutionAction::MigrateFromGpu(_) => 30000, // ~30us for GPU-GPU
            ResolutionAction::WaitForMigration => 1000, // Retry delay
        }
    }

    /// Check for completed resolutions
    pub fn check_completions(&mut self, current_cycle: u64) -> Vec<u64> {
        let mut completed = Vec::new();

        for (vpn, resolution) in &self.in_progress {
            if current_cycle >= resolution.expected_complete {
                completed.push(*vpn);
            }
        }

        for vpn in &completed {
            self.in_progress.remove(vpn);
        }

        completed
    }
}

// ============================================================================
// Access Pattern Detection
// ============================================================================

/// Access pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (stride = 1 page)
    Sequential,
    /// Strided access (regular stride)
    Strided { stride: i64 },
    /// Random access
    Random,
    /// Clustered access (hot spots)
    Clustered,
    /// Unknown pattern
    Unknown,
}

/// Access history for pattern detection
#[derive(Debug, Clone)]
pub struct AccessHistory {
    /// Recent addresses accessed
    recent_addresses: VecDeque<u64>,
    /// Maximum history size
    max_history: usize,
    /// Stride frequency map
    stride_counts: HashMap<i64, u32>,
}

impl AccessHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            recent_addresses: VecDeque::with_capacity(max_history),
            max_history,
            stride_counts: HashMap::new(),
        }
    }

    /// Record an access
    pub fn record(&mut self, address: u64) {
        if let Some(&last) = self.recent_addresses.back() {
            let stride = address as i64 - last as i64;
            *self.stride_counts.entry(stride).or_insert(0) += 1;
        }

        self.recent_addresses.push_back(address);
        if self.recent_addresses.len() > self.max_history {
            self.recent_addresses.pop_front();
        }
    }

    /// Detect the dominant access pattern
    pub fn detect_pattern(&self) -> AccessPattern {
        if self.stride_counts.is_empty() {
            return AccessPattern::Unknown;
        }

        // Find most common stride
        let mut max_count = 0;
        let mut dominant_stride = 0i64;
        let mut total_count = 0u32;

        for (&stride, &count) in &self.stride_counts {
            total_count += count;
            if count > max_count {
                max_count = count;
                dominant_stride = stride;
            }
        }

        // Threshold for pattern detection
        let threshold = (total_count as f64 * 0.5) as u32;

        if max_count > threshold {
            if dominant_stride == 4096 || dominant_stride == 65536 {
                // Page-sized stride
                AccessPattern::Sequential
            } else if dominant_stride != 0 {
                AccessPattern::Strided {
                    stride: dominant_stride,
                }
            } else {
                AccessPattern::Clustered
            }
        } else if self.stride_counts.len() > 10 {
            AccessPattern::Random
        } else {
            AccessPattern::Unknown
        }
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.recent_addresses.clear();
        self.stride_counts.clear();
    }
}

/// Access pattern detector for UVM
#[derive(Debug)]
pub struct AccessPatternDetector {
    /// Per-allocation history
    histories: HashMap<u64, AccessHistory>,
    /// History size
    history_size: usize,
}

impl AccessPatternDetector {
    pub fn new(history_size: usize) -> Self {
        Self {
            histories: HashMap::new(),
            history_size,
        }
    }

    /// Record an access to an allocation
    pub fn record_access(&mut self, allocation_id: u64, address: u64) {
        let history = self
            .histories
            .entry(allocation_id)
            .or_insert_with(|| AccessHistory::new(self.history_size));
        history.record(address);
    }

    /// Get pattern for an allocation
    pub fn get_pattern(&self, allocation_id: u64) -> AccessPattern {
        self.histories
            .get(&allocation_id)
            .map(|h| h.detect_pattern())
            .unwrap_or(AccessPattern::Unknown)
    }
}

// ============================================================================
// Memory Migration
// ============================================================================

/// Migration policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPolicy {
    /// Migrate on first access (eager)
    OnFirstAccess,
    /// Migrate on first touch (like first access)
    FirstTouch,
    /// Migrate based on access frequency
    AccessFrequency { threshold: u32 },
    /// Never auto-migrate
    Manual,
    /// Prefetch based on access patterns
    Prefetch,
}

/// Migration request
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Page to migrate
    pub vpn: u64,
    /// Source location
    pub source: MemoryLocation,
    /// Destination
    pub destination: MemoryLocation,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Request time
    pub request_cycle: u64,
}

/// Migration engine
#[derive(Debug)]
pub struct MigrationEngine {
    /// Migration policy
    policy: MigrationPolicy,
    /// Pending migrations
    pending: VecDeque<MigrationRequest>,
    /// In-flight migrations
    in_flight: HashMap<u64, MigrationInFlight>,
    /// Maximum concurrent migrations
    max_concurrent: usize,
    /// Migration bandwidth in bytes/cycle
    bandwidth_bytes_per_cycle: u64,
    /// Statistics
    pub stats: MigrationStats,
}

/// In-flight migration state
#[derive(Debug, Clone)]
pub struct MigrationInFlight {
    /// Request
    pub request: MigrationRequest,
    /// Bytes transferred so far
    pub bytes_transferred: u64,
    /// Total bytes to transfer
    pub total_bytes: u64,
    /// Start cycle
    pub start_cycle: u64,
}

/// Migration statistics
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    pub total_migrations: u64,
    pub total_bytes_migrated: u64,
    pub cpu_to_gpu: u64,
    pub gpu_to_cpu: u64,
    pub gpu_to_gpu: u64,
    pub avg_migration_cycles: f64,
}

impl MigrationEngine {
    pub fn new(
        policy: MigrationPolicy,
        max_concurrent: usize,
        bandwidth_gbps: f64,
        clock_freq_ghz: f64,
    ) -> Self {
        // Convert bandwidth to bytes per cycle
        let bytes_per_second = bandwidth_gbps * 1e9 / 8.0;
        let cycles_per_second = clock_freq_ghz * 1e9;
        let bandwidth_bytes_per_cycle = (bytes_per_second / cycles_per_second) as u64;

        Self {
            policy,
            pending: VecDeque::new(),
            in_flight: HashMap::new(),
            max_concurrent,
            bandwidth_bytes_per_cycle: bandwidth_bytes_per_cycle.max(1),
            stats: MigrationStats::default(),
        }
    }

    /// Request a migration
    pub fn request_migration(
        &mut self,
        vpn: u64,
        source: MemoryLocation,
        destination: MemoryLocation,
        priority: u32,
        current_cycle: u64,
    ) {
        let request = MigrationRequest {
            vpn,
            source,
            destination,
            priority,
            request_cycle: current_cycle,
        };

        // Insert in priority order
        let pos = self
            .pending
            .iter()
            .position(|r| r.priority < priority)
            .unwrap_or(self.pending.len());
        self.pending.insert(pos, request);
    }

    /// Start pending migrations
    pub fn start_migrations(
        &mut self,
        page_size: PageSize,
        current_cycle: u64,
    ) -> Vec<MigrationRequest> {
        let mut started = Vec::new();

        while self.in_flight.len() < self.max_concurrent {
            if let Some(request) = self.pending.pop_front() {
                // Update type-specific counters before moving request
                match (&request.source, &request.destination) {
                    (MemoryLocation::Cpu, MemoryLocation::Gpu(_)) => {
                        self.stats.cpu_to_gpu += 1;
                    }
                    (MemoryLocation::Gpu(_), MemoryLocation::Cpu) => {
                        self.stats.gpu_to_cpu += 1;
                    }
                    (MemoryLocation::Gpu(_), MemoryLocation::Gpu(_)) => {
                        self.stats.gpu_to_gpu += 1;
                    }
                    _ => {}
                }

                let vpn = request.vpn;
                let in_flight = MigrationInFlight {
                    request: request.clone(),
                    bytes_transferred: 0,
                    total_bytes: page_size.bytes(),
                    start_cycle: current_cycle,
                };

                self.in_flight.insert(vpn, in_flight);
                started.push(request);
                self.stats.total_migrations += 1;
            } else {
                break;
            }
        }

        started
    }

    /// Advance migrations by one cycle
    pub fn tick(&mut self, current_cycle: u64) -> Vec<u64> {
        let mut completed = Vec::new();

        for (vpn, migration) in &mut self.in_flight {
            migration.bytes_transferred += self.bandwidth_bytes_per_cycle;

            if migration.bytes_transferred >= migration.total_bytes {
                completed.push(*vpn);
                self.stats.total_bytes_migrated += migration.total_bytes;
            }
        }

        for vpn in &completed {
            self.in_flight.remove(vpn);
        }

        completed
    }

    /// Check if migration is in progress for a page
    pub fn is_migrating(&self, vpn: u64) -> bool {
        self.in_flight.contains_key(&vpn)
    }

    /// Should migrate based on policy
    pub fn should_migrate(
        &self,
        pte: &PageTableEntry,
        access_count: u64,
        current_location: MemoryLocation,
        requested_location: MemoryLocation,
    ) -> bool {
        if current_location == requested_location {
            return false;
        }

        match self.policy {
            MigrationPolicy::OnFirstAccess | MigrationPolicy::FirstTouch => true,
            MigrationPolicy::AccessFrequency { threshold } => access_count >= threshold as u64,
            MigrationPolicy::Manual => false,
            MigrationPolicy::Prefetch => true, // Handled by prefetcher
        }
    }
}

// ============================================================================
// Prefetcher
// ============================================================================

/// UVM prefetcher
#[derive(Debug)]
pub struct UvmPrefetcher {
    /// Access pattern detector
    detector: AccessPatternDetector,
    /// Prefetch distance (pages ahead)
    prefetch_distance: u32,
    /// Maximum prefetch requests
    max_prefetch: usize,
    /// Pending prefetch requests
    pending: VecDeque<u64>,
    /// Statistics
    pub stats: PrefetchStats,
}

/// Prefetch statistics
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    pub prefetches_issued: u64,
    pub prefetches_useful: u64,
    pub prefetches_wasted: u64,
}

impl PrefetchStats {
    pub fn accuracy(&self) -> f64 {
        let total = self.prefetches_useful + self.prefetches_wasted;
        if total == 0 {
            0.0
        } else {
            self.prefetches_useful as f64 / total as f64
        }
    }
}

impl UvmPrefetcher {
    pub fn new(prefetch_distance: u32, max_prefetch: usize) -> Self {
        Self {
            detector: AccessPatternDetector::new(64),
            prefetch_distance,
            max_prefetch,
            pending: VecDeque::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Record access and generate prefetch requests
    pub fn on_access(&mut self, allocation_id: u64, address: u64, page_size: PageSize) -> Vec<u64> {
        self.detector.record_access(allocation_id, address);

        let pattern = self.detector.get_pattern(allocation_id);
        let mut prefetches = Vec::new();

        match pattern {
            AccessPattern::Sequential => {
                // Prefetch next N pages
                let page_bytes = page_size.bytes();
                for i in 1..=self.prefetch_distance {
                    let prefetch_addr = address + (i as u64 * page_bytes);
                    prefetches.push(prefetch_addr);
                }
            }
            AccessPattern::Strided { stride } => {
                // Prefetch along stride
                for i in 1..=self.prefetch_distance {
                    let prefetch_addr = (address as i64 + (i as i64 * stride)) as u64;
                    prefetches.push(prefetch_addr);
                }
            }
            _ => {
                // No prefetching for random/unknown patterns
            }
        }

        // Limit prefetch count
        prefetches.truncate(self.max_prefetch);
        self.stats.prefetches_issued += prefetches.len() as u64;

        prefetches
    }

    /// Mark prefetch as useful (was actually accessed)
    pub fn mark_useful(&mut self, _address: u64) {
        self.stats.prefetches_useful += 1;
    }

    /// Mark prefetch as wasted (evicted before use)
    pub fn mark_wasted(&mut self, _address: u64) {
        self.stats.prefetches_wasted += 1;
    }
}

// ============================================================================
// Unified Memory Manager
// ============================================================================

/// Unified Virtual Memory manager
#[derive(Debug)]
pub struct UnifiedMemoryManager {
    /// Page tables per GPU
    page_tables: HashMap<u32, GpuPageTable>,
    /// Page fault handler
    fault_handler: PageFaultHandler,
    /// Migration engine
    migration_engine: MigrationEngine,
    /// Prefetcher
    prefetcher: UvmPrefetcher,
    /// Default page size
    page_size: PageSize,
    /// Current cycle
    current_cycle: u64,
    /// Next allocation ID
    next_alloc_id: u64,
}

impl UnifiedMemoryManager {
    pub fn new(num_gpus: u32, page_size: PageSize, migration_policy: MigrationPolicy) -> Self {
        let mut page_tables = HashMap::new();
        for gpu_id in 0..num_gpus {
            page_tables.insert(
                gpu_id,
                GpuPageTable::new(page_size, 1024), // 1K TLB entries
            );
        }

        Self {
            page_tables,
            fault_handler: PageFaultHandler::new(100000),
            migration_engine: MigrationEngine::new(
                migration_policy,
                4,    // Max concurrent migrations
                25.0, // 25 GB/s migration bandwidth
                1.4,  // 1.4 GHz clock
            ),
            prefetcher: UvmPrefetcher::new(4, 8),
            page_size,
            current_cycle: 0,
            next_alloc_id: 0,
        }
    }

    /// Allocate unified memory
    pub fn allocate(&mut self, size: u64) -> UvmAllocation {
        let alloc_id = self.next_alloc_id;
        self.next_alloc_id += 1;

        let page_bytes = self.page_size.bytes();
        let num_pages = (size + page_bytes - 1) / page_bytes;

        // Virtual address range (simplified)
        let base_va = alloc_id * 0x100000000; // 4GB per allocation

        UvmAllocation {
            id: alloc_id,
            base_va,
            size,
            num_pages,
            page_size: self.page_size,
        }
    }

    /// Access memory from a GPU
    pub fn access(&mut self, gpu_id: u32, address: u64, is_write: bool) -> Result<u64, PageFault> {
        self.current_cycle += 1;

        // Try to translate
        let page_table = self.page_tables.get_mut(&gpu_id).ok_or(PageFault {
            virtual_address: address,
            vpn: address >> self.page_size.offset_bits(),
            fault_type: PageFaultType::NotPresent,
        })?;

        let result = page_table.translate(address);

        if let Err(fault) = &result {
            self.fault_handler.report_fault(fault.clone());
        } else {
            let vpn = address >> self.page_size.offset_bits();
            page_table.mark_accessed(vpn, self.current_cycle);
            if is_write {
                page_table.mark_dirty(vpn);
            }
        }

        result
    }

    /// Handle pending faults
    pub fn process_faults(&mut self, gpu_id: u32) -> Vec<PageFaultResolution> {
        if let Some(page_table) = self.page_tables.get(&gpu_id) {
            self.fault_handler
                .handle_faults(page_table, self.current_cycle, gpu_id)
        } else {
            Vec::new()
        }
    }

    /// Tick the manager forward
    pub fn tick(&mut self) {
        self.current_cycle += 1;

        // Complete migrations
        let completed = self.migration_engine.tick(self.current_cycle);

        // Complete fault handling
        let _ = self.fault_handler.check_completions(self.current_cycle);
    }

    /// Prefetch hint
    pub fn prefetch_hint(&mut self, gpu_id: u32, address: u64) {
        let alloc_id = address / 0x100000000; // Simplified
        let prefetches = self.prefetcher.on_access(alloc_id, address, self.page_size);

        for prefetch_addr in prefetches {
            let vpn = prefetch_addr >> self.page_size.offset_bits();
            self.migration_engine.request_migration(
                vpn,
                MemoryLocation::Cpu,
                MemoryLocation::Gpu(gpu_id),
                0, // Low priority
                self.current_cycle,
            );
        }
    }
}

/// UVM allocation descriptor
#[derive(Debug, Clone)]
pub struct UvmAllocation {
    /// Allocation ID
    pub id: u64,
    /// Base virtual address
    pub base_va: u64,
    /// Size in bytes
    pub size: u64,
    /// Number of pages
    pub num_pages: u64,
    /// Page size
    pub page_size: PageSize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size() {
        assert_eq!(PageSize::Small4K.bytes(), 4096);
        assert_eq!(PageSize::Large64K.bytes(), 65536);
        assert_eq!(PageSize::Huge2M.bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_page_table_map_translate() {
        let mut pt = GpuPageTable::new(PageSize::Small4K, 64);

        // Map a page
        pt.map_page(100, 200, MemoryLocation::Gpu(0));

        // Translate address in that page
        let va = 100 * 4096 + 123; // VPN 100, offset 123
        let result = pt.translate(va);

        assert!(result.is_ok());
        let pa = result.unwrap();
        assert_eq!(pa, 200 * 4096 + 123);
    }

    #[test]
    fn test_page_fault() {
        let mut pt = GpuPageTable::new(PageSize::Small4K, 64);

        // Access unmapped page
        let result = pt.translate(0x12345);
        assert!(result.is_err());

        let fault = result.unwrap_err();
        assert_eq!(fault.fault_type, PageFaultType::NotPresent);
    }

    #[test]
    fn test_tlb() {
        let mut pt = GpuPageTable::new(PageSize::Small4K, 4); // Small TLB

        // Map some pages
        for i in 0..10 {
            pt.map_page(i, i + 100, MemoryLocation::Gpu(0));
        }

        // Access first few - should build TLB
        for i in 0..4 {
            let _ = pt.translate(i * 4096);
        }

        assert!(pt.stats.tlb_hits > 0 || pt.stats.tlb_misses > 0);
    }

    #[test]
    fn test_access_pattern_detection() {
        let mut history = AccessHistory::new(32);

        // Sequential accesses
        for i in 0..20 {
            history.record(i * 4096);
        }

        let pattern = history.detect_pattern();
        assert_eq!(pattern, AccessPattern::Sequential);
    }

    #[test]
    fn test_strided_pattern() {
        let mut history = AccessHistory::new(32);

        // Strided accesses (stride = 8KB)
        for i in 0..20 {
            history.record(i * 8192);
        }

        let pattern = history.detect_pattern();
        if let AccessPattern::Strided { stride } = pattern {
            assert_eq!(stride, 8192);
        }
    }

    #[test]
    fn test_migration_request() {
        let mut engine = MigrationEngine::new(MigrationPolicy::OnFirstAccess, 4, 25.0, 1.4);

        engine.request_migration(100, MemoryLocation::Cpu, MemoryLocation::Gpu(0), 1, 0);

        let started = engine.start_migrations(PageSize::Small4K, 0);
        assert_eq!(started.len(), 1);
        assert!(engine.is_migrating(100));
    }

    #[test]
    fn test_migration_completion() {
        let mut engine = MigrationEngine::new(
            MigrationPolicy::OnFirstAccess,
            4,
            1000.0, // Very high bandwidth for quick test
            1.0,
        );

        engine.request_migration(100, MemoryLocation::Cpu, MemoryLocation::Gpu(0), 1, 0);

        engine.start_migrations(PageSize::Small4K, 0);

        // Tick until complete
        let mut completed = Vec::new();
        for cycle in 1..10000 {
            let c = engine.tick(cycle);
            completed.extend(c);
            if !completed.is_empty() {
                break;
            }
        }

        assert!(!completed.is_empty());
        assert_eq!(completed[0], 100);
    }

    #[test]
    fn test_prefetcher() {
        let mut prefetcher = UvmPrefetcher::new(4, 8);

        // Simulate sequential accesses
        for i in 0..10 {
            let addr = i * 4096;
            let _ = prefetcher.on_access(0, addr, PageSize::Small4K);
        }

        // Should have issued prefetches
        assert!(prefetcher.stats.prefetches_issued > 0);
    }

    #[test]
    fn test_uvm_manager() {
        let mut manager =
            UnifiedMemoryManager::new(2, PageSize::Small4K, MigrationPolicy::OnFirstAccess);

        // Allocate
        let alloc = manager.allocate(1024 * 1024); // 1MB
        assert!(alloc.num_pages > 0);

        // Access will fault (not mapped yet)
        let result = manager.access(0, alloc.base_va, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_page_table_stats() {
        let mut pt = GpuPageTable::new(PageSize::Small4K, 64);

        // Map and access
        pt.map_page(100, 200, MemoryLocation::Gpu(0));
        let _ = pt.translate(100 * 4096);
        let _ = pt.translate(100 * 4096);

        assert!(pt.stats.tlb_hits + pt.stats.tlb_misses >= 2);
    }

    #[test]
    fn test_fault_handler() {
        let mut handler = PageFaultHandler::new(10000);

        let fault = PageFault {
            virtual_address: 0x1000,
            vpn: 1,
            fault_type: PageFaultType::NotPresent,
        };

        handler.report_fault(fault);
        assert_eq!(handler.stats.total_faults, 1);

        let pt = GpuPageTable::new(PageSize::Small4K, 64);
        let resolutions = handler.handle_faults(&pt, 0, 0);

        assert_eq!(resolutions.len(), 1);
    }

    #[test]
    fn test_migration_policies() {
        let engine = MigrationEngine::new(
            MigrationPolicy::AccessFrequency { threshold: 5 },
            4,
            25.0,
            1.4,
        );

        let pte = PageTableEntry::new(100, 200, PageSize::Small4K);

        // Should not migrate with low access count
        assert!(!engine.should_migrate(&pte, 3, MemoryLocation::Cpu, MemoryLocation::Gpu(0)));

        // Should migrate with high access count
        assert!(engine.should_migrate(&pte, 10, MemoryLocation::Cpu, MemoryLocation::Gpu(0)));
    }

    #[test]
    fn test_uvm_allocation() {
        let mut manager =
            UnifiedMemoryManager::new(1, PageSize::Large64K, MigrationPolicy::FirstTouch);

        let alloc1 = manager.allocate(1024 * 1024);
        let alloc2 = manager.allocate(2 * 1024 * 1024);

        // Allocations should have different IDs and base addresses
        assert_ne!(alloc1.id, alloc2.id);
        assert_ne!(alloc1.base_va, alloc2.base_va);
    }
}
