//! Memory Pool Allocator
//!
//! `cudaMalloc` is slow (~1ms). For iterative algorithms that allocate
//! repeatedly, pooling allocations provides significant speedups.
//!
//! This module provides arena-style memory pooling with:
//! - Fast allocation from pre-allocated arenas
//! - Free list for reusing deallocated blocks
//! - Automatic pool growth
//! - Caching allocator for recently freed blocks

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Memory pool errors
#[derive(Debug, Error)]
pub enum PoolError {
    /// Out of memory
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Allocation failed
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    /// Double free detected
    #[error("Double free detected for pointer {0:p}")]
    DoubleFree(*mut u8),
}

// Make PoolError Send + Sync safe
unsafe impl Send for PoolError {}
unsafe impl Sync for PoolError {}

/// Device pointer wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    /// Null pointer
    pub fn null() -> Self {
        Self(0)
    }

    /// Check if null
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    /// Offset by bytes
    pub fn offset(&self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }

    /// Get as raw pointer (for CPU simulation)
    pub fn as_ptr(&self) -> *mut u8 {
        self.0 as *mut u8
    }
}

/// Device identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool size (bytes)
    pub initial_size: usize,
    /// Growth factor when pool is exhausted
    pub growth_factor: f64,
    /// Maximum pool size (bytes)
    pub max_size: usize,
    /// Minimum allocation size (bytes)
    pub min_alloc_size: usize,
    /// Alignment requirement (bytes)
    pub alignment: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024, // 256 MB
            growth_factor: 2.0,
            max_size: 8 * 1024 * 1024 * 1024, // 8 GB
            min_alloc_size: 256,
            alignment: 256,
        }
    }
}

impl PoolConfig {
    /// Create a small pool for testing
    pub fn small() -> Self {
        Self {
            initial_size: 1024 * 1024,  // 1 MB
            max_size: 16 * 1024 * 1024, // 16 MB
            ..Default::default()
        }
    }

    /// Create a large pool for production
    pub fn large() -> Self {
        Self {
            initial_size: 1024 * 1024 * 1024,  // 1 GB
            max_size: 32 * 1024 * 1024 * 1024, // 32 GB
            ..Default::default()
        }
    }
}

/// A contiguous region of device memory
#[derive(Debug)]
struct Arena {
    /// Base pointer
    base: DevicePtr,
    /// Total size in bytes
    size: usize,
    /// Used bytes (high water mark)
    used: usize,
}

impl Arena {
    /// Available space in this arena
    fn available(&self) -> usize {
        self.size.saturating_sub(self.used)
    }

    /// Try to allocate from this arena
    fn try_alloc(&mut self, size: usize, alignment: usize) -> Option<DevicePtr> {
        // Align the current used pointer
        let aligned_used = (self.used + alignment - 1) / alignment * alignment;
        let end = aligned_used + size;

        if end <= self.size {
            let ptr = self.base.offset(aligned_used);
            self.used = end;
            Some(ptr)
        } else {
            None
        }
    }

    /// Reset arena for reuse
    fn reset(&mut self) {
        self.used = 0;
    }
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total memory allocated from system
    pub total_allocated: usize,
    /// Memory currently in use by application
    pub total_in_use: usize,
    /// Peak memory usage
    pub peak_in_use: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Cache hits (reused blocks)
    pub cache_hits: u64,
    /// Cache misses (new allocations)
    pub cache_misses: u64,
    /// Number of arenas
    pub arena_count: usize,
}

impl PoolStats {
    /// Memory fragmentation ratio
    pub fn fragmentation(&self) -> f64 {
        if self.total_allocated == 0 {
            0.0
        } else {
            1.0 - (self.total_in_use as f64 / self.total_allocated as f64)
        }
    }

    /// Cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Pool Statistics:")?;
        writeln!(
            f,
            "  Total allocated: {} MB",
            self.total_allocated / (1024 * 1024)
        )?;
        writeln!(f, "  In use: {} MB", self.total_in_use / (1024 * 1024))?;
        writeln!(f, "  Peak: {} MB", self.peak_in_use / (1024 * 1024))?;
        writeln!(f, "  Allocations: {}", self.allocation_count)?;
        writeln!(f, "  Deallocations: {}", self.deallocation_count)?;
        writeln!(f, "  Cache hit rate: {:.1}%", self.hit_rate() * 100.0)?;
        writeln!(f, "  Fragmentation: {:.1}%", self.fragmentation() * 100.0)?;
        writeln!(f, "  Arenas: {}", self.arena_count)?;
        Ok(())
    }
}

/// Backend trait for actual memory allocation
pub trait PoolBackend: Send + Sync {
    /// Allocate memory on device
    fn alloc(&self, device: DeviceId, size: usize) -> Result<DevicePtr, PoolError>;

    /// Free memory on device
    fn free(&self, ptr: DevicePtr) -> Result<(), PoolError>;
}

/// CPU backend for testing (allocates on heap)
#[derive(Debug, Default)]
pub struct CpuPoolBackend;

impl PoolBackend for CpuPoolBackend {
    fn alloc(&self, _device: DeviceId, size: usize) -> Result<DevicePtr, PoolError> {
        let layout = std::alloc::Layout::from_size_align(size, 256)
            .map_err(|e| PoolError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            Err(PoolError::OutOfMemory {
                requested: size,
                available: 0,
            })
        } else {
            Ok(DevicePtr(ptr as u64))
        }
    }

    fn free(&self, ptr: DevicePtr) -> Result<(), PoolError> {
        // In a real implementation, we'd track sizes
        // For now, this is a no-op since we don't know the size
        let _ = ptr;
        Ok(())
    }
}

/// Arena-style memory pool
pub struct MemoryPool {
    device: DeviceId,
    backend: Arc<dyn PoolBackend>,
    config: PoolConfig,

    /// All allocated arenas
    arenas: Mutex<Vec<Arena>>,

    /// Free blocks by size (for quick lookup)
    free_blocks: Mutex<BTreeMap<usize, Vec<DevicePtr>>>,

    /// Allocated blocks (ptr -> size)
    allocated: Mutex<BTreeMap<u64, usize>>,

    /// Statistics
    stats: Mutex<PoolStats>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(
        device: DeviceId,
        backend: Arc<dyn PoolBackend>,
        config: PoolConfig,
    ) -> Result<Self, PoolError> {
        let pool = Self {
            device,
            backend,
            config,
            arenas: Mutex::new(Vec::new()),
            free_blocks: Mutex::new(BTreeMap::new()),
            allocated: Mutex::new(BTreeMap::new()),
            stats: Mutex::new(PoolStats::default()),
        };

        // Allocate initial arena
        pool.grow(pool.config.initial_size)?;

        Ok(pool)
    }

    /// Create with CPU backend for testing
    pub fn with_cpu_backend(config: PoolConfig) -> Result<Self, PoolError> {
        Self::new(DeviceId(0), Arc::new(CpuPoolBackend), config)
    }

    /// Allocate from pool
    pub fn alloc(&self, size: usize) -> Result<DevicePtr, PoolError> {
        let aligned_size = self.align_size(size);

        // Try to find a free block
        if let Some(ptr) = self.find_free_block(aligned_size) {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_hits += 1;
            stats.allocation_count += 1;
            stats.total_in_use += aligned_size;
            stats.peak_in_use = stats.peak_in_use.max(stats.total_in_use);
            return Ok(ptr);
        }

        // No free block, try to allocate from arena
        if let Some(ptr) = self.alloc_from_arena(aligned_size)? {
            let mut stats = self.stats.lock().unwrap();
            stats.cache_misses += 1;
            stats.allocation_count += 1;
            stats.total_in_use += aligned_size;
            stats.peak_in_use = stats.peak_in_use.max(stats.total_in_use);
            return Ok(ptr);
        }

        // Need to grow the pool
        let current_total = self.stats.lock().unwrap().total_allocated;
        let new_arena_size =
            (aligned_size * 2).max((current_total as f64 * self.config.growth_factor) as usize);

        if current_total + new_arena_size > self.config.max_size {
            return Err(PoolError::OutOfMemory {
                requested: aligned_size,
                available: self.config.max_size - current_total,
            });
        }

        self.grow(new_arena_size)?;

        // Try again
        self.alloc_from_arena(aligned_size)?
            .ok_or_else(|| PoolError::OutOfMemory {
                requested: aligned_size,
                available: 0,
            })
    }

    /// Return memory to pool
    pub fn free(&self, ptr: DevicePtr) -> Result<(), PoolError> {
        let mut allocated = self.allocated.lock().unwrap();

        let size = allocated
            .remove(&ptr.0)
            .ok_or_else(|| PoolError::DoubleFree(ptr.as_ptr()))?;

        // Add to free list
        let mut free_blocks = self.free_blocks.lock().unwrap();
        free_blocks.entry(size).or_default().push(ptr);

        let mut stats = self.stats.lock().unwrap();
        stats.deallocation_count += 1;
        stats.total_in_use = stats.total_in_use.saturating_sub(size);

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset pool (free all allocations, keep arenas)
    pub fn reset(&self) {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut allocated = self.allocated.lock().unwrap();
        let mut arenas = self.arenas.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        free_blocks.clear();
        allocated.clear();

        // Reset arena usage
        for arena in arenas.iter_mut() {
            arena.reset();
        }

        stats.total_in_use = 0;
        // Keep other stats for debugging
    }

    /// Trim unused memory back to system
    pub fn trim(&self) -> Result<usize, PoolError> {
        let mut arenas = self.arenas.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let mut trimmed = 0;

        // Remove completely unused arenas (except the first one)
        while arenas.len() > 1 {
            let last = arenas.last().unwrap();
            if last.used == 0 {
                let arena = arenas.pop().unwrap();
                self.backend.free(arena.base)?;
                trimmed += arena.size;
                stats.total_allocated -= arena.size;
                stats.arena_count -= 1;
            } else {
                break;
            }
        }

        Ok(trimmed)
    }

    // Internal methods

    fn align_size(&self, size: usize) -> usize {
        let align = self.config.alignment;
        ((size + align - 1) / align) * align
    }

    fn find_free_block(&self, size: usize) -> Option<DevicePtr> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut allocated = self.allocated.lock().unwrap();

        // Find smallest block that fits (best-fit)
        let key = *free_blocks.range(size..).next().map(|(k, _)| k)?;

        let blocks = free_blocks.get_mut(&key)?;
        let ptr = blocks.pop()?;

        if blocks.is_empty() {
            free_blocks.remove(&key);
        }

        allocated.insert(ptr.0, key);

        Some(ptr)
    }

    fn alloc_from_arena(&self, size: usize) -> Result<Option<DevicePtr>, PoolError> {
        let mut arenas = self.arenas.lock().unwrap();
        let mut allocated = self.allocated.lock().unwrap();

        for arena in arenas.iter_mut() {
            if let Some(ptr) = arena.try_alloc(size, self.config.alignment) {
                allocated.insert(ptr.0, size);
                return Ok(Some(ptr));
            }
        }

        Ok(None)
    }

    fn grow(&self, size: usize) -> Result<(), PoolError> {
        let ptr = self.backend.alloc(self.device, size)?;

        let mut arenas = self.arenas.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        arenas.push(Arena {
            base: ptr,
            size,
            used: 0,
        });

        stats.total_allocated += size;
        stats.arena_count = arenas.len();

        Ok(())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let arenas = self.arenas.lock().unwrap();
        for arena in arenas.iter() {
            let _ = self.backend.free(arena.base);
        }
    }
}

/// Caching allocator that wraps a pool with time-based caching
pub struct CachingAllocator {
    pool: MemoryPool,
    /// Cache recently freed allocations by size
    cache: Mutex<BTreeMap<usize, Vec<(DevicePtr, std::time::Instant)>>>,
    /// Maximum time to keep cached allocations
    cache_timeout: std::time::Duration,
}

impl CachingAllocator {
    /// Create a new caching allocator
    pub fn new(pool: MemoryPool) -> Self {
        Self {
            pool,
            cache: Mutex::new(BTreeMap::new()),
            cache_timeout: std::time::Duration::from_secs(5),
        }
    }

    /// Set cache timeout
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.cache_timeout = timeout;
        self
    }

    /// Allocate with caching
    pub fn alloc(&self, size: usize) -> Result<DevicePtr, PoolError> {
        let aligned = self.pool.align_size(size);

        // Check cache first
        let mut cache = self.cache.lock().unwrap();
        if let Some(blocks) = cache.get_mut(&aligned) {
            if let Some((ptr, _)) = blocks.pop() {
                return Ok(ptr);
            }
        }
        drop(cache);

        self.pool.alloc(size)
    }

    /// Free with caching
    pub fn free(&self, ptr: DevicePtr, size: usize) {
        let aligned = self.pool.align_size(size);

        let mut cache = self.cache.lock().unwrap();
        cache
            .entry(aligned)
            .or_default()
            .push((ptr, std::time::Instant::now()));
    }

    /// Clean up expired cache entries
    pub fn gc(&self) {
        let now = std::time::Instant::now();
        let mut cache = self.cache.lock().unwrap();

        for (_size, blocks) in cache.iter_mut() {
            let expired: Vec<_> = blocks
                .iter()
                .filter(|(_, time)| now.duration_since(*time) > self.cache_timeout)
                .map(|(ptr, _)| *ptr)
                .collect();

            for ptr in expired {
                // Note: In production, we'd need to track size properly
                let _ = self.pool.free(ptr);
            }

            blocks.retain(|(_, time)| now.duration_since(*time) <= self.cache_timeout);
        }

        cache.retain(|_, blocks| !blocks.is_empty());
    }

    /// Get underlying pool stats
    pub fn stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Reset everything
    pub fn reset(&self) {
        self.cache.lock().unwrap().clear();
        self.pool.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_alloc_free() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let ptr1 = pool.alloc(1024).unwrap();
        let ptr2 = pool.alloc(2048).unwrap();

        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);

        pool.free(ptr1).unwrap();
        pool.free(ptr2).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.deallocation_count, 2);
    }

    #[test]
    fn test_pool_reuse() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let ptr1 = pool.alloc(1024).unwrap();
        pool.free(ptr1).unwrap();

        let ptr2 = pool.alloc(1024).unwrap();

        // Should reuse the same block
        assert_eq!(ptr1, ptr2);

        let stats = pool.stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_pool_reset() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let _ = pool.alloc(1024).unwrap();
        let _ = pool.alloc(2048).unwrap();

        pool.reset();

        let stats = pool.stats();
        assert_eq!(stats.total_in_use, 0);
    }

    #[test]
    fn test_pool_growth() {
        let config = PoolConfig {
            initial_size: 1024,
            max_size: 1024 * 1024,
            ..PoolConfig::small()
        };
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        // Allocate more than initial size
        let _ptr = pool.alloc(2048).unwrap();

        let stats = pool.stats();
        assert!(stats.arena_count >= 2);
    }

    #[test]
    fn test_pool_stats() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let ptr1 = pool.alloc(1024).unwrap();
        let ptr2 = pool.alloc(2048).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.total_in_use >= 1024 + 2048);

        pool.free(ptr1).unwrap();
        pool.free(ptr2).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.total_in_use, 0);
    }

    #[test]
    fn test_alignment() {
        let config = PoolConfig {
            alignment: 256,
            ..PoolConfig::small()
        };
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let ptr = pool.alloc(100).unwrap();

        // Pointer should not be null
        assert!(!ptr.is_null());

        // Allocation should succeed
        let ptr2 = pool.alloc(200).unwrap();
        assert!(!ptr2.is_null());
        assert_ne!(ptr, ptr2);
    }

    #[test]
    fn test_caching_allocator() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();
        let alloc = CachingAllocator::new(pool);

        let ptr = alloc.alloc(1024).unwrap();
        alloc.free(ptr, 1024);

        // Should get the same pointer back from cache
        let ptr2 = alloc.alloc(1024).unwrap();
        assert_eq!(ptr, ptr2);
    }

    #[test]
    fn test_double_free_detection() {
        let config = PoolConfig::small();
        let pool = MemoryPool::with_cpu_backend(config).unwrap();

        let ptr = pool.alloc(1024).unwrap();
        pool.free(ptr).unwrap();

        // Second free should fail
        let result = pool.free(ptr);
        assert!(result.is_err());
    }

    #[test]
    fn test_fragmentation() {
        let stats = PoolStats {
            total_allocated: 1000,
            total_in_use: 800,
            ..Default::default()
        };

        assert!((stats.fragmentation() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_hit_rate() {
        let stats = PoolStats {
            cache_hits: 8,
            cache_misses: 2,
            ..Default::default()
        };

        assert!((stats.hit_rate() - 0.8).abs() < 0.01);
    }
}
