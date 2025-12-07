//! Memory Management
//!
//! Provides memory space abstractions and access pattern optimization.

pub mod access;
pub mod spaces;

pub use access::{
    AccessMode, AccessPattern, BankConflictAnalyzer, BankConflictResult, CacheHint,
    CoalescedAccessor, CoalescingAnalyzer, CoalescingResult, MemoryAccess, OptimizationHints,
};
pub use spaces::{
    ConstPtr, Constant, Global, GlobalPtr, GlobalRegion, Local, LocalPtr, MemoryLayout,
    MemoryRegion, MemorySpace, Pinned, PinnedPtr, Ptr, Shared, SharedPtr, SharedRegion, Unified,
    UnifiedPtr, UnifiedRegion,
};
