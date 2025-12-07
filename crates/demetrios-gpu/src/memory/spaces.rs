
//! Memory Space Abstractions
//!
//! Provides type-safe abstractions for GPU memory spaces.

use crate::ir::types::AddressSpace;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Marker trait for memory spaces
pub trait MemorySpace {
    /// Get the address space
    fn address_space() -> AddressSpace;

    /// Check if this space is accessible from the host
    fn is_host_accessible() -> bool;

    /// Check if this space is accessible from the device
    fn is_device_accessible() -> bool;

    /// Get the name of this memory space
    fn name() -> &'static str;
}

/// Global memory space (device memory)
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

impl MemorySpace for Global {
    fn address_space() -> AddressSpace {
        AddressSpace::Global
    }

    fn is_host_accessible() -> bool {
        false
    }

    fn is_device_accessible() -> bool {
        true
    }

    fn name() -> &'static str {
        "global"
    }
}

/// Shared memory space (per-block)
#[derive(Debug, Clone, Copy, Default)]
pub struct Shared;

impl MemorySpace for Shared {
    fn address_space() -> AddressSpace {
        AddressSpace::Shared
    }

    fn is_host_accessible() -> bool {
        false
    }

    fn is_device_accessible() -> bool {
        true
    }

    fn name() -> &'static str {
        "shared"
    }
}

/// Constant memory space (read-only, cached)
#[derive(Debug, Clone, Copy, Default)]
pub struct Constant;

impl MemorySpace for Constant {
    fn address_space() -> AddressSpace {
        AddressSpace::Constant
    }

    fn is_host_accessible() -> bool {
        false
    }

    fn is_device_accessible() -> bool {
        true
    }

    fn name() -> &'static str {
        "constant"
    }
}

/// Local memory space (per-thread)
#[derive(Debug, Clone, Copy, Default)]
pub struct Local;

impl MemorySpace for Local {
    fn address_space() -> AddressSpace {
        AddressSpace::Local
    }

    fn is_host_accessible() -> bool {
        false
    }

    fn is_device_accessible() -> bool {
        true
    }

    fn name() -> &'static str {
        "local"
    }
}

/// Unified memory space (accessible from both host and device)
#[derive(Debug, Clone, Copy, Default)]
pub struct Unified;

impl MemorySpace for Unified {
    fn address_space() -> AddressSpace {
        AddressSpace::Global // Unified uses global space
    }

    fn is_host_accessible() -> bool {
        true
    }

    fn is_device_accessible() -> bool {
        true
    }

    fn name() -> &'static str {
        "unified"
    }
}

/// Pinned memory space (host memory with fast device access)
#[derive(Debug, Clone, Copy, Default)]
pub struct Pinned;

impl MemorySpace for Pinned {
    fn address_space() -> AddressSpace {
        AddressSpace::Global
    }

    fn is_host_accessible() -> bool {
        true
    }

    fn is_device_accessible() -> bool {
        true // Can be accessed via zero-copy
    }

    fn name() -> &'static str {
        "pinned"
    }
}

/// Pointer with memory space information
#[derive(Debug)]
pub struct Ptr<T, S: MemorySpace> {
    ptr: *mut T,
    _space: PhantomData<S>,
}

impl<T, S: MemorySpace> Ptr<T, S> {
    /// Create a new typed pointer
    pub fn new(ptr: *mut T) -> Self {
        Ptr {
            ptr,
            _space: PhantomData,
        }
    }

    /// Create a null pointer
    pub fn null() -> Self {
        Ptr {
            ptr: std::ptr::null_mut(),
            _space: PhantomData,
        }
    }

    /// Check if the pointer is null
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get the mutable raw pointer
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Get the address space
    pub fn address_space(&self) -> AddressSpace {
        S::address_space()
    }

    /// Offset the pointer by count elements
    pub unsafe fn add(self, count: usize) -> Self {
        Ptr::new(self.ptr.add(count))
    }

    /// Offset the pointer backwards by count elements
    pub unsafe fn sub(self, count: usize) -> Self {
        Ptr::new(self.ptr.sub(count))
    }

    /// Read a value from the pointer (unsafe, assumes valid memory)
    pub unsafe fn read(&self) -> T
    where
        T: Copy,
    {
        *self.ptr
    }

    /// Write a value to the pointer (unsafe, assumes valid memory)
    pub unsafe fn write(&self, value: T) {
        *self.ptr = value;
    }
}

impl<T, S: MemorySpace> Clone for Ptr<T, S> {
    fn clone(&self) -> Self {
        Ptr {
            ptr: self.ptr,
            _space: PhantomData,
        }
    }
}

impl<T, S: MemorySpace> Copy for Ptr<T, S> {}

// Safety: Ptr is just a wrapper around a raw pointer
unsafe impl<T: Send, S: MemorySpace> Send for Ptr<T, S> {}
unsafe impl<T: Sync, S: MemorySpace> Sync for Ptr<T, S> {}

/// Type aliases for common pointer types
pub type GlobalPtr<T> = Ptr<T, Global>;
pub type SharedPtr<T> = Ptr<T, Shared>;
pub type ConstPtr<T> = Ptr<T, Constant>;
pub type LocalPtr<T> = Ptr<T, Local>;
pub type UnifiedPtr<T> = Ptr<T, Unified>;
pub type PinnedPtr<T> = Ptr<T, Pinned>;

/// Memory region with space information
#[derive(Debug)]
pub struct MemoryRegion<S: MemorySpace> {
    /// Base pointer
    ptr: *mut u8,
    /// Size in bytes
    size: usize,
    /// Phantom data for space
    _space: PhantomData<S>,
}

impl<S: MemorySpace> MemoryRegion<S> {
    /// Create a new memory region
    pub fn new(ptr: *mut u8, size: usize) -> Self {
        MemoryRegion {
            ptr,
            size,
            _space: PhantomData,
        }
    }

    /// Get the base pointer
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the address space
    pub fn address_space(&self) -> AddressSpace {
        S::address_space()
    }

    /// Check if host accessible
    pub fn is_host_accessible(&self) -> bool {
        S::is_host_accessible()
    }

    /// Check if device accessible
    pub fn is_device_accessible(&self) -> bool {
        S::is_device_accessible()
    }

    /// Get a typed pointer at offset
    pub fn at<T>(&self, offset: usize) -> Option<Ptr<T, S>> {
        if offset + std::mem::size_of::<T>() <= self.size {
            Some(Ptr::new(unsafe { self.ptr.add(offset) as *mut T }))
        } else {
            None
        }
    }

    /// Create a subregion
    pub fn subregion(&self, offset: usize, size: usize) -> Option<Self> {
        if offset + size <= self.size {
            Some(MemoryRegion::new(
                unsafe { self.ptr.add(offset) },
                size,
            ))
        } else {
            None
        }
    }
}

// Safety: MemoryRegion is just pointer + size
unsafe impl<S: MemorySpace> Send for MemoryRegion<S> {}
unsafe impl<S: MemorySpace> Sync for MemoryRegion<S> {}

/// Type aliases for common region types
pub type GlobalRegion = MemoryRegion<Global>;
pub type SharedRegion = MemoryRegion<Shared>;
pub type UnifiedRegion = MemoryRegion<Unified>;

/// Memory layout information
#[derive(Debug, Clone, Copy)]
pub struct MemoryLayout {
    /// Size in bytes
    pub size: usize,
    /// Alignment in bytes
    pub alignment: usize,
}

impl MemoryLayout {
    /// Create a new layout
    pub fn new(size: usize, alignment: usize) -> Self {
        MemoryLayout { size, alignment }
    }

    /// Create layout for a type
    pub fn of<T>() -> Self {
        MemoryLayout {
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
        }
    }

    /// Create layout for an array
    pub fn array<T>(count: usize) -> Self {
        MemoryLayout {
            size: std::mem::size_of::<T>() * count,
            alignment: std::mem::align_of::<T>(),
        }
    }

    /// Calculate padded size for alignment
    pub fn padded_size(&self) -> usize {
        (self.size + self.alignment - 1) / self.alignment * self.alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_spaces() {
        assert_eq!(Global::address_space(), AddressSpace::Global);
        assert_eq!(Shared::address_space(), AddressSpace::Shared);
        assert_eq!(Constant::address_space(), AddressSpace::Constant);

        assert!(!Global::is_host_accessible());
        assert!(Global::is_device_accessible());

        assert!(Unified::is_host_accessible());
        assert!(Unified::is_device_accessible());
    }

    #[test]
    fn test_typed_pointer() {
        let mut value: f32 = 3.14;
        let ptr: GlobalPtr<f32> = Ptr::new(&mut value);

        assert!(!ptr.is_null());
        assert_eq!(ptr.address_space(), AddressSpace::Global);

        unsafe {
            assert_eq!(ptr.read(), 3.14);
            ptr.write(2.71);
            assert_eq!(ptr.read(), 2.71);
        }
    }

    #[test]
    fn test_memory_region() {
        let mut data = [0u8; 64];
        let region: GlobalRegion = MemoryRegion::new(data.as_mut_ptr(), 64);

        assert_eq!(region.size(), 64);
        assert!(!region.is_empty());
        assert!(region.is_device_accessible());
        assert!(!region.is_host_accessible());

        let sub = region.subregion(16, 32).unwrap();
        assert_eq!(sub.size(), 32);
    }

    #[test]
    fn test_memory_layout() {
        let layout = MemoryLayout::of::<f32>();
        assert_eq!(layout.size, 4);
        assert_eq!(layout.alignment, 4);

        let array_layout = MemoryLayout::array::<f32>(100);
        assert_eq!(array_layout.size, 400);
    }
}
