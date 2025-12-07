//! GPU Buffer Management
//!
//! Provides typed, safe GPU buffer abstractions with linear ownership semantics.

use super::device::{Device, DeviceError, DeviceType};
use crate::ir::types::{AddressSpace, GpuType, ScalarType};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::{fmt, ops};
use thiserror::Error;

/// Buffer-related errors
#[derive(Debug, Error)]
pub enum BufferError {
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Invalid buffer state: {0}")]
    InvalidState(String),

    #[error("Transfer failed: {0}")]
    TransferFailed(String),

    #[error("Device error: {0}")]
    Device(#[from] DeviceError),

    #[error("Buffer already consumed")]
    AlreadyConsumed,
}

/// Buffer memory location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferLocation {
    /// Host (CPU) memory
    Host,
    /// Device (GPU) memory
    Device,
    /// Unified memory (accessible from both)
    Unified,
    /// Pinned host memory (fast transfers)
    Pinned,
}

impl BufferLocation {
    pub fn is_host_accessible(self) -> bool {
        matches!(
            self,
            BufferLocation::Host | BufferLocation::Unified | BufferLocation::Pinned
        )
    }

    pub fn is_device_accessible(self) -> bool {
        matches!(self, BufferLocation::Device | BufferLocation::Unified)
    }
}

/// Buffer access mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessMode {
    /// Read-only access
    ReadOnly,
    /// Write-only access
    WriteOnly,
    /// Read-write access
    ReadWrite,
}

/// Raw GPU buffer (untyped)
#[derive(Debug)]
pub struct RawBuffer {
    /// Pointer to data
    ptr: Option<NonNull<u8>>,
    /// Size in bytes
    size: usize,
    /// Memory location
    location: BufferLocation,
    /// Device index
    device_index: usize,
    /// Whether this buffer owns the memory
    owned: bool,
}

impl RawBuffer {
    /// Allocate a new buffer on the host
    pub fn alloc_host(size: usize) -> Result<Self, BufferError> {
        if size == 0 {
            return Ok(RawBuffer {
                ptr: None,
                size: 0,
                location: BufferLocation::Host,
                device_index: 0,
                owned: true,
            });
        }

        let layout = Layout::from_size_align(size, 16)
            .map_err(|e| BufferError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr)
            .ok_or_else(|| BufferError::AllocationFailed("Host allocation failed".to_string()))?;

        Ok(RawBuffer {
            ptr: Some(ptr),
            size,
            location: BufferLocation::Host,
            device_index: 0,
            owned: true,
        })
    }

    /// Allocate a new buffer on the device (CPU simulation)
    pub fn alloc_device(size: usize, device_index: usize) -> Result<Self, BufferError> {
        // For CPU simulation, device memory is just host memory
        let mut buf = Self::alloc_host(size)?;
        buf.location = BufferLocation::Device;
        buf.device_index = device_index;
        Ok(buf)
    }

    /// Allocate unified memory
    pub fn alloc_unified(size: usize, device_index: usize) -> Result<Self, BufferError> {
        let mut buf = Self::alloc_host(size)?;
        buf.location = BufferLocation::Unified;
        buf.device_index = device_index;
        Ok(buf)
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the memory location
    pub fn location(&self) -> BufferLocation {
        self.location
    }

    /// Get the device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get raw pointer (if accessible from host)
    pub fn as_ptr(&self) -> Option<*const u8> {
        if self.location.is_host_accessible() {
            self.ptr.map(|p| p.as_ptr() as *const u8)
        } else {
            None
        }
    }

    /// Get mutable raw pointer (if accessible from host)
    pub fn as_mut_ptr(&mut self) -> Option<*mut u8> {
        if self.location.is_host_accessible() {
            self.ptr.map(|p| p.as_ptr())
        } else {
            None
        }
    }

    /// Copy data from host to this buffer
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), BufferError> {
        if data.len() != self.size {
            return Err(BufferError::SizeMismatch {
                expected: self.size,
                actual: data.len(),
            });
        }

        // For CPU simulation, we can always copy
        if let Some(ptr) = self.ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), self.size);
            }
        }

        Ok(())
    }

    /// Copy data from this buffer to host
    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<(), BufferError> {
        if data.len() != self.size {
            return Err(BufferError::SizeMismatch {
                expected: self.size,
                actual: data.len(),
            });
        }

        if let Some(ptr) = self.ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr() as *const u8,
                    data.as_mut_ptr(),
                    self.size,
                );
            }
        }

        Ok(())
    }

    /// Copy from another buffer
    pub fn copy_from(&mut self, src: &RawBuffer) -> Result<(), BufferError> {
        if src.size != self.size {
            return Err(BufferError::SizeMismatch {
                expected: self.size,
                actual: src.size,
            });
        }

        if let (Some(dst_ptr), Some(src_ptr)) = (self.ptr, src.ptr) {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr.as_ptr(), dst_ptr.as_ptr(), self.size);
            }
        }

        Ok(())
    }

    /// Fill buffer with zeros
    pub fn zero(&mut self) -> Result<(), BufferError> {
        if let Some(ptr) = self.ptr {
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, self.size);
            }
        }
        Ok(())
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        if self.owned && self.size > 0 {
            if let Some(ptr) = self.ptr.take() {
                let layout = Layout::from_size_align(self.size, 16).unwrap();
                unsafe {
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }
}

// Safety: RawBuffer manages its own memory and can be sent between threads
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

/// Typed GPU buffer with element type T
pub struct GpuBuffer<T> {
    /// Underlying raw buffer
    raw: RawBuffer,
    /// Number of elements
    len: usize,
    /// Phantom data for type
    _marker: PhantomData<T>,
}

impl<T: Copy> GpuBuffer<T> {
    /// Allocate a new buffer on the device
    pub fn new(len: usize, device: &Device) -> Result<Self, BufferError> {
        let size = len * std::mem::size_of::<T>();
        let raw = RawBuffer::alloc_device(size, device.index())?;

        Ok(GpuBuffer {
            raw,
            len,
            _marker: PhantomData,
        })
    }

    /// Allocate a new buffer and initialize from slice
    pub fn from_slice(data: &[T], device: &Device) -> Result<Self, BufferError> {
        let mut buf = Self::new(data.len(), device)?;
        buf.upload(data)?;
        Ok(buf)
    }

    /// Allocate a zeroed buffer
    pub fn zeroed(len: usize, device: &Device) -> Result<Self, BufferError> {
        let mut buf = Self::new(len, device)?;
        buf.raw.zero()?;
        Ok(buf)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.raw.size()
    }

    /// Get the memory location
    pub fn location(&self) -> BufferLocation {
        self.raw.location()
    }

    /// Upload data from host
    pub fn upload(&mut self, data: &[T]) -> Result<(), BufferError> {
        if data.len() != self.len {
            return Err(BufferError::SizeMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, self.raw.size()) };
        self.raw.copy_from_host(bytes)
    }

    /// Download data to host
    pub fn download(&self) -> Result<Vec<T>, BufferError> {
        let mut result = vec![unsafe { std::mem::zeroed() }; self.len];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, self.raw.size())
        };
        self.raw.copy_to_host(bytes)?;
        Ok(result)
    }

    /// Download data into existing slice
    pub fn download_into(&self, data: &mut [T]) -> Result<(), BufferError> {
        if data.len() != self.len {
            return Err(BufferError::SizeMismatch {
                expected: self.len,
                actual: data.len(),
            });
        }

        let bytes = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, self.raw.size())
        };
        self.raw.copy_to_host(bytes)
    }

    /// Get raw pointer for kernel launch
    pub fn as_device_ptr(&self) -> *const T {
        self.raw
            .ptr
            .map(|p| p.as_ptr() as *const T)
            .unwrap_or(std::ptr::null())
    }

    /// Get mutable raw pointer for kernel launch
    pub fn as_device_ptr_mut(&mut self) -> *mut T {
        self.raw
            .ptr
            .map(|p| p.as_ptr() as *mut T)
            .unwrap_or(std::ptr::null_mut())
    }

    /// Fill with a value
    pub fn fill(&mut self, value: T) -> Result<(), BufferError>
    where
        T: Clone,
    {
        let data = vec![value; self.len];
        self.upload(&data)
    }

    /// Copy from another buffer of the same type
    pub fn copy_from(&mut self, src: &GpuBuffer<T>) -> Result<(), BufferError> {
        self.raw.copy_from(&src.raw)
    }
}

impl<T> fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.len)
            .field("size_bytes", &self.raw.size())
            .field("location", &self.raw.location())
            .finish()
    }
}

/// Linear GPU buffer - must be consumed (downloaded, freed, or transferred)
pub struct LinearBuffer<T> {
    inner: Option<GpuBuffer<T>>,
    consumed: bool,
}

impl<T: Copy> LinearBuffer<T> {
    /// Create a new linear buffer
    pub fn new(len: usize, device: &Device) -> Result<Self, BufferError> {
        Ok(LinearBuffer {
            inner: Some(GpuBuffer::new(len, device)?),
            consumed: false,
        })
    }

    /// Create from slice
    pub fn from_slice(data: &[T], device: &Device) -> Result<Self, BufferError> {
        Ok(LinearBuffer {
            inner: Some(GpuBuffer::from_slice(data, device)?),
            consumed: false,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.inner.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload data (borrows the buffer)
    pub fn upload(&mut self, data: &[T]) -> Result<(), BufferError> {
        self.inner
            .as_mut()
            .ok_or(BufferError::AlreadyConsumed)?
            .upload(data)
    }

    /// Download and consume the buffer (linear consumption)
    pub fn download(mut self) -> Result<Vec<T>, BufferError> {
        let buf = self.inner.take().ok_or(BufferError::AlreadyConsumed)?;
        self.consumed = true;
        buf.download()
    }

    /// Transfer to another buffer (linear consumption)
    pub fn transfer_to(mut self, other: &mut LinearBuffer<T>) -> Result<(), BufferError> {
        let buf = self.inner.take().ok_or(BufferError::AlreadyConsumed)?;
        let other_buf = other.inner.as_mut().ok_or(BufferError::AlreadyConsumed)?;
        other_buf.copy_from(&buf)?;
        self.consumed = true;
        Ok(())
    }

    /// Free the buffer explicitly (linear consumption)
    pub fn free(mut self) {
        self.inner.take();
        self.consumed = true;
    }

    /// Borrow the underlying buffer for read-only operations
    pub fn borrow(&self) -> Result<&GpuBuffer<T>, BufferError> {
        self.inner.as_ref().ok_or(BufferError::AlreadyConsumed)
    }

    /// Borrow the underlying buffer for read-write operations
    pub fn borrow_mut(&mut self) -> Result<&mut GpuBuffer<T>, BufferError> {
        self.inner.as_mut().ok_or(BufferError::AlreadyConsumed)
    }

    /// Get device pointer for kernel launch
    pub fn as_device_ptr(&self) -> Result<*const T, BufferError> {
        Ok(self.borrow()?.as_device_ptr())
    }

    /// Get mutable device pointer for kernel launch
    pub fn as_device_ptr_mut(&mut self) -> Result<*mut T, BufferError> {
        Ok(self.borrow_mut()?.as_device_ptr_mut())
    }
}

impl<T> Drop for LinearBuffer<T> {
    fn drop(&mut self) {
        if !self.consumed && self.inner.is_some() {
            // In debug mode, panic if buffer wasn't consumed
            #[cfg(debug_assertions)]
            {
                eprintln!(
                    "Warning: LinearBuffer dropped without being consumed (downloaded, freed, or transferred)"
                );
            }
        }
    }
}

impl<T> fmt::Debug for LinearBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearBuffer")
            .field("inner", &self.inner.as_ref().map(|_| "..."))
            .field("consumed", &self.consumed)
            .finish()
    }
}

/// Host buffer with pinned memory for fast transfers
pub struct PinnedBuffer<T> {
    data: Vec<T>,
}

impl<T: Copy + Default> PinnedBuffer<T> {
    /// Create a new pinned buffer
    pub fn new(len: usize) -> Self {
        PinnedBuffer {
            data: vec![T::default(); len],
        }
    }

    /// Create from slice
    pub fn from_slice(data: &[T]) -> Self {
        PinnedBuffer {
            data: data.to_vec(),
        }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Upload to device buffer
    pub fn upload_to(&self, dst: &mut GpuBuffer<T>) -> Result<(), BufferError> {
        dst.upload(&self.data)
    }

    /// Download from device buffer
    pub fn download_from(&mut self, src: &GpuBuffer<T>) -> Result<(), BufferError> {
        src.download_into(&mut self.data)
    }
}

impl<T> ops::Index<usize> for PinnedBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> ops::IndexMut<usize> for PinnedBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_buffer_alloc() {
        let buf = RawBuffer::alloc_host(1024).unwrap();
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.location(), BufferLocation::Host);
    }

    #[test]
    fn test_raw_buffer_copy() {
        let mut buf = RawBuffer::alloc_host(4).unwrap();
        let data = [1u8, 2, 3, 4];
        buf.copy_from_host(&data).unwrap();

        let mut result = [0u8; 4];
        buf.copy_to_host(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_gpu_buffer() {
        let device = Device::cpu();
        let mut buf: GpuBuffer<f32> = GpuBuffer::new(100, &device).unwrap();
        assert_eq!(buf.len(), 100);

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        buf.upload(&data).unwrap();

        let result = buf.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_linear_buffer() {
        let device = Device::cpu();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let buf = LinearBuffer::from_slice(&data, &device).unwrap();

        let result = buf.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_pinned_buffer() {
        let mut pinned = PinnedBuffer::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(pinned.len(), 4);
        assert_eq!(pinned[0], 1.0);

        pinned[0] = 10.0;
        assert_eq!(pinned[0], 10.0);
    }
}
