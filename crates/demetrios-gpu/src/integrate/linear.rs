//! Linear Types Integration for GPU Buffers
//!
//! Provides linear ownership semantics for GPU buffers, ensuring resources
//! are properly consumed (downloaded, freed, or transferred).

use crate::runtime::{BufferError, Device, GpuBuffer};
use std::marker::PhantomData;

/// Linearity kind for GPU resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linearity {
    /// Linear: must be used exactly once
    Linear,
    /// Affine: can be used at most once (can be dropped)
    Affine,
    /// Unrestricted: can be used any number of times
    Unrestricted,
}

impl Linearity {
    /// Check if this linearity allows dropping
    pub fn allows_drop(self) -> bool {
        matches!(self, Linearity::Affine | Linearity::Unrestricted)
    }

    /// Check if this linearity allows cloning
    pub fn allows_clone(self) -> bool {
        matches!(self, Linearity::Unrestricted)
    }
}

/// Linear GPU buffer - must be consumed exactly once
#[derive(Debug)]
pub struct LinearGpuBuffer<T: Copy> {
    buffer: Option<GpuBuffer<T>>,
    consumed: bool,
}

impl<T: Copy> LinearGpuBuffer<T> {
    /// Create a new linear buffer
    pub fn new(len: usize, device: &Device) -> Result<Self, BufferError> {
        Ok(LinearGpuBuffer {
            buffer: Some(GpuBuffer::new(len, device)?),
            consumed: false,
        })
    }

    /// Create from slice
    pub fn from_slice(data: &[T], device: &Device) -> Result<Self, BufferError> {
        Ok(LinearGpuBuffer {
            buffer: Some(GpuBuffer::from_slice(data, device)?),
            consumed: false,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload data (borrows)
    pub fn upload(&mut self, data: &[T]) -> Result<(), BufferError> {
        self.buffer
            .as_mut()
            .ok_or(BufferError::AlreadyConsumed)?
            .upload(data)
    }

    /// Download and consume the buffer (linear consumption)
    pub fn download(mut self) -> Result<Vec<T>, BufferError> {
        let buffer = self.buffer.take().ok_or(BufferError::AlreadyConsumed)?;
        self.consumed = true;
        buffer.download()
    }

    /// Transfer ownership to another linear buffer (linear consumption)
    pub fn transfer(mut self, other: &mut LinearGpuBuffer<T>) -> Result<(), BufferError> {
        let src = self.buffer.take().ok_or(BufferError::AlreadyConsumed)?;
        let dst = other.buffer.as_mut().ok_or(BufferError::AlreadyConsumed)?;
        dst.copy_from(&src)?;
        self.consumed = true;
        Ok(())
    }

    /// Explicitly free the buffer (linear consumption)
    pub fn free(mut self) {
        self.buffer.take();
        self.consumed = true;
    }

    /// Borrow for read operations
    pub fn borrow(&self) -> Result<&GpuBuffer<T>, BufferError> {
        self.buffer.as_ref().ok_or(BufferError::AlreadyConsumed)
    }

    /// Borrow for write operations
    pub fn borrow_mut(&mut self) -> Result<&mut GpuBuffer<T>, BufferError> {
        self.buffer.as_mut().ok_or(BufferError::AlreadyConsumed)
    }

    /// Get device pointer (for kernel launch)
    pub fn as_device_ptr(&self) -> Result<*const T, BufferError> {
        Ok(self.borrow()?.as_device_ptr())
    }

    /// Get mutable device pointer (for kernel launch)
    pub fn as_device_ptr_mut(&mut self) -> Result<*mut T, BufferError> {
        Ok(self.borrow_mut()?.as_device_ptr_mut())
    }
}

impl<T: Copy> Drop for LinearGpuBuffer<T> {
    fn drop(&mut self) {
        if !self.consumed && self.buffer.is_some() {
            #[cfg(debug_assertions)]
            eprintln!(
                "Warning: LinearGpuBuffer dropped without being consumed. \
                 Use download(), transfer(), or free() to properly consume."
            );
        }
    }
}

/// Affine GPU buffer - can be used at most once (can be dropped)
#[derive(Debug)]
pub struct AffineGpuBuffer<T: Copy> {
    buffer: Option<GpuBuffer<T>>,
}

impl<T: Copy> AffineGpuBuffer<T> {
    /// Create a new affine buffer
    pub fn new(len: usize, device: &Device) -> Result<Self, BufferError> {
        Ok(AffineGpuBuffer {
            buffer: Some(GpuBuffer::new(len, device)?),
        })
    }

    /// Create from slice
    pub fn from_slice(data: &[T], device: &Device) -> Result<Self, BufferError> {
        Ok(AffineGpuBuffer {
            buffer: Some(GpuBuffer::from_slice(data, device)?),
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Upload data
    pub fn upload(&mut self, data: &[T]) -> Result<(), BufferError> {
        self.buffer
            .as_mut()
            .ok_or(BufferError::AlreadyConsumed)?
            .upload(data)
    }

    /// Download and consume the buffer
    pub fn download(mut self) -> Result<Vec<T>, BufferError> {
        let buffer = self.buffer.take().ok_or(BufferError::AlreadyConsumed)?;
        buffer.download()
    }

    /// Take ownership of the underlying buffer
    pub fn take(mut self) -> Option<GpuBuffer<T>> {
        self.buffer.take()
    }

    /// Borrow for read operations
    pub fn borrow(&self) -> Result<&GpuBuffer<T>, BufferError> {
        self.buffer.as_ref().ok_or(BufferError::AlreadyConsumed)
    }

    /// Borrow for write operations
    pub fn borrow_mut(&mut self) -> Result<&mut GpuBuffer<T>, BufferError> {
        self.buffer.as_mut().ok_or(BufferError::AlreadyConsumed)
    }
}

/// Linear pair of GPU buffers (both must be consumed)
#[derive(Debug)]
pub struct LinearPair<A: Copy, B: Copy> {
    first: LinearGpuBuffer<A>,
    second: LinearGpuBuffer<B>,
}

impl<A: Copy, B: Copy> LinearPair<A, B> {
    /// Create a new linear pair
    pub fn new(first: LinearGpuBuffer<A>, second: LinearGpuBuffer<B>) -> Self {
        LinearPair { first, second }
    }

    /// Split into components (consumes the pair)
    pub fn split(self) -> (LinearGpuBuffer<A>, LinearGpuBuffer<B>) {
        (self.first, self.second)
    }

    /// Borrow both buffers
    pub fn borrow(&self) -> Result<(&GpuBuffer<A>, &GpuBuffer<B>), BufferError> {
        Ok((self.first.borrow()?, self.second.borrow()?))
    }

    /// Borrow both buffers mutably
    pub fn borrow_mut(&mut self) -> Result<(&mut GpuBuffer<A>, &mut GpuBuffer<B>), BufferError> {
        Ok((self.first.borrow_mut()?, self.second.borrow_mut()?))
    }
}

/// Linear choice between two buffer types (exactly one is present)
#[derive(Debug)]
pub enum LinearChoice<A: Copy, B: Copy> {
    Left(LinearGpuBuffer<A>),
    Right(LinearGpuBuffer<B>),
}

impl<A: Copy, B: Copy> LinearChoice<A, B> {
    /// Create a left choice
    pub fn left(buffer: LinearGpuBuffer<A>) -> Self {
        LinearChoice::Left(buffer)
    }

    /// Create a right choice
    pub fn right(buffer: LinearGpuBuffer<B>) -> Self {
        LinearChoice::Right(buffer)
    }

    /// Check if this is the left variant
    pub fn is_left(&self) -> bool {
        matches!(self, LinearChoice::Left(_))
    }

    /// Check if this is the right variant
    pub fn is_right(&self) -> bool {
        matches!(self, LinearChoice::Right(_))
    }

    /// Match on the choice and consume
    pub fn match_choice<T>(
        self,
        on_left: impl FnOnce(LinearGpuBuffer<A>) -> T,
        on_right: impl FnOnce(LinearGpuBuffer<B>) -> T,
    ) -> T {
        match self {
            LinearChoice::Left(buf) => on_left(buf),
            LinearChoice::Right(buf) => on_right(buf),
        }
    }
}

/// Resource guard that ensures cleanup
pub struct ResourceGuard<T: Copy, F: FnOnce(GpuBuffer<T>)> {
    buffer: Option<GpuBuffer<T>>,
    cleanup: Option<F>,
}

impl<T: Copy, F: FnOnce(GpuBuffer<T>)> ResourceGuard<T, F> {
    /// Create a new resource guard
    pub fn new(buffer: GpuBuffer<T>, cleanup: F) -> Self {
        ResourceGuard {
            buffer: Some(buffer),
            cleanup: Some(cleanup),
        }
    }

    /// Borrow the buffer
    pub fn borrow(&self) -> Option<&GpuBuffer<T>> {
        self.buffer.as_ref()
    }

    /// Borrow the buffer mutably
    pub fn borrow_mut(&mut self) -> Option<&mut GpuBuffer<T>> {
        self.buffer.as_mut()
    }

    /// Release the buffer without running cleanup
    pub fn release(mut self) -> Option<GpuBuffer<T>> {
        self.cleanup.take();
        self.buffer.take()
    }
}

impl<T: Copy, F: FnOnce(GpuBuffer<T>)> Drop for ResourceGuard<T, F> {
    fn drop(&mut self) {
        if let (Some(buffer), Some(cleanup)) = (self.buffer.take(), self.cleanup.take()) {
            cleanup(buffer);
        }
    }
}

/// Linear buffer pool for efficient allocation
pub struct LinearBufferPool<T: Copy> {
    device: Device,
    available: Vec<GpuBuffer<T>>,
    in_use: usize,
    element_count: usize,
}

impl<T: Copy> LinearBufferPool<T> {
    /// Create a new buffer pool
    pub fn new(
        device: Device,
        element_count: usize,
        initial_capacity: usize,
    ) -> Result<Self, BufferError> {
        let mut available = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            available.push(GpuBuffer::new(element_count, &device)?);
        }

        Ok(LinearBufferPool {
            device,
            available,
            in_use: 0,
            element_count,
        })
    }

    /// Acquire a buffer from the pool
    pub fn acquire(&mut self) -> Result<LinearGpuBuffer<T>, BufferError> {
        let buffer = if let Some(buf) = self.available.pop() {
            buf
        } else {
            GpuBuffer::new(self.element_count, &self.device)?
        };

        self.in_use += 1;

        Ok(LinearGpuBuffer {
            buffer: Some(buffer),
            consumed: false,
        })
    }

    /// Return a buffer to the pool
    pub fn release(&mut self, mut buffer: LinearGpuBuffer<T>) {
        if let Some(buf) = buffer.buffer.take() {
            self.available.push(buf);
            buffer.consumed = true;
            self.in_use -= 1;
        }
    }

    /// Get the number of buffers in use
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// Get the number of available buffers
    pub fn available(&self) -> usize {
        self.available.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_buffer_download() {
        let device = Device::cpu();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = LinearGpuBuffer::from_slice(&data, &device).unwrap();

        let result = buffer.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_linear_buffer_free() {
        let device = Device::cpu();
        let buffer: LinearGpuBuffer<f32> = LinearGpuBuffer::new(100, &device).unwrap();
        buffer.free(); // Should not warn
    }

    #[test]
    fn test_affine_buffer() {
        let device = Device::cpu();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = AffineGpuBuffer::from_slice(&data, &device).unwrap();

        let result = buffer.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_linear_pair() {
        let device = Device::cpu();
        let a = LinearGpuBuffer::from_slice(&[1.0f32, 2.0], &device).unwrap();
        let b = LinearGpuBuffer::from_slice(&[3.0f64, 4.0], &device).unwrap();

        let pair = LinearPair::new(a, b);
        let (a, b) = pair.split();

        let result_a = a.download().unwrap();
        let result_b = b.download().unwrap();

        assert_eq!(result_a, vec![1.0f32, 2.0]);
        assert_eq!(result_b, vec![3.0f64, 4.0]);
    }

    #[test]
    fn test_linear_choice() {
        let device = Device::cpu();
        let buffer = LinearGpuBuffer::from_slice(&[1.0f32, 2.0], &device).unwrap();

        let choice: LinearChoice<f32, f64> = LinearChoice::left(buffer);

        let result =
            choice.match_choice(|buf| buf.download().unwrap(), |_| panic!("Should be left"));

        assert_eq!(result, vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_buffer_pool() {
        let device = Device::cpu();
        let mut pool: LinearBufferPool<f32> = LinearBufferPool::new(device, 100, 2).unwrap();

        assert_eq!(pool.available(), 2);
        assert_eq!(pool.in_use(), 0);

        let buf1 = pool.acquire().unwrap();
        assert_eq!(pool.in_use(), 1);

        let buf2 = pool.acquire().unwrap();
        assert_eq!(pool.in_use(), 2);
        assert_eq!(pool.available(), 0);

        pool.release(buf1);
        assert_eq!(pool.in_use(), 1);
        assert_eq!(pool.available(), 1);

        buf2.free();
    }
}
