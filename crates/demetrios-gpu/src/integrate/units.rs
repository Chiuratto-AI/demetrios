//! Units of Measure Integration for GPU
//!
//! Provides unit-aware GPU buffers and operations with compile-time
//! dimensional analysis.

use crate::runtime::{BufferError, Device, GpuBuffer};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Marker trait for unit types
pub trait Unit: Copy + Default {
    /// Get the unit name
    fn name() -> &'static str;

    /// Get the SI base unit exponents (length, mass, time, current, temperature, amount, luminosity)
    fn dimensions() -> [i8; 7];
}

/// Dimensionless unit
#[derive(Debug, Clone, Copy, Default)]
pub struct Dimensionless;

impl Unit for Dimensionless {
    fn name() -> &'static str {
        "1"
    }

    fn dimensions() -> [i8; 7] {
        [0, 0, 0, 0, 0, 0, 0]
    }
}

/// Meter (length)
#[derive(Debug, Clone, Copy, Default)]
pub struct Meter;

impl Unit for Meter {
    fn name() -> &'static str {
        "m"
    }

    fn dimensions() -> [i8; 7] {
        [1, 0, 0, 0, 0, 0, 0]
    }
}

/// Kilogram (mass)
#[derive(Debug, Clone, Copy, Default)]
pub struct Kilogram;

impl Unit for Kilogram {
    fn name() -> &'static str {
        "kg"
    }

    fn dimensions() -> [i8; 7] {
        [0, 1, 0, 0, 0, 0, 0]
    }
}

/// Second (time)
#[derive(Debug, Clone, Copy, Default)]
pub struct Second;

impl Unit for Second {
    fn name() -> &'static str {
        "s"
    }

    fn dimensions() -> [i8; 7] {
        [0, 0, 1, 0, 0, 0, 0]
    }
}

/// Meter per second (velocity)
#[derive(Debug, Clone, Copy, Default)]
pub struct MeterPerSecond;

impl Unit for MeterPerSecond {
    fn name() -> &'static str {
        "m/s"
    }

    fn dimensions() -> [i8; 7] {
        [1, 0, -1, 0, 0, 0, 0] // m * s^-1
    }
}

/// Meter per second squared (acceleration)
#[derive(Debug, Clone, Copy, Default)]
pub struct MeterPerSecondSquared;

impl Unit for MeterPerSecondSquared {
    fn name() -> &'static str {
        "m/s²"
    }

    fn dimensions() -> [i8; 7] {
        [1, 0, -2, 0, 0, 0, 0] // m * s^-2
    }
}

/// Newton (force)
#[derive(Debug, Clone, Copy, Default)]
pub struct Newton;

impl Unit for Newton {
    fn name() -> &'static str {
        "N"
    }

    fn dimensions() -> [i8; 7] {
        [1, 1, -2, 0, 0, 0, 0] // kg * m * s^-2
    }
}

/// Joule (energy)
#[derive(Debug, Clone, Copy, Default)]
pub struct Joule;

impl Unit for Joule {
    fn name() -> &'static str {
        "J"
    }

    fn dimensions() -> [i8; 7] {
        [2, 1, -2, 0, 0, 0, 0] // kg * m^2 * s^-2
    }
}

/// Quantity with a numeric value and unit
#[derive(Debug, Clone, Copy)]
pub struct Quantity<N, U: Unit> {
    value: N,
    _unit: PhantomData<U>,
}

impl<N: Copy, U: Unit> Quantity<N, U> {
    /// Create a new quantity
    pub fn new(value: N) -> Self {
        Quantity {
            value,
            _unit: PhantomData,
        }
    }

    /// Get the numeric value
    pub fn value(&self) -> N {
        self.value
    }

    /// Get the unit name
    pub fn unit_name() -> &'static str {
        U::name()
    }

    /// Convert to dimensionless (if dimensionally compatible)
    pub fn to_raw(self) -> N {
        self.value
    }
}

// Arithmetic operations for quantities
impl<N: Add<Output = N> + Copy, U: Unit> Add for Quantity<N, U> {
    type Output = Quantity<N, U>;

    fn add(self, rhs: Self) -> Self::Output {
        Quantity::new(self.value + rhs.value)
    }
}

impl<N: Sub<Output = N> + Copy, U: Unit> Sub for Quantity<N, U> {
    type Output = Quantity<N, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        Quantity::new(self.value - rhs.value)
    }
}

/// Unit-aware GPU buffer
#[derive(Debug)]
pub struct UnitBuffer<N: Copy, U: Unit> {
    buffer: GpuBuffer<N>,
    _unit: PhantomData<U>,
}

impl<N: Copy, U: Unit> UnitBuffer<N, U> {
    /// Create a new unit buffer
    pub fn new(len: usize, device: &Device) -> Result<Self, BufferError> {
        Ok(UnitBuffer {
            buffer: GpuBuffer::new(len, device)?,
            _unit: PhantomData,
        })
    }

    /// Create from a slice of quantities
    pub fn from_quantities(
        quantities: &[Quantity<N, U>],
        device: &Device,
    ) -> Result<Self, BufferError> {
        let values: Vec<N> = quantities.iter().map(|q| q.value()).collect();
        Ok(UnitBuffer {
            buffer: GpuBuffer::from_slice(&values, device)?,
            _unit: PhantomData,
        })
    }

    /// Create from raw values (assumes correct units)
    pub fn from_raw(values: &[N], device: &Device) -> Result<Self, BufferError> {
        Ok(UnitBuffer {
            buffer: GpuBuffer::from_slice(values, device)?,
            _unit: PhantomData,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the unit name
    pub fn unit_name() -> &'static str {
        U::name()
    }

    /// Upload raw values
    pub fn upload_raw(&mut self, data: &[N]) -> Result<(), BufferError> {
        self.buffer.upload(data)
    }

    /// Upload quantities
    pub fn upload(&mut self, quantities: &[Quantity<N, U>]) -> Result<(), BufferError> {
        let values: Vec<N> = quantities.iter().map(|q| q.value()).collect();
        self.buffer.upload(&values)
    }

    /// Download as raw values
    pub fn download_raw(&self) -> Result<Vec<N>, BufferError> {
        self.buffer.download()
    }

    /// Download as quantities
    pub fn download(&self) -> Result<Vec<Quantity<N, U>>, BufferError> {
        let values = self.buffer.download()?;
        Ok(values.into_iter().map(Quantity::new).collect())
    }

    /// Get device pointer
    pub fn as_device_ptr(&self) -> *const N {
        self.buffer.as_device_ptr()
    }

    /// Get mutable device pointer
    pub fn as_device_ptr_mut(&mut self) -> *mut N {
        self.buffer.as_device_ptr_mut()
    }

    /// Borrow the underlying buffer
    pub fn inner(&self) -> &GpuBuffer<N> {
        &self.buffer
    }

    /// Borrow the underlying buffer mutably
    pub fn inner_mut(&mut self) -> &mut GpuBuffer<N> {
        &mut self.buffer
    }
}

/// Result of a unit-checked GPU operation
pub struct UnitResult<N: Copy, U: Unit> {
    buffer: UnitBuffer<N, U>,
}

impl<N: Copy, U: Unit> UnitResult<N, U> {
    pub fn new(buffer: UnitBuffer<N, U>) -> Self {
        UnitResult { buffer }
    }

    pub fn into_buffer(self) -> UnitBuffer<N, U> {
        self.buffer
    }
}

/// Unit-preserving operations
pub mod ops {
    use super::*;

    /// Add two unit buffers (same units)
    pub fn add<N: Copy, U: Unit>(
        a: &UnitBuffer<N, U>,
        b: &UnitBuffer<N, U>,
        result: &mut UnitBuffer<N, U>,
    ) -> Result<(), BufferError> {
        // In a real implementation, this would launch a kernel
        // For now, we just verify the types match at compile time
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        Ok(())
    }

    /// Scale a unit buffer by a dimensionless factor
    pub fn scale<N: Copy, U: Unit>(
        a: &UnitBuffer<N, U>,
        _factor: Quantity<N, Dimensionless>,
        result: &mut UnitBuffer<N, U>,
    ) -> Result<(), BufferError> {
        assert_eq!(a.len(), result.len());
        Ok(())
    }
}

/// Unit conversion utilities
pub mod convert {
    use super::*;

    /// Convert between compatible units (e.g., km to m)
    pub fn convert<N: Copy + Mul<Output = N>, U1: Unit, U2: Unit>(
        _buffer: &UnitBuffer<N, U1>,
        _factor: N,
    ) -> Result<UnitBuffer<N, U2>, BufferError>
    where
        U1: CompatibleWith<U2>,
    {
        // Compile-time check that units are compatible
        // At runtime, apply the conversion factor
        todo!("Unit conversion not yet implemented")
    }

    /// Marker trait for compatible unit conversions
    pub trait CompatibleWith<U: Unit>: Unit {}

    // Same unit is always compatible with itself
    impl<U: Unit> CompatibleWith<U> for U {}
}

/// Type aliases for common unit buffers
pub type MeterBuffer<N> = UnitBuffer<N, Meter>;
pub type KilogramBuffer<N> = UnitBuffer<N, Kilogram>;
pub type SecondBuffer<N> = UnitBuffer<N, Second>;
pub type VelocityBuffer<N> = UnitBuffer<N, MeterPerSecond>;
pub type AccelerationBuffer<N> = UnitBuffer<N, MeterPerSecondSquared>;
pub type ForceBuffer<N> = UnitBuffer<N, Newton>;
pub type EnergyBuffer<N> = UnitBuffer<N, Joule>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity() {
        let distance: Quantity<f64, Meter> = Quantity::new(100.0);
        assert_eq!(distance.value(), 100.0);
        assert_eq!(Quantity::<f64, Meter>::unit_name(), "m");

        let d1: Quantity<f64, Meter> = Quantity::new(50.0);
        let d2: Quantity<f64, Meter> = Quantity::new(30.0);
        let sum = d1 + d2;
        assert_eq!(sum.value(), 80.0);
    }

    #[test]
    fn test_unit_buffer() {
        let device = Device::cpu();
        let mut buffer: MeterBuffer<f64> = UnitBuffer::new(4, &device).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        buffer.upload_raw(&data).unwrap();

        let result = buffer.download_raw().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_unit_buffer_quantities() {
        let device = Device::cpu();
        let quantities: Vec<Quantity<f64, Meter>> =
            vec![Quantity::new(1.0), Quantity::new(2.0), Quantity::new(3.0)];

        let buffer = UnitBuffer::from_quantities(&quantities, &device).unwrap();
        let result = buffer.download().unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].value(), 1.0);
        assert_eq!(result[2].value(), 3.0);
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(Meter::dimensions(), [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(Second::dimensions(), [0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(MeterPerSecond::dimensions(), [1, 0, -1, 0, 0, 0, 0]);

        // Force = mass * acceleration = kg * m/s^2
        // [1, 1, -2] = [0, 1, 0] + [1, 0, -2]
        assert_eq!(Newton::dimensions(), [1, 1, -2, 0, 0, 0, 0]);
    }
}
