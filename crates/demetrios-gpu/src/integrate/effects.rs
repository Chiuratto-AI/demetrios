//! Effect System Integration for GPU Operations
//!
//! Maps GPU operations to Demetrios algebraic effects for safety tracking.

use crate::ir::effects::{EffectSet, GpuEffect};
use std::collections::HashSet;
use std::marker::PhantomData;

/// Effect handler for GPU operations
pub trait GpuEffectHandler {
    /// Handle global memory read
    fn on_global_read(&mut self) {}

    /// Handle global memory write
    fn on_global_write(&mut self) {}

    /// Handle shared memory read
    fn on_shared_read(&mut self) {}

    /// Handle shared memory write
    fn on_shared_write(&mut self) {}

    /// Handle atomic operation
    fn on_atomic(&mut self) {}

    /// Handle synchronization
    fn on_sync(&mut self) {}
}

/// Effect tracker that records all effects
#[derive(Debug, Clone, Default)]
pub struct EffectTracker {
    effects: EffectSet,
    operation_count: usize,
}

impl EffectTracker {
    pub fn new() -> Self {
        EffectTracker {
            effects: EffectSet::new(),
            operation_count: 0,
        }
    }

    pub fn effects(&self) -> &EffectSet {
        &self.effects
    }

    pub fn operation_count(&self) -> usize {
        self.operation_count
    }

    pub fn has_mutation(&self) -> bool {
        self.effects.has_mutation()
    }

    pub fn is_pure(&self) -> bool {
        self.effects.is_pure()
    }
}

impl GpuEffectHandler for EffectTracker {
    fn on_global_read(&mut self) {
        self.effects.add(GpuEffect::GlobalRead);
        self.operation_count += 1;
    }

    fn on_global_write(&mut self) {
        self.effects.add(GpuEffect::GlobalWrite);
        self.operation_count += 1;
    }

    fn on_shared_read(&mut self) {
        self.effects.add(GpuEffect::SharedRead);
        self.operation_count += 1;
    }

    fn on_shared_write(&mut self) {
        self.effects.add(GpuEffect::SharedWrite);
        self.operation_count += 1;
    }

    fn on_atomic(&mut self) {
        self.effects.add(GpuEffect::Atomic);
        self.operation_count += 1;
    }

    fn on_sync(&mut self) {
        self.effects.add(GpuEffect::Sync);
        self.operation_count += 1;
    }
}

/// Effectful GPU computation marker
pub struct Effectful<T, E: EffectMarker> {
    value: T,
    _effect: PhantomData<E>,
}

impl<T, E: EffectMarker> Effectful<T, E> {
    /// Create a new effectful computation
    pub fn new(value: T) -> Self {
        Effectful {
            value,
            _effect: PhantomData,
        }
    }

    /// Get the inner value
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Map over the value
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Effectful<U, E> {
        Effectful::new(f(self.value))
    }
}

/// Marker trait for effect types
pub trait EffectMarker: Default {}

/// Pure computation marker (no effects)
#[derive(Debug, Clone, Copy, Default)]
pub struct Pure;
impl EffectMarker for Pure {}

/// Global memory read effect
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalReadEffect;
impl EffectMarker for GlobalReadEffect {}

/// Global memory write effect
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalWriteEffect;
impl EffectMarker for GlobalWriteEffect {}

/// Global memory read-write effect
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalRWEffect;
impl EffectMarker for GlobalRWEffect {}

/// Shared memory effect
#[derive(Debug, Clone, Copy, Default)]
pub struct SharedMemEffect;
impl EffectMarker for SharedMemEffect {}

/// Atomic operation effect
#[derive(Debug, Clone, Copy, Default)]
pub struct AtomicEffect;
impl EffectMarker for AtomicEffect {}

/// Synchronization effect
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncEffect;
impl EffectMarker for SyncEffect {}

/// Full GPU effects (all effects enabled)
#[derive(Debug, Clone, Copy, Default)]
pub struct FullGpuEffect;
impl EffectMarker for FullGpuEffect {}

/// Effect-checked kernel parameter
#[derive(Debug)]
pub struct EffectParam<T, E: EffectMarker> {
    ptr: *mut T,
    len: usize,
    _effect: PhantomData<E>,
}

impl<T, E: EffectMarker> EffectParam<T, E> {
    /// Create a new effect parameter
    pub fn new(ptr: *mut T, len: usize) -> Self {
        EffectParam {
            ptr,
            len,
            _effect: PhantomData,
        }
    }

    /// Get the pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get the mutable pointer
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Read-only kernel parameter
pub type ReadParam<T> = EffectParam<T, GlobalReadEffect>;

/// Write-only kernel parameter
pub type WriteParam<T> = EffectParam<T, GlobalWriteEffect>;

/// Read-write kernel parameter
pub type RWParam<T> = EffectParam<T, GlobalRWEffect>;

/// Effect-bounded kernel
pub struct EffectBoundedKernel<E: EffectMarker> {
    name: String,
    declared_effects: EffectSet,
    _effect: PhantomData<E>,
}

impl<E: EffectMarker> EffectBoundedKernel<E> {
    /// Create a new effect-bounded kernel
    pub fn new(name: impl Into<String>, effects: EffectSet) -> Self {
        EffectBoundedKernel {
            name: name.into(),
            declared_effects: effects,
            _effect: PhantomData,
        }
    }

    /// Get kernel name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get declared effects
    pub fn declared_effects(&self) -> &EffectSet {
        &self.declared_effects
    }

    /// Check if an effect is declared
    pub fn has_effect(&self, effect: GpuEffect) -> bool {
        self.declared_effects.has(effect)
    }
}

/// Effect inference for kernel operations
#[derive(Debug, Default)]
pub struct EffectInference {
    inferred: EffectSet,
}

impl EffectInference {
    pub fn new() -> Self {
        EffectInference {
            inferred: EffectSet::new(),
        }
    }

    /// Infer effect from a read operation
    pub fn infer_read(&mut self, is_shared: bool) {
        if is_shared {
            self.inferred.add(GpuEffect::SharedRead);
        } else {
            self.inferred.add(GpuEffect::GlobalRead);
        }
    }

    /// Infer effect from a write operation
    pub fn infer_write(&mut self, is_shared: bool) {
        if is_shared {
            self.inferred.add(GpuEffect::SharedWrite);
        } else {
            self.inferred.add(GpuEffect::GlobalWrite);
        }
    }

    /// Infer effect from an atomic operation
    pub fn infer_atomic(&mut self) {
        self.inferred.add(GpuEffect::Atomic);
        self.inferred.add(GpuEffect::GlobalRead);
    }

    /// Infer effect from a barrier
    pub fn infer_barrier(&mut self) {
        self.inferred.add(GpuEffect::Sync);
    }

    /// Get the inferred effects
    pub fn effects(&self) -> &EffectSet {
        &self.inferred
    }

    /// Check if inferred effects satisfy declared effects
    pub fn satisfies(&self, declared: &EffectSet) -> bool {
        self.inferred.is_subset(declared)
    }

    /// Get missing effects (inferred but not declared)
    pub fn missing(&self, declared: &EffectSet) -> EffectSet {
        let mut missing = EffectSet::new();
        for effect in self.inferred.iter() {
            if !declared.has(effect) {
                missing.add(effect);
            }
        }
        missing
    }
}

/// Effect polymorphism support
#[derive(Debug, Clone)]
pub struct EffectPolymorphic<E> {
    /// Effect variable
    effect_var: String,
    /// Bound constraints
    bounds: Vec<GpuEffect>,
    _phantom: PhantomData<E>,
}

impl<E> EffectPolymorphic<E> {
    /// Create a new effect-polymorphic type
    pub fn new(var: impl Into<String>) -> Self {
        EffectPolymorphic {
            effect_var: var.into(),
            bounds: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add an effect bound
    pub fn with_bound(mut self, effect: GpuEffect) -> Self {
        self.bounds.push(effect);
        self
    }

    /// Get the effect variable name
    pub fn var(&self) -> &str {
        &self.effect_var
    }

    /// Get the bounds
    pub fn bounds(&self) -> &[GpuEffect] {
        &self.bounds
    }

    /// Check if an effect satisfies bounds
    pub fn satisfies_bounds(&self, effects: &EffectSet) -> bool {
        self.bounds.iter().all(|b| effects.has(*b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_tracker() {
        let mut tracker = EffectTracker::new();
        assert!(tracker.is_pure());

        tracker.on_global_read();
        assert!(!tracker.is_pure());
        assert!(!tracker.has_mutation());

        tracker.on_global_write();
        assert!(tracker.has_mutation());
        assert_eq!(tracker.operation_count(), 2);
    }

    #[test]
    fn test_effectful() {
        let value: Effectful<i32, GlobalReadEffect> = Effectful::new(42);
        let mapped = value.map(|x| x * 2);
        assert_eq!(mapped.into_inner(), 84);
    }

    #[test]
    fn test_effect_inference() {
        let mut inference = EffectInference::new();
        inference.infer_read(false);
        inference.infer_write(false);

        let declared = crate::ir::effects::common::global_rw();
        assert!(inference.satisfies(&declared));

        inference.infer_atomic();
        assert!(!inference.satisfies(&declared));

        let missing = inference.missing(&declared);
        assert!(missing.has(GpuEffect::Atomic));
    }

    #[test]
    fn test_effect_bounded_kernel() {
        let effects = crate::ir::effects::common::global_rw();
        let kernel: EffectBoundedKernel<GlobalRWEffect> =
            EffectBoundedKernel::new("vector_add", effects);

        assert_eq!(kernel.name(), "vector_add");
        assert!(kernel.has_effect(GpuEffect::GlobalRead));
        assert!(kernel.has_effect(GpuEffect::GlobalWrite));
        assert!(!kernel.has_effect(GpuEffect::Atomic));
    }
}
