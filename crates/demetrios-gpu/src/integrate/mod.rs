//! Integration Module
//!
//! Integrates GPU operations with Demetrios type system features:
//! - Linear types for resource safety
//! - Algebraic effects for operation tracking
//! - Units of measure for dimensional analysis
//! - Epistemic types for uncertainty quantification

pub mod effects;
pub mod epistemic;
pub mod linear;
pub mod units;

pub use effects::{
    AtomicEffect, EffectBoundedKernel, EffectInference, EffectMarker, EffectParam,
    EffectPolymorphic, EffectTracker, Effectful, FullGpuEffect, GlobalRWEffect, GlobalReadEffect,
    GlobalWriteEffect, GpuEffectHandler, Pure, RWParam, ReadParam, SharedMemEffect, SyncEffect,
    WriteParam,
};

pub use epistemic::{
    propagation, Confidence, Epistemic, EpistemicBuffer, EpistemicStats, KnowledgeSource,
};

pub use linear::{
    AffineGpuBuffer, LinearBufferPool, LinearChoice, LinearGpuBuffer, LinearPair, Linearity,
    ResourceGuard,
};

pub use units::{
    AccelerationBuffer, Dimensionless, EnergyBuffer, ForceBuffer, Joule, Kilogram, KilogramBuffer,
    Meter, MeterBuffer, MeterPerSecond, MeterPerSecondSquared, Newton, Quantity, Second,
    SecondBuffer, Unit, UnitBuffer, UnitResult, VelocityBuffer,
};
