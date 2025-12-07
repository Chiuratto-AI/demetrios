//! Physical Quantity Types with Semantic Substrate Awareness
//!
//! This module provides types that encode not just numerical values but
//! their **physical meaning**. The compiler understands that a `Temperature`
//! is not just a float — it has thermodynamic semantics.
//!
//! # Novel Aspects
//!
//! 1. **Semantic Typing**: Types carry physical meaning, not just dimensions
//! 2. **Constraint Encoding**: Physical constraints are part of the type
//! 3. **Substrate Awareness**: Compiler knows how quantities relate physically
//!
//! # Example
//!
//! ```ignore
//! // Not just f64 — carries thermodynamic semantics
//! let T: Temperature<Kelvin> = 300.0.kelvin();
//!
//! // Compiler knows: T must be positive (absolute temperature)
//! // Compiler knows: dS/dT > 0 for stable systems
//! // Compiler knows: T → 0 requires special handling (third law)
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// DIMENSIONAL ANALYSIS
// ============================================================================

/// SI base dimensions encoded at the type level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dimensions {
    /// Length (meters)
    pub length: i8,
    /// Mass (kilograms)
    pub mass: i8,
    /// Time (seconds)
    pub time: i8,
    /// Electric current (amperes)
    pub current: i8,
    /// Temperature (kelvin)
    pub temperature: i8,
    /// Amount of substance (moles)
    pub amount: i8,
    /// Luminous intensity (candela)
    pub luminosity: i8,
}

impl Dimensions {
    pub const DIMENSIONLESS: Self = Self {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    pub const LENGTH: Self = Self {
        length: 1,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    pub const MASS: Self = Self {
        length: 0,
        mass: 1,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    pub const TIME: Self = Self {
        length: 0,
        mass: 0,
        time: 1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    pub const TEMPERATURE: Self = Self {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 1,
        amount: 0,
        luminosity: 0,
    };

    pub const AMOUNT: Self = Self {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 1,
        luminosity: 0,
    };

    /// Energy: kg⋅m²/s²
    pub const ENERGY: Self = Self {
        length: 2,
        mass: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    /// Force: kg⋅m/s²
    pub const FORCE: Self = Self {
        length: 1,
        mass: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    /// Pressure: kg/(m⋅s²)
    pub const PRESSURE: Self = Self {
        length: -1,
        mass: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    /// Velocity: m/s
    pub const VELOCITY: Self = Self {
        length: 1,
        mass: 0,
        time: -1,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    /// Acceleration: m/s²
    pub const ACCELERATION: Self = Self {
        length: 1,
        mass: 0,
        time: -2,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
    };

    /// Entropy: J/K = kg⋅m²/(s²⋅K)
    pub const ENTROPY: Self = Self {
        length: 2,
        mass: 1,
        time: -2,
        current: 0,
        temperature: -1,
        amount: 0,
        luminosity: 0,
    };

    /// Chemical potential: J/mol
    pub const CHEMICAL_POTENTIAL: Self = Self {
        length: 2,
        mass: 1,
        time: -2,
        current: 0,
        temperature: 0,
        amount: -1,
        luminosity: 0,
    };

    /// Multiply dimensions (for quantity multiplication)
    pub const fn mul(self, other: Self) -> Self {
        Self {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    /// Divide dimensions
    pub const fn div(self, other: Self) -> Self {
        Self {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    /// Check if dimensions match
    pub const fn matches(&self, other: &Self) -> bool {
        self.length == other.length
            && self.mass == other.mass
            && self.time == other.time
            && self.current == other.current
            && self.temperature == other.temperature
            && self.amount == other.amount
            && self.luminosity == other.luminosity
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();

        if self.length != 0 {
            if self.length == 1 {
                parts.push("m".to_string());
            } else {
                parts.push(format!("m^{}", self.length));
            }
        }
        if self.mass != 0 {
            if self.mass == 1 {
                parts.push("kg".to_string());
            } else {
                parts.push(format!("kg^{}", self.mass));
            }
        }
        if self.time != 0 {
            if self.time == 1 {
                parts.push("s".to_string());
            } else {
                parts.push(format!("s^{}", self.time));
            }
        }
        if self.temperature != 0 {
            if self.temperature == 1 {
                parts.push("K".to_string());
            } else {
                parts.push(format!("K^{}", self.temperature));
            }
        }
        if self.amount != 0 {
            if self.amount == 1 {
                parts.push("mol".to_string());
            } else {
                parts.push(format!("mol^{}", self.amount));
            }
}

        if parts.is_empty() {
            write!(f, "1")
        } else {
            write!(f, "{}", parts.join("⋅"))
        }
    }
}

// ============================================================================
// QUANTITY KIND - SEMANTIC MEANING BEYOND DIMENSIONS
// ============================================================================

/// The semantic kind of a physical quantity
///
/// Two quantities can have the same dimensions but different physical meanings.
/// For example, torque and energy both have dimensions of kg⋅m²/s², but they
/// are fundamentally different physical concepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantityKind {
    // Fundamental
    Length,
    Mass,
    Time,
    Temperature,
    AmountOfSubstance,
    ElectricCurrent,

    // Mechanical
    Velocity,
    Acceleration,
    Force,
    Momentum,
    AngularMomentum,
    Energy,
    Work,
    Torque, // Same dimensions as Energy, different meaning!
    Power,
    Pressure,
    Stress, // Same dimensions as Pressure, different meaning!

    // Thermodynamic
    Entropy,
    Enthalpy,
    FreeEnergy,
    ChemicalPotential,
    HeatCapacity,

    // Electromagnetic
    Charge,
    ElectricField,
    MagneticField,
    ElectricPotential,

    // Quantum
    WaveFunction,
    Probability,
    Action, // Planck's constant dimensions

    // Statistical
    PartitionFunction,
    Degeneracy,

    // Field Theory
    ScalarField,
    VectorField,
    TensorField,
    SpinorField,

    // Custom/Domain-specific
    Concentration,
    ReactionRate,
    Diffusivity,
    Viscosity,
    SurfaceTension,

    /// Generic quantity (fallback)
    Generic,
}

impl QuantityKind {
    /// Get the expected dimensions for this quantity kind
    pub fn expected_dimensions(&self) -> Option<Dimensions> {
        match self {
            Self::Length => Some(Dimensions::LENGTH),
            Self::Mass => Some(Dimensions::MASS),
            Self::Time => Some(Dimensions::TIME),
            Self::Temperature => Some(Dimensions::TEMPERATURE),
            Self::AmountOfSubstance => Some(Dimensions::AMOUNT),
            Self::Velocity => Some(Dimensions::VELOCITY),
            Self::Acceleration => Some(Dimensions::ACCELERATION),
            Self::Force => Some(Dimensions::FORCE),
            Self::Energy => Some(Dimensions::ENERGY),
            Self::Work => Some(Dimensions::ENERGY),
            Self::Torque => Some(Dimensions::ENERGY), // Same as energy!
            Self::Pressure => Some(Dimensions::PRESSURE),
            Self::Stress => Some(Dimensions::PRESSURE), // Same as pressure!
            Self::Entropy => Some(Dimensions::ENTROPY),
            Self::ChemicalPotential => Some(Dimensions::CHEMICAL_POTENTIAL),
            Self::Probability => Some(Dimensions::DIMENSIONLESS),
            _ => None, // Complex or context-dependent
        }
    }

    /// Check if this kind is extensive (scales with system size)
    pub fn is_extensive(&self) -> bool {
        matches!(
            self,
            Self::Mass
                | Self::Energy
                | Self::Entropy
                | Self::Momentum
                | Self::AngularMomentum
                | Self::Charge
                | Self::AmountOfSubstance
        )
    }

    /// Check if this kind is intensive (independent of system size)
    pub fn is_intensive(&self) -> bool {
        matches!(
            self,
            Self::Temperature | Self::Pressure | Self::ChemicalPotential | Self::Concentration
        )
    }

    /// Check if this kind must be positive
    pub fn must_be_positive(&self) -> bool {
        matches!(
            self,
            Self::Temperature
                | Self::Mass
                | Self::Pressure
                | Self::Probability
                | Self::Concentration
        )
    }

    /// Check if this kind is bounded
    pub fn bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Probability => Some((0.0, 1.0)),
            Self::Temperature => Some((0.0, f64::INFINITY)), // Absolute zero
            _ => None,
        }
    }
}

// ============================================================================
// PHYSICAL CONSTRAINTS
// ============================================================================

/// Constraints that apply to physical quantities
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalConstraint {
    /// Value must be positive
    Positive,
    /// Value must be non-negative
    NonNegative,
    /// Value must be bounded
    Bounded { min: f64, max: f64 },
    /// Value must be normalized (e.g., probability, wavefunction)
    Normalized,
    /// Value must satisfy a conservation law
    Conserved(ConservedQuantityType),
    /// Value must be symmetric under some transformation
    Symmetric(SymmetryType),
    /// Value must satisfy thermodynamic stability
    ThermodynamicallyStable,
    /// Value must satisfy a custom predicate
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservedQuantityType {
    Mass,
    Charge,
    Energy,
    Momentum,
    AngularMomentum,
    Probability,
    ParticleNumber,
    BaryonNumber,
    LeptonNumber,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymmetryType {
    Translational,
    Rotational,
    TimeReversal,
    ParityInversion,
    GaugeU1,
    GaugeSU2,
    GaugeSU3,
    Lorentz,
    Conformal,
}

// ============================================================================
// PHYSICAL QUANTITY - THE CORE TYPE
// ============================================================================

/// A physical quantity with dimensional analysis and semantic meaning
///
/// This is the foundation of substrate-aware computing. A PhysicalQuantity
/// carries not just a value, but its physical meaning, constraints, and
/// how it should behave under transformations.
#[derive(Debug, Clone)]
pub struct PhysicalQuantity<T = f64> {
    /// The numerical value
    pub value: T,
    /// SI dimensions
    pub dimensions: Dimensions,
    /// Semantic meaning
    pub kind: QuantityKind,
    /// Physical constraints
    pub constraints: Vec<PhysicalConstraint>,
    /// Uncertainty (if tracked)
    pub uncertainty: Option<T>,
}

impl<T: Copy> PhysicalQuantity<T> {
    /// Create a new physical quantity
    pub fn new(value: T, dimensions: Dimensions, kind: QuantityKind) -> Self {
        Self {
            value,
            dimensions,
            kind,
            constraints: Vec::new(),
            uncertainty: None,
        }
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: PhysicalConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add uncertainty
    pub fn with_uncertainty(mut self, uncertainty: T) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }
}

impl PhysicalQuantity<f64> {
    /// Check if all constraints are satisfied
    pub fn validate(&self) -> Result<(), ConstraintViolation> {
        for constraint in &self.constraints {
            match constraint {
                PhysicalConstraint::Positive => {
                    if self.value <= 0.0 {
                        return Err(ConstraintViolation::PositivityViolated {
                            value: self.value,
                            kind: self.kind,
                        });
                    }
                }
                PhysicalConstraint::NonNegative => {
                    if self.value < 0.0 {
                        return Err(ConstraintViolation::NonNegativityViolated {
                            value: self.value,
                            kind: self.kind,
                        });
                    }
                }
                PhysicalConstraint::Bounded { min, max } => {
                    if self.value < *min || self.value > *max {
                        return Err(ConstraintViolation::BoundsViolated {
                            value: self.value,
                            min: *min,
                            max: *max,
                            kind: self.kind,
                        });
                    }
                }
                _ => {} // Other constraints require more context
            }
        }
        Ok(())
    }

    /// Create a dimensionless quantity
    pub fn dimensionless(value: f64) -> Self {
        Self::new(value, Dimensions::DIMENSIONLESS, QuantityKind::Generic)
    }

    /// Create a length
    pub fn length(value: f64) -> Self {
        Self::new(value, Dimensions::LENGTH, QuantityKind::Length)
            .with_constraint(PhysicalConstraint::Positive)
    }

    /// Create a mass
    pub fn mass(value: f64) -> Self {
        Self::new(value, Dimensions::MASS, QuantityKind::Mass)
            .with_constraint(PhysicalConstraint::Positive)
    }

    /// Create a temperature
    pub fn temperature(value: f64) -> Self {
        Self::new(value, Dimensions::TEMPERATURE, QuantityKind::Temperature)
            .with_constraint(PhysicalConstraint::Positive)
    }

    /// Create an energy
    pub fn energy(value: f64) -> Self {
        Self::new(value, Dimensions::ENERGY, QuantityKind::Energy)
    }

    /// Create a pressure
    pub fn pressure(value: f64) -> Self {
        Self::new(value, Dimensions::PRESSURE, QuantityKind::Pressure)
            .with_constraint(PhysicalConstraint::NonNegative)
    }

    /// Create a probability
    pub fn probability(value: f64) -> Self {
        Self::new(value, Dimensions::DIMENSIONLESS, QuantityKind::Probability)
            .with_constraint(PhysicalConstraint::Bounded { min: 0.0, max: 1.0 })
            .with_constraint(PhysicalConstraint::Normalized)
    }

    /// Create a chemical potential
    pub fn chemical_potential(value: f64) -> Self {
        Self::new(
            value,
            Dimensions::CHEMICAL_POTENTIAL,
            QuantityKind::ChemicalPotential,
        )
    }

    /// Create entropy
    pub fn entropy(value: f64) -> Self {
        Self::new(value, Dimensions::ENTROPY, QuantityKind::Entropy)
            .with_constraint(PhysicalConstraint::NonNegative)
    }
}

/// Error when a physical constraint is violated
#[derive(Debug, Clone)]
pub enum ConstraintViolation {
    PositivityViolated {
        value: f64,
        kind: QuantityKind,
    },
    NonNegativityViolated {
        value: f64,
        kind: QuantityKind,
    },
    BoundsViolated {
        value: f64,
        min: f64,
        max: f64,
        kind: QuantityKind,
    },
    NormalizationViolated {
        norm: f64,
        expected: f64,
    },
    ConservationViolated {
        quantity: ConservedQuantityType,
        delta: f64,
    },
    DimensionMismatch {
        expected: Dimensions,
        actual: Dimensions,
    },
    ThermodynamicInstability {
        reason: String,
    },
}

impl fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositivityViolated { value, kind } => {
                write!(f, "{:?} must be positive, got {}", kind, value)
            }
            Self::NonNegativityViolated { value, kind } => {
                write!(f, "{:?} must be non-negative, got {}", kind, value)
            }
            Self::BoundsViolated {
                value,
                min,
                max,
                kind,
            } => {
                write!(f, "{:?} must be in [{}, {}], got {}", kind, min, max, value)
            }
            Self::NormalizationViolated { norm, expected } => {
                write!(
                    f,
                    "Normalization violated: got {}, expected {}",
                    norm, expected
                )
            }
            Self::ConservationViolated { quantity, delta } => {
                write!(f, "{:?} conservation violated: delta = {}", quantity, delta)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::ThermodynamicInstability { reason } => {
                write!(f, "Thermodynamic instability: {}", reason)
            }
        }
    }
}

impl std::error::Error for ConstraintViolation {}

// ============================================================================
// ARITHMETIC OPERATIONS WITH DIMENSIONAL CHECKING
// ============================================================================

impl<T: Add<Output = T> + Copy> Add for PhysicalQuantity<T> {
    type Output = Result<Self, ConstraintViolation>;

    fn add(self, other: Self) -> Self::Output {
        if !self.dimensions.matches(&other.dimensions) {
            return Err(ConstraintViolation::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }
        Ok(PhysicalQuantity {
            value: self.value + other.value,
            dimensions: self.dimensions,
            kind: self.kind, // Keep the first quantity's kind
            constraints: self.constraints,
            uncertainty: None, // Would need proper propagation
        })
    }
}

impl<T: Sub<Output = T> + Copy> Sub for PhysicalQuantity<T> {
    type Output = Result<Self, ConstraintViolation>;

    fn sub(self, other: Self) -> Self::Output {
        if !self.dimensions.matches(&other.dimensions) {
            return Err(ConstraintViolation::DimensionMismatch {
                expected: self.dimensions,
                actual: other.dimensions,
            });
        }
        Ok(PhysicalQuantity {
            value: self.value - other.value,
            dimensions: self.dimensions,
            kind: self.kind,
            constraints: self.constraints,
            uncertainty: None,
        })
    }
}

impl<T: Mul<Output = T> + Copy> Mul for PhysicalQuantity<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        PhysicalQuantity {
            value: self.value * other.value,
            dimensions: self.dimensions.mul(other.dimensions),
            kind: QuantityKind::Generic, // Product may not have simple meaning
            constraints: Vec::new(),
            uncertainty: None,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div for PhysicalQuantity<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        PhysicalQuantity {
            value: self.value / other.value,
            dimensions: self.dimensions.div(other.dimensions),
            kind: QuantityKind::Generic,
            constraints: Vec::new(),
            uncertainty: None,
        }
    }
}

// ============================================================================
// PHYSICAL FIELD - SPATIAL DISTRIBUTION OF QUANTITIES
// ============================================================================

/// A physical field: a quantity defined over space
///
/// This represents things like temperature fields, concentration fields,
/// electric fields, etc. The field knows its physical meaning and can
/// enforce constraints over the entire domain.
#[derive(Debug, Clone)]
pub struct PhysicalField<T = f64> {
    /// Field values on a grid or mesh
    pub values: Vec<T>,
    /// The kind of quantity this field represents
    pub kind: QuantityKind,
    /// Dimensions of the quantity
    pub dimensions: Dimensions,
    /// Spatial dimensions (1D, 2D, 3D)
    pub spatial_dims: usize,
    /// Grid shape (for structured grids)
    pub shape: Vec<usize>,
    /// Physical constraints on the field
    pub constraints: Vec<PhysicalConstraint>,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
}

/// Boundary condition types
#[derive(Debug, Clone)]
pub enum BoundaryConditions {
    /// Periodic boundaries
    Periodic,
    /// Dirichlet (fixed value)
    Dirichlet(Vec<f64>),
    /// Neumann (fixed gradient)
    Neumann(Vec<f64>),
    /// Mixed
    Mixed(Vec<BoundaryCondition>),
    /// No explicit boundary (e.g., isolated system)
    None,
}

#[derive(Debug, Clone)]
pub struct BoundaryCondition {
    pub face: usize,
    pub condition: BoundaryConditionType,
}

#[derive(Debug, Clone)]
pub enum BoundaryConditionType {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64, gamma: f64 },
    Absorbing,
    Reflecting,
}

impl<T: Copy + Default> PhysicalField<T> {
    /// Create a new uniform field
    pub fn uniform(
        value: T,
        shape: Vec<usize>,
        kind: QuantityKind,
        dimensions: Dimensions,
    ) -> Self {
        let total_size: usize = shape.iter().product();
        Self {
            values: vec![value; total_size],
            kind,
            dimensions,
            spatial_dims: shape.len(),
            shape,
            constraints: Vec::new(),
            boundary_conditions: BoundaryConditions::None,
        }
    }

    /// Get field value at index
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.values.get(idx)
    }

    /// Set field value at index
    pub fn set(&mut self, idx: usize, value: T) {
        if idx < self.values.len() {
            self.values[idx] = value;
        }
    }

    /// Total number of grid points
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if field is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl PhysicalField<f64> {
    /// Compute the integral over the domain
    pub fn integrate(&self, cell_volume: f64) -> f64 {
        self.values.iter().sum::<f64>() * cell_volume
    }

    /// Compute the maximum value
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Compute the minimum value
    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Validate all constraints
    pub fn validate(&self) -> Result<(), Vec<ConstraintViolation>> {
        let mut violations = Vec::new();

        for constraint in &self.constraints {
            match constraint {
                PhysicalConstraint::Positive => {
                    if self.values.iter().any(|&v| v <= 0.0) {
                        violations.push(ConstraintViolation::PositivityViolated {
                            value: self.min(),
                            kind: self.kind,
                        });
                    }
                }
                PhysicalConstraint::NonNegative => {
                    if self.values.iter().any(|&v| v < 0.0) {
                        violations.push(ConstraintViolation::NonNegativityViolated {
                            value: self.min(),
                            kind: self.kind,
                        });
                    }
                }
                PhysicalConstraint::Bounded { min, max } => {
                    if self.values.iter().any(|&v| v < *min || v > *max) {
                        violations.push(ConstraintViolation::BoundsViolated {
                            value: self
                                .values
                                .iter()
                                .find(|&&v| v < *min || v > *max)
                                .cloned()
                                .unwrap_or(0.0),
                            min: *min,
                            max: *max,
                            kind: self.kind,
                        });
                    }
                }
                _ => {}
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

// ============================================================================
// TENSOR FIELD - FOR STRESS, STRAIN, ETC.
// ============================================================================

/// A tensor field (rank-2 tensor at each point)
#[derive(Debug, Clone)]
pub struct TensorField {
    /// Tensor components at each point (flattened: [xx, xy, xz, yx, yy, yz, zx, zy, zz])
    pub components: Vec<[f64; 9]>,
    /// The kind of tensor (stress, strain, etc.)
    pub kind: TensorKind,
    /// Dimensions of each component
    pub dimensions: Dimensions,
    /// Spatial dimensions (usually 3)
    pub spatial_dims: usize,
    /// Grid shape
    pub shape: Vec<usize>,
    /// Symmetry of the tensor
    pub symmetry: TensorSymmetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    Stress,
    Strain,
    Inertia,
    Metric,
    Curvature,
    ElectromagneticField,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorSymmetry {
    /// No symmetry
    None,
    /// Symmetric: T_ij = T_ji
    Symmetric,
    /// Antisymmetric: T_ij = -T_ji
    Antisymmetric,
    /// Traceless: T_ii = 0
    Traceless,
    /// Symmetric and traceless
    SymmetricTraceless,
}

impl TensorField {
    /// Create a zero tensor field
    pub fn zeros(shape: Vec<usize>, kind: TensorKind, dimensions: Dimensions) -> Self {
        let total_size: usize = shape.iter().product();
        Self {
            components: vec![[0.0; 9]; total_size],
            kind,
            dimensions,
            spatial_dims: shape.len(),
            shape,
            symmetry: TensorSymmetry::None,
        }
    }

    /// Compute trace at a point
    pub fn trace(&self, idx: usize) -> f64 {
        let t = &self.components[idx];
        t[0] + t[4] + t[8] // xx + yy + zz
    }

    /// Check if symmetry constraint is satisfied
    pub fn check_symmetry(&self) -> bool {
        match self.symmetry {
            TensorSymmetry::None => true,
            TensorSymmetry::Symmetric => {
                self.components.iter().all(|t| {
                    (t[1] - t[3]).abs() < 1e-10 && // xy = yx
                    (t[2] - t[6]).abs() < 1e-10 && // xz = zx
                    (t[5] - t[7]).abs() < 1e-10 // yz = zy
                })
            }
            TensorSymmetry::Antisymmetric => {
                self.components.iter().all(|t| {
                    (t[1] + t[3]).abs() < 1e-10 && // xy = -yx
                    (t[2] + t[6]).abs() < 1e-10 && // xz = -zx
                    (t[5] + t[7]).abs() < 1e-10 && // yz = -zy
                    t[0].abs() < 1e-10 && t[4].abs() < 1e-10 && t[8].abs() < 1e-10
                })
            }
            TensorSymmetry::Traceless => self
                .components
                .iter()
                .all(|t| (t[0] + t[4] + t[8]).abs() < 1e-10),
            TensorSymmetry::SymmetricTraceless => self.components.iter().all(|t| {
                (t[1] - t[3]).abs() < 1e-10
                    && (t[2] - t[6]).abs() < 1e-10
                    && (t[5] - t[7]).abs() < 1e-10
                    && (t[0] + t[4] + t[8]).abs() < 1e-10
            }),
        }
    }
}

// ============================================================================
// SUBSTRATE TYPE - THE UNIFYING ABSTRACTION
// ============================================================================

/// A substrate type encapsulates all physical information about a quantity
///
/// This is the key abstraction: the compiler can reason about physics,
/// not just numbers.
#[derive(Debug, Clone)]
pub struct SubstrateType {
    /// The dimensions
    pub dimensions: Dimensions,
    /// The semantic kind
    pub kind: QuantityKind,
    /// Physical constraints
    pub constraints: Vec<PhysicalConstraint>,
    /// Whether this is extensive or intensive
    pub extensivity: Extensivity,
    /// Allowed transformations
    pub transformations: Vec<AllowedTransformation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extensivity {
    Extensive, // Scales with system size
    Intensive, // Independent of system size
    Neither,   // e.g., chemical potential in some contexts
}

#[derive(Debug, Clone)]
pub enum AllowedTransformation {
    /// Can be scaled by a positive constant
    PositiveScaling,
    /// Can be shifted by a constant
    Translation,
    /// Can be differentiated
    Differentiation,
    /// Can be integrated
    Integration,
    /// Can undergo Fourier transform
    FourierTransform,
    /// Can undergo Laplace transform
    LaplaceTransform,
    /// Custom transformation
    Custom(String),
}

impl SubstrateType {
    /// Create a substrate type for temperature
    pub fn temperature() -> Self {
        Self {
            dimensions: Dimensions::TEMPERATURE,
            kind: QuantityKind::Temperature,
            constraints: vec![PhysicalConstraint::Positive],
            extensivity: Extensivity::Intensive,
            transformations: vec![AllowedTransformation::PositiveScaling],
        }
    }

    /// Create a substrate type for energy
    pub fn energy() -> Self {
        Self {
            dimensions: Dimensions::ENERGY,
            kind: QuantityKind::Energy,
            constraints: vec![PhysicalConstraint::Conserved(ConservedQuantityType::Energy)],
            extensivity: Extensivity::Extensive,
            transformations: vec![
                AllowedTransformation::PositiveScaling,
                AllowedTransformation::Translation,
                AllowedTransformation::Differentiation,
            ],
        }
    }

    /// Create a substrate type for probability
    pub fn probability() -> Self {
        Self {
            dimensions: Dimensions::DIMENSIONLESS,
            kind: QuantityKind::Probability,
            constraints: vec![
                PhysicalConstraint::Bounded { min: 0.0, max: 1.0 },
                PhysicalConstraint::Normalized,
            ],
            extensivity: Extensivity::Neither,
            transformations: vec![],
        }
    }

    /// Check if two substrate types are compatible for an operation
    pub fn compatible_with(&self, other: &SubstrateType, op: &str) -> bool {
        match op {
            "add" | "sub" => self.dimensions.matches(&other.dimensions),
            "mul" | "div" => {
                true // Always dimensionally valid, result has combined dimensions
            }
            _ => false,
        }
    }
}

// ============================================================================
// DIMENSIONAL ANALYSIS UTILITIES
// ============================================================================

/// Utilities for dimensional analysis
pub struct DimensionalAnalysis;

impl DimensionalAnalysis {
    /// Check if an equation is dimensionally consistent
    pub fn check_equation(lhs: &Dimensions, rhs: &Dimensions) -> Result<(), ConstraintViolation> {
        if lhs.matches(rhs) {
            Ok(())
        } else {
            Err(ConstraintViolation::DimensionMismatch {
                expected: *lhs,
                actual: *rhs,
            })
        }
    }

    /// Derive dimensions from a mathematical expression
    pub fn derive_dimensions(operands: &[Dimensions], operators: &[char]) -> Dimensions {
        let mut result = operands[0];

        for (i, op) in operators.iter().enumerate() {
            let next = operands[i + 1];
            result = match op {
                '*' => result.mul(next),
                '/' => result.div(next),
                _ => result, // + and - require matching dimensions
            };
        }

        result
    }

    /// Check Buckingham Pi theorem applicability
    pub fn count_independent_dimensions(quantities: &[Dimensions]) -> usize {
        // Count how many base dimensions are actually used
        let mut used = [false; 7]; // L, M, T, I, Θ, N, J

        for q in quantities {
            if q.length != 0 {
                used[0] = true;
            }
            if q.mass != 0 {
                used[1] = true;
            }
            if q.time != 0 {
                used[2] = true;
            }
            if q.current != 0 {
                used[3] = true;
            }
            if q.temperature != 0 {
                used[4] = true;
            }
            if q.amount != 0 {
                used[5] = true;
            }
            if q.luminosity != 0 {
                used[6] = true;
            }
}

        used.iter().filter(|&&u| u).count()
    }

    /// Compute number of dimensionless groups (Buckingham Pi)
    pub fn dimensionless_groups(quantities: &[Dimensions]) -> usize {
        let n = quantities.len();
        let k = Self::count_independent_dimensions(quantities);
        n.saturating_sub(k)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_arithmetic() {
        let length = Dimensions::LENGTH;
        let time = Dimensions::TIME;

        // Velocity = length / time
        let velocity = length.div(time);
        assert!(velocity.matches(&Dimensions::VELOCITY));

        // Energy = mass * velocity^2
        let mass = Dimensions::MASS;
        let v_squared = velocity.mul(velocity);
        let energy = mass.mul(v_squared);
        assert!(energy.matches(&Dimensions::ENERGY));
    }

    #[test]
    fn test_dimension_display() {
        assert_eq!(format!("{}", Dimensions::VELOCITY), "m⋅s^-1");
        assert_eq!(format!("{}", Dimensions::ENERGY), "m^2⋅kg⋅s^-2");
        assert_eq!(format!("{}", Dimensions::DIMENSIONLESS), "1");
    }

    #[test]
    fn test_physical_quantity_validation() {
        let temp = PhysicalQuantity::temperature(300.0);
        assert!(temp.validate().is_ok());

        let bad_temp = PhysicalQuantity::temperature(-10.0);
        assert!(bad_temp.validate().is_err());

        let prob = PhysicalQuantity::probability(0.5);
        assert!(prob.validate().is_ok());

        let bad_prob = PhysicalQuantity::probability(1.5);
        assert!(bad_prob.validate().is_err());
    }

    #[test]
    fn test_quantity_addition() {
        let e1 = PhysicalQuantity::energy(100.0);
        let e2 = PhysicalQuantity::energy(50.0);

        let sum = (e1 + e2).unwrap();
        assert_eq!(sum.value, 150.0);
        assert!(sum.dimensions.matches(&Dimensions::ENERGY));
    }

    #[test]
    fn test_quantity_dimension_mismatch() {
        let energy = PhysicalQuantity::energy(100.0);
        let mass = PhysicalQuantity::mass(10.0);

        let result = energy + mass;
        assert!(result.is_err());
    }

    #[test]
    fn test_quantity_multiplication() {
        let force = PhysicalQuantity::new(10.0, Dimensions::FORCE, QuantityKind::Force);
        let distance = PhysicalQuantity::length(5.0);

        let work = force * distance;
        assert_eq!(work.value, 50.0);
        assert!(work.dimensions.matches(&Dimensions::ENERGY));
    }

    #[test]
    fn test_physical_field() {
        let temp_field = PhysicalField::uniform(
            300.0,
            vec![10, 10, 10],
            QuantityKind::Temperature,
            Dimensions::TEMPERATURE,
        );

        assert_eq!(temp_field.len(), 1000);
        assert_eq!(temp_field.max(), 300.0);
        assert_eq!(temp_field.min(), 300.0);
    }

    #[test]
    fn test_tensor_symmetry() {
        let mut stress =
            TensorField::zeros(vec![10, 10, 10], TensorKind::Stress, Dimensions::PRESSURE);
        stress.symmetry = TensorSymmetry::Symmetric;

        // Set symmetric values
        stress.components[0] = [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
        assert!(stress.check_symmetry());

        // Break symmetry
        stress.components[0][1] = 999.0;
        assert!(!stress.check_symmetry());
    }

    #[test]
    fn test_substrate_type_compatibility() {
        let energy = SubstrateType::energy();
        let temp = SubstrateType::temperature();

        // Energy + Energy is valid
        assert!(energy.compatible_with(&energy, "add"));

        // Energy + Temperature is invalid
        assert!(!energy.compatible_with(&temp, "add"));

        // Energy * Temperature is valid (gives entropy-like quantity)
        assert!(energy.compatible_with(&temp, "mul"));
    }

    #[test]
    fn test_dimensional_analysis() {
        let quantities = vec![
            Dimensions::LENGTH,       // L
            Dimensions::VELOCITY,     // L T^-1
            Dimensions::ACCELERATION, // L T^-2
            Dimensions::TIME,         // T
        ];

        // Should have 2 independent dimensions (L and T)
        assert_eq!(
            DimensionalAnalysis::count_independent_dimensions(&quantities),
            2
);

        // 4 quantities - 2 dimensions = 2 dimensionless groups
        assert_eq!(DimensionalAnalysis::dimensionless_groups(&quantities), 2);
    }

    #[test]
    fn test_quantity_kind_properties() {
        assert!(QuantityKind::Energy.is_extensive());
        assert!(!QuantityKind::Temperature.is_extensive());
        assert!(QuantityKind::Temperature.is_intensive());
        assert!(QuantityKind::Temperature.must_be_positive());

        assert_eq!(QuantityKind::Probability.bounds(), Some((0.0, 1.0)));
    }
}
