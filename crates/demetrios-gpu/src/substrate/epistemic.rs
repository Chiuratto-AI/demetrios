//! Epistemic Execution Model
//!
//! This module implements uncertainty-guided computation where epistemic state
//! (what we know and how confident we are) directly influences execution.
//!
//! # Novel Aspects
//!
//! 1. **Uncertainty as First-Class**: Not just tracked, but guides execution
//! 2. **Adaptive Precision**: High uncertainty → reduced precision (errors dominated by uncertainty)
//! 3. **Confidence-Guided Sampling**: More samples where uncertainty is high
//! 4. **Provenance Tracking**: Know where every value came from
//!
//! # The Key Insight
//!
//! Traditional computing treats all values equally. But scientific computing
//! has epistemic structure: some values are well-known, others are uncertain.
//! Computation should adapt to this structure.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// CONFIDENCE LEVELS
// ============================================================================

/// Confidence in a value
///
/// This is a semantic wrapper around probability, with physical meaning
/// attached to different levels.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    /// Create a new confidence value (clamped to [0, 1])
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Complete certainty
    pub const CERTAIN: Self = Self(1.0);

    /// High confidence (e.g., validated experimental data)
    pub const HIGH: Self = Self(0.95);

    /// Medium confidence (e.g., model predictions)
    pub const MEDIUM: Self = Self(0.7);

    /// Low confidence (e.g., rough estimates)
    pub const LOW: Self = Self(0.3);

    /// Unknown/no information
    pub const UNKNOWN: Self = Self(0.0);

    /// Get the raw value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if this is high confidence
    pub fn is_high(&self) -> bool {
        self.0 >= 0.9
    }

    /// Check if this is low confidence
    pub fn is_low(&self) -> bool {
        self.0 < 0.5
    }

    /// Combine two confidences (multiplication for independent events)
    pub fn combine(&self, other: Confidence) -> Confidence {
        Confidence(self.0 * other.0)
    }

    /// Update confidence with new evidence (Bayesian update)
    pub fn update(&self, likelihood_ratio: f64) -> Confidence {
        let prior_odds = self.0 / (1.0 - self.0 + 1e-15);
        let posterior_odds = prior_odds * likelihood_ratio;
        Confidence(posterior_odds / (1.0 + posterior_odds))
    }

    /// Interpolate between two confidences
    pub fn interpolate(&self, other: Confidence, t: f64) -> Confidence {
        Confidence(self.0 * (1.0 - t) + other.0 * t)
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self::MEDIUM
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 0.99 {
            write!(f, "certain")
        } else if self.0 >= 0.9 {
            write!(f, "high ({:.1}%)", self.0 * 100.0)
        } else if self.0 >= 0.5 {
            write!(f, "medium ({:.1}%)", self.0 * 100.0)
        } else if self.0 >= 0.1 {
            write!(f, "low ({:.1}%)", self.0 * 100.0)
        } else {
            write!(f, "unknown ({:.1}%)", self.0 * 100.0)
        }
    }
}

// ============================================================================
// PROVENANCE - WHERE DID THIS VALUE COME FROM?
// ============================================================================

/// Provenance: the origin and history of a value
///
/// Every scientific value should know where it came from. This enables:
/// - Reproducibility verification
/// - Uncertainty propagation
/// - Debugging and auditing
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Unique identifier
    pub id: u64,
    /// Source of this value
    pub source: ProvenanceSource,
    /// Parent provenances (for derived values)
    pub parents: Vec<u64>,
    /// Timestamp
    pub timestamp: Instant,
    /// Transformations applied
    pub transformations: Vec<TransformationRecord>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// The original source of a value
#[derive(Debug, Clone)]
pub enum ProvenanceSource {
    /// Measured experimentally
    Experimental {
        instrument: String,
        measurement_id: String,
        uncertainty: f64,
    },
    /// From a database or literature
    Literature {
        reference: String,
        doi: Option<String>,
    },
    /// Computed from other values
    Computed { algorithm: String, version: String },
    /// User-provided constant
    UserInput { description: String },
    /// Default/placeholder value
    Default,
    /// Sampled from a distribution
    Sampled {
        distribution: String,
        seed: Option<u64>,
    },
    /// From a simulation
    Simulated {
        model: String,
        parameters: HashMap<String, f64>,
    },
    /// Interpolated from other values
    Interpolated { method: String },
    /// Unknown origin
    Unknown,
}

/// Record of a transformation applied to a value
#[derive(Debug, Clone)]
pub struct TransformationRecord {
    /// Name of the transformation
    pub name: String,
    /// Description
    pub description: String,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: Instant,
    /// Error introduced (if known)
    pub error_bound: Option<f64>,
}

impl Provenance {
    /// Create a new provenance from an experimental measurement
    pub fn experimental(instrument: &str, measurement_id: &str, uncertainty: f64) -> Self {
        Self {
            id: Self::generate_id(),
            source: ProvenanceSource::Experimental {
                instrument: instrument.to_string(),
                measurement_id: measurement_id.to_string(),
                uncertainty,
            },
            parents: Vec::new(),
            timestamp: Instant::now(),
            transformations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new provenance from computation
    pub fn computed(algorithm: &str, parents: Vec<&Provenance>) -> Self {
        Self {
            id: Self::generate_id(),
            source: ProvenanceSource::Computed {
                algorithm: algorithm.to_string(),
                version: "1.0".to_string(),
            },
            parents: parents.iter().map(|p| p.id).collect(),
            timestamp: Instant::now(),
            transformations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a provenance for a sampled value
    pub fn sampled(distribution: &str, seed: Option<u64>) -> Self {
        Self {
            id: Self::generate_id(),
            source: ProvenanceSource::Sampled {
                distribution: distribution.to_string(),
                seed,
            },
            parents: Vec::new(),
            timestamp: Instant::now(),
            transformations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a provenance for user input
    pub fn user_input(description: &str) -> Self {
        Self {
            id: Self::generate_id(),
            source: ProvenanceSource::UserInput {
                description: description.to_string(),
            },
            parents: Vec::new(),
            timestamp: Instant::now(),
            transformations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create unknown provenance
    pub fn unknown() -> Self {
        Self {
            id: Self::generate_id(),
            source: ProvenanceSource::Unknown,
            parents: Vec::new(),
            timestamp: Instant::now(),
            transformations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Record a transformation
    pub fn record_transformation(
        &mut self,
        name: &str,
        description: &str,
        error_bound: Option<f64>,
    ) {
        self.transformations.push(TransformationRecord {
            name: name.to_string(),
            description: description.to_string(),
            parameters: HashMap::new(),
            timestamp: Instant::now(),
            error_bound,
        });
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Generate a unique ID
    fn generate_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the depth of the provenance chain
    pub fn chain_depth(&self) -> usize {
        self.transformations.len() + if self.parents.is_empty() { 0 } else { 1 }
    }

    /// Check if this value is traceable to experimental data
    pub fn is_experimentally_grounded(&self) -> bool {
        matches!(self.source, ProvenanceSource::Experimental { .. })
    }
}

// ============================================================================
// TEMPORAL VALIDITY - WHEN IS THIS VALUE VALID?
// ============================================================================

/// Temporal validity: when is this value still valid?
///
/// Scientific data can become stale. A measurement from yesterday is more
/// reliable than one from ten years ago. Model parameters may drift.
#[derive(Debug, Clone)]
pub struct TemporalValidity {
    /// When this value was created/measured
    pub created: Instant,
    /// When this value expires (if known)
    pub expires: Option<Instant>,
    /// How validity decays over time
    pub decay: ValidityDecay,
}

/// How validity decays over time
#[derive(Debug, Clone, Copy)]
pub enum ValidityDecay {
    /// Never expires
    Eternal,
    /// Expires at a fixed time
    Fixed,
    /// Exponential decay with half-life
    Exponential { half_life: Duration },
    /// Linear decay
    Linear { lifetime: Duration },
    /// Immediate expiration (single-use)
    Immediate,
}

impl TemporalValidity {
    /// Create eternal validity (never expires)
    pub fn eternal() -> Self {
        Self {
            created: Instant::now(),
            expires: None,
            decay: ValidityDecay::Eternal,
        }
    }

    /// Create fixed expiration
    pub fn expires_in(duration: Duration) -> Self {
        Self {
            created: Instant::now(),
            expires: Some(Instant::now() + duration),
            decay: ValidityDecay::Fixed,
        }
    }

    /// Create exponential decay
    pub fn with_half_life(half_life: Duration) -> Self {
        Self {
            created: Instant::now(),
            expires: None,
            decay: ValidityDecay::Exponential { half_life },
        }
    }

    /// Check if still valid
    pub fn is_valid(&self) -> bool {
        match self.decay {
            ValidityDecay::Eternal => true,
            ValidityDecay::Fixed => self.expires.map_or(true, |e| Instant::now() < e),
            ValidityDecay::Immediate => false,
            _ => self.current_validity() > 0.0,
        }
    }

    /// Get current validity (0 to 1)
    pub fn current_validity(&self) -> f64 {
        let age = self.created.elapsed();

        match self.decay {
            ValidityDecay::Eternal => 1.0,
            ValidityDecay::Fixed => {
                if let Some(expires) = self.expires {
                    if Instant::now() >= expires {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            }
            ValidityDecay::Exponential { half_life } => {
                let t = age.as_secs_f64() / half_life.as_secs_f64();
                0.5_f64.powf(t)
            }
            ValidityDecay::Linear { lifetime } => {
                let t = age.as_secs_f64() / lifetime.as_secs_f64();
                (1.0 - t).max(0.0)
            }
            ValidityDecay::Immediate => 0.0,
        }
    }
}

impl Default for TemporalValidity {
    fn default() -> Self {
        Self::eternal()
    }
}

// ============================================================================
// EPISTEMIC VALUE - THE CORE TYPE
// ============================================================================

/// An epistemic value: a value with confidence, provenance, and validity
///
/// This is the core type for uncertainty-guided computation. Every value
/// knows how confident we are in it, where it came from, and when it expires.
#[derive(Debug, Clone)]
pub struct Epistemic<T> {
    /// The value itself
    pub value: T,
    /// Confidence in this value
    pub confidence: Confidence,
    /// Provenance (where it came from)
    pub provenance: Provenance,
    /// Temporal validity
    pub validity: TemporalValidity,
    /// Epistemic uncertainty (reducible with more information)
    pub epistemic_std: f64,
    /// Aleatoric uncertainty (irreducible randomness)
    pub aleatoric_std: f64,
}

impl<T: Clone> Epistemic<T> {
    /// Create a new epistemic value
    pub fn new(value: T) -> Self {
        Self {
            value,
            confidence: Confidence::default(),
            provenance: Provenance::unknown(),
            validity: TemporalValidity::default(),
            epistemic_std: 0.0,
            aleatoric_std: 0.0,
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set provenance
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Set validity
    pub fn with_validity(mut self, validity: TemporalValidity) -> Self {
        self.validity = validity;
        self
    }

    /// Set epistemic uncertainty
    pub fn with_epistemic_std(mut self, std: f64) -> Self {
        self.epistemic_std = std;
        self
    }

    /// Set aleatoric uncertainty
    pub fn with_aleatoric_std(mut self, std: f64) -> Self {
        self.aleatoric_std = std;
        self
    }

    /// Get total uncertainty (combined epistemic and aleatoric)
    pub fn total_uncertainty(&self) -> f64 {
        (self.epistemic_std.powi(2) + self.aleatoric_std.powi(2)).sqrt()
    }

    /// Check if this value is usable (valid and confident enough)
    pub fn is_usable(&self, min_confidence: Confidence) -> bool {
        self.validity.is_valid() && self.confidence >= min_confidence
    }

    /// Get effective confidence (considering validity decay)
    pub fn effective_confidence(&self) -> Confidence {
        Confidence::new(self.confidence.value() * self.validity.current_validity())
    }

    /// Map the value while preserving epistemic state
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Epistemic<U> {
        let mut new_provenance = self.provenance.clone();
        new_provenance.record_transformation("map", "Value transformation", None);

        Epistemic {
            value: f(self.value),
            confidence: self.confidence,
            provenance: new_provenance,
            validity: self.validity,
            epistemic_std: self.epistemic_std,
            aleatoric_std: self.aleatoric_std,
        }
    }
}

impl<T: Clone + Default> Default for Epistemic<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

// ============================================================================
// UNCERTAINTY PROPAGATION
// ============================================================================

/// Propagate uncertainty through operations
pub struct UncertaintyPropagation;

impl UncertaintyPropagation {
    /// Propagate uncertainty through addition: z = x + y
    pub fn add(x_std: f64, y_std: f64) -> f64 {
        (x_std.powi(2) + y_std.powi(2)).sqrt()
    }

    /// Propagate uncertainty through subtraction: z = x - y
    pub fn sub(x_std: f64, y_std: f64) -> f64 {
        (x_std.powi(2) + y_std.powi(2)).sqrt()
    }

    /// Propagate uncertainty through multiplication: z = x * y
    pub fn mul(x: f64, x_std: f64, y: f64, y_std: f64) -> f64 {
        let z = x * y;
        let rel_x = if x.abs() > 1e-15 {
            x_std / x.abs()
        } else {
            0.0
        };
        let rel_y = if y.abs() > 1e-15 {
            y_std / y.abs()
        } else {
            0.0
        };
        z.abs() * (rel_x.powi(2) + rel_y.powi(2)).sqrt()
    }

    /// Propagate uncertainty through division: z = x / y
    pub fn div(x: f64, x_std: f64, y: f64, y_std: f64) -> f64 {
        let z = x / y;
        let rel_x = if x.abs() > 1e-15 {
            x_std / x.abs()
        } else {
            0.0
        };
        let rel_y = if y.abs() > 1e-15 {
            y_std / y.abs()
        } else {
            0.0
        };
        z.abs() * (rel_x.powi(2) + rel_y.powi(2)).sqrt()
    }

    /// Propagate uncertainty through power: z = x^n
    pub fn pow(x: f64, x_std: f64, n: f64) -> f64 {
        n.abs() * x.powf(n - 1.0) * x_std
    }

    /// Propagate uncertainty through exp: z = exp(x)
    pub fn exp(x: f64, x_std: f64) -> f64 {
        x.exp() * x_std
    }

    /// Propagate uncertainty through ln: z = ln(x)
    pub fn ln(x: f64, x_std: f64) -> f64 {
        x_std / x.abs()
    }

    /// Propagate uncertainty through sin: z = sin(x)
    pub fn sin(x: f64, x_std: f64) -> f64 {
        x.cos().abs() * x_std
    }

    /// Propagate uncertainty through cos: z = cos(x)
    pub fn cos(x: f64, x_std: f64) -> f64 {
        x.sin().abs() * x_std
    }

    /// General formula for f(x) with known derivative
    pub fn general(df_dx: f64, x_std: f64) -> f64 {
        df_dx.abs() * x_std
    }

    /// Multi-variable propagation: z = f(x1, x2, ..., xn)
    pub fn multivariate(partial_derivatives: &[f64], uncertainties: &[f64]) -> f64 {
        partial_derivatives
            .iter()
            .zip(uncertainties.iter())
            .map(|(df, std)| (df * std).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ============================================================================
// ADAPTIVE PRECISION
// ============================================================================

/// Precision level for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    /// 8-bit floating point
    FP8,
    /// 16-bit floating point
    FP16,
    /// Brain float 16
    BF16,
    /// TensorFloat 32
    TF32,
    /// 32-bit floating point
    FP32,
    /// 64-bit floating point
    FP64,
    /// Extended precision (80-bit)
    Extended,
    /// Arbitrary precision
    Arbitrary(u32),
}

impl Precision {
    /// Get the epsilon (machine precision) for this format
    pub fn epsilon(&self) -> f64 {
        match self {
            Self::FP8 => 0.0625,        // 2^-4
            Self::FP16 => 9.77e-4,      // 2^-10
            Self::BF16 => 7.81e-3,      // 2^-7
            Self::TF32 => 4.88e-4,      // 2^-11
            Self::FP32 => 1.19e-7,      // 2^-23
            Self::FP64 => 2.22e-16,     // 2^-52
            Self::Extended => 1.08e-19, // 2^-63
            Self::Arbitrary(bits) => 2.0_f64.powi(-(*bits as i32 - 1)),
        }
    }

    /// Get the number of bits
    pub fn bits(&self) -> u32 {
        match self {
            Self::FP8 => 8,
            Self::FP16 => 16,
            Self::BF16 => 16,
            Self::TF32 => 19,
            Self::FP32 => 32,
            Self::FP64 => 64,
            Self::Extended => 80,
            Self::Arbitrary(bits) => *bits,
        }
    }
}

/// Adaptive precision selection based on uncertainty
pub struct AdaptivePrecision;

impl AdaptivePrecision {
    /// Select precision based on uncertainty
    ///
    /// If uncertainty dominates, we can use lower precision without
    /// losing meaningful information.
    pub fn select_precision(
        epistemic_std: f64,
        aleatoric_std: f64,
        value_magnitude: f64,
    ) -> Precision {
        let total_uncertainty = (epistemic_std.powi(2) + aleatoric_std.powi(2)).sqrt();
        let relative_uncertainty = if value_magnitude.abs() > 1e-15 {
            total_uncertainty / value_magnitude.abs()
        } else {
            total_uncertainty
        };

        // If uncertainty is high, numerical precision doesn't matter
        if relative_uncertainty > 0.1 {
            Precision::FP16
        } else if relative_uncertainty > 0.01 {
            Precision::BF16
        } else if relative_uncertainty > 1e-4 {
            Precision::FP32
        } else if relative_uncertainty > 1e-10 {
            Precision::FP64
        } else {
            Precision::Extended
        }
    }

    /// Check if precision is sufficient for the uncertainty level
    pub fn is_precision_sufficient(precision: Precision, relative_uncertainty: f64) -> bool {
        precision.epsilon() < relative_uncertainty * 0.1
    }

    /// Calculate wasted precision (bits beyond uncertainty level)
    pub fn wasted_precision(precision: Precision, relative_uncertainty: f64) -> u32 {
        let useful_bits = (-relative_uncertainty.log2()).ceil() as u32;
        precision.bits().saturating_sub(useful_bits)
    }
}

// ============================================================================
// UNCERTAINTY-GUIDED SAMPLING
// ============================================================================

/// Guidance for adaptive sampling based on uncertainty
pub struct UncertaintyGuidedSampling;

impl UncertaintyGuidedSampling {
    /// Compute number of samples needed based on uncertainty
    ///
    /// High epistemic uncertainty → more samples (we can learn more)
    /// High aleatoric uncertainty → diminishing returns from more samples
    pub fn recommended_samples(
        epistemic_std: f64,
        aleatoric_std: f64,
        target_precision: f64,
    ) -> usize {
        // Central limit theorem: std decreases as 1/sqrt(n)
        // We want epistemic_std / sqrt(n) < target_precision

        if epistemic_std <= target_precision {
            return 1; // Already precise enough
        }

        let n_for_epistemic = (epistemic_std / target_precision).powi(2);

        // Aleatoric uncertainty can't be reduced, so cap the samples
        let n_max_useful = if aleatoric_std > 1e-15 {
            (aleatoric_std / target_precision).powi(2) * 10.0
        } else {
            1e6
        };

        (n_for_epistemic.min(n_max_useful) as usize).max(1)
    }

    /// Should we continue sampling?
    pub fn should_continue(
        current_epistemic_std: f64,
        target_precision: f64,
        samples_so_far: usize,
        max_samples: usize,
    ) -> bool {
        samples_so_far < max_samples && current_epistemic_std > target_precision
    }

    /// Compute sampling weight for importance sampling
    ///
    /// High uncertainty regions should be sampled more frequently.
    pub fn importance_weight(epistemic_std: f64, baseline_std: f64) -> f64 {
        if baseline_std > 1e-15 {
            (epistemic_std / baseline_std).powi(2).min(100.0)
        } else {
            1.0
        }
    }
}

// ============================================================================
// EPISTEMIC EXECUTION CONTEXT
// ============================================================================

/// Execution context that adapts based on epistemic state
#[derive(Debug)]
pub struct EpistemicExecution {
    /// Minimum confidence for computation to proceed
    pub min_confidence: Confidence,
    /// Target precision
    pub target_precision: f64,
    /// Maximum samples for Monte Carlo
    pub max_samples: usize,
    /// Whether to track provenance
    pub track_provenance: bool,
    /// Whether to use adaptive precision
    pub adaptive_precision: bool,
    /// Statistics
    pub stats: EpistemicStats,
}

/// Statistics about epistemic execution
#[derive(Debug, Default)]
pub struct EpistemicStats {
    /// Total computations performed
    pub computations: u64,
    /// Computations skipped due to low confidence
    pub skipped_low_confidence: u64,
    /// Computations with reduced precision
    pub reduced_precision: u64,
    /// Samples taken
    pub total_samples: u64,
    /// Provenance records created
    pub provenance_records: u64,
}

impl EpistemicExecution {
    /// Create a new execution context
    pub fn new() -> Self {
        Self {
            min_confidence: Confidence::LOW,
            target_precision: 1e-6,
            max_samples: 10000,
            track_provenance: true,
            adaptive_precision: true,
            stats: EpistemicStats::default(),
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: Confidence) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Set target precision
    pub fn with_target_precision(mut self, precision: f64) -> Self {
        self.target_precision = precision;
        self
    }

    /// Enable/disable adaptive precision
    pub fn with_adaptive_precision(mut self, enabled: bool) -> Self {
        self.adaptive_precision = enabled;
        self
    }

    /// Execute a computation with epistemic awareness
    pub fn execute<T, F>(&mut self, input: &Epistemic<T>, f: F) -> Option<Epistemic<T>>
    where
        T: Clone,
        F: FnOnce(&T) -> T,
    {
        self.stats.computations += 1;

        // Check confidence threshold
        if input.effective_confidence() < self.min_confidence {
            self.stats.skipped_low_confidence += 1;
            return None;
        }

        // Record precision choice
        if self.adaptive_precision {
            let precision = AdaptivePrecision::select_precision(
                input.epistemic_std,
                input.aleatoric_std,
                1.0, // Would need actual magnitude
            );
            if precision < Precision::FP64 {
                self.stats.reduced_precision += 1;
            }
        }

        // Perform computation
        let result = f(&input.value);

        // Create provenance
        let mut provenance = if self.track_provenance {
            self.stats.provenance_records += 1;
            Provenance::computed("execute", vec![&input.provenance])
        } else {
            Provenance::unknown()
        };

        Some(Epistemic {
            value: result,
            confidence: input.confidence,
            provenance,
            validity: input.validity.clone(),
            epistemic_std: input.epistemic_std,
            aleatoric_std: input.aleatoric_std,
        })
    }
}

impl Default for EpistemicExecution {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_levels() {
        assert!(Confidence::CERTAIN > Confidence::HIGH);
        assert!(Confidence::HIGH > Confidence::MEDIUM);
        assert!(Confidence::MEDIUM > Confidence::LOW);
        assert!(Confidence::LOW > Confidence::UNKNOWN);
    }

    #[test]
    fn test_confidence_combination() {
        let c1 = Confidence::new(0.9);
        let c2 = Confidence::new(0.9);
        let combined = c1.combine(c2);
        assert!((combined.value() - 0.81).abs() < 1e-10);
    }

    #[test]
    fn test_provenance_chain() {
        let p1 = Provenance::experimental("sensor1", "meas001", 0.01);
        let p2 = Provenance::computed("average", vec![&p1]);

        assert!(p1.is_experimentally_grounded());
        assert!(!p2.is_experimentally_grounded());
        assert!(p2.parents.contains(&p1.id));
    }

    #[test]
    fn test_temporal_validity_exponential() {
        let validity = TemporalValidity::with_half_life(Duration::from_secs(1));

        // Initially should be ~1.0
        assert!(validity.current_validity() > 0.99);

        // After creation it starts decaying
        std::thread::sleep(Duration::from_millis(10));
        let v = validity.current_validity();
        assert!(v < 1.0 && v > 0.9);
    }

    #[test]
    fn test_epistemic_value() {
        let e = Epistemic::new(42.0)
            .with_confidence(Confidence::HIGH)
            .with_epistemic_std(1.0)
            .with_aleatoric_std(0.5);

        assert!((e.total_uncertainty() - (1.0_f64.powi(2) + 0.5_f64.powi(2)).sqrt()).abs() < 1e-10);
        assert!(e.is_usable(Confidence::MEDIUM));
    }

    #[test]
    fn test_uncertainty_propagation_addition() {
        // z = x + y with independent uncertainties
        let x_std = 1.0;
        let y_std = 1.0;
        let z_std = UncertaintyPropagation::add(x_std, y_std);

        // Should be sqrt(1^2 + 1^2) = sqrt(2)
        assert!((z_std - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_uncertainty_propagation_multiplication() {
        let x = 10.0;
        let x_std = 1.0; // 10% relative uncertainty
        let y = 5.0;
        let y_std = 0.5; // 10% relative uncertainty

        let z_std = UncertaintyPropagation::mul(x, x_std, y, y_std);

        // z = 50, relative uncertainty = sqrt(0.1^2 + 0.1^2) ≈ 14.14%
        // absolute uncertainty ≈ 7.07
        assert!((z_std - 50.0 * (0.1_f64.powi(2) + 0.1_f64.powi(2)).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_precision() {
        // High uncertainty → low precision is fine
        let p1 = AdaptivePrecision::select_precision(0.5, 0.1, 1.0);
        assert!(p1 <= Precision::FP16);

        // Low uncertainty → high precision needed
        let p2 = AdaptivePrecision::select_precision(1e-12, 0.0, 1.0);
        assert!(p2 >= Precision::FP64);
    }

    #[test]
    fn test_recommended_samples() {
        // High epistemic uncertainty needs many samples
        let n1 = UncertaintyGuidedSampling::recommended_samples(1.0, 0.0, 0.01);
        assert!(n1 > 1000);

        // Low epistemic uncertainty needs few samples
        let n2 = UncertaintyGuidedSampling::recommended_samples(0.001, 0.0, 0.01);
        assert!(n2 == 1);

        // High aleatoric uncertainty caps useful samples
        let n3 = UncertaintyGuidedSampling::recommended_samples(1.0, 1.0, 0.01);
        assert!(n3 < 1_000_000);
    }

    #[test]
    fn test_epistemic_execution() {
        let mut ctx = EpistemicExecution::new().with_min_confidence(Confidence::LOW);

        let input = Epistemic::new(10.0).with_confidence(Confidence::HIGH);

        let result = ctx.execute(&input, |x| x * 2.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, 20.0);
    }

    #[test]
    fn test_epistemic_execution_low_confidence_skip() {
        let mut ctx = EpistemicExecution::new().with_min_confidence(Confidence::HIGH);

        let input = Epistemic::new(10.0).with_confidence(Confidence::LOW);

        let result = ctx.execute(&input, |x| x * 2.0);
        assert!(result.is_none());
        assert_eq!(ctx.stats.skipped_low_confidence, 1);
    }
}
