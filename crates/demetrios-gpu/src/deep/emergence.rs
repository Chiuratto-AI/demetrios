//! Emergence and Self-Organization
//!
//! This module implements computational concepts from complexity science:
//!
//! # Key Concepts
//!
//! ## Self-Organized Criticality (SOC)
//! Systems that naturally evolve to critical states where small perturbations
//! can cause cascades of all sizes (power-law distributions).
//!
//! ## Renormalization Group (RG)
//! A method for analyzing systems at different scales, revealing
//! which details matter and which can be coarse-grained away.
//!
//! ## Computational Irreducibility
//! Some computations cannot be shortcut - you must run them to know the result.
//! This is a fundamental limit on prediction.
//!
//! ## Phase Transitions
//! Abrupt changes in system behavior as parameters cross critical values.
//! Order parameters characterize the transition.

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// SCALE
// ============================================================================

/// A scale in a multi-scale system
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Scale(pub f64);

impl Scale {
    /// Microscopic scale
    pub const MICRO: Scale = Scale(1.0);

    /// Mesoscopic scale
    pub const MESO: Scale = Scale(1e3);

    /// Macroscopic scale
    pub const MACRO: Scale = Scale(1e6);

    /// Create a scale
    pub fn new(value: f64) -> Self {
        assert!(value > 0.0, "Scale must be positive");
        Scale(value)
    }

    /// Ratio between scales
    pub fn ratio(&self, other: &Scale) -> f64 {
        self.0 / other.0
    }

    /// Coarsen by a factor
    pub fn coarsen(&self, factor: f64) -> Self {
        Scale(self.0 * factor)
    }

    /// Refine by a factor
    pub fn refine(&self, factor: f64) -> Self {
        Scale(self.0 / factor)
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1e6 {
            write!(f, "{:.1}M", self.0 / 1e6)
        } else if self.0 >= 1e3 {
            write!(f, "{:.1}K", self.0 / 1e3)
        } else {
            write!(f, "{:.1}", self.0)
        }
    }
}

// ============================================================================
// SELF-ORGANIZATION
// ============================================================================

/// Characteristics of self-organization
#[derive(Debug, Clone)]
pub struct SelfOrganization {
    /// The order parameter (measures organization)
    pub order_parameter: f64,
    /// Entropy of the system
    pub entropy: f64,
    /// Free energy (if applicable)
    pub free_energy: Option<f64>,
    /// Is the system at a critical point?
    pub is_critical: bool,
    /// Power law exponent (if scale-free)
    pub power_law_exponent: Option<f64>,
}

impl SelfOrganization {
    /// Create a new self-organization measurement
    pub fn new(order_parameter: f64, entropy: f64) -> Self {
        Self {
            order_parameter,
            entropy,
            free_energy: None,
            is_critical: false,
            power_law_exponent: None,
        }
    }

    /// Mark as critical with power law exponent
    pub fn critical(mut self, exponent: f64) -> Self {
        self.is_critical = true;
        self.power_law_exponent = Some(exponent);
        self
    }

    /// Set free energy
    pub fn with_free_energy(mut self, f: f64) -> Self {
        self.free_energy = Some(f);
        self
    }

    /// Check if system shows signs of self-organization
    pub fn is_organized(&self) -> bool {
        self.order_parameter > 0.5
    }

    /// Check if system is in edge of chaos
    pub fn is_edge_of_chaos(&self) -> bool {
        self.is_critical && self.order_parameter > 0.3 && self.order_parameter < 0.7
    }
}

// ============================================================================
// CRITICALITY
// ============================================================================

/// Measures of criticality
#[derive(Debug, Clone)]
pub struct Criticality {
    /// Distance from critical point
    pub distance: f64,
    /// Correlation length (diverges at criticality)
    pub correlation_length: f64,
    /// Susceptibility (response to perturbation)
    pub susceptibility: f64,
    /// Critical exponents
    pub exponents: CriticalExponents,
}

/// Critical exponents characterize universality classes
#[derive(Debug, Clone, Copy)]
pub struct CriticalExponents {
    /// β: order parameter ~ |T-Tc|^β
    pub beta: f64,
    /// γ: susceptibility ~ |T-Tc|^(-γ)
    pub gamma: f64,
    /// ν: correlation length ~ |T-Tc|^(-ν)
    pub nu: f64,
    /// α: specific heat ~ |T-Tc|^(-α)
    pub alpha: f64,
    /// η: correlation function at Tc ~ r^(-(d-2+η))
    pub eta: f64,
}

impl CriticalExponents {
    /// Mean field exponents
    pub fn mean_field() -> Self {
        Self {
            beta: 0.5,
            gamma: 1.0,
            nu: 0.5,
            alpha: 0.0,
            eta: 0.0,
        }
    }

    /// 2D Ising model exponents
    pub fn ising_2d() -> Self {
        Self {
            beta: 0.125,
            gamma: 1.75,
            nu: 1.0,
            alpha: 0.0, // Logarithmic
            eta: 0.25,
        }
    }

    /// 3D Ising model exponents (approximate)
    pub fn ising_3d() -> Self {
        Self {
            beta: 0.326,
            gamma: 1.237,
            nu: 0.630,
            alpha: 0.110,
            eta: 0.036,
        }
    }

    /// Check hyperscaling relation: 2 - α = νd
    pub fn check_hyperscaling(&self, d: f64) -> f64 {
        let lhs = 2.0 - self.alpha;
        let rhs = self.nu * d;
        (lhs - rhs).abs()
    }

    /// Check Rushbrooke inequality: α + 2β + γ >= 2
    pub fn check_rushbrooke(&self) -> f64 {
        self.alpha + 2.0 * self.beta + self.gamma - 2.0
    }
}

impl Criticality {
    /// Create criticality measure
    pub fn new(distance: f64, correlation_length: f64, susceptibility: f64) -> Self {
        Self {
            distance,
            correlation_length,
            susceptibility,
            exponents: CriticalExponents::mean_field(),
        }
    }

    /// Set critical exponents
    pub fn with_exponents(mut self, exponents: CriticalExponents) -> Self {
        self.exponents = exponents;
        self
    }

    /// Is the system at criticality?
    pub fn is_critical(&self) -> bool {
        self.distance.abs() < 0.01 && self.correlation_length > 100.0
    }

    /// Estimate order parameter from critical exponents
    pub fn order_parameter(&self) -> f64 {
        if self.distance > 0.0 {
            0.0 // Above Tc
        } else {
            (-self.distance).powf(self.exponents.beta)
        }
    }
}

// ============================================================================
// RENORMALIZATION GROUP
// ============================================================================

/// Renormalization group transformation
///
/// The RG systematically integrates out short-distance degrees of freedom,
/// revealing the effective theory at larger scales.
#[derive(Debug, Clone)]
pub struct RenormalizationGroup {
    /// Current scale
    pub scale: Scale,
    /// Coupling constants at this scale
    pub couplings: HashMap<String, f64>,
    /// RG flow history
    flow_history: Vec<RGFlowPoint>,
}

/// A point in RG flow
#[derive(Debug, Clone)]
pub struct RGFlowPoint {
    pub scale: Scale,
    pub couplings: HashMap<String, f64>,
}

impl RenormalizationGroup {
    /// Create a new RG at initial scale
    pub fn new(scale: Scale) -> Self {
        Self {
            scale,
            couplings: HashMap::new(),
            flow_history: Vec::new(),
        }
    }

    /// Set a coupling constant
    pub fn set_coupling(&mut self, name: &str, value: f64) {
        self.couplings.insert(name.to_string(), value);
    }

    /// Get a coupling constant
    pub fn get_coupling(&self, name: &str) -> Option<f64> {
        self.couplings.get(name).copied()
    }

    /// Perform one RG step (coarse-graining)
    ///
    /// The beta functions describe how couplings change with scale:
    /// dg/d(ln μ) = β(g)
    pub fn step<F>(&mut self, factor: f64, beta_functions: F)
    where
        F: Fn(&str, f64) -> f64,
    {
        // Record current state
        self.flow_history.push(RGFlowPoint {
            scale: self.scale,
            couplings: self.couplings.clone(),
        });

        // Update scale
        self.scale = self.scale.coarsen(factor);

        // Update couplings according to beta functions
        let log_factor = factor.ln();
        for (name, value) in self.couplings.iter_mut() {
            let beta = beta_functions(name, *value);
            *value += beta * log_factor;
        }
    }

    /// Find fixed points (where β = 0)
    pub fn find_fixed_points(&self) -> Vec<HashMap<String, f64>> {
        // This would require solving β(g*) = 0
        // Simplified: return current couplings if they're small
        let is_fixed = self.couplings.values().all(|&v| v.abs() < 0.01);
        if is_fixed {
            vec![self.couplings.clone()]
        } else {
            Vec::new()
        }
    }

    /// Check if a coupling is relevant, marginal, or irrelevant
    /// at the current fixed point
    pub fn coupling_relevance(&self, name: &str, dimension: f64) -> CouplingRelevance {
        // In general, relevance depends on scaling dimension
        // Simplified: use mass dimension
        if dimension > 0.0 {
            CouplingRelevance::Relevant
        } else if dimension == 0.0 {
            CouplingRelevance::Marginal
        } else {
            CouplingRelevance::Irrelevant
        }
    }

    /// Get the flow trajectory
    pub fn trajectory(&self) -> &[RGFlowPoint] {
        &self.flow_history
    }
}

/// Relevance of a coupling under RG flow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingRelevance {
    /// Grows under RG flow (important at large scales)
    Relevant,
    /// Stays constant (logarithmic corrections)
    Marginal,
    /// Shrinks under RG flow (unimportant at large scales)
    Irrelevant,
}

// ============================================================================
// COMPUTATIONAL IRREDUCIBILITY
// ============================================================================

/// Measures of computational irreducibility
///
/// A computation is irreducible if there's no shortcut -
/// you must run it step by step to know the result.
#[derive(Debug, Clone)]
pub struct Irreducibility {
    /// Estimated complexity class
    pub complexity: ComplexityClass,
    /// Is there a known shortcut?
    pub has_shortcut: bool,
    /// Ratio of actual to optimal computation
    pub efficiency: f64,
    /// Evidence of irreducibility
    pub evidence: Vec<String>,
}

/// Complexity classes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexityClass {
    /// Constant time
    O1,
    /// Logarithmic
    OLogN,
    /// Linear
    ON,
    /// Linearithmic
    ONLogN,
    /// Quadratic
    ON2,
    /// Polynomial
    OPoly,
    /// Exponential
    OExp,
    /// Computationally irreducible
    Irreducible,
}

impl Irreducibility {
    /// Create an irreducibility assessment
    pub fn new(complexity: ComplexityClass, has_shortcut: bool) -> Self {
        Self {
            complexity,
            has_shortcut,
            efficiency: if has_shortcut { 0.5 } else { 1.0 },
            evidence: Vec::new(),
        }
    }

    /// Add evidence for (ir)reducibility
    pub fn add_evidence(&mut self, evidence: &str) {
        self.evidence.push(evidence.to_string());
    }

    /// Is this computation fundamentally irreducible?
    pub fn is_irreducible(&self) -> bool {
        self.complexity == ComplexityClass::Irreducible && !self.has_shortcut
    }

    /// Can we predict the outcome without running the full computation?
    pub fn is_predictable(&self) -> bool {
        self.has_shortcut || self.complexity != ComplexityClass::Irreducible
    }
}

/// Check if a cellular automaton rule is computationally irreducible
pub fn check_ca_irreducibility(rule: u8, steps: usize) -> Irreducibility {
    // Rules 30, 110, etc. are known to be computationally irreducible
    let known_irreducible = [30, 45, 73, 89, 101, 110, 135, 149, 169];

    let is_irreducible = known_irreducible.contains(&rule);

    let mut irr = Irreducibility::new(
        if is_irreducible {
            ComplexityClass::Irreducible
        } else {
            ComplexityClass::ON
        },
        !is_irreducible,
    );

    if is_irreducible {
        irr.add_evidence(&format!(
            "Rule {} is proven computationally irreducible",
            rule
        ));
    }

    irr
}

// ============================================================================
// PHASE TRANSITION
// ============================================================================

/// A phase transition in a computational system
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Critical parameter value
    pub critical_value: f64,
    /// Name of the control parameter
    pub control_parameter: String,
    /// Type of transition
    pub transition_type: TransitionType,
    /// Order parameter name
    pub order_parameter_name: String,
}

/// Types of phase transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// First-order (discontinuous order parameter)
    FirstOrder,
    /// Second-order (continuous, diverging susceptibility)
    SecondOrder,
    /// Infinite-order (e.g., BKT transition)
    InfiniteOrder,
    /// Crossover (no true singularity)
    Crossover,
}

impl PhaseTransition {
    /// Create a phase transition
    pub fn new(
        critical_value: f64,
        control_parameter: &str,
        transition_type: TransitionType,
        order_parameter: &str,
    ) -> Self {
        Self {
            critical_value,
            control_parameter: control_parameter.to_string(),
            transition_type,
            order_parameter_name: order_parameter.to_string(),
        }
    }

    /// Is this a continuous transition?
    pub fn is_continuous(&self) -> bool {
        matches!(
            self.transition_type,
            TransitionType::SecondOrder | TransitionType::InfiniteOrder
        )
    }

    /// Distance from critical point
    pub fn reduced_parameter(&self, value: f64) -> f64 {
        (value - self.critical_value) / self.critical_value
    }
}

// ============================================================================
// ORDER PARAMETER
// ============================================================================

/// An order parameter that characterizes a phase
#[derive(Debug, Clone)]
pub struct OrderParameter {
    /// Name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Symmetry that's broken when non-zero
    pub broken_symmetry: String,
    /// Is it currently ordered?
    pub is_ordered: bool,
}

impl OrderParameter {
    /// Create an order parameter
    pub fn new(name: &str, value: f64, symmetry: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            broken_symmetry: symmetry.to_string(),
            is_ordered: value.abs() > 1e-10,
        }
    }

    /// Magnetization (Ising model)
    pub fn magnetization(value: f64) -> Self {
        Self::new("magnetization", value, "Z2 (spin flip)")
    }

    /// Superfluid order parameter
    pub fn superfluid(value: f64) -> Self {
        Self::new("superfluid_density", value, "U(1) (phase)")
    }

    /// Crystal order parameter
    pub fn crystal(value: f64) -> Self {
        Self::new("crystalline_order", value, "Translation")
    }

    /// Update the value
    pub fn update(&mut self, new_value: f64) {
        self.value = new_value;
        self.is_ordered = new_value.abs() > 1e-10;
    }
}

// ============================================================================
// AVALANCHE STATISTICS
// ============================================================================

/// Statistics of avalanches (for SOC systems)
#[derive(Debug, Clone)]
pub struct AvalancheStatistics {
    /// Size distribution (size → count)
    pub size_distribution: Vec<(usize, usize)>,
    /// Power law exponent (if scale-free)
    pub exponent: Option<f64>,
    /// Maximum observed size
    pub max_size: usize,
    /// Total number of avalanches
    pub total_count: usize,
}

impl AvalancheStatistics {
    /// Create from a list of avalanche sizes
    pub fn from_sizes(sizes: &[usize]) -> Self {
        if sizes.is_empty() {
            return Self {
                size_distribution: Vec::new(),
                exponent: None,
                max_size: 0,
                total_count: 0,
            };
        }

        // Count sizes
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &s in sizes {
            *counts.entry(s).or_insert(0) += 1;
        }

        let mut distribution: Vec<(usize, usize)> = counts.into_iter().collect();
        distribution.sort_by_key(|(s, _)| *s);

        let max_size = *sizes.iter().max().unwrap_or(&0);

        // Estimate power law exponent using maximum likelihood
        let exponent = Self::estimate_exponent(sizes);

        Self {
            size_distribution: distribution,
            exponent,
            max_size,
            total_count: sizes.len(),
        }
    }

    /// Estimate power law exponent using Hill estimator
    fn estimate_exponent(sizes: &[usize]) -> Option<f64> {
        if sizes.len() < 10 {
            return None;
        }

        let mut sorted: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Use top 10% for Hill estimator
        let k = sizes.len() / 10;
        if k < 2 {
            return None;
        }

        let log_ratio_sum: f64 = sorted[..k].iter().map(|&x| (x / sorted[k]).ln()).sum();

        let alpha = 1.0 + k as f64 / log_ratio_sum;

        if alpha > 0.5 && alpha < 5.0 {
            Some(alpha)
        } else {
            None
        }
    }

    /// Is this a power-law distribution?
    pub fn is_power_law(&self) -> bool {
        self.exponent.is_some()
    }

    /// Is this showing signs of SOC?
    pub fn is_soc(&self) -> bool {
        if let Some(exp) = self.exponent {
            // SOC typically has exponents between 1 and 3
            exp > 1.0 && exp < 3.5
        } else {
            false
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        let micro = Scale::MICRO;
        let macro_scale = Scale::MACRO;

        assert_eq!(macro_scale.ratio(&micro), 1e6);

        let coarsened = micro.coarsen(10.0);
        assert_eq!(coarsened.0, 10.0);
    }

    #[test]
    fn test_self_organization() {
        let so = SelfOrganization::new(0.8, 2.5).critical(2.0);

        assert!(so.is_organized());
        assert!(so.is_critical);
        assert_eq!(so.power_law_exponent, Some(2.0));
    }

    #[test]
    fn test_critical_exponents() {
        let mf = CriticalExponents::mean_field();
        let ising = CriticalExponents::ising_2d();

        // Check Rushbrooke inequality
        assert!(mf.check_rushbrooke() >= -1e-10);
        assert!(ising.check_rushbrooke() >= -1e-10);
    }

    #[test]
    fn test_renormalization_group() {
        let mut rg = RenormalizationGroup::new(Scale::MICRO);
        rg.set_coupling("g", 0.1);

        // Simple beta function: asymptotic freedom (coupling decreases)
        rg.step(2.0, |name, g| if name == "g" { -0.1 * g * g } else { 0.0 });

        assert!(rg.scale.0 > Scale::MICRO.0);
        assert!(rg.get_coupling("g").unwrap() < 0.1);
    }

    #[test]
    fn test_ca_irreducibility() {
        let rule30 = check_ca_irreducibility(30, 100);
        let rule0 = check_ca_irreducibility(0, 100);

        assert!(rule30.is_irreducible());
        assert!(!rule0.is_irreducible());
    }

    #[test]
    fn test_phase_transition() {
        let pt = PhaseTransition::new(
            2.269, // Ising Tc
            "temperature",
            TransitionType::SecondOrder,
            "magnetization",
        );

        assert!(pt.is_continuous());

        // Above Tc
        assert!(pt.reduced_parameter(3.0) > 0.0);
        // Below Tc
        assert!(pt.reduced_parameter(2.0) < 0.0);
    }

    #[test]
    fn test_order_parameter() {
        let mut m = OrderParameter::magnetization(0.0);
        assert!(!m.is_ordered);

        m.update(0.5);
        assert!(m.is_ordered);
    }

    #[test]
    fn test_avalanche_statistics() {
        // Power law-ish sizes
        let sizes: Vec<usize> = (1..=100).flat_map(|s| vec![s; (100 / s)]).collect();

        let stats = AvalancheStatistics::from_sizes(&sizes);

        assert!(stats.total_count > 0);
        assert!(stats.exponent.is_some());
    }
}
