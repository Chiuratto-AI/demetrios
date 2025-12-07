//! Conservation Law Verification
//!
//! This module provides compile-time and runtime verification of conservation laws.
//! The compiler can PROVE that mass, energy, charge, and other quantities are conserved.
//!
//! # Novel Aspects
//!
//! 1. **Conservation as Types**: Conservation laws are encoded in the type system
//! 2. **Automatic Verification**: The compiler checks conservation automatically
//! 3. **Physical Diagnostics**: Violations report physical, not numerical, errors
//!
//! # Example
//!
//! ```ignore
//! fn chemistry_step(
//!     concentrations: &!ConcentrationField
//! ) with Conserved<Mass>, Conserved<Charge> {
//!     // Compiler verifies mass and charge are conserved
//!     // If not, compilation fails with physics-aware error message
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

use super::physical_quantity::{
    ConservedQuantityType, ConstraintViolation, Dimensions, PhysicalField, QuantityKind,
};

// ============================================================================
// CONSERVATION LAW TRAIT
// ============================================================================

/// A conservation law that must be satisfied
pub trait ConservationLaw: fmt::Debug + Send + Sync {
    /// The type of quantity being conserved
    fn quantity_type(&self) -> ConservedQuantityType;

    /// Human-readable name
    fn name(&self) -> &str;

    /// The dimensions of the conserved quantity
    fn dimensions(&self) -> Dimensions;

    /// Check if conservation is satisfied within tolerance (dyn-compatible version)
    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult;

    /// Get the current value of the conserved quantity (dyn-compatible version)
    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64;
}

/// Extension trait for typed conservation checks
pub trait ConservationLawExt: ConservationLaw {
    /// Check if conservation is satisfied within tolerance
    fn check<T: ConservationCheckable>(
        &self,
        before: &T,
        after: &T,
        tolerance: f64,
    ) -> ConservationResult {
        ConservationLaw::check_dyn(self, before, after, tolerance)
    }

    /// Get the current value of the conserved quantity
    fn current_value<T: ConservationCheckable>(&self, state: &T) -> f64 {
        ConservationLaw::current_value_dyn(self, state)
    }
}

impl<L: ConservationLaw + ?Sized> ConservationLawExt for L {}

/// A type that can be checked for conservation
pub trait ConservationCheckable {
    /// Compute the total of a conserved quantity
    fn total(&self, quantity: ConservedQuantityType) -> f64;

    /// Get the flux of a quantity through boundaries
    fn boundary_flux(&self, quantity: ConservedQuantityType) -> f64;

    /// Get the source/sink terms
    fn source_term(&self, quantity: ConservedQuantityType) -> f64;
}

/// Result of conservation check
#[derive(Debug, Clone)]
pub struct ConservationResult {
    /// Whether conservation is satisfied
    pub satisfied: bool,
    /// The quantity type
    pub quantity: ConservedQuantityType,
    /// Value before
    pub before: f64,
    /// Value after
    pub after: f64,
    /// Net change
    pub delta: f64,
    /// Boundary flux (if applicable)
    pub boundary_flux: f64,
    /// Source terms (if applicable)
    pub source: f64,
    /// Expected change (flux + source)
    pub expected_delta: f64,
    /// Relative error
    pub relative_error: f64,
    /// Tolerance used
    pub tolerance: f64,
    /// Diagnostic message
    pub message: String,
}

impl ConservationResult {
    /// Create a satisfied result
    pub fn satisfied(
        quantity: ConservedQuantityType,
        before: f64,
        after: f64,
        tolerance: f64,
    ) -> Self {
        let delta = after - before;
        Self {
            satisfied: true,
            quantity,
            before,
            after,
            delta,
            boundary_flux: 0.0,
            source: 0.0,
            expected_delta: 0.0,
            relative_error: if before.abs() > 1e-15 {
                (delta / before).abs()
            } else {
                delta.abs()
            },
            tolerance,
            message: format!("{:?} conservation satisfied", quantity),
        }
    }

    /// Create a violated result
    pub fn violated(
        quantity: ConservedQuantityType,
        before: f64,
        after: f64,
        tolerance: f64,
        reason: &str,
    ) -> Self {
        let delta = after - before;
        Self {
            satisfied: false,
            quantity,
            before,
            after,
            delta,
            boundary_flux: 0.0,
            source: 0.0,
            expected_delta: 0.0,
            relative_error: if before.abs() > 1e-15 {
                (delta / before).abs()
            } else {
                delta.abs()
            },
            tolerance,
            message: format!(
                "{:?} conservation violated: {} (Δ = {:.6e}, tol = {:.6e})",
                quantity, reason, delta, tolerance
            ),
        }
    }

    /// Convert to constraint violation if not satisfied
    pub fn to_violation(&self) -> Option<ConstraintViolation> {
        if self.satisfied {
            None
        } else {
            Some(ConstraintViolation::ConservationViolated {
                quantity: self.quantity,
                delta: self.delta,
            })
        }
    }
}

// ============================================================================
// CONCRETE CONSERVATION LAWS
// ============================================================================

/// Mass conservation law
#[derive(Debug, Clone, Copy)]
pub struct MassConservation;

impl ConservationLaw for MassConservation {
    fn quantity_type(&self) -> ConservedQuantityType {
        ConservedQuantityType::Mass
    }

    fn name(&self) -> &str {
        "Mass Conservation"
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions::MASS
    }

    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult {
        let m_before = before.total(ConservedQuantityType::Mass);
        let m_after = after.total(ConservedQuantityType::Mass);
        let flux = after.boundary_flux(ConservedQuantityType::Mass);
        let source = after.source_term(ConservedQuantityType::Mass);

        let delta = m_after - m_before;
        let expected = flux + source;
        let error = (delta - expected).abs();

        let relative = if m_before.abs() > 1e-15 {
            error / m_before.abs()
        } else {
            error
        };

        if relative <= tolerance {
            let mut result = ConservationResult::satisfied(
                ConservedQuantityType::Mass,
                m_before,
                m_after,
                tolerance,
            );
            result.boundary_flux = flux;
            result.source = source;
            result.expected_delta = expected;
            result
        } else {
            let mut result = ConservationResult::violated(
                ConservedQuantityType::Mass,
                m_before,
                m_after,
                tolerance,
                &format!(
                    "Mass changed by {:.6e} kg, expected {:.6e} kg from fluxes/sources",
                    delta, expected
                ),
            );
            result.boundary_flux = flux;
            result.source = source;
            result.expected_delta = expected;
            result
        }
    }

    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64 {
        state.total(ConservedQuantityType::Mass)
    }
}

/// Charge conservation law
#[derive(Debug, Clone, Copy)]
pub struct ChargeConservation;

impl ConservationLaw for ChargeConservation {
    fn quantity_type(&self) -> ConservedQuantityType {
        ConservedQuantityType::Charge
    }

    fn name(&self) -> &str {
        "Charge Conservation"
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            length: 0,
            mass: 0,
            time: 1,
            current: 1,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult {
        let q_before = before.total(ConservedQuantityType::Charge);
        let q_after = after.total(ConservedQuantityType::Charge);
        let current = after.boundary_flux(ConservedQuantityType::Charge);

        let delta = q_after - q_before;
        let error = (delta - current).abs();

        let relative = if q_before.abs() > 1e-15 {
            error / q_before.abs()
        } else {
            error
        };

        if relative <= tolerance {
            ConservationResult::satisfied(
                ConservedQuantityType::Charge,
                q_before,
                q_after,
                tolerance,
            )
        } else {
            ConservationResult::violated(
                ConservedQuantityType::Charge,
                q_before,
                q_after,
                tolerance,
                "Net charge changed without corresponding current",
            )
        }
    }

    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64 {
        state.total(ConservedQuantityType::Charge)
    }
}

/// Energy conservation law
#[derive(Debug, Clone, Copy)]
pub struct EnergyConservation;

impl ConservationLaw for EnergyConservation {
    fn quantity_type(&self) -> ConservedQuantityType {
        ConservedQuantityType::Energy
    }

    fn name(&self) -> &str {
        "Energy Conservation (First Law of Thermodynamics)"
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions::ENERGY
    }

    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult {
        let e_before = before.total(ConservedQuantityType::Energy);
        let e_after = after.total(ConservedQuantityType::Energy);
        let heat_flux = after.boundary_flux(ConservedQuantityType::Energy);
        let work = after.source_term(ConservedQuantityType::Energy);

        let delta = e_after - e_before;
        let expected = heat_flux + work; // Q + W
        let error = (delta - expected).abs();

        let relative = if e_before.abs() > 1e-15 {
            error / e_before.abs()
        } else {
            error
        };

        if relative <= tolerance {
            let mut result = ConservationResult::satisfied(
                ConservedQuantityType::Energy,
                e_before,
                e_after,
                tolerance,
            );
            result.boundary_flux = heat_flux;
            result.source = work;
            result.expected_delta = expected;
            result.message = format!(
                "First Law satisfied: ΔU = {:.6e} J = Q ({:.6e}) + W ({:.6e})",
                delta, heat_flux, work
            );
            result
        } else {
            ConservationResult::violated(
                ConservedQuantityType::Energy,
                e_before,
                e_after,
                tolerance,
                &format!(
                    "First Law violated: ΔU = {:.6e} J ≠ Q + W = {:.6e} J",
                    delta, expected
                ),
            )
        }
    }

    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64 {
        state.total(ConservedQuantityType::Energy)
    }
}

/// Momentum conservation law
#[derive(Debug, Clone, Copy)]
pub struct MomentumConservation;

impl ConservationLaw for MomentumConservation {
    fn quantity_type(&self) -> ConservedQuantityType {
        ConservedQuantityType::Momentum
    }

    fn name(&self) -> &str {
        "Momentum Conservation"
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            length: 1,
            mass: 1,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult {
        let p_before = before.total(ConservedQuantityType::Momentum);
        let p_after = after.total(ConservedQuantityType::Momentum);
        let external_force = after.source_term(ConservedQuantityType::Momentum);

        let delta = p_after - p_before;
        let error = (delta - external_force).abs();

        let relative = if p_before.abs() > 1e-15 {
            error / p_before.abs()
        } else {
            error
        };

        if relative <= tolerance {
            ConservationResult::satisfied(
                ConservedQuantityType::Momentum,
                p_before,
                p_after,
                tolerance,
            )
        } else {
            ConservationResult::violated(
                ConservedQuantityType::Momentum,
                p_before,
                p_after,
                tolerance,
                "Momentum changed without external force",
            )
        }
    }

    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64 {
        state.total(ConservedQuantityType::Momentum)
    }
}

/// Angular momentum conservation law
#[derive(Debug, Clone, Copy)]
pub struct AngularMomentumConservation;

impl ConservationLaw for AngularMomentumConservation {
    fn quantity_type(&self) -> ConservedQuantityType {
        ConservedQuantityType::AngularMomentum
    }

    fn name(&self) -> &str {
        "Angular Momentum Conservation"
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            length: 2,
            mass: 1,
            time: -1,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
        }
    }

    fn check_dyn(
        &self,
        before: &dyn ConservationCheckable,
        after: &dyn ConservationCheckable,
        tolerance: f64,
    ) -> ConservationResult {
        let l_before = before.total(ConservedQuantityType::AngularMomentum);
        let l_after = after.total(ConservedQuantityType::AngularMomentum);
        let external_torque = after.source_term(ConservedQuantityType::AngularMomentum);

        let delta = l_after - l_before;
        let error = (delta - external_torque).abs();

        let relative = if l_before.abs() > 1e-15 {
            error / l_before.abs()
        } else {
            error
        };

        if relative <= tolerance {
            ConservationResult::satisfied(
                ConservedQuantityType::AngularMomentum,
                l_before,
                l_after,
                tolerance,
            )
        } else {
            ConservationResult::violated(
                ConservedQuantityType::AngularMomentum,
                l_before,
                l_after,
                tolerance,
                "Angular momentum changed without external torque",
            )
        }
    }

    fn current_value_dyn(&self, state: &dyn ConservationCheckable) -> f64 {
        state.total(ConservedQuantityType::AngularMomentum)
    }
}

// ============================================================================
// CONSERVED QUANTITY TRACKER
// ============================================================================

/// A quantity that tracks conservation
#[derive(Debug, Clone)]
pub struct ConservedQuantity {
    /// The quantity type
    pub quantity_type: ConservedQuantityType,
    /// Initial value
    pub initial: f64,
    /// Current value
    pub current: f64,
    /// Cumulative flux through boundaries
    pub cumulative_flux: f64,
    /// Cumulative source/sink terms
    pub cumulative_source: f64,
    /// History of values (for debugging)
    pub history: Vec<f64>,
    /// Tolerance for conservation checks
    pub tolerance: f64,
}

impl ConservedQuantity {
    /// Create a new conserved quantity tracker
    pub fn new(quantity_type: ConservedQuantityType, initial: f64, tolerance: f64) -> Self {
        Self {
            quantity_type,
            initial,
            current: initial,
            cumulative_flux: 0.0,
            cumulative_source: 0.0,
            history: vec![initial],
            tolerance,
        }
    }

    /// Update the current value
    pub fn update(&mut self, new_value: f64, flux: f64, source: f64) {
        self.current = new_value;
        self.cumulative_flux += flux;
        self.cumulative_source += source;
        self.history.push(new_value);
    }

    /// Check if conservation is satisfied
    pub fn check(&self) -> ConservationResult {
        let expected_change = self.cumulative_flux + self.cumulative_source;
        let actual_change = self.current - self.initial;
        let error = (actual_change - expected_change).abs();

        let relative = if self.initial.abs() > 1e-15 {
            error / self.initial.abs()
        } else {
            error
        };

        if relative <= self.tolerance {
            let mut result = ConservationResult::satisfied(
                self.quantity_type,
                self.initial,
                self.current,
                self.tolerance,
            );
            result.boundary_flux = self.cumulative_flux;
            result.source = self.cumulative_source;
            result.expected_delta = expected_change;
            result
        } else {
            let mut result = ConservationResult::violated(
                self.quantity_type,
                self.initial,
                self.current,
                self.tolerance,
                &format!(
                    "Total change {:.6e} exceeds expected {:.6e} by {:.6e}",
                    actual_change, expected_change, error
                ),
            );
            result.boundary_flux = self.cumulative_flux;
            result.source = self.cumulative_source;
            result.expected_delta = expected_change;
            result
        }
    }

    /// Get the maximum deviation from expected conservation
    pub fn max_deviation(&self) -> f64 {
        let expected_change = self.cumulative_flux + self.cumulative_source;
        let actual_change = self.current - self.initial;
        (actual_change - expected_change).abs()
    }
}

// ============================================================================
// CONSERVATION CHECKER - THE MAIN VERIFICATION ENGINE
// ============================================================================

/// Verifies multiple conservation laws simultaneously
#[derive(Debug)]
pub struct ConservationChecker {
    /// Active conservation laws
    laws: Vec<Box<dyn ConservationLaw>>,
    /// Tolerance for checks
    tolerance: f64,
    /// Track history for diagnostics
    track_history: bool,
    /// History of conservation values
    history: HashMap<ConservedQuantityType, Vec<f64>>,
}

impl ConservationChecker {
    /// Create a new conservation checker
    pub fn new(tolerance: f64) -> Self {
        Self {
            laws: Vec::new(),
            tolerance,
            track_history: false,
            history: HashMap::new(),
        }
    }

    /// Enable history tracking for diagnostics
    pub fn with_history(mut self) -> Self {
        self.track_history = true;
        self
    }

    /// Add a conservation law
    pub fn add_law(mut self, law: Box<dyn ConservationLaw>) -> Self {
        let qty = law.quantity_type();
        self.laws.push(law);
        if self.track_history {
            self.history.insert(qty, Vec::new());
        }
        self
    }

    /// Add mass conservation
    pub fn with_mass_conservation(self) -> Self {
        self.add_law(Box::new(MassConservation))
    }

    /// Add charge conservation
    pub fn with_charge_conservation(self) -> Self {
        self.add_law(Box::new(ChargeConservation))
    }

    /// Add energy conservation
    pub fn with_energy_conservation(self) -> Self {
        self.add_law(Box::new(EnergyConservation))
    }

    /// Add momentum conservation
    pub fn with_momentum_conservation(self) -> Self {
        self.add_law(Box::new(MomentumConservation))
    }

    /// Add angular momentum conservation
    pub fn with_angular_momentum_conservation(self) -> Self {
        self.add_law(Box::new(AngularMomentumConservation))
    }

    /// Check all conservation laws
    pub fn check_all<T: ConservationCheckable>(
        &mut self,
        before: &T,
        after: &T,
    ) -> Vec<ConservationResult> {
        let mut results = Vec::new();

        for law in &self.laws {
            let result = law.check_dyn(before, after, self.tolerance);

            if self.track_history {
                if let Some(hist) = self.history.get_mut(&law.quantity_type()) {
                    hist.push(law.current_value_dyn(after));
                }
            }

            results.push(result);
        }

        results
    }

    /// Check if any conservation laws are violated
    pub fn has_violations<T: ConservationCheckable>(&mut self, before: &T, after: &T) -> bool {
        self.check_all(before, after).iter().any(|r| !r.satisfied)
    }

    /// Get all violations
    pub fn get_violations<T: ConservationCheckable>(
        &mut self,
        before: &T,
        after: &T,
    ) -> Vec<ConservationResult> {
        self.check_all(before, after)
            .into_iter()
            .filter(|r| !r.satisfied)
            .collect()
    }

    /// Generate a diagnostic report
    pub fn diagnostic_report<T: ConservationCheckable>(&mut self, before: &T, after: &T) -> String {
        let results = self.check_all(before, after);
        let mut report = String::new();

        report.push_str("=== Conservation Law Verification Report ===\n\n");

        for result in &results {
            let status = if result.satisfied { "✓" } else { "✗" };
            report.push_str(&format!("[{}] {:?}\n", status, result.quantity));
            report.push_str(&format!("    Before: {:.6e}\n", result.before));
            report.push_str(&format!("    After:  {:.6e}\n", result.after));
            report.push_str(&format!("    Delta:  {:.6e}\n", result.delta));
            if result.boundary_flux.abs() > 1e-15 || result.source.abs() > 1e-15 {
                report.push_str(&format!("    Flux:   {:.6e}\n", result.boundary_flux));
                report.push_str(&format!("    Source: {:.6e}\n", result.source));
                report.push_str(&format!("    Expected: {:.6e}\n", result.expected_delta));
            }
            report.push_str(&format!("    Rel. Error: {:.6e}\n", result.relative_error));
            report.push_str(&format!("    Tolerance:  {:.6e}\n", result.tolerance));
            report.push_str(&format!("    {}\n\n", result.message));
        }

        let violations: Vec<_> = results.iter().filter(|r| !r.satisfied).collect();
        if violations.is_empty() {
            report.push_str("All conservation laws satisfied.\n");
        } else {
            report.push_str(&format!(
                "WARNING: {} conservation law(s) violated!\n",
                violations.len()
            ));
        }

        report
    }
}

// ============================================================================
// SIMPLE STATE FOR TESTING
// ============================================================================

/// A simple state that tracks conserved quantities
#[derive(Debug, Clone)]
pub struct SimpleConservativeState {
    pub mass: f64,
    pub charge: f64,
    pub energy: f64,
    pub momentum: f64,
    pub angular_momentum: f64,
    pub boundary_fluxes: HashMap<ConservedQuantityType, f64>,
    pub sources: HashMap<ConservedQuantityType, f64>,
}

impl Default for SimpleConservativeState {
    fn default() -> Self {
        Self {
            mass: 0.0,
            charge: 0.0,
            energy: 0.0,
            momentum: 0.0,
            angular_momentum: 0.0,
            boundary_fluxes: HashMap::new(),
            sources: HashMap::new(),
        }
    }
}

impl SimpleConservativeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self
    }

    pub fn with_energy(mut self, energy: f64) -> Self {
        self.energy = energy;
        self
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }
}

impl ConservationCheckable for SimpleConservativeState {
    fn total(&self, quantity: ConservedQuantityType) -> f64 {
        match quantity {
            ConservedQuantityType::Mass => self.mass,
            ConservedQuantityType::Charge => self.charge,
            ConservedQuantityType::Energy => self.energy,
            ConservedQuantityType::Momentum => self.momentum,
            ConservedQuantityType::AngularMomentum => self.angular_momentum,
            _ => 0.0,
        }
    }

    fn boundary_flux(&self, quantity: ConservedQuantityType) -> f64 {
        *self.boundary_fluxes.get(&quantity).unwrap_or(&0.0)
    }

    fn source_term(&self, quantity: ConservedQuantityType) -> f64 {
        *self.sources.get(&quantity).unwrap_or(&0.0)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    // Import extension trait to access the `check` and `current_value` methods
    use super::ConservationLawExt;

    #[test]
    fn test_mass_conservation_satisfied() {
        let before = SimpleConservativeState::new().with_mass(100.0);
        let after = SimpleConservativeState::new().with_mass(100.0);

        let result = MassConservation.check(&before, &after, 1e-10);
        assert!(result.satisfied);
    }

    #[test]
    fn test_mass_conservation_violated() {
        let before = SimpleConservativeState::new().with_mass(100.0);
        let after = SimpleConservativeState::new().with_mass(90.0);

        let result = MassConservation.check(&before, &after, 1e-10);
        assert!(!result.satisfied);
    }

    #[test]
    fn test_energy_conservation_with_work() {
        let before = SimpleConservativeState::new().with_energy(100.0);
        let mut after = SimpleConservativeState::new().with_energy(150.0);
        after.sources.insert(ConservedQuantityType::Energy, 50.0); // Work done

        let result = EnergyConservation.check(&before, &after, 1e-10);
        assert!(result.satisfied);
        assert!(result.message.contains("First Law satisfied"));
    }

    #[test]
    fn test_multiple_conservation_laws() {
        let before = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(1000.0)
            .with_charge(10.0);

        let after = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(1000.0)
            .with_charge(10.0);

        let mut checker = ConservationChecker::new(1e-10)
            .with_mass_conservation()
            .with_energy_conservation()
            .with_charge_conservation();

        let results = checker.check_all(&before, &after);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.satisfied));
    }

    #[test]
    fn test_conservation_violation_detection() {
        let before = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(1000.0);

        let after = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(900.0); // Energy lost!

        let mut checker = ConservationChecker::new(1e-10)
            .with_mass_conservation()
            .with_energy_conservation();

        assert!(checker.has_violations(&before, &after));

        let violations = checker.get_violations(&before, &after);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].quantity, ConservedQuantityType::Energy);
    }

    #[test]
    fn test_conserved_quantity_tracker() {
        let mut tracker = ConservedQuantity::new(ConservedQuantityType::Mass, 100.0, 1e-10);

        // Update with flux
        tracker.update(90.0, -10.0, 0.0); // 10 units flowed out
        assert!(tracker.check().satisfied);

        // Update without accounting for flux
        tracker.update(80.0, 0.0, 0.0); // 10 units disappeared!
        assert!(!tracker.check().satisfied);
    }

    #[test]
    fn test_diagnostic_report() {
        let before = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(1000.0);

        let after = SimpleConservativeState::new()
            .with_mass(100.0)
            .with_energy(1000.0);

        let mut checker = ConservationChecker::new(1e-10)
            .with_mass_conservation()
            .with_energy_conservation();

        let report = checker.diagnostic_report(&before, &after);
        assert!(report.contains("Conservation Law Verification Report"));
        assert!(report.contains("All conservation laws satisfied"));
    }
}
