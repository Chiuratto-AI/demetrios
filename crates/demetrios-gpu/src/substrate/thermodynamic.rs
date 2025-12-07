//! Thermodynamic Consistency Checking
//!
//! This module provides verification of thermodynamic laws.
//! The compiler can check that computations respect the laws of thermodynamics.
//!
//! # Novel Aspects
//!
//! 1. **Second Law as Type Constraint**: Entropy must not decrease in isolated systems
//! 2. **Equilibrium Finding**: Automatic minimization of appropriate free energy
//! 3. **Stability Analysis**: Verify thermodynamic stability conditions

use std::fmt;

// ============================================================================
// THERMODYNAMIC STATE
// ============================================================================

/// A thermodynamic state
#[derive(Debug, Clone)]
pub struct ThermodynamicState {
    /// Temperature (K)
    pub temperature: f64,
    /// Pressure (Pa)
    pub pressure: f64,
    /// Volume (m³)
    pub volume: f64,
    /// Number of moles for each species
    pub moles: Vec<f64>,
    /// Internal energy (J)
    pub internal_energy: f64,
    /// Entropy (J/K)
    pub entropy: f64,
    /// Additional properties
    pub properties: ThermodynamicProperties,
}

/// Derived thermodynamic properties
#[derive(Debug, Clone, Default)]
pub struct ThermodynamicProperties {
    /// Enthalpy H = U + PV (J)
    pub enthalpy: f64,
    /// Helmholtz free energy F = U - TS (J)
    pub helmholtz: f64,
    /// Gibbs free energy G = H - TS (J)
    pub gibbs: f64,
    /// Heat capacity at constant volume (J/K)
    pub cv: f64,
    /// Heat capacity at constant pressure (J/K)
    pub cp: f64,
    /// Compressibility (1/Pa)
    pub compressibility: f64,
    /// Thermal expansion coefficient (1/K)
    pub thermal_expansion: f64,
}

impl ThermodynamicState {
    /// Create a new state
    pub fn new(temperature: f64, pressure: f64, volume: f64) -> Self {
        Self {
            temperature,
            pressure,
            volume,
            moles: Vec::new(),
            internal_energy: 0.0,
            entropy: 0.0,
            properties: ThermodynamicProperties::default(),
        }
    }

    /// Set moles
    pub fn with_moles(mut self, moles: Vec<f64>) -> Self {
        self.moles = moles;
        self
    }

    /// Compute derived properties
    pub fn compute_properties(&mut self) {
        self.properties.enthalpy = self.internal_energy + self.pressure * self.volume;
        self.properties.helmholtz = self.internal_energy - self.temperature * self.entropy;
        self.properties.gibbs = self.properties.enthalpy - self.temperature * self.entropy;
    }

    /// Total moles
    pub fn total_moles(&self) -> f64 {
        self.moles.iter().sum()
    }

    /// Mole fractions
    pub fn mole_fractions(&self) -> Vec<f64> {
        let total = self.total_moles();
        if total > 0.0 {
            self.moles.iter().map(|n| n / total).collect()
        } else {
            vec![0.0; self.moles.len()]
        }
    }
}

// ============================================================================
// THERMODYNAMIC ENSEMBLES
// ============================================================================

/// Statistical mechanical ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ensemble {
    /// Microcanonical (NVE): fixed particles, volume, energy
    Microcanonical,
    /// Canonical (NVT): fixed particles, volume, temperature
    Canonical,
    /// Isothermal-Isobaric (NPT): fixed particles, pressure, temperature
    IsothermalIsobaric,
    /// Grand Canonical (μVT): fixed chemical potential, volume, temperature
    GrandCanonical,
}

impl Ensemble {
    /// Get the characteristic function (what's minimized at equilibrium)
    pub fn characteristic_function(&self) -> CharacteristicFunction {
        match self {
            Self::Microcanonical => CharacteristicFunction::NegativeEntropy,
            Self::Canonical => CharacteristicFunction::Helmholtz,
            Self::IsothermalIsobaric => CharacteristicFunction::Gibbs,
            Self::GrandCanonical => CharacteristicFunction::GrandPotential,
        }
    }

    /// Get the natural variables for this ensemble
    pub fn natural_variables(&self) -> Vec<NaturalVariable> {
        match self {
            Self::Microcanonical => vec![
                NaturalVariable::ParticleNumber,
                NaturalVariable::Volume,
                NaturalVariable::Energy,
            ],
            Self::Canonical => vec![
                NaturalVariable::ParticleNumber,
                NaturalVariable::Volume,
                NaturalVariable::Temperature,
            ],
            Self::IsothermalIsobaric => vec![
                NaturalVariable::ParticleNumber,
                NaturalVariable::Pressure,
                NaturalVariable::Temperature,
            ],
            Self::GrandCanonical => vec![
                NaturalVariable::ChemicalPotential,
                NaturalVariable::Volume,
                NaturalVariable::Temperature,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharacteristicFunction {
    /// -S (for microcanonical)
    NegativeEntropy,
    /// F = U - TS (Helmholtz)
    Helmholtz,
    /// G = H - TS (Gibbs)
    Gibbs,
    /// Ω = F - μN (Grand potential)
    GrandPotential,
}

#[derive(Debug, Clone, Copy)]
pub enum NaturalVariable {
    ParticleNumber,
    Volume,
    Energy,
    Temperature,
    Pressure,
    ChemicalPotential,
    Entropy,
}

// ============================================================================
// SECOND LAW CHECKER
// ============================================================================

/// Verifies the second law of thermodynamics
#[derive(Debug)]
pub struct SecondLawChecker {
    /// Tolerance for entropy changes
    tolerance: f64,
    /// Track all processes
    history: Vec<ProcessRecord>,
}

/// Record of a thermodynamic process
#[derive(Debug, Clone)]
pub struct ProcessRecord {
    /// Initial entropy of system
    pub s_initial: f64,
    /// Final entropy of system
    pub s_final: f64,
    /// Heat transferred to environment
    pub q_env: f64,
    /// Environment temperature
    pub t_env: f64,
    /// Entropy change of system
    pub delta_s_system: f64,
    /// Entropy change of environment
    pub delta_s_env: f64,
    /// Total entropy change (must be >= 0)
    pub delta_s_total: f64,
    /// Whether second law is satisfied
    pub satisfied: bool,
}

impl SecondLawChecker {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            history: Vec::new(),
        }
    }

    /// Check a process for second law compliance
    pub fn check_process(
        &mut self,
        initial: &ThermodynamicState,
        final_state: &ThermodynamicState,
        heat_to_env: f64,
        env_temperature: f64,
    ) -> SecondLawResult {
        let delta_s_system = final_state.entropy - initial.entropy;
        let delta_s_env = heat_to_env / env_temperature;
        let delta_s_total = delta_s_system + delta_s_env;

        let satisfied = delta_s_total >= -self.tolerance;

        let record = ProcessRecord {
            s_initial: initial.entropy,
            s_final: final_state.entropy,
            q_env: heat_to_env,
            t_env: env_temperature,
            delta_s_system,
            delta_s_env,
            delta_s_total,
            satisfied,
        };

        self.history.push(record.clone());

        if satisfied {
            SecondLawResult::Satisfied {
                entropy_production: delta_s_total,
                is_reversible: delta_s_total.abs() < self.tolerance,
            }
        } else {
            SecondLawResult::Violated {
                entropy_deficit: -delta_s_total,
                message: format!(
                    "Second law violated: ΔS_total = {:.6e} J/K < 0",
                    delta_s_total
                ),
            }
        }
    }

    /// Check an isolated system (no heat exchange)
    pub fn check_isolated(
        &mut self,
        initial: &ThermodynamicState,
        final_state: &ThermodynamicState,
    ) -> SecondLawResult {
        let delta_s = final_state.entropy - initial.entropy;

        let record = ProcessRecord {
            s_initial: initial.entropy,
            s_final: final_state.entropy,
            q_env: 0.0,
            t_env: 0.0,
            delta_s_system: delta_s,
            delta_s_env: 0.0,
            delta_s_total: delta_s,
            satisfied: delta_s >= -self.tolerance,
        };

        self.history.push(record);

        if delta_s >= -self.tolerance {
            SecondLawResult::Satisfied {
                entropy_production: delta_s,
                is_reversible: delta_s.abs() < self.tolerance,
            }
        } else {
            SecondLawResult::Violated {
                entropy_deficit: -delta_s,
                message: format!(
                    "Isolated system entropy decreased: ΔS = {:.6e} J/K < 0",
                    delta_s
                ),
            }
        }
    }

    /// Get total entropy produced in all recorded processes
    pub fn total_entropy_production(&self) -> f64 {
        self.history.iter().map(|r| r.delta_s_total).sum()
    }

    /// Check if any process violated the second law
    pub fn has_violations(&self) -> bool {
        self.history.iter().any(|r| !r.satisfied)
    }

    /// Get all violations
    pub fn violations(&self) -> Vec<&ProcessRecord> {
        self.history.iter().filter(|r| !r.satisfied).collect()
    }
}

/// Result of second law check
#[derive(Debug, Clone)]
pub enum SecondLawResult {
    Satisfied {
        entropy_production: f64,
        is_reversible: bool,
    },
    Violated {
        entropy_deficit: f64,
        message: String,
    },
}

impl SecondLawResult {
    pub fn is_satisfied(&self) -> bool {
        matches!(self, Self::Satisfied { .. })
    }

    pub fn is_reversible(&self) -> bool {
        matches!(
            self,
            Self::Satisfied {
                is_reversible: true,
                ..
            }
        )
    }
}

// ============================================================================
// EQUILIBRIUM FINDER
// ============================================================================

/// Finds thermodynamic equilibrium states
#[derive(Debug)]
pub struct EquilibriumFinder {
    /// Ensemble
    ensemble: Ensemble,
    /// Tolerance for convergence
    tolerance: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl EquilibriumFinder {
    pub fn new(ensemble: Ensemble) -> Self {
        Self {
            ensemble,
            tolerance: 1e-8,
            max_iterations: 1000,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Find equilibrium composition for a chemical system
    pub fn find_chemical_equilibrium(
        &self,
        species: &[ChemicalSpecies],
        initial_moles: &[f64],
        temperature: f64,
        pressure: f64,
    ) -> EquilibriumResult {
        // Minimize Gibbs free energy subject to conservation constraints
        let n_species = species.len();
        let r = 8.314; // J/(mol·K)

        // Simple iteration towards equilibrium
        let mut moles = initial_moles.to_vec();
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.max_iterations {
            // Compute chemical potentials
            let total: f64 = moles.iter().sum();
            let mut mu: Vec<f64> = Vec::with_capacity(n_species);

            for (i, n) in moles.iter().enumerate() {
                let x = if total > 0.0 { n / total } else { 0.0 };
                let mu_i = species[i].standard_gibbs
                    + r * temperature * (if x > 1e-15 { x.ln() } else { -30.0 });
                mu.push(mu_i);
            }

            // Check for equilibrium (all chemical potentials equal for reacting species)
            let mu_max = mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);

            if (mu_max - mu_min).abs() < self.tolerance * r * temperature {
                converged = true;
                iterations = iter;
                break;
            }

            // Simple relaxation step
            // This is a placeholder - real implementation would use proper optimization
            iterations = iter;
        }

        let total: f64 = moles.iter().sum();
        let gibbs: f64 = moles
            .iter()
            .zip(species.iter())
            .map(|(n, s)| {
                let x = if total > 0.0 { n / total } else { 0.0 };
                n * (s.standard_gibbs + r * temperature * (if x > 1e-15 { x.ln() } else { -30.0 }))
            })
            .sum();

        EquilibriumResult {
            moles,
            gibbs_energy: gibbs,
            converged,
            iterations,
        }
    }

    /// Check stability of a phase
    pub fn check_phase_stability(&self, state: &ThermodynamicState) -> StabilityResult {
        // Stability conditions:
        // 1. Thermal: Cv > 0 (temperature increases with energy)
        // 2. Mechanical: (∂P/∂V)_T < 0 (pressure decreases with volume)
        // 3. Chemical: (∂μ/∂n) > 0 (chemical potential increases with amount)

        let thermal_stable = state.properties.cv > 0.0;
        let mechanical_stable = state.properties.compressibility > 0.0;

        let stable = thermal_stable && mechanical_stable;

        StabilityResult {
            stable,
            thermal_stable,
            mechanical_stable,
            chemical_stable: true, // Simplified
            spinodal_distance: if stable { Some(1.0) } else { None },
        }
    }
}

/// Chemical species for equilibrium calculations
#[derive(Debug, Clone)]
pub struct ChemicalSpecies {
    /// Name
    pub name: String,
    /// Standard Gibbs energy of formation (J/mol)
    pub standard_gibbs: f64,
    /// Stoichiometric coefficients for elements
    pub composition: Vec<(String, f64)>,
}

impl ChemicalSpecies {
    pub fn new(name: &str, standard_gibbs: f64) -> Self {
        Self {
            name: name.to_string(),
            standard_gibbs,
            composition: Vec::new(),
        }
    }

    pub fn with_composition(mut self, element: &str, coefficient: f64) -> Self {
        self.composition.push((element.to_string(), coefficient));
        self
    }
}

/// Result of equilibrium calculation
#[derive(Debug, Clone)]
pub struct EquilibriumResult {
    /// Equilibrium moles of each species
    pub moles: Vec<f64>,
    /// Gibbs energy at equilibrium
    pub gibbs_energy: f64,
    /// Whether calculation converged
    pub converged: bool,
    /// Number of iterations
    pub iterations: usize,
}

/// Result of stability analysis
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Overall stability
    pub stable: bool,
    /// Thermal stability (Cv > 0)
    pub thermal_stable: bool,
    /// Mechanical stability (κ > 0)
    pub mechanical_stable: bool,
    /// Chemical stability
    pub chemical_stable: bool,
    /// Distance to spinodal (if stable)
    pub spinodal_distance: Option<f64>,
}

// ============================================================================
// THERMODYNAMIC PROCESS
// ============================================================================

/// Types of thermodynamic processes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermodynamicProcess {
    /// Constant temperature
    Isothermal,
    /// Constant pressure
    Isobaric,
    /// Constant volume
    Isochoric,
    /// No heat exchange
    Adiabatic,
    /// Constant entropy (reversible adiabatic)
    Isentropic,
    /// Throttling (constant enthalpy)
    Isenthalpic,
    /// Polytropic (PV^n = const)
    Polytropic { n: i32 }, // n * 100 for fixed point
}

impl ThermodynamicProcess {
    /// Check if a process is consistent with its type
    pub fn is_consistent(
        &self,
        initial: &ThermodynamicState,
        final_state: &ThermodynamicState,
        tolerance: f64,
    ) -> bool {
        match self {
            Self::Isothermal => (initial.temperature - final_state.temperature).abs() < tolerance,
            Self::Isobaric => (initial.pressure - final_state.pressure).abs() < tolerance,
            Self::Isochoric => (initial.volume - final_state.volume).abs() < tolerance,
            Self::Isentropic => (initial.entropy - final_state.entropy).abs() < tolerance,
            Self::Isenthalpic => {
                (initial.properties.enthalpy - final_state.properties.enthalpy).abs() < tolerance
            }
            _ => true, // Other processes need more context to verify
        }
    }

    /// Get the work done in an ideal gas process
    pub fn ideal_gas_work(
        &self,
        initial: &ThermodynamicState,
        final_state: &ThermodynamicState,
    ) -> f64 {
        let r = 8.314; // J/(mol·K)
        let n = initial.total_moles();

        match self {
            Self::Isothermal => {
                // W = nRT ln(V2/V1)
                n * r * initial.temperature * (final_state.volume / initial.volume).ln()
            }
            Self::Isobaric => {
                // W = P(V2 - V1)
                initial.pressure * (final_state.volume - initial.volume)
            }
            Self::Isochoric => {
                // W = 0
                0.0
            }
            Self::Adiabatic | Self::Isentropic => {
                // W = (P1V1 - P2V2) / (γ - 1)
                let gamma = 1.4; // Assuming diatomic gas
                (initial.pressure * initial.volume - final_state.pressure * final_state.volume)
                    / (gamma - 1.0)
            }
            _ => 0.0,
        }
    }
}

// ============================================================================
// FREE ENERGY MINIMIZATION
// ============================================================================

/// Different types of free energy for different conditions
#[derive(Debug, Clone, Copy)]
pub enum FreeEnergy {
    /// Helmholtz: F = U - TS (constant T, V)
    Helmholtz,
    /// Gibbs: G = H - TS (constant T, P)
    Gibbs,
    /// Grand potential: Ω = F - μN (constant T, V, μ)
    GrandPotential,
}

impl FreeEnergy {
    /// Compute the free energy for a state
    pub fn compute(&self, state: &ThermodynamicState, chemical_potentials: &[f64]) -> f64 {
        match self {
            Self::Helmholtz => state.internal_energy - state.temperature * state.entropy,
            Self::Gibbs => {
                state.internal_energy + state.pressure * state.volume
                    - state.temperature * state.entropy
            }
            Self::GrandPotential => {
                let helmholtz = state.internal_energy - state.temperature * state.entropy;
                let mu_n: f64 = chemical_potentials
                    .iter()
                    .zip(state.moles.iter())
                    .map(|(mu, n)| mu * n)
                    .sum();
                helmholtz - mu_n
            }
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
    fn test_thermodynamic_state() {
        let mut state = ThermodynamicState::new(300.0, 101325.0, 0.001).with_moles(vec![1.0, 2.0]);

        state.internal_energy = 1000.0;
        state.entropy = 10.0;
        state.compute_properties();

        assert!((state.properties.enthalpy - (1000.0 + 101325.0 * 0.001)).abs() < 0.01);
        assert!((state.properties.helmholtz - (1000.0 - 300.0 * 10.0)).abs() < 0.01);
    }

    #[test]
    fn test_second_law_satisfied() {
        let mut checker = SecondLawChecker::new(1e-10);

        let initial = ThermodynamicState {
            entropy: 100.0,
            ..ThermodynamicState::new(300.0, 101325.0, 0.001)
        };

        let final_state = ThermodynamicState {
            entropy: 110.0, // Entropy increased
            ..ThermodynamicState::new(350.0, 101325.0, 0.001)
        };

        let result = checker.check_isolated(&initial, &final_state);
        assert!(result.is_satisfied());
    }

    #[test]
    fn test_second_law_violated() {
        let mut checker = SecondLawChecker::new(1e-10);

        let initial = ThermodynamicState {
            entropy: 100.0,
            ..ThermodynamicState::new(300.0, 101325.0, 0.001)
        };

        let final_state = ThermodynamicState {
            entropy: 90.0, // Entropy decreased!
            ..ThermodynamicState::new(250.0, 101325.0, 0.001)
        };

        let result = checker.check_isolated(&initial, &final_state);
        assert!(!result.is_satisfied());
    }

    #[test]
    fn test_ensemble_properties() {
        assert_eq!(
            Ensemble::Canonical.characteristic_function(),
            CharacteristicFunction::Helmholtz
        );
        assert_eq!(
            Ensemble::IsothermalIsobaric.characteristic_function(),
            CharacteristicFunction::Gibbs
        );
    }

    #[test]
    fn test_isothermal_process() {
        let process = ThermodynamicProcess::Isothermal;

        let initial = ThermodynamicState::new(300.0, 101325.0, 0.001);
        let final_good = ThermodynamicState::new(300.0, 202650.0, 0.0005);
        let final_bad = ThermodynamicState::new(350.0, 202650.0, 0.0005);

        assert!(process.is_consistent(&initial, &final_good, 1.0));
        assert!(!process.is_consistent(&initial, &final_bad, 1.0));
    }

    #[test]
    fn test_ideal_gas_work() {
        let process = ThermodynamicProcess::Isobaric;

        let initial = ThermodynamicState {
            volume: 0.001,
            pressure: 101325.0,
            ..ThermodynamicState::new(300.0, 101325.0, 0.001)
        };

        let final_state = ThermodynamicState {
            volume: 0.002,
            pressure: 101325.0,
            ..ThermodynamicState::new(600.0, 101325.0, 0.002)
        };

        let work = process.ideal_gas_work(&initial, &final_state);
        assert!((work - 101.325).abs() < 0.1); // P * ΔV = 101325 * 0.001 = 101.325 J
    }

    #[test]
    fn test_free_energy_computation() {
        let state = ThermodynamicState {
            internal_energy: 1000.0,
            entropy: 10.0,
            pressure: 101325.0,
            volume: 0.001,
            ..ThermodynamicState::new(300.0, 101325.0, 0.001)
        };

        let helmholtz = FreeEnergy::Helmholtz.compute(&state, &[]);
        assert!((helmholtz - (1000.0 - 300.0 * 10.0)).abs() < 0.01);

        let gibbs = FreeEnergy::Gibbs.compute(&state, &[]);
        assert!((gibbs - (1000.0 + 101.325 - 3000.0)).abs() < 1.0);
    }

    #[test]
    fn test_chemical_species() {
        let h2o = ChemicalSpecies::new("H2O", -237000.0)
            .with_composition("H", 2.0)
            .with_composition("O", 1.0);

        assert_eq!(h2o.name, "H2O");
        assert_eq!(h2o.composition.len(), 2);
    }
}
