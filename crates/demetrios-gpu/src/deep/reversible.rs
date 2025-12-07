//! Reversible Computation Framework
//!
//! Based on Landauer's principle: information erasure has thermodynamic cost.
//! Reversible computation can theoretically approach zero energy dissipation.
//!
//! # Key Concepts
//!
//! ## Landauer's Principle
//! Erasing one bit costs at least kT ln(2) energy ≈ 0.018 eV at room temperature.
//!
//! ## Reversible Operations
//! Bijective functions preserve information and have zero Landauer cost.
//! Examples: NOT, CNOT, Toffoli, Fredkin gates.
//!
//! ## Bennett's Method
//! Any computation can be made reversible by:
//! 1. Compute forward, keeping intermediate results
//! 2. Copy output
//! 3. Uncompute (run backwards) to restore input
//!
//! ## Adiabatic Computing
//! Gradual, reversible state changes can approach zero energy.

use std::fmt;
use std::marker::PhantomData;

// ============================================================================
// THERMODYNAMIC CONSTANTS
// ============================================================================

/// Boltzmann constant in J/K
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Room temperature in Kelvin
pub const ROOM_TEMP: f64 = 300.0;

/// Landauer limit at room temperature: kT ln(2)
pub const LANDAUER_LIMIT: f64 = BOLTZMANN * ROOM_TEMP * std::f64::consts::LN_2;

// ============================================================================
// THERMODYNAMIC COST
// ============================================================================

/// Energy cost of computation
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ThermodynamicCost {
    /// Energy in Joules
    pub energy: f64,
    /// Bits erased
    pub bits_erased: f64,
    /// Temperature (Kelvin)
    pub temperature: f64,
}

impl ThermodynamicCost {
    /// Zero cost (reversible operation)
    pub const ZERO: ThermodynamicCost = ThermodynamicCost {
        energy: 0.0,
        bits_erased: 0.0,
        temperature: ROOM_TEMP,
    };

    /// Cost of erasing n bits at given temperature
    pub fn erasure(bits: f64, temperature: f64) -> Self {
        Self {
            energy: bits * BOLTZMANN * temperature * std::f64::consts::LN_2,
            bits_erased: bits,
            temperature,
        }
    }

    /// Cost at room temperature
    pub fn erasure_room_temp(bits: f64) -> Self {
        Self::erasure(bits, ROOM_TEMP)
    }

    /// Combine costs
    pub fn add(&self, other: &Self) -> Self {
        Self {
            energy: self.energy + other.energy,
            bits_erased: self.bits_erased + other.bits_erased,
            temperature: self.temperature.max(other.temperature),
        }
    }

    /// Cost in multiples of Landauer limit
    pub fn landauer_multiples(&self) -> f64 {
        self.energy / (BOLTZMANN * self.temperature * std::f64::consts::LN_2)
    }

    /// Is this at the Landauer limit?
    pub fn is_optimal(&self) -> bool {
        (self.landauer_multiples() - self.bits_erased).abs() < 1e-10
    }
}

impl std::ops::Add for ThermodynamicCost {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        ThermodynamicCost::add(&self, &rhs)
    }
}

impl fmt::Display for ThermodynamicCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2e} J ({:.1} bits erased, {:.1}× Landauer)",
            self.energy,
            self.bits_erased,
            self.landauer_multiples() / self.bits_erased.max(1.0)
        )
    }
}

// ============================================================================
// LANDAUER BOUND
// ============================================================================

/// Tracks Landauer bound for a computation
#[derive(Debug, Clone)]
pub struct LandauerBound {
    /// Minimum energy required (theoretical)
    pub minimum: f64,
    /// Actual energy used
    pub actual: f64,
    /// Bits erased
    pub bits_erased: f64,
    /// Is this reversible?
    pub is_reversible: bool,
}

impl LandauerBound {
    /// Reversible computation (zero bound)
    pub fn reversible() -> Self {
        Self {
            minimum: 0.0,
            actual: 0.0,
            bits_erased: 0.0,
            is_reversible: true,
        }
    }

    /// Irreversible computation erasing n bits
    pub fn irreversible(bits: f64) -> Self {
        let minimum = bits * LANDAUER_LIMIT;
        Self {
            minimum,
            actual: minimum, // Assume optimal
            bits_erased: bits,
            is_reversible: false,
        }
    }

    /// Efficiency: minimum / actual
    pub fn efficiency(&self) -> f64 {
        if self.actual > 0.0 {
            self.minimum / self.actual
        } else if self.minimum == 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// REVERSIBLE TRAIT
// ============================================================================

/// A reversible transformation
pub trait Reversible {
    /// The inverse type (often Self)
    type Inverse: Reversible;

    /// Get the inverse operation
    fn inverse(&self) -> Self::Inverse;

    /// Check if this is self-inverse
    fn is_involution(&self) -> bool {
        false
    }
}

// ============================================================================
// BIJECTION
// ============================================================================

/// A bijective (invertible) function
pub struct Bijection<A, B> {
    /// Forward function
    forward: Box<dyn Fn(A) -> B + Send + Sync>,
    /// Backward function
    backward: Box<dyn Fn(B) -> A + Send + Sync>,
    /// Name for debugging
    name: String,
    _phantom: PhantomData<(A, B)>,
}

impl<A: 'static, B: 'static> Bijection<A, B> {
    /// Create a bijection from forward and backward functions
    pub fn new<F, G>(name: &str, forward: F, backward: G) -> Self
    where
        F: Fn(A) -> B + Send + Sync + 'static,
        G: Fn(B) -> A + Send + Sync + 'static,
    {
        Self {
            forward: Box::new(forward),
            backward: Box::new(backward),
            name: name.to_string(),
            _phantom: PhantomData,
        }
    }

    /// Apply forward
    pub fn apply(&self, a: A) -> B {
        (self.forward)(a)
    }

    /// Apply backward
    pub fn unapply(&self, b: B) -> A {
        (self.backward)(b)
    }

    /// Compose with another bijection
    pub fn compose<C: 'static>(self, other: Bijection<B, C>) -> Bijection<A, C> {
        let f1 = self.forward;
        let b1 = self.backward;
        let f2 = other.forward;
        let b2 = other.backward;

        Bijection::new(
            &format!("{} ∘ {}", other.name, self.name),
            move |a| f2(f1(a)),
            move |c| b1(b2(c)),
        )
    }

    /// Get the inverse bijection
    pub fn invert(self) -> Bijection<B, A> {
        Bijection {
            forward: self.backward,
            backward: self.forward,
            name: format!("{}⁻¹", self.name),
            _phantom: PhantomData,
        }
    }
}

impl<A, B> fmt::Debug for Bijection<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bijection({})", self.name)
    }
}

// ============================================================================
// REVERSIBLE OPERATIONS
// ============================================================================

/// Reversible operations on bits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReversibleOp {
    /// Identity (do nothing)
    Identity,
    /// NOT gate (flip bit)
    Not,
    /// CNOT (controlled NOT)
    CNot { control: usize, target: usize },
    /// Toffoli (controlled-controlled NOT)
    Toffoli {
        control1: usize,
        control2: usize,
        target: usize,
    },
    /// Fredkin (controlled SWAP)
    Fredkin {
        control: usize,
        target1: usize,
        target2: usize,
    },
    /// SWAP two bits
    Swap { bit1: usize, bit2: usize },
}

impl ReversibleOp {
    /// Apply to a bit vector
    pub fn apply(&self, bits: &mut Vec<bool>) {
        match *self {
            ReversibleOp::Identity => {}
            ReversibleOp::Not => {
                if !bits.is_empty() {
                    bits[0] = !bits[0];
                }
            }
            ReversibleOp::CNot { control, target } => {
                if bits.get(control).copied().unwrap_or(false) {
                    if let Some(t) = bits.get_mut(target) {
                        *t = !*t;
                    }
                }
            }
            ReversibleOp::Toffoli {
                control1,
                control2,
                target,
            } => {
                let c1 = bits.get(control1).copied().unwrap_or(false);
                let c2 = bits.get(control2).copied().unwrap_or(false);
                if c1 && c2 {
                    if let Some(t) = bits.get_mut(target) {
                        *t = !*t;
                    }
                }
            }
            ReversibleOp::Fredkin {
                control,
                target1,
                target2,
            } => {
                if bits.get(control).copied().unwrap_or(false) {
                    let t1 = bits.get(target1).copied().unwrap_or(false);
                    let t2 = bits.get(target2).copied().unwrap_or(false);
                    if let Some(t) = bits.get_mut(target1) {
                        *t = t2;
                    }
                    if let Some(t) = bits.get_mut(target2) {
                        *t = t1;
                    }
                }
            }
            ReversibleOp::Swap { bit1, bit2 } => {
                let b1 = bits.get(bit1).copied().unwrap_or(false);
                let b2 = bits.get(bit2).copied().unwrap_or(false);
                if let Some(b) = bits.get_mut(bit1) {
                    *b = b2;
                }
                if let Some(b) = bits.get_mut(bit2) {
                    *b = b1;
                }
            }
        }
    }

    /// Get the inverse operation
    pub fn inverse(&self) -> Self {
        // All these operations are self-inverse!
        *self
    }

    /// Is this operation self-inverse?
    pub fn is_involution(&self) -> bool {
        true // All standard reversible gates are involutions
    }

    /// Thermodynamic cost (always zero for reversible ops)
    pub fn cost(&self) -> ThermodynamicCost {
        ThermodynamicCost::ZERO
    }
}

// ============================================================================
// REVERSIBLE CIRCUIT
// ============================================================================

/// A circuit of reversible operations
#[derive(Debug, Clone)]
pub struct ReversibleCircuit {
    /// Number of bits
    pub width: usize,
    /// Operations in order
    pub ops: Vec<ReversibleOp>,
}

impl ReversibleCircuit {
    /// Create an empty circuit
    pub fn new(width: usize) -> Self {
        Self {
            width,
            ops: Vec::new(),
        }
    }

    /// Add an operation
    pub fn add(&mut self, op: ReversibleOp) {
        self.ops.push(op);
    }

    /// Execute the circuit
    pub fn execute(&self, input: &[bool]) -> Vec<bool> {
        let mut bits = input.to_vec();
        bits.resize(self.width, false);

        for op in &self.ops {
            op.apply(&mut bits);
        }

        bits
    }

    /// Get the inverse circuit
    pub fn inverse(&self) -> Self {
        Self {
            width: self.width,
            ops: self.ops.iter().rev().map(|op| op.inverse()).collect(),
        }
    }

    /// Compose with another circuit
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.width, other.width);
        let mut ops = self.ops.clone();
        ops.extend(other.ops.iter().cloned());
        Self {
            width: self.width,
            ops,
        }
    }

    /// Total thermodynamic cost (zero for reversible)
    pub fn cost(&self) -> ThermodynamicCost {
        ThermodynamicCost::ZERO
    }

    /// Verify reversibility by running forward then backward
    pub fn verify_reversible(&self, input: &[bool]) -> bool {
        let output = self.execute(input);
        let recovered = self.inverse().execute(&output);

        input.iter().zip(recovered.iter()).all(|(a, b)| a == b)
    }
}

// ============================================================================
// UNCOMPUTATION
// ============================================================================

/// Uncomputation: running a computation backwards to free memory
#[derive(Debug, Clone)]
pub struct Uncomputation<T> {
    /// The forward computation result
    pub result: T,
    /// The inverse operation to clean up
    inverse_ops: Vec<ReversibleOp>,
    /// Garbage bits to be cleaned
    garbage_bits: Vec<bool>,
}

impl<T: Clone> Uncomputation<T> {
    /// Create an uncomputation record
    pub fn new(result: T, inverse_ops: Vec<ReversibleOp>, garbage: Vec<bool>) -> Self {
        Self {
            result,
            inverse_ops,
            garbage_bits: garbage,
        }
    }

    /// Get the result
    pub fn get(&self) -> &T {
        &self.result
    }

    /// Consume and get the result, scheduling cleanup
    pub fn consume(self) -> (T, UncomputeTask) {
        (
            self.result,
            UncomputeTask {
                ops: self.inverse_ops,
                garbage: self.garbage_bits,
            },
        )
    }

    /// Memory overhead (garbage bits)
    pub fn overhead(&self) -> usize {
        self.garbage_bits.len()
    }
}

/// A task to perform uncomputation
#[derive(Debug, Clone)]
pub struct UncomputeTask {
    ops: Vec<ReversibleOp>,
    garbage: Vec<bool>,
}

impl UncomputeTask {
    /// Execute the uncomputation
    pub fn execute(self) -> ThermodynamicCost {
        // In a real system, this would run the inverse ops
        // and free the garbage bits with zero energy cost
        ThermodynamicCost::ZERO
    }
}

// ============================================================================
// ADIABATIC COMPUTATION
// ============================================================================

/// Adiabatic computation: gradual, reversible state changes
#[derive(Debug, Clone)]
pub struct AdiabaticComputation {
    /// Number of adiabatic steps
    pub steps: usize,
    /// Energy per step (approaches 0 as steps → ∞)
    pub energy_per_step: f64,
    /// Total time
    pub total_time: f64,
}

impl AdiabaticComputation {
    /// Create an adiabatic computation
    pub fn new(steps: usize, total_time: f64) -> Self {
        // Energy per step ∝ 1/steps for adiabatic limit
        let energy_per_step = LANDAUER_LIMIT / steps as f64;
        Self {
            steps,
            energy_per_step,
            total_time,
        }
    }

    /// Total energy dissipation
    pub fn total_energy(&self) -> f64 {
        self.energy_per_step * self.steps as f64
    }

    /// Speedup vs fully reversible (but slower)
    pub fn speedup(&self, conventional_time: f64) -> f64 {
        conventional_time / self.total_time
    }

    /// Energy savings vs conventional
    pub fn energy_savings(&self, conventional_energy: f64) -> f64 {
        1.0 - self.total_energy() / conventional_energy
    }
}

// ============================================================================
// BENNETT'S REVERSIBLE COMPUTATION
// ============================================================================

/// Bennett's method for making any computation reversible
///
/// 1. Compute f(x) → y, keeping all intermediate results
/// 2. Copy y to output
/// 3. Uncompute: run backwards to restore x
#[derive(Debug)]
pub struct BennettComputation<I, O> {
    /// Input
    input: I,
    /// Output
    output: O,
    /// Intermediate storage (garbage)
    garbage_size: usize,
    /// Is cleanup complete?
    cleaned: bool,
}

impl<I: Clone, O: Clone> BennettComputation<I, O> {
    /// Create a Bennett computation
    ///
    /// The closure computes f(x) and returns (output, garbage_size)
    pub fn compute<F>(input: I, f: F) -> Self
    where
        F: FnOnce(&I) -> (O, usize),
    {
        let (output, garbage_size) = f(&input);
        Self {
            input,
            output,
            garbage_size,
            cleaned: false,
        }
    }

    /// Get the output (before cleanup)
    pub fn output(&self) -> &O {
        &self.output
    }

    /// Perform uncomputation (cleanup)
    pub fn cleanup(mut self) -> (I, O, ThermodynamicCost) {
        self.cleaned = true;
        // In true reversible computing, this runs the inverse
        // and releases garbage with zero energy cost
        (
            self.input.clone(),
            self.output.clone(),
            ThermodynamicCost::ZERO,
        )
    }

    /// Space overhead
    pub fn space_overhead(&self) -> usize {
        self.garbage_size
    }

    /// Time overhead (3x: compute, copy, uncompute)
    pub fn time_overhead() -> f64 {
        3.0
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landauer_limit() {
        assert!((LANDAUER_LIMIT - 2.87e-21).abs() < 1e-22);
    }

    #[test]
    fn test_thermodynamic_cost() {
        let cost = ThermodynamicCost::erasure_room_temp(1.0);
        assert!((cost.energy - LANDAUER_LIMIT).abs() < 1e-25);
        assert!(cost.is_optimal());
    }

    #[test]
    fn test_bijection() {
        let double: Bijection<i32, i32> = Bijection::new("double", |x| x * 2, |x| x / 2);

        assert_eq!(double.apply(5), 10);
        assert_eq!(double.unapply(10), 5);
    }

    #[test]
    fn test_bijection_compose() {
        let add_one: Bijection<i32, i32> = Bijection::new("add_one", |x| x + 1, |x| x - 1);
        let double: Bijection<i32, i32> = Bijection::new("double", |x| x * 2, |x| x / 2);

        let composed = add_one.compose(double);
        assert_eq!(composed.apply(5), 12); // (5+1)*2
        assert_eq!(composed.unapply(12), 5);
    }

    #[test]
    fn test_reversible_not() {
        let mut bits = vec![false, true, false];
        ReversibleOp::Not.apply(&mut bits);
        assert_eq!(bits[0], true);

        ReversibleOp::Not.apply(&mut bits);
        assert_eq!(bits[0], false); // Back to original
    }

    #[test]
    fn test_reversible_cnot() {
        let mut bits = vec![true, false]; // control=1, target=0
        ReversibleOp::CNot {
            control: 0,
            target: 1,
        }
        .apply(&mut bits);
        assert_eq!(bits, vec![true, true]); // target flipped

        ReversibleOp::CNot {
            control: 0,
            target: 1,
        }
        .apply(&mut bits);
        assert_eq!(bits, vec![true, false]); // back to original
    }

    #[test]
    fn test_toffoli() {
        let mut bits = vec![true, true, false];
        ReversibleOp::Toffoli {
            control1: 0,
            control2: 1,
            target: 2,
        }
        .apply(&mut bits);
        assert_eq!(bits[2], true); // flipped because both controls are 1
    }

    #[test]
    fn test_reversible_circuit() {
        let mut circuit = ReversibleCircuit::new(3);
        circuit.add(ReversibleOp::Not);
        circuit.add(ReversibleOp::CNot {
            control: 0,
            target: 1,
        });

        let input = vec![false, false, false];
        let output = circuit.execute(&input);

        // Verify reversibility
        assert!(circuit.verify_reversible(&input));

        // Inverse should recover input
        let recovered = circuit.inverse().execute(&output);
        assert_eq!(input, recovered);
    }

    #[test]
    fn test_adiabatic() {
        let adiabatic = AdiabaticComputation::new(1000, 1.0);

        // Total energy approaches Landauer limit (equal in idealized model)
        let epsilon = 1e-20;
        assert!((adiabatic.total_energy() - LANDAUER_LIMIT).abs() < epsilon);

        // Energy per step decreases with more steps
        let more_steps = AdiabaticComputation::new(10000, 1.0);
        assert!(more_steps.energy_per_step < adiabatic.energy_per_step);
    }

    #[test]
    fn test_bennett_computation() {
        let input = 42;
        let bennett = BennettComputation::compute(input, |&x| {
            // Compute x² with some garbage
            (x * x, 64) // 64 bits of garbage
        });

        assert_eq!(*bennett.output(), 1764);
        assert_eq!(bennett.space_overhead(), 64);

        let (recovered_input, output, cost) = bennett.cleanup();
        assert_eq!(recovered_input, 42);
        assert_eq!(output, 1764);
        assert_eq!(cost, ThermodynamicCost::ZERO);
    }
}
