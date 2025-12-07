# Substrate-Aware Epistemic Computing

**Status:** Implemented  
**Location:** `crates/demetrios-gpu/src/substrate/`  
**Tests:** 61 passing

---

## Overview

Substrate-Aware Epistemic Computing is a novel GPU programming paradigm that encodes physical meaning directly into the type system and execution model. Unlike traditional GPU languages that treat computation as abstract numeric operations, Demetrios understands the *physical substrate* of computation.

## The Core Insight

Scientific computing is fundamentally about physical reality:
- **Chemistry:** Electrons minimize energy on potential surfaces
- **Biology:** Systems reach thermodynamic equilibrium  
- **Physics:** Particles follow least-action paths
- **Materials:** Crystals relax to ground states

Demetrios encodes this insight at the language level.

---

## Three Pillars

### 1. Semantic Substrate Types

Types encode what physical quantity they represent, not just their computational representation.

```rust
// A ChemicalPotential isn't just f64 - it carries thermodynamic semantics
pub struct PhysicalQuantity<T = f64> {
    pub value: T,
    pub dimensions: Dimensions,      // SI dimensional analysis
    pub kind: QuantityKind,          // Semantic meaning (Energy, Entropy, etc.)
    pub constraints: Vec<PhysicalConstraint>,
    pub uncertainty: Option<T>,
}
```

**Key Types:**
- `Dimensions` - SI base units with compile-time arithmetic
- `QuantityKind` - 30+ physical quantity types (Energy, Entropy, ChemicalPotential, etc.)
- `PhysicalConstraint` - Domain constraints (NonNegative, Bounded, Conserved)
- `SubstrateType` - Full semantic type with extensivity and transformation rules

### 2. Epistemic Execution Model

Computation guided by what we *know*, not just what we compute.

```rust
pub struct Epistemic<T> {
    pub value: T,
    pub confidence: Confidence,       // [0.0, 1.0] certainty
    pub provenance: Provenance,       // Source and transformation history
    pub validity: TemporalValidity,   // When is this knowledge valid?
    pub epistemic_std: f64,           // Reducible uncertainty
    pub aleatoric_std: f64,           // Irreducible uncertainty
}
```

**Key Features:**
- **Uncertainty Propagation:** Automatic error propagation through computations
- **Adaptive Precision:** High uncertainty → use f32; high confidence → use f64
- **Confidence Decay:** Knowledge validity decreases over time/transformations
- **Provenance Tracking:** Full history of data sources and transformations

### 3. Physical Memory Topology

Memory layout mirrors physical reality. Elements near in physical space are near in memory.

```rust
// Space-filling curves for cache-efficient physical layouts
pub trait SpaceFillingCurve {
    fn encode(&self, coords: &[u32]) -> u64;
    fn decode(&self, index: u64) -> Vec<u32>;
}

// O(1) neighbor queries for particle simulations
pub struct CellList<T> {
    cells: Vec<Vec<usize>>,
    items: Vec<T>,
    positions: Vec<Coord3D>,
    // ...
}
```

**Key Features:**
- **Morton (Z-order) Curves:** Simple, fast encoding for 2D/3D
- **Hilbert Curves:** Better locality preservation for stencil operations
- **Cell Lists:** O(1) neighbor queries within cutoff radius
- **Physical Arrays:** Automatic coordinate-to-index mapping

---

## Module Structure

```
substrate/
├── mod.rs                 # Module root and re-exports
├── physical_quantity.rs   # Dimensional analysis, semantic types (~700 lines)
├── conservation.rs        # Conservation law verification (~900 lines)
├── epistemic.rs           # Uncertainty-guided computation (~650 lines)
├── topology.rs            # Space-filling curves, cell lists (~800 lines)
├── symmetry.rs            # Lie groups, equivariance (~600 lines)
├── variational.rs         # Unified optimization framework (~600 lines)
└── thermodynamic.rs       # Second Law, equilibrium (~550 lines)
```

---

## Conservation Laws

The system can verify physical conservation laws at runtime:

```rust
pub trait ConservationLaw {
    fn quantity_type(&self) -> ConservedQuantityType;
    fn check_dyn(&self, before: &dyn ConservationCheckable, 
                 after: &dyn ConservationCheckable, 
                 tolerance: f64) -> ConservationResult;
}

// Built-in laws
pub struct MassConservation;
pub struct ChargeConservation;
pub struct EnergyConservation;      // First Law of Thermodynamics
pub struct MomentumConservation;
pub struct AngularMomentumConservation;
```

**Usage:**
```rust
let checker = ConservationChecker::new(1e-10)
    .with_mass_conservation()
    .with_energy_conservation();

let violations = checker.check_all(&before_state, &after_state);
```

---

## Symmetry and Lie Groups

First-class support for continuous symmetries:

```rust
pub trait LieGroup {
    const DIMENSION: usize;
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;
    fn exp(tangent: &[f64]) -> Self;  // Exponential map
    fn log(&self) -> Vec<f64>;         // Logarithmic map
}

// Implemented groups
pub struct SO3;   // 3D rotations
pub struct SE3;   // 3D rigid motions
pub struct U1;    // Phase/gauge transformations
pub struct SU2;   // Spin transformations
pub struct SU3;   // Color charge (QCD)
```

**Equivariance Checking:**
```rust
pub trait Equivariant<G: LieGroup> {
    fn transform(&self, g: &G) -> Self;
    fn is_equivariant(&self, g: &G, tolerance: f64) -> bool;
}
```

---

## Variational Framework

Unified optimization across physics, chemistry, and ML:

```rust
pub trait VariationalPrinciple {
    type State: Clone;
    type ActionValue: Into<f64>;
    
    fn action(&self, state: &Self::State) -> Self::ActionValue;
    fn variation(&self, state: &Self::State) -> Self::State;
    fn find_stationary(&self, initial: Self::State) -> StationaryResult<Self::State>;
}

// Built-in principles
pub struct HamiltonPrinciple<L>;  // Classical mechanics
pub struct RayleighRitz;          // Quantum mechanics, eigenproblems
pub struct MaxEntropy;            // Statistical mechanics
pub struct GibbsMinimization;     // Chemical equilibrium
pub struct EmpiricalRisk;         // Machine learning
```

---

## Thermodynamic Consistency

Automatic verification of thermodynamic laws:

```rust
pub struct SecondLawChecker {
    tolerance: f64,
    history: Vec<ProcessRecord>,
}

impl SecondLawChecker {
    // Verify ΔS_universe >= 0
    pub fn check_process(&mut self, process: &ThermodynamicProcess) -> SecondLawResult;
}

pub struct EquilibriumFinder {
    // Find equilibrium by minimizing appropriate free energy
    pub fn find_equilibrium(&self, initial: ThermodynamicState) -> EquilibriumResult;
}
```

---

## Integration with Compiler

The substrate module complements the compiler's type system:

| Compiler (compile-time) | Substrate (runtime) |
|------------------------|---------------------|
| `types/units.rs` - Dimensional analysis | `physical_quantity.rs` - Runtime quantities |
| `types/epistemic.rs` - Knowledge types | `epistemic.rs` - Uncertainty propagation |
| `effects/` - Effect tracking | `conservation.rs` - Conservation verification |

---

## Example: Molecular Dynamics

```rust
use demetrios_gpu::substrate::*;

// Create physical space with periodic boundaries
let space = Space3D::new(
    Coord3D::new(0.0, 0.0, 0.0),
    Coord3D::new(10.0, 10.0, 10.0),
);

// Build cell list for O(1) neighbor queries
let mut cell_list = CellList::new(space.clone(), 2.5); // 2.5 Å cutoff
for (i, pos) in positions.iter().enumerate() {
    cell_list.insert(atoms[i].clone(), *pos);
}

// Compute forces with neighbor list
for i in 0..n_atoms {
    for j in cell_list.neighbors_of(i) {
        let r = positions[j] - positions[i];
        let force = lennard_jones(r);
        forces[i] += force;
    }
}

// Verify energy conservation
let checker = ConservationChecker::new(1e-6)
    .with_energy_conservation();
let result = checker.check_all(&state_before, &state_after);
assert!(result.iter().all(|r| r.satisfied));
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Morton encode/decode | O(1) | Bit interleaving |
| Hilbert encode/decode | O(log n) | Better locality |
| Cell list build | O(n) | Linear in particles |
| Neighbor query | O(k) | k = neighbors within cutoff |
| Conservation check | O(1) | Per quantity |

---

## Future Work

1. **GPU Kernels:** Generate optimized CUDA/PTX from substrate types
2. **Automatic Differentiation:** Integrate with variational framework
3. **Parallel Cell Lists:** GPU-accelerated spatial partitioning
4. **Compile-Time Conservation:** Static verification where possible
5. **Domain-Specific Optimizations:** PDE type → optimal stencil layout

---

## References

1. Morton, G.M. (1966). "A Computer Oriented Geodetic Data Base"
2. Hilbert, D. (1891). "Über die stetige Abbildung einer Linie auf ein Flächenstück"
3. Verlet, L. (1967). "Computer Experiments on Classical Fluids"
4. Allen, M.P. & Tildesley, D.J. (2017). "Computer Simulation of Liquids"
