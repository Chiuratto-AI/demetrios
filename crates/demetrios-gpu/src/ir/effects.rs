//! GPU Effect Annotations
//!
//! Integrates Demetrios algebraic effects with GPU operations.
//! Maps GPU operations to effect signatures for safety checking.

use std::collections::HashSet;
use std::fmt;

/// GPU-specific effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuEffect {
    /// Reading from global memory
    GlobalRead,
    /// Writing to global memory
    GlobalWrite,
    /// Reading from shared memory
    SharedRead,
    /// Writing to shared memory
    SharedWrite,
    /// Atomic operations
    Atomic,
    /// Thread synchronization (barriers)
    Sync,
    /// Warp-level operations
    Warp,
    /// Texture sampling
    Texture,
    /// Surface read/write
    Surface,
    /// Memory allocation
    Alloc,
    /// Non-termination (loops that may not terminate)
    Diverge,
}

impl GpuEffect {
    /// Check if this effect involves memory mutation
    pub fn is_mutation(self) -> bool {
        matches!(
            self,
            GpuEffect::GlobalWrite
                | GpuEffect::SharedWrite
                | GpuEffect::Atomic
                | GpuEffect::Surface
        )
    }

    /// Check if this effect requires synchronization
    pub fn requires_sync(self) -> bool {
        matches!(
            self,
            GpuEffect::SharedWrite | GpuEffect::Atomic | GpuEffect::Sync
        )
    }

    /// Get a human-readable description
    pub fn description(self) -> &'static str {
        match self {
            GpuEffect::GlobalRead => "reads from global memory",
            GpuEffect::GlobalWrite => "writes to global memory",
            GpuEffect::SharedRead => "reads from shared memory",
            GpuEffect::SharedWrite => "writes to shared memory",
            GpuEffect::Atomic => "performs atomic operations",
            GpuEffect::Sync => "synchronizes threads",
            GpuEffect::Warp => "uses warp-level primitives",
            GpuEffect::Texture => "samples textures",
            GpuEffect::Surface => "accesses surfaces",
            GpuEffect::Alloc => "allocates memory",
            GpuEffect::Diverge => "may not terminate",
        }
    }
}

impl fmt::Display for GpuEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuEffect::GlobalRead => write!(f, "GlobalRead"),
            GpuEffect::GlobalWrite => write!(f, "GlobalWrite"),
            GpuEffect::SharedRead => write!(f, "SharedRead"),
            GpuEffect::SharedWrite => write!(f, "SharedWrite"),
            GpuEffect::Atomic => write!(f, "Atomic"),
            GpuEffect::Sync => write!(f, "Sync"),
            GpuEffect::Warp => write!(f, "Warp"),
            GpuEffect::Texture => write!(f, "Texture"),
            GpuEffect::Surface => write!(f, "Surface"),
            GpuEffect::Alloc => write!(f, "Alloc"),
            GpuEffect::Diverge => write!(f, "Diverge"),
        }
    }
}

/// Effect set for a GPU operation or function
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectSet {
    effects: HashSet<GpuEffect>,
}

impl EffectSet {
    /// Create an empty effect set
    pub fn new() -> Self {
        EffectSet {
            effects: HashSet::new(),
        }
    }

    /// Create a pure effect set (no effects)
    pub fn pure() -> Self {
        Self::new()
    }

    /// Create an effect set with a single effect
    pub fn single(effect: GpuEffect) -> Self {
        let mut set = Self::new();
        set.add(effect);
        set
    }

    /// Add an effect
    pub fn add(&mut self, effect: GpuEffect) {
        self.effects.insert(effect);
    }

    /// Add multiple effects
    pub fn add_all(&mut self, effects: impl IntoIterator<Item = GpuEffect>) {
        self.effects.extend(effects);
    }

    /// Check if an effect is present
    pub fn has(&self, effect: GpuEffect) -> bool {
        self.effects.contains(&effect)
    }

    /// Check if the set is pure (no effects)
    pub fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }

    /// Check if any effect involves mutation
    pub fn has_mutation(&self) -> bool {
        self.effects.iter().any(|e| e.is_mutation())
    }

    /// Check if any effect requires synchronization
    pub fn requires_sync(&self) -> bool {
        self.effects.iter().any(|e| e.requires_sync())
    }

    /// Union with another effect set
    pub fn union(&self, other: &EffectSet) -> EffectSet {
        EffectSet {
            effects: self.effects.union(&other.effects).copied().collect(),
        }
    }

    /// Intersection with another effect set
    pub fn intersection(&self, other: &EffectSet) -> EffectSet {
        EffectSet {
            effects: self.effects.intersection(&other.effects).copied().collect(),
        }
    }

    /// Check if this set is a subset of another
    pub fn is_subset(&self, other: &EffectSet) -> bool {
        self.effects.is_subset(&other.effects)
    }

    /// Iterate over effects
    pub fn iter(&self) -> impl Iterator<Item = GpuEffect> + '_ {
        self.effects.iter().copied()
    }

    /// Get the number of effects
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "pure");
        }

        let effects: Vec<_> = self.effects.iter().map(|e| e.to_string()).collect();
        write!(f, "{}", effects.join(", "))
    }
}

impl FromIterator<GpuEffect> for EffectSet {
    fn from_iter<I: IntoIterator<Item = GpuEffect>>(iter: I) -> Self {
        EffectSet {
            effects: iter.into_iter().collect(),
        }
    }
}

/// Effect requirements for GPU operations
#[derive(Debug, Clone)]
pub struct EffectRequirements {
    /// Declared effects (what the function claims)
    pub declared: EffectSet,
    /// Actual effects (computed from analysis)
    pub actual: EffectSet,
}

impl EffectRequirements {
    pub fn new(declared: EffectSet) -> Self {
        EffectRequirements {
            declared,
            actual: EffectSet::new(),
        }
    }

    /// Check if the actual effects satisfy the declared effects
    pub fn is_satisfied(&self) -> bool {
        self.actual.is_subset(&self.declared)
    }

    /// Get missing effects (actual but not declared)
    pub fn missing(&self) -> EffectSet {
        EffectSet {
            effects: self
                .actual
                .effects
                .difference(&self.declared.effects)
                .copied()
                .collect(),
        }
    }
}

/// Commonly used effect sets
pub mod common {
    use super::*;

    /// Pure computation (no effects)
    pub fn pure() -> EffectSet {
        EffectSet::pure()
    }

    /// Read-only global memory access
    pub fn global_read() -> EffectSet {
        EffectSet::single(GpuEffect::GlobalRead)
    }

    /// Read-write global memory access
    pub fn global_rw() -> EffectSet {
        [GpuEffect::GlobalRead, GpuEffect::GlobalWrite]
            .into_iter()
            .collect()
    }

    /// Shared memory access (read-write with sync)
    pub fn shared_rw() -> EffectSet {
        [
            GpuEffect::SharedRead,
            GpuEffect::SharedWrite,
            GpuEffect::Sync,
        ]
        .into_iter()
        .collect()
    }

    /// Atomic operations
    pub fn atomic() -> EffectSet {
        [GpuEffect::GlobalRead, GpuEffect::Atomic]
            .into_iter()
            .collect()
    }

    /// Full effectful computation
    pub fn full() -> EffectSet {
        [
            GpuEffect::GlobalRead,
            GpuEffect::GlobalWrite,
            GpuEffect::SharedRead,
            GpuEffect::SharedWrite,
            GpuEffect::Atomic,
            GpuEffect::Sync,
        ]
        .into_iter()
        .collect()
    }
}

/// Effect analysis for GPU IR
pub mod analysis {
    use super::*;
    use crate::ir::inst::{AtomicOp, Instruction};
    use crate::ir::types::AddressSpace;

    /// Compute the effects of a single instruction
    pub fn instruction_effects(inst: &Instruction) -> EffectSet {
        let mut effects = EffectSet::new();

        match inst {
            // Load from memory
            Instruction::Load { space, .. } => {
                match space {
                    AddressSpace::Global => effects.add(GpuEffect::GlobalRead),
                    AddressSpace::Shared => effects.add(GpuEffect::SharedRead),
                    AddressSpace::Constant => {} // Pure
                    AddressSpace::Local => {}    // Thread-local, no effect
                    AddressSpace::Texture => effects.add(GpuEffect::Texture),
                    AddressSpace::Generic => {
                        // Conservative: could be any
                        effects.add(GpuEffect::GlobalRead);
                        effects.add(GpuEffect::SharedRead);
                    }
                }
            }

            // Store to memory
            Instruction::Store { space, .. } => {
                match space {
                    AddressSpace::Global => effects.add(GpuEffect::GlobalWrite),
                    AddressSpace::Shared => effects.add(GpuEffect::SharedWrite),
                    AddressSpace::Local => {}    // Thread-local
                    AddressSpace::Constant => {} // Should be error
                    AddressSpace::Texture => {}  // Should be error
                    AddressSpace::Generic => {
                        effects.add(GpuEffect::GlobalWrite);
                        effects.add(GpuEffect::SharedWrite);
                    }
                }
            }

            // Atomic operations
            Instruction::Atomic { .. } | Instruction::AtomicCAS { .. } => {
                effects.add(GpuEffect::Atomic);
                effects.add(GpuEffect::GlobalRead);
            }

            // Synchronization
            Instruction::Barrier { .. } | Instruction::MemFence { .. } => {
                effects.add(GpuEffect::Sync);
            }

            // Warp operations
            Instruction::WarpShuffle { .. }
            | Instruction::WarpVote { .. }
            | Instruction::WarpReduce { .. } => {
                effects.add(GpuEffect::Warp);
            }

            // Shared memory allocation
            Instruction::SharedAlloc { .. } => {
                effects.add(GpuEffect::Alloc);
            }

            // Function calls need special handling
            Instruction::Call { .. } => {
                // Conservative: assume full effects
                // TODO: Use function effect signatures
                effects.add(GpuEffect::GlobalRead);
                effects.add(GpuEffect::GlobalWrite);
            }

            // Pure operations (no effects)
            Instruction::Const { .. }
            | Instruction::BinOp { .. }
            | Instruction::UnaryOp { .. }
            | Instruction::Cmp { .. }
            | Instruction::Convert { .. }
            | Instruction::Bitcast { .. }
            | Instruction::GetElementPtr { .. }
            | Instruction::Branch { .. }
            | Instruction::CondBranch { .. }
            | Instruction::Return { .. }
            | Instruction::ThreadIdx { .. }
            | Instruction::BlockIdx { .. }
            | Instruction::BlockDim { .. }
            | Instruction::GridDim { .. }
            | Instruction::WarpId { .. }
            | Instruction::LaneId { .. }
            | Instruction::Select { .. }
            | Instruction::Phi { .. }
            | Instruction::FMA { .. } => {}
        }

        effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_set_operations() {
        let mut set1 = EffectSet::new();
        set1.add(GpuEffect::GlobalRead);
        set1.add(GpuEffect::GlobalWrite);

        let set2 = EffectSet::single(GpuEffect::GlobalRead);

        assert!(set2.is_subset(&set1));
        assert!(!set1.is_subset(&set2));

        let union = set1.union(&set2);
        assert_eq!(union.len(), 2);

        let intersection = set1.intersection(&set2);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.has(GpuEffect::GlobalRead));
    }

    #[test]
    fn test_effect_requirements() {
        let declared = common::global_rw();
        let mut req = EffectRequirements::new(declared);

        req.actual.add(GpuEffect::GlobalRead);
        assert!(req.is_satisfied());

        req.actual.add(GpuEffect::Atomic);
        assert!(!req.is_satisfied());

        let missing = req.missing();
        assert!(missing.has(GpuEffect::Atomic));
    }

    #[test]
    fn test_effect_mutation() {
        assert!(GpuEffect::GlobalWrite.is_mutation());
        assert!(!GpuEffect::GlobalRead.is_mutation());
        assert!(GpuEffect::Atomic.is_mutation());
    }
}
