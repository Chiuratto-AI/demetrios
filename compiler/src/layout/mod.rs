//! Layout Synthesis Module - Day 38
//!
//! Uses semantic distance from the ontology to inform memory layout decisions.
//! The hypothesis: concepts that are semantically close should be physically
//! close in memory to improve cache performance.
//!
//! # Algorithm
//!
//! 1. **Extract** concepts used in HIR (Knowledge[T, ...] types)
//! 2. **Build** distance matrix using ontology hierarchy
//! 3. **Cluster** concepts by semantic proximity + co-occurrence
//! 4. **Generate** layout plan assigning clusters to memory regions
//! 5. **Measure** cache performance to validate the hypothesis
//!
//! # The Hypothesis
//!
//! ```text
//! If concepts A and B are semantically close (low ontology distance),
//! and they are accessed together in code,
//! then placing them physically close in memory will improve cache hit rate.
//! ```
//!
//! Day 38 must validate this hypothesis through measurement.

pub mod cluster;
pub mod distance;
pub mod extract;
pub mod instrument;
pub mod plan;
pub mod report;

use std::collections::HashMap;

pub use cluster::{Cluster, ClusteringResult, cluster_concepts};
pub use distance::DistanceMatrix;
pub use extract::{ConceptUsage, extract_concepts_from_hir, extract_concepts_from_types};
pub use instrument::{CacheInstrumentation, CacheStats, LayoutComparison, compare_layouts};
pub use plan::{LayoutConfig, LayoutPlan, MemoryRegion, generate_layout};
pub use report::generate_report;

use crate::ontology::native::NativeOntology;

/// Main entry point for layout synthesis
pub struct LayoutSynthesizer<'a> {
    /// Reference to the ontology
    ontology: &'a NativeOntology,
    /// Configuration
    config: LayoutConfig,
}

impl<'a> LayoutSynthesizer<'a> {
    /// Create a new layout synthesizer
    pub fn new(ontology: &'a NativeOntology, config: LayoutConfig) -> Self {
        Self { ontology, config }
    }

    /// Synthesize a layout plan from concept usage
    pub fn synthesize(&self, usage: &ConceptUsage) -> LayoutPlan {
        if usage.concepts.is_empty() {
            return LayoutPlan::empty();
        }

        // Build distance matrix
        let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
        let distances = DistanceMatrix::build(&concepts, self.ontology);

        // Cluster by semantic proximity + co-occurrence
        let clustering = cluster_concepts(usage, &distances, self.config.max_clusters);

        // Generate layout plan
        generate_layout(clustering, self.config.clone())
    }

    /// Synthesize and measure cache effectiveness
    pub fn synthesize_and_measure(
        &self,
        usage: &ConceptUsage,
        access_pattern: &[String],
    ) -> (LayoutPlan, LayoutComparison) {
        let plan = self.synthesize(usage);

        // Convert access pattern to concept accesses
        let accesses: Vec<_> = access_pattern
            .iter()
            .filter(|s| usage.concepts.contains(*s))
            .cloned()
            .collect();

        // Measure baseline vs optimized
        let comparison = instrument::compare_layouts(&accesses, &plan, self.config.cache_size);

        (plan, comparison)
    }
}

/// Layout hint for HIR nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutHint {
    /// Allocate on stack (hot data, L1/L2 friendly)
    Stack,
    /// Allocate in bump arena (warm data, L2/L3)
    Arena,
    /// Allocate on heap (cold data, RAM)
    Heap,
}

impl From<MemoryRegion> for LayoutHint {
    fn from(region: MemoryRegion) -> Self {
        match region {
            MemoryRegion::Hot => LayoutHint::Stack,
            MemoryRegion::Warm => LayoutHint::Arena,
            MemoryRegion::Cold => LayoutHint::Heap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_hint_from_region() {
        assert_eq!(LayoutHint::from(MemoryRegion::Hot), LayoutHint::Stack);
        assert_eq!(LayoutHint::from(MemoryRegion::Warm), LayoutHint::Arena);
        assert_eq!(LayoutHint::from(MemoryRegion::Cold), LayoutHint::Heap);
    }
}
