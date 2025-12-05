//! Semantic-Physical Duality: Locality Types and Memory Optimization
//!
//! This module implements the L0 principle that information is physical.
//! By bridging semantic relationships (from ontologies) with physical memory
//! hierarchy, the compiler can make intelligent decisions about data placement
//! and prefetching.
//!
//! # Key Concepts
//!
//! - **Locality Types**: Type-level encoding of memory hierarchy position
//! - **Semantic Prefetch**: Using ontology relationships to predict access patterns
//! - **NUMA Awareness**: Topology detection and semantic-aware placement
//! - **Cache-Line Packing**: Struct field reordering for co-access optimization

pub mod access;
pub mod codegen;
pub mod numa;
pub mod packing;
pub mod prefetch;
pub mod subtyping;
pub mod types;

pub use access::{AccessAnalyzer, AccessPattern, FieldAccess};
pub use codegen::{PrefetchCodegen, PrefetchInstruction};
pub use numa::{NumaNode, NumaTopology, PlacementStrategy, SemanticPlacement};
pub use packing::{CacheLinePacker, FieldGroup, PackedLayout};
pub use prefetch::{PrefetchHint, PrefetchTable, SemanticDistance};
pub use subtyping::{LocalityLattice, SubtypeResult};
pub use types::{Locality, LocalityBound, LocalityConstraint, LocalityParam};

use std::collections::HashMap;

/// A concept from an ontology for locality analysis.
#[derive(Debug, Clone)]
pub struct Concept {
    /// The CURIE identifier (e.g., "CHEBI:15365")
    pub curie: String,
    /// Human-readable label
    pub label: String,
}

impl Concept {
    /// Create a new concept.
    pub fn new(curie: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            curie: curie.into(),
            label: label.into(),
        }
    }

    /// Get the CURIE.
    pub fn curie(&self) -> &str {
        &self.curie
    }
}

/// Trait for ontology access in locality analysis.
/// This provides the minimal interface needed for semantic prefetch generation.
pub trait Ontology {
    /// Search for concepts matching a query.
    fn search(&self, query: &str, limit: usize) -> Vec<Concept>;

    /// Get ancestors of a concept.
    fn ancestors(&self, curie: &str) -> Vec<String>;

    /// Get descendants of a concept.
    fn descendants(&self, curie: &str) -> Vec<String>;

    /// Check if child is a subclass of parent.
    fn is_subclass(&self, child: &str, parent: &str) -> bool;
}

/// Adapter to use NativeOntology with the Ontology trait.
pub struct NativeOntologyAdapter {
    ontology: crate::ontology::native::NativeOntology,
}

impl NativeOntologyAdapter {
    /// Create a new adapter.
    pub fn new(ontology: crate::ontology::native::NativeOntology) -> Self {
        Self { ontology }
    }
}

impl Ontology for NativeOntologyAdapter {
    fn search(&self, query: &str, limit: usize) -> Vec<Concept> {
        self.ontology
            .search(query, limit)
            .into_iter()
            .map(|(curie, label)| Concept::new(curie, label))
            .collect()
    }

    fn ancestors(&self, curie: &str) -> Vec<String> {
        self.ontology
            .get_ancestors(curie)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn descendants(&self, curie: &str) -> Vec<String> {
        // NativeOntology doesn't have a descendants method
        // Return empty for now
        Vec::new()
    }

    fn is_subclass(&self, child: &str, parent: &str) -> bool {
        self.ontology.is_subclass(child, parent)
    }
}

/// The semantic-physical bridge: connects ontology knowledge to memory optimization.
pub struct SemanticPhysicalBridge {
    /// The ontology providing semantic relationships
    ontology: Option<Box<dyn Ontology>>,

    /// Prefetch table generated from semantic analysis
    prefetch_table: PrefetchTable,

    /// NUMA topology of the current system
    numa_topology: NumaTopology,

    /// Access patterns observed during analysis
    access_patterns: HashMap<String, AccessPattern>,

    /// Cache line size (typically 64 bytes)
    cache_line_size: usize,
}

impl SemanticPhysicalBridge {
    /// Create a new bridge with default settings.
    pub fn new() -> Self {
        Self {
            ontology: None,
            prefetch_table: PrefetchTable::new(),
            numa_topology: NumaTopology::detect(),
            access_patterns: HashMap::new(),
            cache_line_size: 64,
        }
    }

    /// Create a bridge with a specific ontology.
    pub fn with_ontology(ontology: Box<dyn Ontology>) -> Self {
        let mut bridge = Self::new();
        bridge.ontology = Some(ontology);
        bridge
    }

    /// Set the cache line size (for testing or non-standard architectures).
    pub fn with_cache_line_size(mut self, size: usize) -> Self {
        self.cache_line_size = size;
        self
    }

    /// Analyze semantic relationships and build prefetch tables.
    pub fn analyze(&mut self) {
        if let Some(ref ont) = self.ontology {
            self.prefetch_table = PrefetchTable::from_ontology(ont.as_ref());
        }
    }

    /// Get prefetch hints for a given type/field access.
    pub fn get_prefetch_hints(&self, type_name: &str, field: &str) -> Vec<PrefetchHint> {
        self.prefetch_table.get_hints(type_name, field)
    }

    /// Get optimal NUMA placement for semantically related data.
    pub fn get_placement(&self, types: &[&str]) -> PlacementStrategy {
        self.numa_topology
            .suggest_placement(types, &self.prefetch_table)
    }

    /// Pack struct fields for cache efficiency.
    pub fn pack_struct(&self, fields: &[(&str, usize, &str)]) -> PackedLayout {
        let packer = CacheLinePacker::new(self.cache_line_size);
        packer.pack(fields, &self.access_patterns)
    }

    /// Record an access pattern for future optimization.
    pub fn record_access(&mut self, pattern: AccessPattern) {
        self.access_patterns.insert(pattern.name.clone(), pattern);
    }

    /// Get the NUMA topology.
    pub fn numa_topology(&self) -> &NumaTopology {
        &self.numa_topology
    }

    /// Get the prefetch table.
    pub fn prefetch_table(&self) -> &PrefetchTable {
        &self.prefetch_table
    }
}

impl Default for SemanticPhysicalBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = SemanticPhysicalBridge::new();
        assert_eq!(bridge.cache_line_size, 64);
    }

    #[test]
    fn test_bridge_with_cache_line_size() {
        let bridge = SemanticPhysicalBridge::new().with_cache_line_size(128);
        assert_eq!(bridge.cache_line_size, 128);
    }
}
