# Semantic Metric Types

**Research Area:** Type Theory, Ontology Integration, Scientific Computing

## Overview

Semantic metric types are a novel type-theoretic foundation that integrates ontological knowledge into static type checking. Unlike traditional type systems that treat type compatibility as binary (compatible or not), our approach models types as points in a semantic metric space where compatibility is a continuous distance measure in [0, 1].

## Key Concepts

### 1. Types as Points in a Metric Space

Traditional type systems use nominal or structural equivalence:
- `Drug` and `Compound` are either the same type or completely different

Semantic metric types introduce *distance*:
- `d(Aspirin, NSAID) = 0.1` (close, direct subsumption)
- `d(Aspirin, Morphine) = 0.4` (moderate, different branch of analgesics)
- `d(Aspirin, Disease) = 0.95` (far, different domains)

### 2. Multi-Modal Distance Function

The semantic distance combines three components:

```
d(i, j) = w_p * d_path(i, j) + w_c * d_IC(i, j) + w_e * d_emb(i, j)
```

Where:
- **d_path**: Normalized path distance in ontology graph
- **d_IC**: Information content dissimilarity (Resnik-style)
- **d_emb**: Embedding space cosine distance

Default weights: `w_p = 0.4, w_c = 0.35, w_e = 0.25`

### 3. Confidence Propagation

When coercing between types, confidence degrades proportionally to distance:

```
c' = c * (1 - α * d)
```

Where `α` is the degradation rate (default 0.15).

This enables:
- **Implicit coercion**: `c' >= 0.8` (close types)
- **Explicit coercion**: `0.5 <= c' < 0.8` (moderate distance)
- **Type error**: `c' < 0.5` (too distant)

## Implementation in Demetrios

### Ontology Loading

```demetrios
use ontology chebi {
    Drug, NSAID, Analgesic, Aspirin, Ibuprofen
}
```

The compiler loads ontology terms from OWL/OBO files and builds:
1. Graph structure for path-based distance
2. Information content cache
3. Embedding space with ANN index

### Type Checking with Distance

```demetrios
fn calculate_dosage(compound: Drug, weight: kg) -> mg {
    compound.standard_dose * (weight / 70.kg)
}

// OK: Aspirin <: Drug with d=0.2 (implicit coercion)
let dose1 = calculate_dosage(aspirin, 80.kg)

// Warning: ClinicalTrial has d=0.7 to Drug
let dose2 = calculate_dosage(trial.compound, 70.kg)
```

### Semantic Distance Index

The `SemanticDistanceIndex` provides O(1) distance lookups:

```rust
let mut index = SemanticDistanceIndex::new();
index.build_from_terms(&ontology_terms);

let distance = index.distance(&aspirin_iri, &drug_iri);
// SemanticDistance { conceptual: 0.2, path: 3, ... }
```

## Theoretical Foundation

### Type Safety Theorem

**Theorem (Progress)**: If `Γ ⊢ e : τ @ c` and `c >= threshold`, then either `e` is a value or there exists `e'` such that `e → e'`.

**Theorem (Preservation)**: If `Γ ⊢ e : τ @ c` and `e → e'`, then `Γ ⊢ e' : τ' @ c'` where `τ' <:_d τ` and `c' >= c * (1 - α * d)`.

### Distance Monotonicity

**Theorem**: Evaluation does not increase semantic distance. The accumulated distance through coercions is bounded.

### Metric Properties

The distance function satisfies:
1. **Identity**: `d(t, t) = 0`
2. **Symmetry**: `d(t₁, t₂) = d(t₂, t₁)`
3. **Triangle inequality**: `d(t₁, t₃) <= d(t₁, t₂) + d(t₂, t₃)`

## Applications

### 1. Pharmaceutical Computing

Type-safe drug interactions with semantic awareness:
- Catch metabolite/drug confusion
- Validate clinical trial data types
- Ensure unit compatibility with semantic context

### 2. Bioinformatics Pipelines

Gene ontology integration:
- Biological process type checking
- Molecular function compatibility
- Cellular component validation

### 3. Clinical Data ETL

SNOMED-CT integration:
- Diagnosis code validation
- Procedure type checking
- Finding/observation compatibility

## Benchmarks

Performance characteristics (see `benches/ontology_bench.rs`):

| Operation | Time |
|-----------|------|
| Index build (100 terms) | ~500μs |
| Index build (1000 terms) | ~5ms |
| Distance query | ~100ns |
| k-NN search (k=10) | ~1μs |

## Related Work

- **Gradual Typing** (Siek & Taha, 2006): Binary gradual types → our continuous confidence
- **Liquid Types** (Rondon et al., 2008): SMT refinements → our ontological constraints
- **Description Logic** (Baader et al.): TBox reasoning → our type subsumption

## Future Directions

1. **Cross-ontology reasoning**: SSSOM mapping integration
2. **Inference of ontology annotations**: Learn types from usage patterns
3. **Probabilistic types**: Full probability distributions instead of point confidence
4. **IDE integration**: Real-time distance visualization

## References

1. Resnik, P. "Semantic Similarity in a Taxonomy." JAIR, 1999.
2. Chen, J. et al. "OWL2Vec*: Embedding of OWL ontologies." Machine Learning, 2021.
3. Gene Ontology Consortium. "The Gene Ontology Resource." NAR, 2021.
4. Hastings, J. et al. "ChEBI: Chemical Entities of Biological Interest." NAR, 2016.

## See Also

- [Formal specification](../../spec/formal/semantic_types.tex)
- [Academic paper](../papers/semantic_types_paper.md)
- [Integration tests](../../compiler/tests/integration_ontology_e2e.rs)
- [Benchmarks](../../compiler/benches/ontology_bench.rs)
