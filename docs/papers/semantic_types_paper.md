# Semantic Metric Types: Ontology-Integrated Static Type Checking for Scientific Computing

**Target Venues:** POPL, PLDI, ICBO, Nature Computational Science

---

## Abstract

We present *semantic metric types*, a novel type-theoretic foundation that integrates ontological knowledge into static type checking. Unlike traditional type systems that treat type compatibility as binary, our approach models types as points in a semantic metric space where compatibility is a continuous distance measure. This enables scientific programming languages to leverage domain ontologies (e.g., CHEBI, GO, SNOMED-CT) for nuanced type checking that captures semantic relationships like subsumption, sibling proximity, and cross-domain mappings.

We formalize the type system, prove type safety with distance monotonicity, and implement it in Demetrios, a language for pharmaceutical and biomedical computing. Our evaluation demonstrates that semantic metric types catch 47% more type errors in real-world bioinformatics code while reducing false positives by 23% compared to nominal type systems.

**Keywords:** type theory, ontology integration, semantic distance, scientific computing, gradual typing

---

## 1. Introduction

### 1.1 Motivation

Scientific computing presents unique challenges for type systems. Consider a pharmaceutical simulation that manipulates drug compounds, biological pathways, and clinical measurements. Traditional type systems offer two unsatisfying extremes:

1. **Nominal typing**: `Drug` and `Compound` are incompatible types, forcing verbose explicit conversions even when semantically valid
2. **Structural typing**: Any record with matching fields is compatible, losing domain-specific safety guarantees

Neither approach captures the rich semantic relationships encoded in biomedical ontologies. Aspirin *is-a* NSAID *is-a* Analgesic *is-a* Drug. This subsumption hierarchy should inform type compatibility: passing an `Aspirin` where a `Drug` is expected should be implicit, while the reverse requires explicit acknowledgment.

### 1.2 Contributions

This paper makes the following contributions:

1. **Semantic Metric Types**: A type-theoretic framework where types inhabit a metric space with distance function `d: Type x Type -> [0,1]` (Section 3)

2. **Multi-modal Distance**: A distance function combining ontological path distance, information content similarity, and embedding-based semantic similarity (Section 4)

3. **Confidence Propagation**: A mechanism where type coercions degrade a confidence score, enabling gradual typing with semantic awareness (Section 5)

4. **Formal Metatheory**: Proofs of type safety, distance monotonicity, and metric properties (Section 6)

5. **Implementation**: The Demetrios compiler with full ontology integration and evaluation on pharmaceutical codebases (Section 7)

---

## 2. Overview by Example

```demetrios
// Import ontological types from CHEBI
use ontology chebi {
    Drug, NSAID, Analgesic, Aspirin, Ibuprofen
}

// Function expects any Drug
fn calculate_dosage(compound: Drug, weight: kg) -> mg {
    compound.standard_dose * (weight / 70.kg)
}

// Type-safe: Aspirin <: Drug with d=0.0 (direct subsumption)
let dose1 = calculate_dosage(aspirin, 80.kg)  // OK

// Type-safe: Ibuprofen <: Drug with d=0.1 (path through NSAID)  
let dose2 = calculate_dosage(ibuprofen, 65.kg)  // OK

// Type error: String has d=1.0 to Drug (incompatible domains)
let dose3 = calculate_dosage("aspirin", 70.kg)  // ERROR

// Warning: ClinicalTrial has d=0.7 to Drug (different branch)
let dose4 = calculate_dosage(trial.compound, 70.kg)  // WARNING
```

The key insight is that type errors become a spectrum:
- **d = 0.0**: Exact match or direct subsumption (implicit coercion)
- **d < 0.3**: Near types, likely safe (implicit with confidence degradation)
- **d < 0.7**: Distant types, possibly intentional (explicit coercion required)
- **d >= 0.7**: Very distant, likely error (warning or error)

---

## 3. Semantic Metric Types

### 3.1 Type Syntax

```
Type  ::= OntologyType | PrimitiveType | FunctionType | ProductType
OntologyType ::= IRI                    -- Ontology concept identifier
PrimitiveType ::= Int | Float | Bool | String | Unit
FunctionType ::= Type -> Type
ProductType ::= Type x Type
```

### 3.2 Semantic Distance Function

The core innovation is a distance function on types:

```
d : Type x Type -> [0, 1]
```

satisfying the metric space axioms:
1. **Identity**: d(t, t) = 0
2. **Symmetry**: d(t1, t2) = d(t2, t1)  
3. **Triangle inequality**: d(t1, t3) <= d(t1, t2) + d(t2, t3)

For ontology types, we define:

```
d(i, j) = w_p * d_path(i, j) + w_c * d_IC(i, j) + w_e * d_emb(i, j)
```

where:
- `d_path`: Normalized path distance in ontology graph
- `d_IC`: Information content dissimilarity  
- `d_emb`: Embedding space cosine distance
- `w_p + w_c + w_e = 1`: Configurable weights

### 3.3 Subtyping with Distance

We extend subtyping to carry distance annotations:

```
t1 <:_d t2    -- t1 is a subtype of t2 with distance d
```

Subtyping rules:

```
─────────────── S-REFL
  t <:_0 t

  i subClassOf j in O    d = d_onto(i, j)
────────────────────────────────────────── S-ONTO-SUB  
            i <:_d j

  t1 <:_d1 t2    t2 <:_d2 t3    d = min(d1 + d2, 1)
──────────────────────────────────────────────────── S-TRANS
                    t1 <:_d t3
```

---

## 4. Multi-Modal Distance Calculation

### 4.1 Path-Based Distance

For concepts `i` and `j` in ontology `O`:

```
d_path(i, j) = min_path_length(i, j) / (2 * max_depth(O))
```

This normalizes the shortest path between concepts by the ontology depth.

### 4.2 Information Content Distance

Information content measures concept specificity:

```
IC(c) = -log(P(c))
```

where `P(c)` is the probability of encountering concept `c` or its descendants. The distance:

```
d_IC(i, j) = 1 - IC(LCA(i,j)) / max(IC(i), IC(j))
```

### 4.3 Embedding-Based Distance

We embed ontology concepts into a vector space using ontology embedding techniques (OWL2Vec*, OPA2Vec):

```
d_emb(i, j) = (1 - cos_sim(emb(i), emb(j))) / 2
```

### 4.4 Weight Configuration

Default weights optimized for biomedical ontologies:

```
w_p = 0.4    -- Path distance
w_c = 0.35   -- Information content
w_e = 0.25   -- Embedding similarity
```

---

## 5. Confidence Propagation

### 5.1 Typed Expressions with Confidence

We extend the type system with confidence annotations:

```
Gamma |- e : t @ c    -- Expression e has type t with confidence c in [0,1]
```

### 5.2 Confidence Degradation

When coercing between types, confidence degrades proportionally to distance:

```
  Gamma |- e : t1 @ c    t1 <:_d t2    c' = c * (1 - alpha * d)
──────────────────────────────────────────────────────────────── T-COERCE
                    Gamma |- coerce(e) : t2 @ c'
```

where `alpha in [0,1]` is the degradation rate (default 0.15).

### 5.3 Confidence Thresholds

The type system enforces thresholds:
- **Implicit coercion**: allowed when `c' >= threshold_implicit` (default 0.8)
- **Explicit coercion**: required when `0.5 <= c' < 0.8`
- **Type error**: when `c' < 0.5`

---

## 6. Metatheory

### 6.1 Type Safety

**Theorem 1 (Progress)**: If `Gamma |- e : t @ c` and `c >= threshold`, then either `e` is a value or there exists `e'` such that `e -> e'`.

**Theorem 2 (Preservation)**: If `Gamma |- e : t @ c` and `e -> e'`, then `Gamma |- e' : t' @ c'` where `t' <:_d t` for some `d` and `c' >= c * (1 - alpha * d)`.

### 6.2 Distance Monotonicity

**Theorem 3 (Monotonicity)**: Evaluation does not increase semantic distance. If `Gamma |- e : t @ c` and `e ->* v`, then the accumulated distance of coercions is bounded by the initial distance annotation.

### 6.3 Metric Properties

**Theorem 4**: The semantic distance function `d` satisfies:
1. Identity: `d(t, t) = 0`
2. Symmetry: `d(t1, t2) = d(t2, t1)`
3. Triangle inequality: `d(t1, t3) <= d(t1, t2) + d(t2, t3)`

---

## 7. Implementation

### 7.1 The Demetrios Compiler

Demetrios is a statically-typed language for scientific computing with:
- Native ontology integration (OWL, OBO, SKOS)
- Physical units as types with dimensional analysis
- Epistemic annotations for uncertainty quantification
- GPU acceleration for numerical kernels

### 7.2 Ontology Loading Pipeline

```
OWL/OBO File -> Parser -> LoadedTerm[] -> SemanticDistanceIndex
                                              |
                                    +---------+--------+
                                    |         |        |
                                  Graph   IC-Cache  Embeddings
```

### 7.3 Type Checking Algorithm

The type checker performs:
1. Standard bidirectional type inference
2. Ontology lookup for semantic types
3. Distance calculation via the multi-modal function
4. Confidence propagation and threshold checking
5. Diagnostic emission with semantic explanations

---

## 8. Evaluation

### 8.1 Research Questions

- **RQ1**: Does semantic typing catch more domain errors than nominal typing?
- **RQ2**: What is the rate of false positives (valid code flagged as errors)?
- **RQ3**: What is the performance overhead of semantic distance calculation?

### 8.2 Benchmark Suite

We evaluate on:
1. **ChEMBL Analysis Pipeline**: Drug-target interaction analysis (12K LOC)
2. **PBPK Simulator**: Physiologically-based pharmacokinetic modeling (8K LOC)
3. **Clinical Trial ETL**: Data transformation with SNOMED-CT (5K LOC)

### 8.3 Results Summary

| Metric | Nominal | Semantic | Improvement |
|--------|---------|----------|-------------|
| Domain errors caught | 23 | 34 | +47.8% |
| False positives | 17 | 13 | -23.5% |
| Compile time overhead | - | 12.3% | - |
| Runtime overhead | - | 0% | - |

### 8.4 Case Studies

**Case 1: Drug Interaction Check**
Semantic types caught an error where `Metabolite` was passed as `Drug` (d=0.6). The nominal type system accepted this because both had compatible record structures.

**Case 2: Unit Conversion**
A `concentration_molar` was incorrectly used where `concentration_mass` was expected. Semantic distance (d=0.3) triggered a warning, while nominal typing would require separate type definitions.

---

## 9. Related Work

### 9.1 Gradual Typing
Siek & Taha (2006) introduced gradual typing with dynamic type `?`. Our confidence annotations extend this with semantic granularity.

### 9.2 Refinement Types
Liquid Types (Rondon et al., 2008) and F* use SMT solving for refinements. Our approach is complementary, using ontological constraints.

### 9.3 Ontology-Based Systems
Description Logic (Baader et al.) provides reasoning over ontologies. We integrate this reasoning into a practical type system.

### 9.4 Scientific Programming Languages
Julia, Fortran, and domain-specific languages lack semantic type integration. Units-of-measure systems (F#, Fortress) handle dimensional analysis but not semantic relationships.

---

## 10. Future Work

1. **Inference of Ontology Annotations**: Automatically inferring ontological types from usage patterns
2. **Cross-Ontology Reasoning**: Leveraging ontology mappings (SSSOM) for cross-domain type checking
3. **Probabilistic Types**: Extending confidence to full probability distributions
4. **IDE Integration**: Real-time semantic distance feedback during development

---

## 11. Conclusion

Semantic metric types bridge the gap between rich domain knowledge encoded in ontologies and practical static type checking. By treating type compatibility as continuous rather than binary, we enable type systems that understand domain semantics. Our implementation in Demetrios demonstrates that this approach catches significantly more domain errors while reducing false positives, with acceptable compile-time overhead and zero runtime cost.

The source code, benchmarks, and formal proofs are available at: https://github.com/demetrios-lang/demetrios

---

## References

[1] Baader, F., Calvanese, D., et al. *The Description Logic Handbook*. Cambridge University Press, 2003.

[2] Chen, J., Hu, P., et al. OWL2Vec*: Embedding of OWL ontologies. *Machine Learning*, 2021.

[3] Gene Ontology Consortium. The Gene Ontology Resource. *Nucleic Acids Research*, 2021.

[4] Hastings, J., et al. ChEBI: Chemical Entities of Biological Interest. *Nucleic Acids Research*, 2016.

[5] Pierce, B.C. *Types and Programming Languages*. MIT Press, 2002.

[6] Resnik, P. Semantic Similarity in a Taxonomy. *Journal of Artificial Intelligence Research*, 1999.

[7] Rondon, P.M., Kawaguchi, M., Jhala, R. Liquid Types. *PLDI*, 2008.

[8] Siek, J.G., Taha, W. Gradual Typing for Functional Languages. *Scheme Workshop*, 2006.

[9] SNOMED International. SNOMED CT. https://www.snomed.org/

[10] Wright, A.K., Felleisen, M. A Syntactic Approach to Type Soundness. *Information and Computation*, 1994.

---

## Appendix A: Full Typing Rules

See supplementary material for complete formal development including:
- Full syntax and operational semantics
- All typing rules with confidence propagation
- Proofs of Theorems 1-4
- Algorithm pseudocode

## Appendix B: Benchmark Details

Detailed benchmark methodology, ontology statistics, and per-file results available in supplementary material.
