# Demetrios v0.50.0 Release Notes

## The 50-Day Milestone

Today marks 50 days of intensive development on Demetrios, and we're proud to
release version 0.50.0 — the first programming language where 15+ million
ontological terms are first-class types.

## What's New

### Types Have Meaning

In Demetrios, types aren't just structural descriptions — they're ontological
assertions about reality:

```demetrios
use ontology::ChEBI

// Aspirin is a TYPE, not just a value
let medication: ChEBI.Aspirin = ChEBI.Aspirin::new()

// The compiler knows Aspirin is-a Drug
fn prescribe(drug: ChEBI.Drug, dose: mg) { ... }
prescribe(medication, 500.0 : mg)  // Compiles
```

### Semantic Distance

Type compatibility is based on semantic distance, not just structural equality:

```demetrios
// Aspirin and OrganicCompound are semantically close (d ~ 0.15)
// So implicit coercion is allowed
fn process(c: ChEBI.OrganicCompound) { ... }
process(aspirin)  // Implicit coercion
```

### Cross-Ontology Interoperability

Different ontologies can work together via SSSOM mappings:

```demetrios
use ontology::{ChEBI, DrugBank}

let chebi_drug: ChEBI.Drug = ChEBI.Aspirin::new()
let db_drug: DrugBank.Drug = chebi_drug as DrugBank.Drug  // Explicit cast
```

### Epistemic Tracking

Track confidence through type conversions:

```demetrios
use epistemic::Knowledge

let measurement: Knowledge<ChEBI.Concentration> = Knowledge::new(
    value: 10.5,
    confidence: 0.95,
)

// Coercion to supertype degrades confidence
let general: Knowledge<ChEBI.ChemicalEntity> = measurement as _
assert(general.confidence < 0.95)  // Confidence reduced by semantic distance
```

### Rich Error Messages

Error messages now explain *why* types are incompatible:

```
error: semantic distance too large: 0.82 > 0.30 threshold
   --> src/main.d:15:12
    |
 12 | fn treat(d: DOID.Disease) {
    |             ------------ expected `DOID.Disease`
    |
 15 |     treat(aspirin)
    |           ^^^^^^^ found `ChEBI.Aspirin`
    |
    = note: semantic distance: 0.823
      (distant: types are semantically unrelated)
    = note: ChEBI.Aspirin is a drug, not a disease
    = help: did you mean to treat a condition?
```

## Performance

| Operation | Latency |
|-----------|---------|
| L1 cache hit (hot 10K terms) | 50 ns |
| L2 cache hit (warm 100K terms) | 5 μs |
| Federated resolution | 100 ms |
| Distance calculation | 10 μs |
| Embedding similarity (SIMD) | 500 ns |
| Bloom filter check | 100 ns |

## Getting Started

```bash
# Install
cargo install demetrios

# Create a project
dc new my-biomedical-app --ontology=ChEBI,DOID

# Check types
dc check src/main.d

# Run
dc run src/main.d

# Search ontologies
dc ontology search "aspirin"
```

## New CLI Commands

```bash
# Ontology operations
dc ontology search <query>           # Search for terms
dc ontology info <CURIE>             # Get term information
dc ontology distance <from> <to>     # Calculate semantic distance
dc ontology similar <term> --k=10    # Find similar terms

# Profiling
dc check src/main.d --profile        # Show compilation profile
dc check src/main.d --profile=json   # Export profile as JSON
```

## What's Next

- **ICBO 2025**: Paper submission in March
- **POPL 2026**: Formal type theory submission in July
- **Community**: Discord server launching soon
- **IDE Support**: VS Code extension in development

## Acknowledgments

This 50-day journey has been one of intense focus and discovery. Demetrios
represents a new paradigm in programming language design: types grounded in
formal ontologies, with semantic meaning as a first-class concern.

*"The purpose of types is meaning, not structure."*

---

Questions? Issues? Contributions?
- GitHub: https://github.com/Chiuratto-AI/demetrios
- Documentation: https://demetrios-lang.org
