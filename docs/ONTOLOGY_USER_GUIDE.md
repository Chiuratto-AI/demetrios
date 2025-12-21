# Demetrios Ontology Integration - User Guide

## Overview

Demetrios is the **first programming language** where 15+ million ontological terms serve as **first-class types**. This guide shows how to use biomedical ontologies (CHEBI, GO, SNOMED, FHIR, ICD-10, etc.) directly in your code.

## Quick Start

### 1. Enable Ontology Features

Add to your `Cargo.toml`:
```toml
[dependencies]
demetrios = { version = "0.78", features = ["ontology", "network"] }
```

Features:
- `ontology`: SQLite-backed domain ontologies (ChEBI, GO, etc.)
- `network`: BioPortal API access for federated resolution
- `ontology-build`: OWL/RDF parsing tools

### 2. Initialize the Ontology Resolver

```rust
use demetrios::ontology::{OntologyResolver, ResolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the resolver
    let config = ResolverConfig::default()
        .with_data_dir("./ontology_data");  // Local cache

    let mut resolver = OntologyResolver::new(config)?;

    Ok(())
}
```

### 3. Resolve a Term

```rust
use demetrios::ontology::OntologyResolver;

let mut resolver = OntologyResolver::default_resolver()?;

// Resolve aspirin (CHEBI:15365)
let aspirin = resolver.resolve("CHEBI:15365")?;

println!("Term: {}", aspirin.curie);
println!("Label: {}", aspirin.label.unwrap_or_default());
println!("Definition: {}", aspirin.definition.unwrap_or_default());
println!("Layer: {:?}", aspirin.layer);

// Output:
// Term: CHEBI:15365
// Label: aspirin
// Definition: A member of the class of benzoic acids...
// Layer: Domain
```

## 4-Layer Resolution Architecture

Demetrios resolves terms through 4 layers, from fastest to slowest:

```text
┌─────────────────────────────────────────────────────────────┐
│ L1: Primitive (BFO, RO, COB) - Compiled in, instant         │
│     ~850 terms, HashMap lookup                              │
├─────────────────────────────────────────────────────────────┤
│ L2: Foundation (PATO, UO, IAO, FHIR, Schema.org)           │
│     ~8,000 terms, file-based, <1ms                          │
├─────────────────────────────────────────────────────────────┤
│ L3: Domain (ChEBI, GO, DOID, HP, MONDO, etc.)              │
│     ~500,000 terms, SQLite, 10-100ms                        │
├─────────────────────────────────────────────────────────────┤
│ L4: Federated (BioPortal, OLS4) - Network, 100-1000ms      │
│     ~15 million terms, HTTP API with caching                │
└─────────────────────────────────────────────────────────────┘
```

### Resolution Examples by Layer

#### L1: Primitive Ontologies (BFO, RO, COB)
```rust
// Resolve BFO:0000001 (entity)
let entity = resolver.resolve("BFO:0000001")?;
assert_eq!(entity.label, Some("entity".to_string()));
assert_eq!(entity.layer, OntologyLayer::Primitive);

// Resolve RO:0002162 (in taxon)
let in_taxon = resolver.resolve("RO:0002162")?;
assert_eq!(entity.layer, OntologyLayer::Primitive);
```

#### L2: Foundation Ontologies (FHIR, PATO, UO, etc.)
```rust
// FHIR R5 resources
let patient = resolver.resolve("FHIR:Patient")?;
let observation = resolver.resolve("FHIR:Observation")?;
let medication = resolver.resolve("FHIR:Medication")?;

// Units of Measurement (UO)
let gram = resolver.resolve("UO:0000021")?;
let liter = resolver.resolve("UO:0000099")?;

// Phenotypic Quality (PATO)
let size = resolver.resolve("PATO:0000117")?;
let color = resolver.resolve("PATO:0000014")?;
```

#### L3: Domain Ontologies (ChEBI, GO, DOID, HP, MONDO)
```rust
// Chemical Entities (ChEBI)
let aspirin = resolver.resolve("CHEBI:15365")?;
let ibuprofen = resolver.resolve("CHEBI:5855")?;
let glucose = resolver.resolve("CHEBI:17234")?;

// Gene Ontology (GO)
let biological_process = resolver.resolve("GO:0008150")?;
let cellular_process = resolver.resolve("GO:0009987")?;
let metabolism = resolver.resolve("GO:0008152")?;

// Disease Ontology (DOID)
let cancer = resolver.resolve("DOID:162")?;
let diabetes = resolver.resolve("DOID:9351")?;

// Human Phenotype Ontology (HP)
let fever = resolver.resolve("HP:0001945")?;
let seizure = resolver.resolve("HP:0001250")?;

// Mondo Disease Ontology
let parkinsons = resolver.resolve("MONDO:0005180")?;
```

#### L4: Federated (BioPortal API)
```rust
// Requires BIOPORTAL_API_KEY environment variable
let config = ResolverConfig::default()
    .with_data_dir("./ontology_data");

let mut resolver = OntologyResolver::new(config)?;

// Any ontology term not in L1-L3
let rare_term = resolver.resolve("NCIT:C12345")?;  // NCI Thesaurus
let mesh_term = resolver.resolve("MESH:D001249")?;  // Medical Subject Headings
```

## Subsumption Checking (Is-A Hierarchy)

### Check if a term is a subclass of another

```rust
use demetrios::ontology::SubsumptionResult;

// Is aspirin a drug?
let result = resolver.is_subclass_of("CHEBI:15365", "CHEBI:23888")?;
assert!(matches!(result, SubsumptionResult::IsSubclass));

// Is aspirin a biological_process? (No)
let result = resolver.is_subclass_of("CHEBI:15365", "GO:0008150")?;
assert!(matches!(result, SubsumptionResult::NotSubclass));

// Same term (equivalent)
let result = resolver.is_subclass_of("CHEBI:15365", "CHEBI:15365")?;
assert!(matches!(result, SubsumptionResult::Equivalent));
```

### Get All Ancestors

```rust
let ancestors = resolver.get_ancestors("CHEBI:15365")?;

for ancestor in ancestors {
    println!("Aspirin is-a {}", ancestor);
}

// Output (simplified):
// Aspirin is-a CHEBI:35472  (anti-inflammatory agent)
// Aspirin is-a CHEBI:35623  (antipyretic)
// Aspirin is-a CHEBI:23888  (drug)
// Aspirin is-a CHEBI:24431  (chemical entity)
// ...
```

## Cross-Ontology Translation (SSSOM Mappings)

### Load Mappings

```rust
// Load SSSOM mapping file
resolver.load_mappings("mappings/chebi_to_fhir.sssom.tsv")?;
```

### Translate Between Ontologies

```rust
// Translate ChEBI term to FHIR
if let Some(fhir_code) = resolver.translate("CHEBI:15365", "FHIR")? {
    println!("CHEBI:15365 (aspirin) maps to FHIR:{}", fhir_code);
}

// Translate SNOMED to ICD-10
resolver.load_mappings("mappings/snomed_to_icd10.sssom.tsv")?;
if let Some(icd10) = resolver.translate("SNOMED:386661006", "ICD10")? {
    println!("SNOMED:386661006 (fever) maps to ICD-10:{}", icd10);
}
```

## Offline vs. Online Mode

### Offline Mode (No Network)
```rust
let config = ResolverConfig::default()
    .offline();  // Disables L4 federated queries

let mut resolver = OntologyResolver::new(config)?;

// Only resolves L1-L3 terms
let aspirin = resolver.resolve("CHEBI:15365")?;  // OK (L3)
let rare = resolver.resolve("RARE:123456");       // Error (not in L1-L3)
```

### Online Mode (with BioPortal)
```rust
// Set API key
std::env::set_var("BIOPORTAL_API_KEY", "your-key-here");

let config = ResolverConfig::default();  // Federated enabled by default
let mut resolver = OntologyResolver::new(config)?;

// Can resolve any term in BioPortal
let term = resolver.resolve("NCIT:C12345")?;  // NCI Thesaurus via API
```

## Loading Local OBO Files

```rust
use demetrios::ontology::loader::{OntologyLoader, OntologyLoaderConfig};

let config = OntologyLoaderConfig::default();
let loader = OntologyLoader::new(config)?;

// Load ChEBI from local OBO file
let count = loader.load_obo_file(
    std::path::Path::new("./ontology_files/chebi.obo"),
    OntologyId::ChEBI
)?;

println!("Loaded {} ChEBI terms", count);

// Now resolve locally
let aspirin = loader.resolve_curie("CHEBI:15365")?;
```

## Caching and Performance

### Cache Configuration

```rust
use demetrios::ontology::cache::CacheConfig;

let cache_config = CacheConfig::default()
    .with_max_entries(100_000)        // Total cache capacity
    .with_ttl_seconds(3600)           // 1 hour TTL
    .with_negative_caching(true);     // Cache "not found" results

let config = ResolverConfig {
    cache: cache_config,
    ..Default::default()
};

let mut resolver = OntologyResolver::new(config)?;
```

### Cache Statistics

```rust
// After some resolutions...
let stats = resolver.cache_stats();

println!("Total entries: {}", stats.total_hits());
println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
println!("Hot cache hits: {}", stats.hot_hits);
println!("Warm cache hits: {}", stats.warm_hits);
println!("Cold cache hits: {}", stats.cold_hits);
println!("Cache misses: {}", stats.misses);
```

### Ontology Statistics

```rust
let stats = resolver.stats();

println!("Primitive resolutions (L1): {}", stats.primitive_hits);
println!("Foundation resolutions (L2): {}", stats.foundation_hits);
println!("Domain resolutions (L3): {}", stats.domain_hits);
println!("Federated resolutions (L4): {}", stats.federated_hits);
println!("Total resolutions: {}", stats.total_resolutions());
println!("Cache hit rate: {:.2}%", stats.cache_hit_rate() * 100.0);
```

## Supported Ontologies

### L1: Primitive (Compiled In)
- **BFO**: Basic Formal Ontology (~100 terms)
- **RO**: Relation Ontology (~600 terms)
- **COB**: Core Ontology for Biology (~150 terms)

### L2: Foundation (Shipped with Stdlib)
- **PATO**: Phenotypic Quality Ontology (~2,500 terms)
- **UO**: Units of Measurement (~1,000 terms)
- **IAO**: Information Artifact Ontology (~300 terms)
- **FHIR**: Fast Healthcare Interoperability Resources (~1,150 resources)
- **Schema.org**: Web vocabulary (~2,850 types)

### L3: Domain (SQLite Database)
- **ChEBI**: Chemical Entities of Biological Interest (~200,000 terms)
- **GO**: Gene Ontology (~45,000 terms)
- **DOID**: Disease Ontology (~10,000 terms)
- **HP**: Human Phenotype Ontology (~16,000 terms)
- **MONDO**: Mondo Disease Ontology (~25,000 terms)
- **UBERON**: Uber-anatomy Ontology (~15,000 terms)
- **CL**: Cell Ontology (~2,000 terms)
- **NCBITaxon**: NCBI Taxonomy (~2,000,000 terms)
- **PR**: Protein Ontology (~50,000 terms)
- **SO**: Sequence Ontology (~2,500 terms)
- **MAXO**: Medical Action Ontology (~1,000 terms)

### L4: Federated (BioPortal/OLS4 API)
- **SNOMED-CT**: Systematized Nomenclature of Medicine (~350,000 concepts)
- **ICD-10**: International Classification of Diseases (~14,000 codes)
- **LOINC**: Logical Observation Identifiers (~90,000 terms)
- **RxNorm**: Normalized Names for Drugs (~200,000 terms)
- **MeSH**: Medical Subject Headings (~30,000 terms)
- **NCIT**: NCI Thesaurus (~150,000 terms)
- **DrugBank**: Drug and drug target database (~14,000 drugs)
- **UniProt**: Universal Protein Resource (~200,000,000 proteins)
- ... and 500+ more ontologies via BioPortal

## Example: Medical Prescription System

```rust
use demetrios::ontology::{OntologyResolver, ResolverConfig, SubsumptionResult};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut resolver = OntologyResolver::default_resolver()?;

    // Define medication
    let aspirin = resolver.resolve("CHEBI:15365")?;
    println!("Medication: {}", aspirin.label.unwrap());

    // Check if it's an NSAID (non-steroidal anti-inflammatory drug)
    let nsaid = resolver.resolve("CHEBI:35475")?;  // NSAID class
    let is_nsaid = resolver.is_subclass_of("CHEBI:15365", "CHEBI:35475")?;

    if matches!(is_nsaid, SubsumptionResult::IsSubclass) {
        println!("Aspirin is an NSAID");
    }

    // Get therapeutic indications (ancestors in CHEBI)
    let ancestors = resolver.get_ancestors("CHEBI:15365")?;
    for ancestor in ancestors.iter().take(5) {
        if let Ok(term) = resolver.resolve(ancestor) {
            println!("  - {}", term.label.unwrap_or_default());
        }
    }

    Ok(())
}
```

## Example: Disease Classification

```rust
use demetrios::ontology::OntologyResolver;

fn classify_disease(disease_code: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut resolver = OntologyResolver::default_resolver()?;

    // Resolve disease
    let disease = resolver.resolve(disease_code)?;
    println!("Disease: {}", disease.label.unwrap_or_default());

    // Get disease hierarchy
    let ancestors = resolver.get_ancestors(disease_code)?;

    // Check if it's a cancer
    for ancestor in &ancestors {
        if ancestor.contains("DOID:162") {  // Cancer
            println!("This is a type of cancer");
            break;
        }
    }

    // Check if it's infectious
    for ancestor in &ancestors {
        if ancestor.contains("DOID:0050117") {  // Infectious disease
            println!("This is an infectious disease");
            break;
        }
    }

    Ok(())
}

// Usage
classify_disease("DOID:3910")?;  // Lung cancer
```

## Example: FHIR Resource Types

```rust
use demetrios::ontology::OntologyResolver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut resolver = OntologyResolver::default_resolver()?;

    // Resolve FHIR resources (L2 Foundation)
    let patient = resolver.resolve("FHIR:Patient")?;
    let observation = resolver.resolve("FHIR:Observation")?;
    let medication = resolver.resolve("FHIR:Medication")?;
    let condition = resolver.resolve("FHIR:Condition")?;

    println!("FHIR Resources:");
    println!("  Patient: {}", patient.label.unwrap_or_default());
    println!("  Observation: {}", observation.label.unwrap_or_default());
    println!("  Medication: {}", medication.label.unwrap_or_default());
    println!("  Condition: {}", condition.label.unwrap_or_default());

    Ok(())
}
```

## Integration with Demetrios Type System

Ontology terms can be used as **types** in Demetrios code:

```d
// Hypothetical Demetrios code (syntax subject to change)

linear struct Prescription {
    patient_id: string,
    medication: CHEBI:23888,     // Type: drug (any subclass)
    dosage: mg = 500.0,
    condition: DOID:4,           // Type: disease (any subclass)
}

fn prescribe(
    patient: &!Patient,
    drug: CHEBI:23888,           // Must be-a drug
    dose: mg
) -> Result<Prescription, Error> with IO {
    // Type checker ensures drug is-a CHEBI:23888
    // CHEBI:15365 (aspirin) is-a CHEBI:23888 (drug), so this typechecks

    let prescription = Prescription {
        patient_id: patient.id,
        medication: drug,
        dosage: dose,
        condition: patient.condition,
    };

    Ok(prescription)
}

fn main() with IO {
    var patient = Patient { ... };

    // OK: aspirin is-a drug
    let rx = prescribe(&!patient, CHEBI:15365, 500.0)?;

    // ERROR: biological_process is not a drug
    // let rx = prescribe(&!patient, GO:0008150, 500.0)?;
}
```

## Advanced: Custom Ontology Indexing

```rust
use demetrios::ontology::distance::{SemanticDistanceIndex, DistanceConfig};

let config = DistanceConfig::default();
let mut distance_index = SemanticDistanceIndex::new(config)?;

// Index ChEBI ontology for semantic distance queries
distance_index.index_ontology(OntologyId::ChEBI)?;

// Compute semantic distance
let distance = distance_index.distance("CHEBI:15365", "CHEBI:5855")?;
println!("Distance between aspirin and ibuprofen: {}", distance);
```

## Environment Variables

```bash
# BioPortal API key (for L4 federated queries)
export BIOPORTAL_API_KEY=your-api-key-here

# Ontology cache directory (default: .demetrios/ontology_cache)
export DEMETRIOS_ONTOLOGY_CACHE=/path/to/cache

# Enable debug logging
export RUST_LOG=demetrios::ontology=debug
```

## File Structure

```
.demetrios/ontology_cache/
├── l2_cache.db                    # SQLite L2 cache
├── chebi.db                       # ChEBI domain ontology
├── go.db                          # GO domain ontology
├── doid.db                        # DOID domain ontology
├── mappings/
│   ├── chebi_to_fhir.sssom.tsv
│   ├── snomed_to_icd10.sssom.tsv
│   └── go_to_mondo.sssom.tsv
└── downloads/
    ├── chebi.obo
    ├── go.obo
    └── doid.obo
```

## Troubleshooting

### Term Not Found
```rust
match resolver.resolve("UNKNOWN:123") {
    Ok(term) => println!("Found: {}", term.curie),
    Err(OntologyError::TermNotFound { ontology, term }) => {
        eprintln!("Term {} not found in ontology {}", term, ontology);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

### Network Errors (L4 Federated)
```rust
match resolver.resolve("NCIT:C12345") {
    Err(OntologyError::NetworkError(msg)) => {
        eprintln!("Network error: {}", msg);
        eprintln!("Check your internet connection or BIOPORTAL_API_KEY");
    }
    _ => {}
}
```

### Rate Limiting
```rust
// BioPortal rate limits: 15 requests/second
// The client automatically handles backoff and retry
```

## Performance Tips

1. **Use offline mode** for faster resolution if you don't need L4 federated terms
2. **Preload commonly used ontologies** at startup
3. **Configure cache size** based on available memory
4. **Enable negative caching** to avoid repeated failed lookups
5. **Use batch operations** when resolving multiple terms

## Next Steps

- Explore the [API Reference](https://docs.rs/demetrios/latest/demetrios/ontology/)
- See [Integration Tests](/mnt/e/workspace/demetrios/compiler/tests/integration_ontology_e2e.rs)
- Read the [Architecture Documentation](/mnt/e/workspace/demetrios/ONTOLOGY_INTEGRATION_SUMMARY.md)
- Join the community at [github.com/Chiuratto-AI/demetrios](https://github.com/Chiuratto-AI/demetrios)

---

**Version**: Demetrios v0.78.1
**Last Updated**: 2025-12-21
**License**: MIT OR Apache-2.0
