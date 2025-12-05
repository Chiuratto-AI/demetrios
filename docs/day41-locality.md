# Day 41: Semantic-Physical Duality

**Date:** 2025-12-05  
**Version:** v0.41.0  
**Module:** `compiler/src/locality/`

## Overview

Day 41 implements *Semantic-Physical Duality* — the bridge between abstract ontological knowledge and concrete memory optimization. The key insight is Landauer's principle applied to compilation: **information is physical**. Ontology relationships that describe semantic proximity between concepts predict physical data access patterns, enabling the compiler to optimize memory layout and prefetching.

## Core Concepts

### The Duality Principle

```
Semantic Distance ↔ Physical Locality

Close in ontology → Likely accessed together → Place in same cache line
Far in ontology   → Rarely co-accessed      → Can be separated
```

When a program accesses `patient.diagnosis`, the ontology tells us that `diagnosis` is semantically close to `symptoms`, `treatment`, and `prognosis`. This semantic proximity predicts that these fields will likely be accessed together, guiding cache-line packing decisions.

### Locality Hierarchy

The type system encodes memory hierarchy as a subtype lattice:

```
Register < L1 < L2 < L3 < Local < Remote < Persistent < Network
   ↑                                                        ↓
 faster                                                   slower
 closer                                                  farther
```

Subtyping rule: faster localities are subtypes of slower ones. A value in `Register` can be used where `L1` is expected (covariant), but not vice versa.

## Architecture

### Module Structure

```
compiler/src/locality/
├── mod.rs          # Main exports, Ontology trait, SemanticPhysicalBridge
├── types.rs        # Locality enum, LocalityBound, LocalityParam, constraints
├── subtyping.rs    # Subtype lattice, constraint solving, variance
├── prefetch.rs     # Semantic prefetch table, distance calculation
├── access.rs       # Access pattern analysis, co-access detection
├── codegen.rs      # Prefetch instruction generation
├── numa.rs         # NUMA topology detection, placement strategies
└── packing.rs      # Cache-line packing, field grouping
```

### Key Types

```rust
/// Memory hierarchy levels
pub enum Locality {
    Register,   // CPU register
    L1,         // L1 cache (~1-3 cycles)
    L2,         // L2 cache (~10-15 cycles)
    L3,         // L3 cache (~30-50 cycles)
    Local,      // Local DRAM (~100-300 cycles)
    Remote,     // Remote NUMA (~300-500 cycles)
    Persistent, // NVM/SSD (~10K+ cycles)
    Network,    // Network storage (~1M+ cycles)
}

/// Bounds on locality parameters
pub enum LocalityBound {
    Exact(Locality),           // Must be exactly this locality
    AtMost(Locality),          // Can be this fast or faster
    AtLeast(Locality),         // Can be this slow or slower
    Between(Locality, Locality), // Within range
    Any,                       // No constraint
}

/// Locality type parameter for generic types
pub struct LocalityParam {
    pub name: String,
    pub bound: LocalityBound,
    pub variance: Variance,
}
```

## Features

### 1. Locality Types

Annotate data with locality requirements:

```d
// Data must reside in L1 cache
fn hot_path(data: &[f64; L1]) -> f64 {
    data.iter().sum()
}

// Data can be anywhere from L2 to Local
fn cold_path<L: L2..Local>(data: &[f64; L]) -> f64 {
    data.iter().sum()
}

// Locality is polymorphic
fn generic<L>(data: &[f64; L]) -> f64 where L: AtMost<L3> {
    data.iter().sum()
}
```

### 2. Semantic Prefetch Table

The compiler builds a prefetch table from ontology relationships:

```rust
let table = PrefetchTable::from_ontology(&ontology);

// Given access to "diagnosis", table suggests prefetching:
// - symptoms (distance: 1, priority: High)
// - treatment (distance: 1, priority: High)
// - patient (distance: 2, priority: Medium)
// - medication (distance: 2, priority: Medium)
```

Distance calculation combines:
- **Inheritance distance**: Steps in class hierarchy
- **Relationship distance**: Steps through object properties
- **Usage correlation**: Historical co-access patterns

### 3. Access Pattern Analysis

The analyzer tracks field access patterns:

```rust
let mut analyzer = AccessAnalyzer::new();
analyzer.enter_function("process_patient");

// Track accesses
analyzer.record_access("Patient", "name", AccessKind::Read);
analyzer.record_access("Patient", "diagnosis", AccessKind::Read);
analyzer.record_access("Patient", "treatment", AccessKind::Write);

// Analyzer detects co-access patterns
let co_access = analyzer.co_access_for("Patient");
// Returns: [("name", "diagnosis", 0.9), ("diagnosis", "treatment", 0.8)]
```

### 4. NUMA Topology & Placement

Detect system topology and suggest data placement:

```rust
let topology = NumaTopology::detect()?;

// Topology info:
// Node 0: CPUs 0-7, 16GB memory
// Node 1: CPUs 8-15, 16GB memory
// Distance matrix: [[10, 20], [20, 10]]

let placement = topology.suggest_placement(&types, &ontology);
// Returns: {
//   "Patient": Node(0),
//   "Treatment": Node(0),  // Close to Patient
//   "Analytics": Node(1),  // Separate workload
// }
```

### 5. Cache-Line Packing

Reorder struct fields for optimal cache utilization:

```rust
let packer = CacheLinePacker::new(64); // 64-byte cache lines

// Original struct:
// struct Data { a: u8, b: u64, c: u8, d: u64 }
// Size: 32 bytes (with padding)

let packed = packer.pack(&fields, &access_patterns);

// Optimized struct:
// struct Data { b: u64, d: u64, a: u8, c: u8 }
// Size: 18 bytes, hot fields in first cache line
```

### 6. Prefetch Code Generation

Generate architecture-specific prefetch instructions:

```rust
let codegen = PrefetchCodegen::new(Architecture::X86_64);

// For stride pattern (64-byte stride, 8 iterations ahead)
let instructions = codegen.for_stride(&stride, "data", 8);

// Output:
// prefetcht0 [rax + 512]   ; 8 * 64 = 512 bytes ahead
// prefetcht1 [rax + 1024]  ; 16 * 64 = 1024 bytes ahead
```

Supported architectures:
- **x86_64**: `prefetcht0`, `prefetcht1`, `prefetcht2`, `prefetchnta`
- **ARM64**: `prfm pldl1keep`, `prfm pldl2keep`, `prfm pstl1keep`
- **LLVM**: `@llvm.prefetch` intrinsic
- **RISC-V**: `prefetch.r`, `prefetch.w` (Zicbop extension)

## CLI Commands

### `dc locality numa`

Display NUMA topology:

```bash
$ dc locality numa --format json
{
  "nodes": [
    {"id": 0, "cpus": [0,1,2,3], "memory_mb": 16384},
    {"id": 1, "cpus": [4,5,6,7], "memory_mb": 16384}
  ],
  "distances": [[10, 20], [20, 10]]
}
```

### `dc locality analyze`

Analyze access patterns in source file:

```bash
$ dc locality analyze src/main.d --recommend
=== Access Pattern Analysis ===

Function: process_data
  Hotness: Hot
  Accesses: 47
  Co-access clusters:
    - {x, y, z} correlation: 0.95
    - {timestamp, value} correlation: 0.82

Recommendations:
  - Pack fields {x, y, z} in same cache line
  - Consider prefetching 'value' after 'timestamp' access
```

### `dc locality prefetch`

Generate semantic prefetch table:

```bash
$ dc locality prefetch --ontology ~/.demetrios/ontology
=== Semantic Prefetch Table ===

Type: Patient
  → diagnosis (distance: 1, priority: High)
  → treatment (distance: 1, priority: High)
  → symptoms (distance: 2, priority: Medium)

Type: Molecule
  → atoms (distance: 1, priority: High)
  → bonds (distance: 1, priority: High)
  → properties (distance: 2, priority: Medium)
```

### `dc locality pack`

Suggest cache-line packing for structs:

```bash
$ dc locality pack src/types.d --struct Data --cache-line 64
=== Cache-Line Packing Analysis ===

Struct: Data
Original size: 48 bytes (spans 1 cache line + 16 bytes)

Recommended layout:
  offset 0:  hot_field_1 (8 bytes) - 95% access rate
  offset 8:  hot_field_2 (8 bytes) - 87% access rate
  offset 16: hot_field_3 (4 bytes) - 76% access rate
  --- cache line boundary (64 bytes) ---
  offset 64: cold_field_1 (8 bytes) - 12% access rate
  offset 72: cold_field_2 (8 bytes) - 8% access rate

Optimized size: 40 bytes (spans 1 cache line)
Estimated speedup: 15-20% for hot path
```

### `dc locality lattice`

Display locality subtype lattice:

```bash
$ dc locality lattice
Locality Subtype Lattice:
========================

Register ─┬─> L1 ─┬─> L2 ─┬─> L3 ─┬─> Local ─┬─> Remote ─┬─> Persistent ─┬─> Network
          │       │       │       │          │           │               │
          └───────┴───────┴───────┴──────────┴───────────┴───────────────┘
                              (faster is subtype of slower)

Latencies (cycles):
  Register:   1
  L1:         3
  L2:         12
  L3:         40
  Local:      200
  Remote:     400
  Persistent: 10000
  Network:    1000000
```

### `dc locality codegen`

Generate prefetch instructions:

```bash
$ dc locality codegen src/main.d --function hot_loop --arch x86_64
=== Prefetch Code Generation ===

Function: hot_loop
Architecture: x86_64

Generated instructions:
  ; Prefetch for stride pattern (64 bytes, 8 ahead)
  prefetcht0 [rax + 512]
  prefetcht1 [rax + 1024]
  
  ; Semantic prefetch for Patient.diagnosis access
  prefetcht0 [rbx + 24]   ; treatment field
  prefetcht1 [rbx + 48]   ; symptoms field
```

## Implementation Details

### Semantic Distance Calculation

```rust
impl SemanticDistance {
    pub fn from_ontology(ont: &dyn Ontology, from: &str, to: &str) -> Self {
        // 1. Check direct relationship
        if ont.is_subclass(from, to) || ont.is_subclass(to, from) {
            return SemanticDistance::new(1);
        }
        
        // 2. Find common ancestor
        let from_ancestors = ont.ancestors(from);
        let to_ancestors = ont.ancestors(to);
        
        for (i, fa) in from_ancestors.iter().enumerate() {
            if let Some(j) = to_ancestors.iter().position(|ta| ta == fa) {
                return SemanticDistance::new(i + j + 2);
            }
        }
        
        // 3. No relationship found
        SemanticDistance::infinity()
    }
}
```

### Constraint Solving

```rust
impl LocalitySolver {
    pub fn solve(&mut self) -> Result<HashMap<String, Locality>, LocalityError> {
        // 1. Propagate exact constraints
        self.propagate_exact()?;
        
        // 2. Apply subtype constraints
        for constraint in &self.constraints {
            match constraint {
                LocalityConstraint::Subtype { sub, sup } => {
                    let sub_loc = self.get_bound(sub)?;
                    let sup_loc = self.get_bound(sup)?;
                    if !sub_loc.is_subtype_of(&sup_loc) {
                        return Err(LocalityError::Unsatisfiable);
                    }
                }
                // ...
            }
        }
        
        // 3. Find minimal satisfying assignment
        self.minimize()
    }
}
```

### NUMA Detection (Linux)

```rust
impl NumaTopology {
    pub fn detect() -> Result<Self, NumaError> {
        let mut nodes = Vec::new();
        
        // Parse /sys/devices/system/node/
        for entry in fs::read_dir("/sys/devices/system/node")? {
            let path = entry?.path();
            if path.file_name()?.to_str()?.starts_with("node") {
                let node_id = parse_node_id(&path)?;
                let cpus = parse_cpulist(&path.join("cpulist"))?;
                let memory = parse_meminfo(&path.join("meminfo"))?;
                nodes.push(NumaNode { id: node_id, cpus, memory_bytes: memory });
            }
        }
        
        // Parse distance matrix
        let distances = parse_distance_matrix(&nodes)?;
        
        Ok(NumaTopology { nodes, distances })
    }
}
```

## Testing

71 tests covering all modules:

```
locality::types::tests          - 12 tests
locality::subtyping::tests      - 14 tests
locality::prefetch::tests       - 10 tests
locality::access::tests         - 11 tests
locality::codegen::tests        - 10 tests
locality::numa::tests           - 10 tests
locality::packing::tests        - 10 tests
locality::tests                 - 4 tests
```

Run tests:
```bash
cargo test locality
```

## Future Work

1. **Profile-guided optimization**: Use runtime profiling data to refine access patterns
2. **Automatic prefetch insertion**: Insert prefetch instructions during HIR lowering
3. **NUMA-aware allocator**: Custom allocator respecting placement decisions
4. **GPU locality**: Extend hierarchy for GPU memory (shared, L1, L2, global, host)
5. **Persistent memory**: Special handling for Intel Optane / CXL memory

## References

- Landauer, R. (1961). "Irreversibility and Heat Generation in the Computing Process"
- Drepper, U. (2007). "What Every Programmer Should Know About Memory"
- Intel 64 and IA-32 Architectures Optimization Reference Manual
- ARM Architecture Reference Manual (Prefetch instructions)
- RISC-V Zicbop Extension Specification

## Changelog

### v0.41.0 (2025-12-05)
- Initial implementation of locality module
- 8 submodules: types, subtyping, prefetch, access, codegen, numa, packing, mod
- 6 CLI commands: numa, analyze, prefetch, pack, lattice, codegen
- 71 tests passing
