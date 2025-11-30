# Day 21: Build System, Caching & Incremental Compilation

## Overview

Day 21 implements a comprehensive build system for the Demetrios compiler with:
- **Build Graph**: DAG-based dependency tracking with topological sort
- **Change Detection**: Content-based file monitoring with glob patterns
- **Artifact Cache**: Content-addressed storage with multiple eviction policies
- **Query System**: Salsa-style incremental compilation
- **Parallel Execution**: Work-stealing scheduler for multi-core builds
- **Build Configuration**: Profiles (dev/release/test/bench) and target triples

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BuildManager                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Graph   │  │  Change  │  │  Cache   │  │  Query   │        │
│  │          │  │ Detector │  │          │  │   Db     │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       └─────────────┴──────┬──────┴─────────────┘               │
│                            │                                    │
│                    ┌───────┴───────┐                            │
│                    │   Parallel    │                            │
│                    │   Executor    │                            │
│                    └───────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Part A: Build Graph (`build/graph.rs`)

The build graph tracks compilation units and their dependencies using a directed acyclic graph (DAG).

```rust
// Key types
pub struct BuildGraph {
    units: HashMap<UnitId, CompilationUnit>,
    dependencies: HashMap<UnitId, HashSet<UnitId>>,
    dependents: HashMap<UnitId, HashSet<UnitId>>,
}

pub struct CompilationUnit {
    pub id: UnitId,
    pub path: PathBuf,
    pub kind: UnitKind,
    pub content_hash: ContentHash,
    pub status: CompilationStatus,
}
```

**Features:**
- Topological sort using Kahn's algorithm
- Cycle detection with detailed reporting
- Invalidation propagation (mark dependents as dirty)
- Parallel-ready level computation

### Part B: Change Detection (`build/change.rs`)

Monitors source files for changes using content hashing.

```rust
pub struct ChangeDetector {
    file_states: HashMap<PathBuf, FileState>,
    watch_patterns: Vec<String>,
}

pub enum ChangeType {
    Added,
    Modified,
    Deleted,
}
```

**Features:**
- SHA-256 content hashing
- Glob pattern matching for watch paths
- Incremental scanning
- Integration with build graph invalidation

### Part C: Artifact Cache (`build/cache.rs`)

Content-addressed storage for build artifacts with configurable eviction.

```rust
pub struct ArtifactCache {
    cache_dir: PathBuf,
    index: HashMap<CacheKey, CacheEntry>,
    config: CacheConfig,
    stats: CacheStats,
}

pub enum EvictionPolicy {
    Lru,        // Least Recently Used
    Lfu,        // Least Frequently Used
    Fifo,       // First In First Out
    SizeBased,  // Largest first
}
```

**Features:**
- Content-addressed keys (SHA-256)
- Multiple eviction policies
- Cache statistics tracking
- Compression support (future)
- Remote cache stub (feature-gated)

### Part D: Query System (`build/query.rs`)

Salsa-style incremental computation with automatic dependency tracking.

```rust
pub struct QueryDb {
    storage: HashMap<QueryKey, QueryValue>,
    dependencies: HashMap<QueryKey, HashSet<QueryKey>>,
    revisions: HashMap<QueryKey, Revision>,
    current_revision: Revision,
}

pub enum QueryKey {
    Parse(PathBuf),
    Resolve(PathBuf),
    TypeCheck(PathBuf),
    LowerHir(PathBuf),
    Codegen(PathBuf),
}
```

**Features:**
- Revision-based change tracking
- Automatic dependency recording
- On-demand recomputation
- Query memoization

### Part E: Parallel Executor (`build/parallel.rs`)

Work-stealing scheduler for parallel compilation.

```rust
pub struct ParallelExecutor {
    thread_pool: Vec<JoinHandle<()>>,
    job_queue: Arc<Mutex<BinaryHeap<PrioritizedJob>>>,
    results: Arc<Mutex<HashMap<UnitId, JobResult>>>,
}

pub struct BuildScheduler {
    executor: ParallelExecutor,
    graph: BuildGraph,
    running: HashSet<UnitId>,
}
```

**Features:**
- Priority-based job scheduling
- Work stealing between threads
- Dependency-aware execution order
- Configurable parallelism (default: CPU count)

### Part F: Build Configuration (`build/config.rs`)

Build profiles and target configuration.

```rust
pub struct BuildConfig {
    pub profile: BuildProfile,
    pub target: TargetTriple,
    pub features: HashSet<String>,
    pub output_dir: PathBuf,
}

pub enum BuildProfile {
    Dev,      // -O0, debug info
    Release,  // -O3, LTO
    Test,     // -O0, debug, test harness
    Bench,    // -O3, debug info
}

pub struct TargetTriple {
    pub arch: Architecture,
    pub vendor: Vendor,
    pub os: OperatingSystem,
    pub env: Option<Environment>,
}
```

**Features:**
- Standard profiles with sensible defaults
- Target triple parsing (e.g., `x86_64-unknown-linux-gnu`)
- Feature flags
- Environment variable overrides

## CLI Integration

### Build Command

```bash
# Full build
dc build

# Incremental build
dc build --incremental

# Release build
dc build --profile release

# Parallel build with 8 threads
dc build --jobs 8

# Cross-compilation
dc build --target aarch64-apple-darwin
```

### Clean Command

```bash
# Clean all artifacts
dc clean

# Clean only cache
dc clean --cache

# Clean specific target
dc clean --target release
```

## Build Manager

The `BuildManager` coordinates all components:

```rust
pub struct BuildManager {
    graph: BuildGraph,
    change_detector: ChangeDetector,
    cache: ArtifactCache,
    query_db: QueryDb,
    config: BuildConfig,
}

impl BuildManager {
    pub fn build(&mut self) -> Result<BuildReport, BuildError>;
    pub fn incremental_build(&mut self) -> Result<BuildReport, BuildError>;
    pub fn clean(&mut self) -> Result<(), BuildError>;
}
```

## Build Report

```rust
pub struct BuildReport {
    pub units_compiled: usize,
    pub units_cached: usize,
    pub units_skipped: usize,
    pub errors: Vec<BuildError>,
    pub warnings: Vec<String>,
    pub duration: Duration,
}
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Topological sort | O(V + E) | Kahn's algorithm |
| Cycle detection | O(V + E) | DFS-based |
| Cache lookup | O(1) | Hash map |
| Change detection | O(n) | Per-file hash |
| Invalidation | O(dependents) | Transitive closure |

## Configuration Files

### `demetrios.toml`

```toml
[build]
profile = "dev"
target = "x86_64-unknown-linux-gnu"
jobs = 4

[build.cache]
enabled = true
max_size = "10GB"
eviction = "lru"

[profile.dev]
opt_level = 0
debug = true
lto = false

[profile.release]
opt_level = 3
debug = false
lto = true
```

## Module Structure

```
compiler/src/build/
├── mod.rs       # BuildManager, module exports (362 lines)
├── graph.rs     # BuildGraph, dependencies (672 lines)
├── change.rs    # ChangeDetector (343 lines)
├── cache.rs     # ArtifactCache (654 lines)
├── query.rs     # QueryDb, incremental (473 lines)
├── parallel.rs  # ParallelExecutor (455 lines)
└── config.rs    # BuildConfig, profiles (578 lines)

Total: ~3,537 lines
```

## Dependencies Added

```toml
sha2 = "0.10"      # Content hashing
hex = "0.4"        # Hash encoding
bincode = "1.3"    # Serialization
glob = "0.3"       # File patterns
num_cpus = "1.16"  # Thread count
tempfile = "3.10"  # Test utilities
```

## Future Enhancements

1. **Remote Cache**: Shared cache server for team builds
2. **Distributed Builds**: Build across multiple machines
3. **Build Tracing**: Detailed timing and dependency visualization
4. **Hermetic Builds**: Reproducible builds with sandboxing
5. **Watch Mode**: Continuous incremental compilation

## Example Usage

```rust
use demetrios::build::{BuildManager, BuildConfig, BuildProfile};

fn main() -> Result<(), BuildError> {
    // Create build configuration
    let config = BuildConfig::dev();
    
    // Initialize build manager
    let mut manager = BuildManager::new(config)?;
    
    // Add source files
    manager.add_source("src/main.d")?;
    manager.add_source("src/lib.d")?;
    
    // Perform incremental build
    let report = manager.incremental_build()?;
    
    println!("Compiled: {} units", report.units_compiled);
    println!("Cached: {} units", report.units_cached);
    println!("Duration: {:?}", report.duration);
    
    Ok(())
}
```

## Testing

```bash
# Run build system tests
cargo test --lib build::

# Test specific component
cargo test --lib build::graph::tests
cargo test --lib build::cache::tests
cargo test --lib build::query::tests
```

## Conclusion

Day 21 provides a robust, production-quality build system with:
- Fast incremental compilation via content-based change detection
- Efficient caching with configurable eviction policies
- Parallel execution with work stealing
- Flexible configuration through profiles and targets

The system is designed for scalability and can handle large codebases with thousands of compilation units while maintaining fast rebuild times.
