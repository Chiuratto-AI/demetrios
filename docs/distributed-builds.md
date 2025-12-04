# Distributed Builds & CI Integration

This document describes the distributed build system and CI integration features introduced in Day 24 of the Demetrios compiler development.

## Overview

The distributed build system enables:
- **Remote build execution**: Offload compilation to powerful build servers
- **Shared build caching**: Share compilation artifacts across machines
- **Reproducible builds**: Ensure builds are deterministic and verifiable
- **CI/CD integration**: Generate GitHub Actions and GitLab CI pipelines

## Feature Flag

The distributed build features are optional and require the `distributed` feature flag:

```bash
cargo build --features distributed
```

Or enable all features:

```bash
cargo build --features full
```

## Architecture

### Components

```
+----------------+     +----------------+     +----------------+
|  Build Client  |<--->|  Build Server  |<--->|  Build Worker  |
+----------------+     +----------------+     +----------------+
        |                     |
        v                     v
+----------------+     +----------------+
|  Local Cache   |     |  Cache Server  |
+----------------+     +----------------+
```

### Protocol

The distributed build system uses a length-prefixed JSON protocol over TCP:

```
+--------+------------------+
| Length |   JSON Message   |
| 4 bytes|   (variable)     |
+--------+------------------+
```

## CLI Commands

### Distributed Build Commands

```bash
# Submit a build to a remote server
dc distributed build --server build.example.com

# Check server status
dc distributed status --server build.example.com

# Start a build server
dc distributed server --address 0.0.0.0:9876 --workers 4
```

### Cache Commands

```bash
# Start a cache server
dc cache server --address 0.0.0.0:9877 --storage ~/.d/cache --max-size 10GB

# View cache statistics
dc cache stats
dc cache stats --url http://cache.example.com

# Clean the cache
dc cache clean --all
dc cache clean --older-than 7d
dc cache clean --dry-run
```

### CI Commands

```bash
# Generate GitHub Actions workflow
dc ci github --output .github/workflows/ci.yml
dc ci github --output .github/workflows/ci.yml --release

# Generate GitLab CI pipeline
dc ci gitlab --output .gitlab-ci.yml

# Generate build provenance (SLSA format)
dc ci provenance --output provenance.json

# Check build reproducibility
dc ci reproducible --builds 3 --check-env
```

## Programmatic API

### Build Client

```rust
use demetrios::distributed::{BuildClient, ClientConfig, BuildRequest};

// Create client
let config = ClientConfig::default();
let client = BuildClient::new(config);

// Connect to server
client.connect("build.example.com:9876").await?;

// Submit build job
let request = BuildRequest {
    sources: collect_sources(".")?,
    target: "x86_64-unknown-linux-gnu".into(),
    profile: "release".into(),
    ..Default::default()
};

let job = client.submit_job(request).await?;

// Wait for completion
let result = client.wait_for_job(job.id).await?;
```

### Build Server

```rust
use demetrios::distributed::{BuildServer, ServerConfig};

let config = ServerConfig {
    listen_addr: "0.0.0.0:9876".parse()?,
    max_workers: 8,
    cache_enabled: true,
    ..Default::default()
};

let server = BuildServer::new(config)?;
server.start().await?;
```

### Cache Client

```rust
use demetrios::distributed::cache::{CacheClient, CacheConfig};

let config = CacheConfig {
    url: "http://cache.example.com:9877".into(),
    token: Some("secret".into()),
    ..Default::default()
};

let client = CacheClient::new(config);

// Check cache
if client.contains("abc123").await? {
    let data = client.get("abc123").await?;
}

// Store in cache
client.put("abc123", data, metadata).await?;
```

### Reproducible Builds

```rust
use demetrios::distributed::reproducible::{BuildEnvironment, BuildInputs};

// Capture environment
let env = BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release");

// Check reproducibility requirements
if !env.is_reproducible() {
    eprintln!("Set SOURCE_DATE_EPOCH for reproducible builds");
}

// Collect build inputs
let inputs = BuildInputs::collect(".", "x86_64-unknown-linux-gnu", "release")?;

// Generate provenance
let provenance = BuildProvenance::new(env, inputs);
let slsa_json = provenance.to_slsa_json()?;
```

### CI Workflow Generation

```rust
use demetrios::distributed::ci::github::{Workflow, WorkflowGenerator};

let mut generator = WorkflowGenerator::new();
generator.add_target("x86_64-unknown-linux-gnu");
generator.add_target("aarch64-apple-darwin");

// Generate CI workflow
let ci_workflow = generator.generate_ci();
std::fs::write(".github/workflows/ci.yml", ci_workflow.to_yaml()?)?;

// Generate release workflow
let release_workflow = generator.generate_release();
std::fs::write(".github/workflows/release.yml", release_workflow.to_yaml()?)?;
```

## Build Cache

### Cache Key Format

Cache keys are computed from:
- Source file content hash
- Compiler version
- Target triple
- Build profile
- Compiler flags

```
key = sha256(source_hash || compiler_version || target || profile || flags)
```

### Cache Entry Types

- `object` - Compiled object files
- `executable` - Linked executables
- `library` - Static/dynamic libraries
- `metadata` - Build metadata and dependencies
- `test_result` - Cached test results

### Eviction Policy

The cache uses LRU (Least Recently Used) eviction:
1. Entries are evicted when cache exceeds size limit
2. Entries older than configured TTL are automatically purged
3. Manual eviction via `dc cache clean`

## Reproducible Builds

### Requirements

For reproducible builds, set the `SOURCE_DATE_EPOCH` environment variable:

```bash
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
```

### SLSA Provenance

The compiler generates SLSA v0.2 provenance attestations:

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [...],
  "predicate": {
    "builder": {
      "id": "demetrios-compiler-v0.14.0"
    },
    "buildType": "https://demetrios-lang.org/build/v1",
    "invocation": {
      "configSource": {},
      "parameters": {
        "profile": "release",
        "target": "x86_64-unknown-linux-gnu"
      }
    }
  }
}
```

## CI/CD Integration

### GitHub Actions

Generated workflows include:
- Multi-platform matrix builds
- Caching of dependencies and build artifacts
- Test execution with coverage
- Release automation with artifact publishing

### GitLab CI

Generated pipelines include:
- Build and test stages
- Cache configuration for faster builds
- Artifact preservation

## Security Considerations

### Authentication

The build server supports token-based authentication:

```rust
let config = ServerConfig {
    require_auth: true,
    auth_tokens: vec!["secret-token".into()],
    ..Default::default()
};
```

### TLS

For production deployments, use TLS:

```rust
let config = ServerConfig {
    tls_cert: Some("cert.pem".into()),
    tls_key: Some("key.pem".into()),
    ..Default::default()
};
```

### Build Isolation

Build workers execute in isolated environments:
- Separate working directories
- Limited network access
- Resource quotas (CPU, memory, disk)

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `D_BUILD_SERVER` | Default build server address |
| `D_CACHE_URL` | Default cache server URL |
| `D_CACHE_TOKEN` | Cache authentication token |
| `SOURCE_DATE_EPOCH` | Timestamp for reproducible builds |

### Configuration File

Create `.d/config.toml`:

```toml
[distributed]
server = "build.example.com:9876"
cache_url = "http://cache.example.com:9877"

[cache]
local_path = "~/.d/cache"
max_size = "10GB"
ttl = "7d"

[ci]
default_targets = ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"]
```

## Troubleshooting

### Connection Issues

```bash
# Test connectivity
dc distributed status --server build.example.com

# Check server logs
dc distributed server --verbose
```

### Cache Misses

Common causes:
- Different compiler versions
- Different build flags
- Source file changes

```bash
# View cache statistics
dc cache stats --verbose

# Clear stale entries
dc cache clean --older-than 1d
```

### Reproducibility Failures

```bash
# Check environment
dc ci reproducible --check-env

# Verify SOURCE_DATE_EPOCH
echo $SOURCE_DATE_EPOCH
```

## Performance Tuning

### Parallel Builds

Configure worker count based on available resources:

```rust
let config = ServerConfig {
    max_workers: num_cpus::get(),
    ..Default::default()
};
```

### Cache Sizing

Recommended cache sizes:
- Local development: 1-5 GB
- CI runners: 5-10 GB
- Shared cache server: 50-100 GB

### Network Optimization

Enable compression for remote cache:

```rust
let config = CacheConfig {
    compression: true,
    ..Default::default()
};
```
