# Demetrios Language Documentation

This directory contains comprehensive documentation for the Demetrios (D) programming language compiler.

## Quick Navigation

| Category | Description |
|----------|-------------|
| [Architecture](#architecture) | Compiler design and structure |
| [Language Features](#language-features) | Core language capabilities |
| [Development Logs](#development-logs) | Daily development progress |
| [API Reference](#api-reference) | Detailed API documentation |
| [Releases](#releases) | Version release notes |

---

## Architecture

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Overview of compiler architecture, module organization, and design decisions

---

## Language Features

### Type System

- **[refinement-types.md](refinement-types.md)** - Refinement types with SMT verification for compile-time constraint checking

### Ontological Type System

- **[research/semantic_metric_types.md](research/semantic_metric_types.md)** - Semantic metric types integrating ontological knowledge into static type checking
- **[papers/semantic_types_paper.md](papers/semantic_types_paper.md)** - Academic paper on semantic metric types (POPL/PLDI target)
- **[spec/formal/semantic_types.tex](../spec/formal/semantic_types.tex)** - Formal type theory specification in LaTeX

### Scientific Computing

- **[SCIENTIFIC_COMPUTING.md](SCIENTIFIC_COMPUTING.md)** - Units of measure, dimensional analysis, and scientific computing features

### Epistemic Computing

- **[EPISTEMIC_AGENTS.md](EPISTEMIC_AGENTS.md)** - Epistemic knowledge representation, confidence tracking, and agent-based reasoning

### Macro System

- **[MACRO_SYSTEM.md](MACRO_SYSTEM.md)** - Hygenic macro system overview
- **[MACRO_INTEGRATION.md](MACRO_INTEGRATION.md)** - Integration of macros with the compiler pipeline

### Memory and Layout

- **[day38-layout-synthesis.md](day38-layout-synthesis.md)** - Semantic-aware struct layout synthesis
- **[day41-locality.md](day41-locality.md)** - Cache locality optimization and semantic-physical duality

---

## Tooling

### IDE Support

- **[lsp.md](lsp.md)** - Language Server Protocol implementation for IDE integration

### Build System

- **[day21_build_system.md](day21_build_system.md)** - Incremental build system with content-addressable caching
- **[day22_watch_mode.md](day22_watch_mode.md)** - File watching and automatic recompilation
- **[day23_cross_compilation.md](day23_cross_compilation.md)** - Cross-compilation support for multiple targets
- **[distributed-builds.md](distributed-builds.md)** - Distributed compilation across multiple machines

### Diagnostics

- **[day25-diagnostics.md](day25-diagnostics.md)** - Rich diagnostic output with source highlighting

---

## API Reference

Detailed API documentation for library authors:

- **[api/SCIENTIFIC_API.md](api/SCIENTIFIC_API.md)** - Scientific computing API reference
- **[api/MACRO_API.md](api/MACRO_API.md)** - Macro system API reference

---

## Releases

Version history and release notes:

| Version | Highlights |
|---------|------------|
| [v0.41.0](../compiler/docs/RELEASE_v0.41.0.md) | Semantic-Physical Duality, Locality Analysis |
| [v0.39.0](releases/v0.39.0.md) | Layout Synthesis |
| [v0.30.0](releases/v0.30.0-epistemic-paradigm.md) | Epistemic Paradigm |
| [v0.28.0](RELEASE_SUMMARY_v0.28.0.md) | Scientific Computing |
| [v0.13.0](releases/v0.13.0.md) | Type System Enhancements |
| [v0.12.0](releases/v0.12.0.md) | Core Language Features |

---

## Module Index

The compiler is organized into the following major modules:

### Frontend
- `lexer/` - Tokenization using Logos
- `parser/` - Recursive descent + Pratt parsing
- `ast/` - Abstract syntax tree definitions
- `resolve/` - Name resolution

### Type System
- `types/` - Type definitions and inference
- `check/` - Type checking
- `effects/` - Algebraic effect system
- `epistemic/` - Epistemic knowledge tracking

### Middle-End
- `hir/` - High-level intermediate representation
- `hlir/` - SSA-based intermediate representation
- `locality/` - Cache locality optimization
- `layout/` - Memory layout synthesis

### Backend
- `codegen/` - Code generation (LLVM, Cranelift, GPU)
- `mlir/` - MLIR integration

### Tooling
- `build/` - Build system
- `watch/` - File watching
- `lsp/` - Language server
- `diagnostic/` - Error reporting

### Domain-Specific
- `ontology/` - Ontology integration for scientific computing
  - `distance/` - Semantic distance calculation (path, IC, embedding)
  - `embedding/` - Ontology embedding space with ANN index
  - `loader/` - OWL/OBO ontology loading
  - `storage/` - Backend storage (memory, SQLite, mmap)
- `causal/` - Causal reasoning
- `temporal/` - Temporal logic

---

## Getting Started

```bash
# Build the compiler
cargo build

# Run tests
cargo test

# Check a D source file
cargo run -- check examples/minimal.d --show-ast --show-types

# Run with all features
cargo build --features full
```

## Feature Flags

The compiler supports optional features:

| Feature | Description |
|---------|-------------|
| `llvm` | LLVM backend (requires LLVM installation) |
| `jit` | Cranelift JIT compilation |
| `gpu` | GPU codegen (PTX and SPIR-V) |
| `smt` | Z3 SMT solver for refinement types |
| `lsp` | Language Server Protocol |
| `distributed` | Distributed build support |
| `ontology` | Ontology integration with SQLite |
| `llm` | LLM integration for epistemic agents |
| `full` | Enable all features |

---

## Contributing

See the main [DEVELOPER.md](../DEVELOPER.md) for coding standards and contribution guidelines.
