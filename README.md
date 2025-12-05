# Demetrios (D)

**A novel systems + scientific programming language with epistemic computing.**

[![Version](https://img.shields.io/badge/version-0.42.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](LICENSE-MIT)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

---

## Overview

Demetrios is a new programming language designed for systems programming and scientific computing, with first-class support for:

- **Algebraic Effects** - Composable effect handlers for IO, mutation, allocation, GPU, and more
- **Linear/Affine Types** - Compile-time resource management without garbage collection
- **Units of Measure** - Dimensional analysis catches unit errors at compile time
- **Refinement Types** - SMT-backed verification of value constraints
- **Epistemic Computing** - Track confidence, provenance, and uncertainty through computations
- **GPU-Native** - First-class GPU memory regions and kernel syntax

## Quick Start

```bash
# Build the compiler
cd compiler
cargo build --release

# Check a program
./target/release/dc check examples/hello.d

# Run with JIT (requires --features jit)
./target/release/dc run examples/hello.d

# Start REPL
./target/release/dc repl
```

## Example

```d
module pharmacokinetics

use units::{mg, mL, h}

/// Two-compartment PK model with effect tracking
fn simulate_pk(
    dose: mg,
    volume: mL,
    clearance: mL/h,
) -> Vec<(h, mg/mL)> with Alloc, Prob {
    let initial_conc = dose / volume
    let eta = sample(Normal(0.0, 0.3))  // Random effect
    
    // Simulate with variability
    (0..24).map(|t| {
        let time = t as f64 : h
        let conc = initial_conc * exp(-clearance/volume * time * (1.0 + eta))
        (time, conc)
    }).collect()
}

kernel fn parallel_solve(data: &[f32], out: &mut [f32]) {
    let i = gpu.thread_id.x
    out[i] = expensive_computation(data[i])
}
```

## Repository Structure

```
demetrios/
├── compiler/           # Rust compiler implementation
│   ├── src/            # Compiler source
│   │   ├── lexer/      # Tokenization (Logos)
│   │   ├── parser/     # Recursive descent + Pratt parsing
│   │   ├── ast/        # Abstract syntax tree
│   │   ├── check/      # Type checking
│   │   ├── effects/    # Algebraic effect system
│   │   ├── hir/        # High-level IR
│   │   ├── hlir/       # SSA-based IR
│   │   ├── codegen/    # LLVM/Cranelift/GPU backends
│   │   ├── epistemic/  # Epistemic computing support
│   │   ├── locality/   # Cache optimization
│   │   └── ontology/   # Scientific ontology integration
│   ├── benches/        # Performance benchmarks
│   └── tests/          # Compiler unit tests
├── stdlib/             # Standard library (D code)
│   ├── core/           # Core types and traits
│   ├── collections/    # Data structures
│   ├── iter/           # Iterators
│   ├── sync/           # Synchronization primitives
│   └── ...
├── spec/               # Language specification
├── docs/               # Documentation
│   ├── README.md       # Documentation index
│   ├── api/            # API reference
│   └── releases/       # Release notes
├── examples/           # Example programs
├── editors/            # Editor integrations
│   └── vscode/         # VS Code extension
├── tools/              # Development tools
└── tests/              # Language test suite
    ├── ui/             # Error message tests
    ├── run-pass/       # Should compile and run
    └── compile-fail/   # Should fail to compile
```

## Key Features

### Algebraic Effects

```d
fn read_config(path: string) -> Config with IO, Panic {
    let content = read_file(path)?
    parse_config(content)
}

// Handle effects at call site
handle read_config("app.toml") {
    IO => filesystem_handler,
    Panic => |e| default_config(),
}
```

### Units of Measure

```d
let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume  // Type-checked!

// Compile error: incompatible units
// let bad: kg = dose + volume
```

### Linear Types

```d
linear struct FileHandle {
    fd: i32,
}

fn process(file: FileHandle) {
    // file must be used exactly once
    file.close()
}  // Compile error if file not consumed
```

### Refinement Types

```d
type Positive = { x: i32 | x > 0 }
type Percentage = { x: f64 | 0.0 <= x && x <= 100.0 }

fn divide(a: i32, b: { x: i32 | x != 0 }) -> i32 {
    a / b  // Division by zero impossible
}
```

### Epistemic Computing

```d
// Values carry confidence and provenance
let measurement: Knowledge<f64> = Knowledge::new(
    value: 98.6,
    confidence: 0.95,
    source: Source::Measurement("thermometer"),
)

// Confidence propagates through computations
let derived = measurement.map(|t| (t - 32.0) * 5.0/9.0)
assert(derived.confidence <= measurement.confidence)
```

## Building from Source

### Prerequisites

- Rust 1.75+ (edition 2024)
- Optional: LLVM 15+ (for native compilation)
- Optional: Z3 (for refinement type verification)

### Build Commands

```bash
cd compiler

# Development build
cargo build

# Release build
cargo build --release

# With all features
cargo build --release --features full

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `jit` | Cranelift JIT for fast development iteration |
| `llvm` | LLVM backend for optimized native binaries |
| `gpu` | GPU codegen (PTX and SPIR-V) |
| `smt` | Z3 SMT solver for refinement types |
| `lsp` | Language Server Protocol for IDE support |
| `distributed` | Distributed build support |
| `ontology` | Scientific ontology integration |
| `full` | Enable all features |

## IDE Support

Full LSP implementation with:
- Real-time diagnostics
- Hover information with types and effects
- Go to definition
- Find references
- Code completion
- Semantic highlighting

Install VS Code extension from `editors/vscode/`.

## Documentation

- **[Language Specification](spec/LANGUAGE_SPECIFICATION.md)** - Formal language definition
- **[Documentation Index](docs/README.md)** - All documentation
- **[Architecture](docs/ARCHITECTURE.md)** - Compiler design
- **[Contributing](CONTRIBUTING.md)** - How to contribute

## License

Dual-licensed under MIT and Apache 2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

## Authors

- Demetrios Chiuratto Agourakis
- Dionisio Chiuratto Agourakis
