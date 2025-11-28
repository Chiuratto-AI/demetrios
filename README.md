# Demetrios (D)

A novel L0 systems + scientific programming language.

[![Version](https://img.shields.io/badge/version-0.12.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green.svg)](LICENSE)

## Features

- **Novel Syntax**: Designed for scientific and medical computing
- **Full Algebraic Effects**: IO, Mut, Alloc, GPU, Prob with handlers
- **Linear/Affine Types**: Safe resource management
- **Units of Measure**: Compile-time dimensional analysis
- **Refinement Types**: SMT-backed constraint verification
- **GPU-Native**: First-class GPU memory and kernels
- **IDE Support**: Full LSP server with VS Code extension
- **LLVM Backend**: Native AOT compilation with optimizations
- **Documentation Generator**: HTML docs, mdBook integration, doctests

## Building

```bash
cd compiler
cargo build --release

# With all features
cargo build --release --features "jit,smt,lsp"
```

## Usage

```bash
# Compile
dc compile program.d -o program

# Build native executable (requires --features llvm)
dc build program.d -O2

# Type check only
dc check program.d

# Run with JIT (requires --features jit)
dc run program.d

# REPL
dc repl

# Generate documentation
dc doc --open

# Generate mdBook
dc doc-book

# Run doctests
dc doctest

# Start LSP server (requires --features lsp)
demetrios-lsp --stdio
```

## IDE Support

Demetrios includes a full-featured Language Server Protocol (LSP) implementation:

- **Real-time Diagnostics**: Syntax, type, effect, and ownership errors
- **Hover Information**: Type info, documentation, and effect signatures
- **Go to Definition**: Navigate to function, type, and variable definitions
- **Find References**: Find all usages across the codebase
- **Code Completion**: Context-aware completions with snippets
- **Semantic Highlighting**: Rich syntax highlighting with custom token types

### VS Code Extension

Install the VS Code extension from `editors/vscode/`:

```bash
cd editors/vscode
npm install
npm run compile
# Then install the .vsix or use VS Code's "Developer: Install Extension from Location"
```

See [docs/lsp.md](docs/lsp.md) for detailed LSP documentation.

## Example

```d
module example

let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume

fn simulate(params: PKParams) -> Vec<f64> with Prob, Alloc {
    let eta = sample(Normal(0.0, 0.3))
    // ...
}

kernel fn matmul(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = gpu.thread_id.x
    // ...
}
```

## Architecture

```
Source -> Lexer -> Parser -> AST -> Type Checker -> HIR -> HLIR -> Codegen
                                           |
                                           v
                                    LSP Server (IDE)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Documentation

Demetrios includes a comprehensive documentation generator:

```d
/// Calculate drug concentration from dose and volume.
///
/// @param dose The drug dose in milligrams
/// @param volume The solution volume in milliliters
/// @returns Concentration in mg/mL
///
/// @example
/// ```d
/// let conc = calculate_concentration(500_mg, 10_mL)
/// assert(conc == 50_mg/mL)
/// ```
fn calculate_concentration(dose: mg, volume: mL) -> mg/mL {
    dose / volume
}
```

- **HTML Documentation**: Responsive, themed API documentation
- **mdBook Integration**: Generate readable guides and tutorials
- **Doctests**: Run code examples from documentation as tests
- **Coverage**: Track documentation coverage statistics

## Feature Flags

| Feature | Description |
|---------|-------------|
| `jit`   | Cranelift JIT backend for fast development |
| `llvm`  | LLVM backend for optimized native code |
| `smt`   | Z3 SMT solver for refinement type verification |
| `lsp`   | Language Server Protocol for IDE integration |
| `full`  | Enable all features |

## License

MIT OR Apache-2.0
