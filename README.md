# Demetrios (D)

A novel L0 systems + scientific programming language.

## Features

- **Novel Syntax**: Designed for scientific and medical computing
- **Full Algebraic Effects**: IO, Mut, Alloc, GPU, Prob with handlers
- **Linear/Affine Types**: Safe resource management
- **Units of Measure**: Compile-time dimensional analysis
- **Refinement Types**: SMT-backed constraint verification
- **GPU-Native**: First-class GPU memory and kernels

## Building

```bash
cd compiler
cargo build --release
```

## Usage

```bash
# Compile
dc compile program.d -o program

# Type check only
dc check program.d

# Run with JIT
dc run program.d

# REPL
dc repl
```

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
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## License

MIT OR Apache-2.0
