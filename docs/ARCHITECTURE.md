# Demetrios Compiler Architecture

## Overview

The Demetrios (D) compiler is a multi-stage compiler that transforms D source code
into optimized native executables or JIT-compiled code.

## Pipeline

```
Source Code
    |
    v
+-------+
| Lexer |  Tokenization using Logos
+-------+
    |
    v
+--------+
| Parser |  Recursive descent, produces AST
+--------+
    |
    v
+-----+
| AST |  Abstract Syntax Tree (untyped)
+-----+
    |
    v
+--------------+
| Type Checker |  Bidirectional type inference
+--------------+
    |
    v
+-----+
| HIR |  High-level IR (typed AST)
+-----+
    |
    v
+------+
| HLIR |  SSA-based low-level IR
+------+
    |
    v
+------+     +-------------+     +-----+
| MLIR | --> | Optimization| --> | GPU |
+------+     +-------------+     +-----+
    |
    v
+---------+
| Codegen |  LLVM or Cranelift
+---------+
    |
    v
+------------+
| Executable |
+------------+
```

## Directory Structure

```
compiler/src/
|-- main.rs          # CLI entry point
|-- lib.rs           # Library root
|-- common.rs        # Shared types (Span, NodeId, etc.)
|
|-- lexer/           # Tokenization
|   |-- mod.rs       # Lexer implementation
|   +-- tokens.rs    # Token definitions (Logos)
|
|-- parser/          # Parsing
|   +-- mod.rs       # Recursive descent parser
|
|-- ast/             # Abstract Syntax Tree
|   +-- mod.rs       # AST node definitions
|
|-- hir/             # High-level IR
|   +-- mod.rs       # Typed AST after type checking
|
|-- hlir/            # SSA-based IR
|   +-- mod.rs       # Low-level IR for optimization
|
|-- types/           # Type system
|   |-- mod.rs       # Module root
|   |-- core.rs      # Core type definitions
|   |-- ownership.rs # Ownership rules (linear, affine)
|   |-- effects.rs   # Algebraic effect system
|   |-- refinement.rs# Refinement types with predicates
|   +-- units.rs     # Units of measure
|
|-- check/           # Type checker
|   +-- mod.rs       # Bidirectional type inference
|
|-- effects/         # Effect system runtime
|   +-- mod.rs       # Effect handlers
|
|-- mlir/            # MLIR integration
|   +-- mod.rs       # MLIR lowering (stub)
|
+-- codegen/         # Code generation backends
    |-- mod.rs       # Backend selection
    |-- llvm.rs      # LLVM backend
    |-- cranelift.rs # Cranelift JIT
    +-- gpu.rs       # GPU codegen (CUDA/SPIR-V)
```

## Key Features

### Type System

D has a sophisticated type system with:

1. **Bidirectional Type Inference**: Types flow both up and down the AST
2. **Linear/Affine Types**: Resources that must be used exactly once (linear)
   or at most once (affine)
3. **Effect Types**: Functions declare their effects (IO, Mut, Alloc, etc.)
4. **Refinement Types**: Types with predicates verified at compile time
5. **Units of Measure**: Dimensional analysis prevents unit errors

### Effect System

Effects are first-class in D:

```d
effect IO {
    fn print(s: String);
    fn read_line() -> String;
}

fn greet(name: String) with IO {
    perform IO.print("Hello, " + name)
}
```

### Ownership Model

D uses ownership tracking similar to Rust but with explicit modes:

- `copy`: Values can be freely copied (primitives)
- `affine`: Values can be used at most once (default)
- `linear`: Values must be used exactly once (resources)

### GPU Support

D has first-class GPU support:

```d
kernel fn vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}
```

## Compilation Backends

### LLVM

The LLVM backend produces optimized native code. Enable with `--features llvm`.

### Cranelift

The Cranelift backend provides fast JIT compilation for development.
Enable with `--features jit`.

### GPU

The GPU backend generates CUDA PTX or SPIR-V code for compute kernels.
Enable with `--features gpu`.

## Future Work

- [ ] Complete MLIR integration
- [ ] Incremental compilation
- [ ] IDE support (LSP)
- [ ] Package manager
- [ ] Standard library
