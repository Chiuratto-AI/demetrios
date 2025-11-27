# Demetrios Language Compiler — Claude Code Rules

## Project Identity

**Demetrios (D)** is a novel L0 systems + scientific programming language created by Demetrios Chiuratto Agourakis. This is NOT a dialect of Rust, Julia, or any existing language. D has its own syntax, semantics, and design philosophy.

## Repository Structure

```
/mnt/e/workspace/demetrios/
├── compiler/              # Rust compiler implementation
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs        # CLI entry point
│       ├── lib.rs         # Library root
│       ├── lexer/         # Tokenization (Logos)
│       ├── parser/        # Recursive descent + Pratt
│       ├── ast/           # Abstract syntax tree
│       ├── resolve/       # Name resolution
│       ├── check/         # Type checker
│       ├── types/         # Type system
│       ├── effects/       # Algebraic effects
│       ├── hir/           # High-level IR
│       ├── hlir/          # SSA-based IR
│       ├── mlir/          # MLIR integration
│       └── codegen/       # LLVM/Cranelift/GPU backends
├── stdlib/                # Standard library (D code)
├── docs/                  # Documentation
├── examples/              # Example programs
└── tests/                 # Integration tests
```

## Language Design Principles

### 1. Core Features
- **Algebraic Effects**: Full effect system with handlers (IO, Mut, Alloc, GPU, Prob, etc.)
- **Linear/Affine Types**: Resource safety, must-use semantics
- **Units of Measure**: Compile-time dimensional analysis (mg, mL, etc.)
- **Refinement Types**: Predicate constraints with SMT verification
- **GPU-Native**: First-class GPU memory and kernel syntax

### 2. Syntax Style
- `let` for immutable, `var` for mutable, `const` for compile-time
- `fn` for functions, `kernel fn` for GPU kernels
- `&T` for shared reference, `&!T` for exclusive reference (NOT `&mut`)
- `with Effect` for effect annotations
- `linear struct`, `affine struct` for resource types

### 3. Effect System
```d
fn read_file(path: string) -> string with IO, Panic { ... }
fn simulate() -> f64 with Prob, Alloc { ... }
```

Built-in effects: IO, Mut, Alloc, Panic, Async, GPU, Prob, Div

### 4. Type System
- Hindley-Milner with bidirectional checking
- Substructural types (linear, affine)
- Effect polymorphism
- Row polymorphism for records (future)

## Coding Standards

### Rust Code (Compiler)
- Use `thiserror` for error types
- Use `miette` for diagnostics with source spans
- Prefer `logos` for lexing
- No `unwrap()` in library code—use `?` or proper error handling
- All public items must have doc comments
- Tests for every module

### D Code (Examples/Stdlib)
- Follow D syntax as defined in the language spec
- Include type annotations for clarity
- Document effects in function signatures

## File Naming Conventions
- Rust: `snake_case.rs`
- D source: `snake_case.d`
- Documentation: `UPPER_CASE.md` for top-level, `Title_Case.md` for sections

## Commit Message Format
```
[component] Brief description

- Detail 1
- Detail 2

Closes #issue (if applicable)
```

Components: `lexer`, `parser`, `ast`, `resolve`, `check`, `types`, `effects`, `hir`, `hlir`, `codegen`, `cli`, `docs`, `stdlib`, `tests`

## Development Workflow

### Before Each Session
1. `cargo build` — must pass
2. `cargo test` — must pass
3. `cargo clippy` — no warnings

### After Each Session
1. Update documentation in `docs/`
2. Add/update tests
3. Commit with descriptive message

## Documentation Requirements

Every significant feature must have:
1. **Specification** in `docs/spec/`
2. **Tutorial** in `docs/tutorial/`
3. **API reference** in `docs/api/`
4. **Examples** in `examples/`

Documentation must be Q1-journal quality:
- Precise terminology
- Formal definitions where appropriate
- Cross-references
- Examples for every concept

## Error Handling Philosophy

Errors should:
1. Have precise source locations (spans)
2. Explain what went wrong
3. Suggest how to fix it
4. Be visually clear (use miette)

Example:
```
error[E0001]: Type mismatch
  ┌─ src/main.d:5:12
  │
5 │     return true
  │            ^^^^ expected `int`, found `bool`
  │
  = help: the function signature declares return type `int`
```

## Testing Strategy

1. **Unit tests**: In each module (`#[cfg(test)]`)
2. **Integration tests**: In `tests/` directory
3. **Example programs**: In `examples/`, must compile and run
4. **Fuzzing**: For parser (future)

## Performance Considerations

- Use arena allocation for AST nodes
- Intern all strings
- Avoid cloning where possible
- Profile before optimizing

## Dependencies Policy

- Minimize dependencies
- Prefer well-maintained crates
- No `unsafe` without justification
- Pin versions in Cargo.toml

## Current Phase

**Bootstrap Phase** (Days 1-7):
- [x] Day 1: Scaffold
- [x] Day 2: Stub files
- [x] Day 3: First pipeline
- [x] Day 4: Name resolution + type checking
- [ ] Day 5: Effects + ownership
- [ ] Day 6: HIR lowering
- [ ] Day 7: First codegen

## Key Commands

```bash
# Build
cargo build

# Test
cargo test

# Check specific file
cargo run -- check examples/minimal.d --show-ast --show-types

# Format
cargo fmt

# Lint
cargo clippy
```

## Important Notes

1. **This is a new language** — don't assume Rust/Julia semantics
2. **Effects are first-class** — every function has an effect signature
3. **Linear types matter** — track resource usage carefully
4. **Documentation is essential** — write it as you code
5. **Quality over speed** — this is a long-term project

## Contact

Creator: Demetrios Chiuratto Agourakis & Dionisio Chiuratto Agourakis
Project: Demetrios Programming Language