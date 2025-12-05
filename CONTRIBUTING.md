# Contributing to Demetrios

Thank you for your interest in contributing to the Demetrios programming language!

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Rust 1.75+ (edition 2024)
- Git
- Optional: LLVM 15+ (for LLVM backend)
- Optional: Z3 (for SMT-based refinement type verification)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/Chiuratto-AI/demetrios.git
cd demetrios

# Build the compiler
cd compiler
cargo build

# Run tests
cargo test

# Build with all features
cargo build --features full
```

### Project Structure

```
demetrios/
├── compiler/          # Rust implementation of the D compiler
│   ├── src/           # Compiler source code
│   ├── benches/       # Benchmarks
│   └── tests/         # Compiler tests
├── stdlib/            # Standard library (D code)
├── spec/              # Language specification
├── docs/              # Documentation
├── examples/          # Example programs
├── editors/           # Editor integrations
├── tools/             # Development tools
└── tests/             # Language test suite
    ├── ui/            # UI tests (error messages)
    ├── run-pass/      # Tests that should compile and run
    └── compile-fail/  # Tests that should fail to compile
```

## How to Contribute

### Reporting Bugs

1. Search existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Demetrios version (`dc --version`)
   - Operating system
   - Minimal reproduction case
   - Expected vs actual behavior

### Suggesting Features

1. Check the roadmap and existing proposals
2. Open a discussion or issue
3. Describe the use case and motivation

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Ensure tests pass: `cargo test`
5. Ensure no warnings: `cargo clippy`
6. Format code: `cargo fmt`
7. Commit with a descriptive message
8. Push and open a pull request

### Commit Message Format

```
[component] Brief description

- Detail 1
- Detail 2

Closes #issue (if applicable)
```

Components: `lexer`, `parser`, `ast`, `resolve`, `check`, `types`, `effects`, `hir`, `hlir`, `codegen`, `cli`, `docs`, `stdlib`, `tests`

### Code Style

- Follow Rust idioms
- Use `thiserror` for error types
- Use `miette` for diagnostics
- No `unwrap()` in library code
- Document public APIs
- Write tests for new functionality

## Areas to Contribute

### Good First Issues

Look for issues labeled `good-first-issue` for beginner-friendly tasks.

### Documentation

- Improve existing documentation
- Add examples
- Write tutorials
- Translate documentation

### Standard Library

- Implement missing functions
- Add unit tests
- Improve documentation

### Compiler

- Optimize performance
- Improve error messages
- Add new features
- Fix bugs

### Tooling

- Editor integrations
- Build system improvements
- Debugging tools

## Testing

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture
```

### Writing Tests

- Unit tests go in the same file as the code
- Integration tests go in `tests/`
- Language tests go in the top-level `tests/` directory

## Review Process

1. All changes require review
2. CI must pass
3. Documentation must be updated
4. Tests must be added for new functionality

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.

## Questions?

- Open a discussion on GitHub
- Check the documentation at `docs/`

Thank you for contributing!
