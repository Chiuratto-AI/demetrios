# Contributing to Demetrios

Thank you for your interest in contributing to Demetrios! This document provides
guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and collaborative. We're building something new
together.

## How to Contribute

### Reporting Issues

1. Check existing issues first
2. Use the issue template
3. Include:
   - Demetrios version (`dc --version`)
   - Operating system
   - Minimal reproduction code
   - Expected vs actual behavior

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`cargo test`)
5. Run lints (`cargo clippy`)
6. Commit with clear messages
7. Open a pull request

### Development Setup

```bash
# Clone
git clone https://github.com/Chiuratto-AI/demetrios
cd demetrios

# Build
cargo build

# Test
cargo test

# Run compiler
cargo run -- check examples/hello.d

# Run with all features
cargo build --features full
```

### Code Style

- Follow Rust idioms
- Use `rustfmt` formatting
- Write doc comments for public items
- Add tests for new features
- Use meaningful variable and function names
- Keep functions focused and small

### Commit Messages

Follow conventional commits:

```
feat: add semantic distance caching
fix: correct bloom filter false positive rate
docs: update ontology loading guide
perf: optimize embedding similarity with SIMD
test: add integration tests for cross-ontology coercion
```

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Error message enhancements
- Additional examples
- Typo fixes
- Test coverage improvements

### Medium Complexity

- New ontology loaders (e.g., SNOMED-CT, ICD)
- Performance optimizations
- LSP features (hover, completion)
- CLI improvements

### Advanced

- Type system extensions
- Embedding model improvements
- Cross-ontology mapping algorithms
- GPU compute optimizations
- Formal verification integration

## Project Structure

```
demetrios/
├── compiler/           # The Demetrios compiler
│   ├── src/
│   │   ├── lexer/      # Tokenization
│   │   ├── parser/     # Syntax analysis
│   │   ├── ast/        # Abstract syntax tree
│   │   ├── check/      # Type checking
│   │   ├── ontology/   # Ontology infrastructure
│   │   │   ├── loader/     # OWL/OBO loading
│   │   │   ├── distance/   # Semantic distance
│   │   │   ├── embedding/  # Vector embeddings
│   │   │   └── storage/    # Backend storage
│   │   ├── epistemic/  # Knowledge types
│   │   ├── profiling/  # Performance profiling
│   │   ├── diagnostics/# Error reporting
│   │   └── codegen/    # Code generation
│   ├── tests/          # Integration tests
│   └── benches/        # Benchmarks
├── std/                # Standard library
├── examples/           # Example programs
├── docs/               # Documentation
│   ├── papers/         # Academic papers
│   └── research/       # Research notes
└── spec/               # Formal specifications
    └── formal/         # LaTeX type theory
```

## Testing

### Running Tests

```bash
# All tests
cargo test

# Specific test file
cargo test --test integration_ontology_e2e

# With output
cargo test -- --nocapture

# Benchmarks
cargo bench
```

### Writing Tests

- Unit tests go in the same file as the code
- Integration tests go in `tests/`
- Use descriptive test names
- Test edge cases and error conditions

```rust
#[test]
fn test_semantic_distance_reflexivity() {
    let index = build_test_index();
    let d = index.distance(&iri("Drug"), &iri("Drug"));
    assert!(d.is_exact(), "Self-distance should be zero");
}
```

## Documentation

### Doc Comments

```rust
/// Calculate semantic distance between two ontology terms.
///
/// # Arguments
///
/// * `from` - Source IRI
/// * `to` - Target IRI
///
/// # Returns
///
/// A `SemanticDistance` with conceptual, path, IC, and embedding components.
///
/// # Example
///
/// ```
/// let d = index.distance(&aspirin_iri, &drug_iri);
/// assert!(d.conceptual < 0.3);
/// ```
pub fn distance(&self, from: &IRI, to: &IRI) -> SemanticDistance {
    // ...
}
```

### README Updates

When adding features, update:
- Feature list in README.md
- Quick start guide if applicable
- API documentation in docs/

## Review Process

1. All PRs require at least one review
2. CI must pass (tests, clippy, formatting)
3. Documentation must be updated
4. Breaking changes need discussion

## Release Process

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. CI builds and publishes

## Questions?

- Open a discussion on GitHub
- Check existing documentation
- Review closed issues for similar questions

Thank you for helping make Demetrios better!
