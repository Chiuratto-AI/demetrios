# Demetrios Documentation — Daily Update Template

## Purpose

At the end of each development day, update documentation to Q1+ journal standards. This ensures:

1. **Traceability** — Every feature is documented when implemented
2. **Quality** — Documentation is written while context is fresh
3. **Usability** — Users can learn as features become available

---

## Files to Update

After each day, update the following files as applicable:

### 1. `docs/LANGUAGE_SPECIFICATION.md`
- Add new syntax constructs
- Update grammar in Appendix A
- Add new type rules
- Document new effects

### 2. `docs/IMPLEMENTATION_STATUS.md`
- Mark completed items
- Add new items to "In Progress"
- Update percentages

### 3. `docs/CHANGELOG.md`
- Add entry for the day
- List all changes with PR/commit references

### 4. `docs/tutorial/` (as features complete)
- `01_getting_started.md`
- `02_basic_types.md`
- `03_functions.md`
- `04_ownership.md`
- `05_effects.md`
- `06_linear_types.md`
- `07_units.md`
- `08_gpu.md`

### 5. `docs/api/` (compiler internals)
- `lexer.md`
- `parser.md`
- `ast.md`
- `resolver.md`
- `typechecker.md`
- `effects.md`
- `ownership.md`
- `hir.md`
- `codegen.md`

---

## Daily Update Checklist

```markdown
## Day N Update Checklist

### Before Starting
- [ ] `cargo build` passes
- [ ] `cargo test` passes
- [ ] All previous docs are current

### Documentation Updates

#### Language Specification
- [ ] New syntax documented
- [ ] Grammar updated
- [ ] Type rules added
- [ ] Examples added

#### Implementation Status
- [ ] Completed items checked
- [ ] New items added
- [ ] Progress percentages updated

#### Changelog
- [ ] Entry added for Day N
- [ ] All changes listed
- [ ] Breaking changes noted

#### Tutorial (if applicable)
- [ ] New tutorial section added
- [ ] Examples tested
- [ ] Cross-references added

#### API Docs (if applicable)
- [ ] New modules documented
- [ ] Public API documented
- [ ] Internal notes added

### Code Documentation
- [ ] All public items have doc comments
- [ ] Module-level docs updated
- [ ] Examples in doc comments compile

### Commit
- [ ] Documentation committed
- [ ] Commit message follows format
```

---

## Documentation Quality Standards

### Q1 Journal Quality

1. **Precision**
   - Use exact terminology
   - Define terms before use
   - No ambiguous language

2. **Formalism**
   - Type rules in inference notation
   - Grammar in EBNF
   - Algorithms in pseudocode

3. **Completeness**
   - All edge cases documented
   - Error conditions explained
   - Examples for every feature

4. **Cross-References**
   - Link related sections
   - Reference external sources
   - Cite prior art

### Example: Type Rule

```markdown
### 4.3.2 Function Application

**Syntax:**
```
call_expr ::= expr '(' [ expr { ',' expr } ] ')'
```

**Typing Rule:**
```
Γ ⊢ e₀ : (τ₁, ..., τₙ) → τ with ε₀
Γ ⊢ eᵢ : τᵢ with εᵢ  for i ∈ 1..n
────────────────────────────────────
Γ ⊢ e₀(e₁, ..., eₙ) : τ with ε₀ ∪ ε₁ ∪ ... ∪ εₙ
```

**Example:**
```d
fn add(a: int, b: int) -> int { a + b }

let result = add(1, 2)  // result : int
```

**Errors:**
- `E0001`: Arity mismatch — wrong number of arguments
- `E0002`: Type mismatch — argument type doesn't match parameter
- `E0003`: Effect escape — function effects not declared by caller
```

### Example: Feature Section

```markdown
## 7. Effect System

### 7.1 Overview

D features a complete algebraic effect system based on the work of
Plotkin and Pretnar [1]. Effects describe computational side effects
and are tracked in function signatures.

### 7.2 Syntax

Effects are declared after the return type using `with`:

```d
fn read_file(path: string) -> string with IO, Panic {
    // ...
}
```

### 7.3 Semantics

Effects form a commutative monoid under union...

### 7.4 Implementation Notes

The effect system is implemented in `src/effects/`:
- `mod.rs` — Module exports
- `inference.rs` — Effect inference algorithm
- `types.rs` — Effect type definitions

### 7.5 References

[1] Plotkin, G., & Pretnar, M. (2009). Handlers of algebraic effects.
    In European Symposium on Programming (pp. 80-94). Springer.
```

---

## File Templates

### CHANGELOG.md Template

```markdown
# Changelog

All notable changes to the Demetrios compiler are documented here.

## [Unreleased]

### Added
- 

### Changed
-

### Fixed
-

### Removed
-

---

## [Day 5] - 2024-XX-XX

### Added
- Effect inference (`src/effects/inference.rs`)
- Ownership checker (`src/ownership/`)
- Linear type enforcement
- Source-located diagnostics with miette
- Comprehensive error types in `src/diagnostics.rs`

### Changed
- `Span` type now includes file ID
- Error messages include source snippets

### Fixed
- Parser recovery after errors

---

## [Day 4] - 2024-XX-XX

### Added
- Symbol table (`src/resolve/symbols.rs`)
- Name resolution pass (`src/resolve/resolver.rs`)
- Bidirectional type checking
- Type environment with DefId-based lookups

---

## [Day 3] - 2024-XX-XX

### Added
- Lexer with Logos
- Recursive descent parser
- Pratt parser for expressions
- AST printer

---

## [Day 2] - 2024-XX-XX

### Added
- Type system stubs
- Effect definitions
- Unit of measure types
- Refinement type predicates

---

## [Day 1] - 2024-XX-XX

### Added
- Initial project scaffold
- Cargo.toml with dependencies
- CLI framework
- Module structure
```

### IMPLEMENTATION_STATUS.md Template

```markdown
# Implementation Status

Last updated: Day N

## Summary

| Component | Status | Progress |
|-----------|--------|----------|
| Lexer | ✅ Complete | 100% |
| Parser | ✅ Complete | 100% |
| AST | ✅ Complete | 100% |
| Resolver | ✅ Complete | 100% |
| Type Checker | 🟡 Partial | 60% |
| Effect System | 🟡 Partial | 40% |
| Ownership | 🟡 Partial | 50% |
| HIR | 🔴 Not Started | 0% |
| HLIR | 🔴 Not Started | 0% |
| LLVM Backend | 🔴 Not Started | 0% |
| GPU Backend | 🔴 Not Started | 0% |

## Detailed Status

### Lexer (100%)
- [x] Keywords
- [x] Operators
- [x] Literals
- [x] Unit literals
- [x] Comments
- [x] Error recovery

### Parser (100%)
- [x] Items (fn, struct, enum, etc.)
- [x] Statements
- [x] Expressions
- [x] Types
- [x] Patterns
- [x] Effect annotations

### Type Checker (60%)
- [x] Literal inference
- [x] Variable lookup
- [x] Binary operators
- [x] Function calls
- [ ] Generics
- [ ] Trait bounds
- [ ] Associated types

### Effect System (40%)
- [x] Effect definitions
- [x] Effect inference
- [x] Effect checking
- [ ] Effect handlers
- [ ] Effect polymorphism

### Ownership (50%)
- [x] Move tracking
- [x] Borrow checking
- [x] Linear types
- [ ] Lifetime inference
- [ ] Non-lexical lifetimes

## Next Milestones

1. **Week 2**: Complete type checker, begin HIR
2. **Week 3**: HIR generation, begin LLVM
3. **Week 4**: First compiled program
```

---

## End of Session Script

Run this at the end of each coding session:

```bash
#!/bin/bash
# update_docs.sh

echo "=== Demetrios Documentation Update ==="

cd /mnt/e/workspace/demetrios

# 1. Run tests
echo "Running tests..."
cargo test

# 2. Check for undocumented items
echo "Checking documentation..."
cargo doc --no-deps 2>&1 | grep -i "missing"

# 3. Format code
echo "Formatting..."
cargo fmt

# 4. Lint
echo "Linting..."
cargo clippy

# 5. Show git status
echo "Changes:"
git status --short

# 6. Reminder
echo ""
echo "=== Don't forget to update: ==="
echo "  - docs/LANGUAGE_SPECIFICATION.md"
echo "  - docs/IMPLEMENTATION_STATUS.md"
echo "  - docs/CHANGELOG.md"
echo ""
```

---

## Summary

**Documentation is not optional.** Every feature implemented must be documented before the session ends. This ensures:

1. Users can learn the language as it develops
2. Contributors understand the design
3. Future you remembers why things work the way they do

**Write docs as you code, not after.**
