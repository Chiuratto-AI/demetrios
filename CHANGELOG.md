# Changelog

All notable changes to the Demetrios (D) compiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2025-11-28

### Added - Day 15: Documentation Generator

- **Doc Comment Parsing** (`doc/parser.rs`)
  - Support for `///` outer line doc comments
  - Support for `//!` inner line doc comments
  - Support for `/** */` outer block doc comments
  - Support for `/*! */` inner block doc comments
  - Markdown support via `pulldown-cmark`
  - Attribute-style docs: `@param`, `@returns`, `@example`, `@since`, `@deprecated`
  - Cross-reference syntax: `[item]` linking

- **Documentation Model** (`doc/model.rs`)
  - `CrateDoc`, `ModuleDoc`, `FunctionDoc`, `TypeDoc`, `TraitDoc`, `ConstantDoc`
  - `SearchIndex` with name-based and full-text term indexing
  - Type information: parameters, generics, where clauses
  - Source file tracking and line number references

- **HTML Generation** (`doc/html/`)
  - Responsive HTML templates for all documentation pages
  - Dark/light theme support with CSS variables
  - Syntax highlighting for D language code
  - Interactive features: theme toggle, copy code buttons, keyboard search
  - Breadcrumb navigation and sidebar

- **mdBook Integration** (`doc/book/`)
  - Complete mdBook structure generation
  - Auto-generated chapters: Introduction, Getting Started, Reference, API
  - Cross-linking between guide and API documentation
  - `book.toml` and `SUMMARY.md` generation

- **Doctest Runner** (`doc/doctest.rs`)
  - Extract and run code examples from documentation
  - Support for `should_panic`, `ignore`, `no_run`, `compile_fail` attributes
  - Documentation coverage statistics
  - Test summary with pass/fail counts and timing

- **CLI Commands**
  - `dc doc` - Generate HTML documentation
  - `dc doc-book` - Generate mdBook documentation
  - `dc doctest` - Run documentation tests
  - `dc doc-coverage` - Show documentation coverage statistics

- **Lexer Updates**
  - Added `DocCommentOuter`, `DocCommentInner`, `DocBlockOuter`, `DocBlockInner` tokens
  - Updated comment skip patterns to preserve doc comments

### Changed
- Updated `lib.rs` to export `doc` module
- Added `pulldown-cmark` dependency for markdown rendering

## [0.11.0] - 2025-11-28

### Added - Day 11: LLVM Backend for AOT Compilation

- **LLVM Code Generation** (`codegen/llvm/`)
  - Full LLVM IR generation from HLIR
  - Type mapping to LLVM types
  - Function compilation with proper calling conventions
  - Control flow: if/else, loops, match expressions
  - Arithmetic and comparison operations

- **Optimization Passes**
  - Multiple optimization levels (O0, O1, O2, O3, Os, Oz)
  - Standard LLVM optimization pipeline
  - Function inlining and dead code elimination
  - Loop optimizations

- **Native Code Emission**
  - Object file generation
  - Assembly output option
  - Executable linking with system linker
  - Cross-platform target support

- **CLI Enhancements**
  - `dc build` command for AOT compilation
  - `--emit-llvm` for LLVM IR output
  - `--emit-asm` for assembly output
  - `-O` flag for optimization level
  - `--target` for cross-compilation

## [0.10.0] - 2025-11-27

### Added - Day 10: LSP Server for IDE Integration

- **LSP Server Core** (`tower-lsp` based)
  - Full Language Server Protocol implementation
  - Async architecture with `tokio` runtime
  - Document synchronization with incremental updates
  - Rope-based text storage for efficient editing (`ropey`)
  - Thread-safe document management (`dashmap`)

- **Real-time Diagnostics**
  - Syntax error reporting from parser
  - Type error reporting from type checker
  - Effect system violation detection
  - Ownership/linearity error reporting
  - Refinement type constraint failures

- **Hover Information**
  - Type information for variables and expressions
  - Documentation for keywords and built-ins
  - Effect signatures for functions
  - Unit annotations for scientific values

- **Go to Definition**
  - Jump to function definitions
  - Jump to type definitions
  - Jump to module declarations
  - Cross-file navigation support

- **Find All References**
  - Find all usages of variables
  - Find all usages of functions
  - Find all usages of types

- **Intelligent Code Completion**
  - Context-aware completions (top-level, expressions, types)
  - Keyword completions with snippets
  - Type name completions
  - Effect name completions
  - Unit completions for scientific computing
  - Built-in function completions

- **Semantic Tokens**
  - Rich syntax highlighting
  - Custom token types: effect, unit, refinement, lifetime
  - Custom modifiers: mutable, linear, affine, unsafe
  - Full token classification from lexer

- **VS Code Extension**
  - Language configuration for `.d` and `.dem` files
  - TextMate grammar for syntax highlighting
  - Extension commands: restart server, show HIR/HLIR, run file
  - Configurable settings for server path and trace level

### Changed
- Updated `Cargo.toml` with LSP feature flag and dependencies
- Added `demetrios-lsp` binary entry point

## [0.9.0] - 2025-11-27

### Added - Day 9: Refinement Types with Z3 SMT Solver

- **Refinement Type System**
  - Predicate-based type refinements
  - SMT-backed constraint verification via Z3
  - Compile-time proof of numeric constraints
  - Subtyping based on logical implication

- **Z3 Integration**
  - Optional `smt` feature flag
  - Automatic constraint extraction from types
  - Proof caching for performance
  - Detailed error messages for failed proofs

- **Refinement Syntax**
  - `x: {v: i32 | v > 0}` - positive integers
  - `x: {v: f64 | v >= 0.0 && v <= 1.0}` - probabilities
  - Array bounds refinements
  - Function pre/post conditions

## [0.8.0] - 2025-11-27

### Added - Day 8: Units of Measure, Source Maps, and Parser Recovery

- **Units of Measure System**
  - Compile-time dimensional analysis
  - SI base units (m, kg, s, A, K, mol, cd)
  - Common derived units (N, J, W, Pa, Hz, etc.)
  - Medical/pharmacological units (mg, mL, mg/mL, etc.)
  - Unit inference and checking
  - Automatic unit conversion

- **Source Maps**
  - Bidirectional source location mapping
  - Span tracking through all compiler phases
  - Debug info generation for source-level debugging
  - Error location precision

- **Parser Recovery**
  - Graceful error recovery
  - Multiple error collection
  - Synchronization tokens
  - Continued parsing after errors

## [0.7.0] - 2025-11-27

### Added - Day 7: HLIR (SSA-Based IR) + Cranelift JIT + REPL

- **HLIR (High-Level Low-Level IR)**
  - SSA-form intermediate representation
  - Basic blocks with phi nodes
  - Explicit control flow graph
  - Type-preserving lowering from HIR

- **Cranelift JIT Backend**
  - Just-in-time compilation
  - Fast development iteration
  - Native code execution
  - Optional via `jit` feature flag

- **Interactive REPL**
  - Read-Eval-Print-Loop
  - Expression evaluation
  - Definition persistence
  - Command history

## [0.6.0] - 2025-11-26

### Added - Day 6: HIR and Type Checking

- **HIR (High-level IR)**
  - Typed AST representation
  - Resolved symbols and types
  - Desugared language constructs

- **Type Checker**
  - Bidirectional type inference
  - Effect type checking
  - Ownership/linearity verification
  - Generic instantiation

## [0.5.0] - 2025-11-26

### Added - Day 5: Effect System

- **Algebraic Effects**
  - Effect declarations
  - Effect handlers
  - Effect polymorphism
  - Built-in effects: IO, Mut, Alloc, GPU, Prob

## [0.4.0] - 2025-11-26

### Added - Day 4: Ownership and Linearity

- **Ownership System**
  - Linear types (must use exactly once)
  - Affine types (use at most once)
  - Copy types (freely copyable)
  - Move semantics

## [0.3.0] - 2025-11-25

### Added - Day 3: AST and Parser

- **Abstract Syntax Tree**
  - Complete AST node definitions
  - Module structure
  - Expressions and statements
  - Pattern matching

- **Recursive Descent Parser**
  - Full language grammar
  - Operator precedence
  - Error messages

## [0.2.0] - 2025-11-25

### Added - Day 2: Lexer

- **Lexer Implementation**
  - Logos-based tokenization
  - All language tokens
  - String and numeric literals
  - Comments and whitespace

## [0.1.0] - 2025-11-25

### Added - Day 1: Project Setup

- Initial project structure
- Cargo workspace configuration
- Basic CLI scaffolding
- Documentation templates
