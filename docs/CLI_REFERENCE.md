# Demetrios CLI Reference

Complete reference for the `dc` (Demetrios Compiler) command-line interface.

## Usage

```bash
dc [OPTIONS] <COMMAND>
```

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help information |
| `-V, --version` | Print version information |

---

## Core Commands

### `compile`

Compile a D source file to an object file or library.

```bash
dc compile <FILE> [OPTIONS]
```

**Options:**
- `--output, -o <FILE>` - Output file path
- `--target <TARGET>` - Target triple (e.g., `x86_64-unknown-linux-gnu`)
- `--opt-level <LEVEL>` - Optimization level (0-3)
- `--emit <TYPE>` - Emit type: `obj`, `asm`, `llvm-ir`, `llvm-bc`

**Example:**
```bash
dc compile src/main.d -o main.o --opt-level 2
```

---

### `build`

Build a D source file to a native executable.

> **Note:** Requires `--features llvm` at compile time.

```bash
dc build <FILE> [OPTIONS]
```

**Options:**
- `--output, -o <FILE>` - Output executable path
- `--release` - Build with optimizations
- `--target <TARGET>` - Target triple

**Example:**
```bash
dc build src/main.d -o myapp --release
```

---

### `check`

Type-check a D source file without compiling.

```bash
dc check <FILE> [OPTIONS]
```

**Options:**
- `--show-ast` - Display the parsed AST
- `--show-types` - Display inferred types
- `--show-effects` - Display effect annotations

**Example:**
```bash
dc check src/main.d --show-types
```

---

### `run`

Run a D program using the interpreter.

```bash
dc run <FILE> [ARGS...]
```

**Example:**
```bash
dc run examples/hello.d
dc run src/main.d -- arg1 arg2
```

---

### `jit`

Run a D program using JIT (Just-In-Time) compilation.

> **Note:** Requires `--features jit` at compile time.

```bash
dc jit <FILE> [ARGS...]
```

**Example:**
```bash
dc jit examples/compute.d
```

---

### `repl`

Start the interactive Read-Eval-Print Loop.

```bash
dc repl
```

**REPL Commands:**
- `:help` - Show help
- `:quit` or `:q` - Exit REPL
- `:type <expr>` - Show type of expression
- `:load <file>` - Load a D file
- `:clear` - Clear the screen

---

## Code Quality Commands

### `fmt`

Format D source code according to style guidelines.

```bash
dc fmt <FILE> [OPTIONS]
```

**Options:**
- `--check` - Check formatting without modifying files
- `--write, -w` - Write changes to files (default)
- `--config <FILE>` - Use custom configuration file

**Example:**
```bash
dc fmt src/*.d --check
dc fmt src/main.d -w
```

---

### `lint`

Lint D source code for potential issues.

```bash
dc lint <FILE> [OPTIONS]
```

**Options:**
- `--fix` - Automatically fix issues where possible
- `--warn <LINT>` - Enable specific warning
- `--deny <LINT>` - Treat warning as error

**Example:**
```bash
dc lint src/main.d
dc lint src/*.d --fix
```

---

### `analyze`

Analyze code for metrics and structural issues.

```bash
dc analyze <FILE> [OPTIONS]
```

**Options:**
- `--complexity` - Show cyclomatic complexity
- `--dependencies` - Show dependency graph
- `--dead-code` - Detect unused code

---

### `fix`

Apply automatic fixes to source code.

```bash
dc fix <FILE> [OPTIONS]
```

---

## Documentation Commands

### `doc`

Generate documentation for a D package.

```bash
dc doc <PATH> [OPTIONS]
```

**Options:**
- `--output, -o <DIR>` - Output directory
- `--format <FMT>` - Output format: `html`, `markdown`, `json`
- `--private` - Include private items

**Example:**
```bash
dc doc src/ -o docs/api --format html
```

---

### `doc-book`

Generate mdBook-style documentation.

```bash
dc doc-book <PATH> [OPTIONS]
```

---

### `doctest`

Run documentation tests (code examples in doc comments).

```bash
dc doctest <FILE>
```

---

### `doc-coverage`

Show documentation coverage statistics.

```bash
dc doc-coverage <PATH>
```

---

## Testing & Benchmarking

### `test`

Run tests defined in D source files.

```bash
dc test <PATH> [OPTIONS]
```

**Options:**
- `--filter <PATTERN>` - Run only tests matching pattern
- `--jobs, -j <N>` - Number of parallel test jobs
- `--no-capture` - Don't capture stdout/stderr

**Example:**
```bash
dc test tests/
dc test src/ --filter "test_parser"
```

---

### `benchmark`

Run benchmark tests.

```bash
dc benchmark <PATH> [OPTIONS]
```

---

### `bench`

Benchmark interpreter vs JIT performance.

```bash
dc bench <FILE>
```

---

## Debugging & Profiling

### `debug`

Debug a D program with integrated debugger.

```bash
dc debug <FILE> [OPTIONS]
```

---

### `profile`

Profile a D program for performance analysis.

```bash
dc profile <FILE> [OPTIONS]
```

**Options:**
- `--output, -o <FILE>` - Profile output file
- `--format <FMT>` - Output format: `flamegraph`, `json`, `text`

---

### `debug-info`

Generate debug information for a compiled binary.

```bash
dc debug-info <FILE>
```

---

### `source-map`

Generate source map for compiled code.

```bash
dc source-map <FILE>
```

---

## Error Handling

### `explain`

Explain an error code with examples and solutions.

```bash
dc explain <ERROR_CODE>
```

**Example:**
```bash
dc explain E0001
dc explain E0042
```

---

### `error-index`

Show all error codes with brief descriptions.

```bash
dc error-index
```

---

### `diagnostics`

Diagnostics management commands.

```bash
dc diagnostics <SUBCOMMAND>
```

---

## Development Server

### `watch`

Watch files and rebuild on changes.

```bash
dc watch <PATH> [OPTIONS]
```

**Options:**
- `--command, -c <CMD>` - Command to run on rebuild
- `--clear` - Clear screen before each rebuild

**Example:**
```bash
dc watch src/ -c "dc run src/main.d"
```

---

### `serve`

Start development server with live reload.

```bash
dc serve <PATH> [OPTIONS]
```

**Options:**
- `--port, -p <PORT>` - Server port (default: 8080)
- `--host <HOST>` - Server host (default: localhost)

---

## Build System

### `build-system`

Build project using the integrated build system.

```bash
dc build-system <SUBCOMMAND>
```

**Subcommands:**
- `init` - Initialize a new project
- `build` - Build the project
- `clean` - Clean build artifacts
- `run` - Build and run

---

### `clean`

Clean build artifacts and cache.

```bash
dc clean [OPTIONS]
```

**Options:**
- `--all` - Remove all cached data
- `--target` - Clean only target directory

---

### `hook`

Run build hooks.

```bash
dc hook <HOOK_NAME>
```

---

## Target Management

### `target`

Target management commands for cross-compilation.

```bash
dc target <SUBCOMMAND>
```

**Subcommands:**
- `list` - List available targets
- `add <TARGET>` - Add a target
- `remove <TARGET>` - Remove a target
- `info <TARGET>` - Show target information

**Example:**
```bash
dc target list
dc target info x86_64-unknown-linux-gnu
```

---

### `sysroot`

Sysroot management for cross-compilation.

```bash
dc sysroot <SUBCOMMAND>
```

---

## Domain-Specific Commands

### `ontology`

Native ontology management commands.

```bash
dc ontology <SUBCOMMAND>
```

**Subcommands:**
- `build` - Build ontology database
- `query` - Query ontology
- `validate` - Validate ontology mappings

---

### `units`

Units of measure commands (dimensional analysis).

```bash
dc units <SUBCOMMAND>
```

**Subcommands:**
- `check` - Check unit consistency
- `convert` - Unit conversion calculator
- `list` - List defined units

---

### `linear`

Linear/affine types commands (resource safety).

```bash
dc linear <SUBCOMMAND>
```

**Subcommands:**
- `check` - Check linearity constraints
- `report` - Generate linearity report

---

### `layout`

Layout synthesis commands.

```bash
dc layout <SUBCOMMAND>
```

---

### `locality`

Locality analysis and optimization commands (Semantic-Physical Duality).

```bash
dc locality <SUBCOMMAND>
```

---

## Information

### `info`

Show information about the compiler.

```bash
dc info
```

Shows:
- Compiler version
- Build configuration
- Available features
- Target information

---

## Feature Flags

The compiler supports optional features that can be enabled at build time:

| Feature | Description |
|---------|-------------|
| `jit` | Cranelift JIT compilation |
| `llvm` | LLVM backend for native compilation |
| `lsp` | Language Server Protocol support |
| `smt` | Z3 SMT solver for refinement types |
| `gpu` | GPU codegen support |
| `cuda` | CUDA runtime integration |
| `distributed` | Distributed build support |
| `ontology` | Scientific ontology integration |
| `full` | All features |

**Building with features:**
```bash
cd compiler
cargo build --release --features jit
cargo build --release --features "jit lsp"
cargo build --release --features full
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error |
| 2 | Type error |
| 3 | Runtime error |
| 101 | Internal compiler error |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DC_LOG` | Set log level (trace, debug, info, warn, error) |
| `DC_BACKTRACE` | Enable backtrace on errors (0 or 1) |
| `DC_CACHE_DIR` | Override cache directory |
| `DC_SYSROOT` | Override sysroot path |

---

## Examples

### Typical Development Workflow

```bash
# Check code for errors
dc check src/main.d

# Format code
dc fmt src/*.d

# Run with interpreter
dc run src/main.d

# Run with JIT for better performance
dc jit src/main.d

# Build release binary
dc build src/main.d -o myapp --release
```

### Watch Mode Development

```bash
# Auto-rebuild on changes
dc watch src/ -c "dc run src/main.d"

# Live reload server
dc serve src/ --port 3000
```

### Testing

```bash
# Run all tests
dc test tests/

# Run specific tests
dc test tests/ --filter "parser"

# Run with verbose output
dc test tests/ -v
```

---

## See Also

- [Language Specification](spec/LANGUAGE_SPECIFICATION.md)
- [Architecture Guide](ARCHITECTURE.md)
- [LLM Programming Guide](LLM_PROGRAMMING_GUIDE.md)
