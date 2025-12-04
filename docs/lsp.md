# Demetrios LSP Server

The Demetrios Language Server Protocol (LSP) implementation provides IDE integration
for the D programming language.

## Installation

### Building the LSP Server

```bash
cd compiler
cargo build --release --features lsp
```

The LSP binary will be at `target/release/demetrios-lsp`.

### VS Code Extension

1. Navigate to the extension directory:
   ```bash
   cd editors/vscode
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Compile the extension:
   ```bash
   npm run compile
   ```

4. Install in VS Code:
   - Open VS Code
   - Run "Developer: Install Extension from Location..."
   - Select the `editors/vscode` directory

## Features

### Real-time Diagnostics

The LSP server provides real-time error reporting:

- **Syntax Errors**: Invalid syntax detected by the parser
- **Type Errors**: Type mismatches, undefined types
- **Effect Errors**: Missing effect declarations, unhandled effects
- **Ownership Errors**: Use-after-move, linearity violations
- **Refinement Errors**: Failed SMT constraint proofs

Diagnostics update as you type with debouncing for performance.

### Hover Information

Hover over any identifier to see:

- **Variables**: Type, ownership mode, effects
- **Functions**: Full signature with effects and return type
- **Types**: Type definition and documentation
- **Keywords**: Language documentation
- **Units**: Unit of measure information

Example hover for a variable:
```
dose: mg (linear)
Unit: milligram
```

### Go to Definition

Press F12 or Ctrl+Click to navigate to:

- Function definitions
- Type definitions
- Variable declarations
- Module declarations

Works across files in the same workspace.

### Find All References

Find all usages of:

- Variables
- Functions
- Types
- Effects

Access via Shift+F12 or right-click menu.

### Code Completion

Context-aware completions for:

**Top-level declarations:**
- `fn`, `let`, `type`, `effect`, `module`, `import`, `struct`, `enum`, `kernel`

**Type context:**
- Built-in types: `i32`, `i64`, `f32`, `f64`, `bool`, `String`, etc.
- User-defined types
- Generic type parameters

**Expression context:**
- Variables in scope
- Function names
- Keywords: `if`, `match`, `for`, `while`, `return`, etc.

**Effect context:**
- Built-in effects: `IO`, `Mut`, `Alloc`, `GPU`, `Prob`
- User-defined effects

**Unit context:**
- SI units: `m`, `kg`, `s`, `A`, `K`, `mol`, `cd`
- Derived units: `N`, `J`, `W`, `Pa`, `Hz`
- Medical units: `mg`, `mL`, `mg/mL`

Snippets are provided for common patterns:
```
fn → fn name(params) -> ReturnType { }
if → if condition { } else { }
match → match expr { pattern => result }
```

### Semantic Tokens

Rich syntax highlighting with custom token types:

| Token Type | Description |
|------------|-------------|
| `effect` | Effect names (IO, Mut, etc.) |
| `unit` | Unit annotations (kg, m/s) |
| `refinement` | Refinement predicates |
| `lifetime` | Lifetime parameters |

Custom modifiers:

| Modifier | Description |
|----------|-------------|
| `mutable` | Mutable bindings |
| `linear` | Linear types |
| `affine` | Affine types |
| `unsafe` | Unsafe blocks |

## Configuration

### VS Code Settings

```json
{
  "demetrios.serverPath": "/path/to/demetrios-lsp",
  "demetrios.trace.server": "verbose",
  "demetrios.inlayHints.enable": true
}
```

### Server Options

```bash
# Start in stdio mode (for editors)
demetrios-lsp --stdio

# Show version
demetrios-lsp --version

# Show help
demetrios-lsp --help
```

## Architecture

```
┌─────────────┐     LSP/JSON-RPC      ┌──────────────────┐
│   VS Code   │ ◄──────────────────► │  demetrios-lsp   │
│  Extension  │                       │                  │
└─────────────┘                       │  ┌────────────┐  │
                                      │  │  Document  │  │
                                      │  │  Manager   │  │
                                      │  └────────────┘  │
                                      │        │         │
                                      │        ▼         │
                                      │  ┌────────────┐  │
                                      │  │  Analysis  │  │
                                      │  │    Host    │  │
                                      │  └────────────┘  │
                                      │        │         │
                                      │        ▼         │
                                      │  ┌────────────┐  │
                                      │  │   Lexer    │  │
                                      │  │   Parser   │  │
                                      │  │   Checker  │  │
                                      │  └────────────┘  │
                                      └──────────────────┘
```

### Components

- **Document Manager**: Rope-based text storage with incremental updates
- **Analysis Host**: Coordinates semantic analysis and caching
- **Feature Providers**: Hover, completion, definition, references, etc.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `tower-lsp` | LSP protocol implementation |
| `tokio` | Async runtime |
| `dashmap` | Thread-safe document storage |
| `ropey` | Efficient text rope |
| `url` | URI handling |
| `async-trait` | Async trait support |

## Troubleshooting

### Server not starting

1. Ensure the LSP binary is built with the `lsp` feature
2. Check the server path in VS Code settings
3. Enable trace logging: `"demetrios.trace.server": "verbose"`

### Slow performance

1. Large files may cause delays - consider splitting
2. Check for recursive imports
3. SMT verification can be slow - disable if not needed

### Missing completions

1. Ensure the file is saved
2. Check for syntax errors that prevent parsing
3. Verify the document is recognized (`.d` or `.dem` extension)

## Development

### Running tests

```bash
cd compiler
cargo test --features lsp
```

### Debug logging

Set environment variable:
```bash
RUST_LOG=demetrios_lsp=debug demetrios-lsp --stdio
```
