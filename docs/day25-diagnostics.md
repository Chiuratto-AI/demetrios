# Day 25: Error Recovery & Diagnostics Enhancement

## Overview

Day 25 focuses on making the Demetrios compiler produce high-quality, helpful error messages that guide developers toward fixing issues quickly. This includes typo detection with "did you mean" suggestions, rich type error visualization, and multiple output formats for IDE integration.

## Features Implemented

### Part A: Error Recovery Framework

The parser recovery system (`compiler/src/parser/recovery.rs`) provides:

- **Synchronization Points**: Automatic recovery to statement/item boundaries
- **Panic Mode Recovery**: Skip tokens until a sync point is found
- **Error Limits**: Configurable maximum errors before aborting
- **Partial Results**: Continue parsing after errors to find more issues

### Part B: Rich Diagnostics System

The diagnostic infrastructure (`compiler/src/diagnostic/`) includes:

- **Diagnostic Structure**: Level, code, message, labels, notes, help, suggestions
- **Error Codes**: Categorized codes (L=Lexer, P=Parser, R=Resolve, T=Type, etc.)
- **Source Spans**: Precise location tracking with file IDs
- **DiagnosticBuilder**: Fluent API for constructing diagnostics

```rust
use demetrios::diagnostic::{Diagnostic, DiagnosticBuilder, Span};

let diagnostic = DiagnosticBuilder::error("T0001", "Type mismatch")
    .with_label(expr_span, "expected `int`, found `bool`")
    .with_help("the function signature declares return type `int`")
    .build();
```

### Part C: Error Rendering

Three output formats are supported (`compiler/src/diagnostic/emitter.rs`):

#### Human (Terminal) Output
```
error[T0001]: Type mismatch
 --> src/main.d:5:12
  |
5 |     return true
  |            ^^^^ expected `int`, found `bool`
  |
  = help: the function signature declares return type `int`
```

#### JSON Output
```json
{
  "level": "error",
  "code": "T0001",
  "message": "Type mismatch",
  "spans": [
    {
      "file": "src/main.d",
      "start_line": 5,
      "start_col": 12,
      "end_line": 5,
      "end_col": 16
    }
  ]
}
```

#### SARIF Output
Static Analysis Results Interchange Format for integration with GitHub, Azure DevOps, and other tools.

### Part D: Typo Detection & Suggestions

The typo detection module (`compiler/src/diagnostic/typo.rs`) provides:

#### String Similarity Algorithms
- **Levenshtein Distance**: Basic edit distance
- **Damerau-Levenshtein**: Edit distance with transpositions
- **Jaro Similarity**: Character-based similarity (0.0 to 1.0)
- **Jaro-Winkler**: Favors common prefixes

#### Usage
```rust
use demetrios::diagnostic::typo::{find_similar, SuggestionBuilder};

// Find similar names
let suggestions = find_similar("prnt", ["print", "println", "panic"].iter().copied(), 2, 3);
// Returns: [TypoSuggestion { text: "print", score: 0.95, distance: 1 }]

// Context-aware suggestions
let builder = SuggestionBuilder::new()
    .with_scope_names(vec!["myVariable".into()]);
    
if let Some(msg) = builder.did_you_mean_variable("myVarible") {
    println!("{}", msg); // "did you mean `myVariable`?"
}
```

### Part E: Type Error Improvements

The type diff module (`compiler/src/diagnostic/type_diff.rs`) provides:

#### Type Difference Visualization
```rust
use demetrios::diagnostic::type_diff::{TypeDiff, render_type_diff};

let diff = TypeDiff::compute("Array<int, 10>", "Array<float, 10>");
// Detects structural difference at argument position 0

let diff = TypeDiff::compute("fn() -> int with IO, Alloc", "fn() -> int with IO");
// Detects missing effect: Alloc
```

#### TypeErrorBuilder
```rust
use demetrios::diagnostic::type_diff::TypeErrorBuilder;

let diagnostic = TypeErrorBuilder::new("int", "bool", span)
    .with_context("in return statement")
    .with_unification_step("matching return type", "int", "bool", Some(ret_span))
    .build();
```

## CLI Commands

### `dc explain <CODE>`
Show detailed explanation of an error code:
```bash
dc explain T0001
```

### `dc error-index`
List all error codes:
```bash
dc error-index --category type --format markdown
```

### `dc diagnostics` subcommands

#### `dc diagnostics check <FILE>`
Check a file with rich diagnostic output:
```bash
dc diagnostics check src/main.d --format human
dc diagnostics check src/main.d --format json
dc diagnostics check src/main.d --format sarif
```

#### `dc diagnostics similar <NAME>`
Find similar names for typo detection:
```bash
dc diagnostics similar prnt
# Output:
# Similar functions:
#   print (distance: 1, score: 0.95)
#   println (distance: 3, score: 0.89)
```

#### `dc diagnostics render <MESSAGE>`
Test diagnostic rendering:
```bash
dc diagnostics render "Type mismatch" --level error --code T0001 --format human
```

#### `dc diagnostics stats <FILE>`
Show diagnostic statistics:
```bash
dc diagnostics stats src/main.d
```

## Error Code Categories

| Prefix | Category | Description |
|--------|----------|-------------|
| L | Lexer | Tokenization errors |
| P | Parser | Syntax errors |
| R | Resolve | Name resolution errors |
| T | Type | Type checking errors |
| F | Effect | Effect system errors |
| O | Ownership | Ownership/borrowing errors |
| M | Pattern | Pattern matching errors |
| X | Macro | Macro expansion errors |
| I | Module | Import/module errors |
| C | Codegen | Code generation errors |
| E | Internal | Internal compiler errors |

## Architecture

```
compiler/src/diagnostic/
├── mod.rs          # Core types: Span, Diagnostic, DiagnosticHandler
├── codes.rs        # Error codes registry and documentation
├── emitter.rs      # HumanEmitter, JsonEmitter, SarifEmitter
├── suggestion.rs   # Code fix suggestions
├── typo.rs         # Typo detection algorithms
└── type_diff.rs    # Type mismatch visualization
```

## Integration Points

### IDE Integration
- JSON output for real-time error display
- SARIF for GitHub code scanning
- Source spans for jump-to-definition

### CI/CD Integration
- Machine-readable output formats
- Error counts for build status
- SARIF upload to security dashboards

## Testing

Run the diagnostic tests:
```bash
cargo test --lib diagnostic
```

All 32 diagnostic tests should pass:
- Span operations
- Source file line/column mapping
- Error code categories
- Suggestion patterns
- Typo detection algorithms
- Type diff computation
- JSON/SARIF emitter output

## Performance Considerations

- String interning reduces memory for repeated identifiers
- Lazy evaluation of suggestions (only computed when needed)
- Bounded edit distance computation for typo detection
- Configurable error limits prevent runaway diagnostics

## Future Enhancements

1. **LSP Integration**: Real-time diagnostics in editors
2. **Quick Fixes**: Automated code fixes for common errors
3. **Error Groups**: Collapse related errors
4. **Historical Context**: "This error often occurs when..."
5. **Multilingual Messages**: Localized error messages
