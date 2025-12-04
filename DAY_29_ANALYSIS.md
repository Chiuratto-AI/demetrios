# Day 29: Parser Integration Analysis

## Current Parser Structure

### Parser Organization
- **File**: `compiler/src/parser/mod.rs`
- **Size**: ~1500 lines
- **Key Functions**:
  - `parse_expr()` — Expression parsing (line 1233)
  - `parse_primary()` — Primary expression parsing (line 1427)
  - `parse_stmt()` — Statement parsing
  - `parse_item()` — Item parsing

### Expression Parsing Flow
```
parse_expr()
  ├── parse_assignment()
  ├── parse_logical_or()
  ├── parse_logical_and()
  ├── parse_equality()
  ├── parse_comparison()
  ├── parse_additive()
  ├── parse_multiplicative()
  ├── parse_unary()
  └── parse_primary()
```

### Primary Expression Parsing
The `parse_primary()` function handles:
- Literals (numbers, strings, booleans)
- Identifiers and paths
- Parenthesized expressions
- Block expressions
- If expressions
- Match expressions
- Loop expressions
- Closures
- Tuples
- Arrays
- Struct literals

## AST Structure

### Expr Enum (lines 574-700+)
Current variants include:
- Literal, Path, Binary, Unary
- Call, MethodCall, Field, TupleField, Index
- Cast, Block, If, Match
- Loop, While, For, Return, Break, Continue
- Closure, Tuple, Array, StructLit
- Try, Perform, Handle

### Item Enum (lines 48-60)
Current variants include:
- Function, Struct, Enum, Trait, Impl
- TypeAlias, Effect, Handler, Import, Extern, Global

### Stmt Enum
Need to check for statement types

## Integration Points

### 1. Expression Macros
**Location**: `parse_primary()` function
**Syntax**: `macro_name!(args)`
**Implementation**:
- Detect identifier followed by `!`
- Parse macro arguments (token tree)
- Create MacroInvocation expr variant

### 2. Statement Macros
**Location**: `parse_stmt()` function
**Syntax**: `macro_name!(args);`
**Implementation**:
- Similar to expression macros
- Create MacroInvocation stmt variant

### 3. Item Macros
**Location**: `parse_item()` function
**Syntax**: `macro_name!(args);`
**Implementation**:
- Similar to expression macros
- Create MacroInvocation item variant

## Design Decisions

### 1. MacroInvocation Structure
```rust
pub struct MacroInvocation {
    pub id: NodeId,
    pub name: String,
    pub args: Vec<TokenTree>,
    pub span: Span,
}
```

### 2. AST Integration
- Add `MacroInvocation { id, name, args, span }` to Expr enum
- Add `MacroInvocation { id, name, args, span }` to Stmt enum
- Add `MacroInvocation { id, name, args, span }` to Item enum

### 3. Parser Functions
```rust
fn parse_macro_invocation(&mut self) -> Result<MacroInvocation, ParseError>
fn parse_macro_args(&mut self) -> Result<Vec<TokenTree>, ParseError>
```

## Implementation Plan

### Step 1: Add MacroInvocation to AST
- Add struct definition
- Add variants to Expr, Stmt, Item enums

### Step 2: Implement Parser Functions
- `parse_macro_invocation()` — Main macro parsing
- `parse_macro_args()` — Argument parsing
- Integrate with `parse_primary()`
- Integrate with `parse_stmt()`
- Integrate with `parse_item()`

### Step 3: Handle Token Trees
- Import TokenTree from macro_system
- Implement token tree parsing
- Handle nested delimiters

### Step 4: Error Handling
- Invalid macro names
- Missing arguments
- Unclosed delimiters
- Invalid token sequences

## Testing Strategy

### Unit Tests
- Macro invocation parsing
- Argument parsing
- Error cases

### Integration Tests
- Expression macros
- Statement macros
- Item macros
- Nested macros

### Edge Cases
- Empty arguments
- Nested delimiters
- Multiple macros
- Macro in macro

## Expected Changes

### Files to Modify
1. `compiler/src/ast/mod.rs` — Add MacroInvocation
2. `compiler/src/parser/mod.rs` — Implement parsing

### Lines of Code
- AST changes: ~50 lines
- Parser changes: ~200-300 lines
- Tests: ~300-400 lines
- Total: ~600-800 lines

## Success Criteria

- ✅ Parser recognizes macro invocations
- ✅ Macro invocations parsed into AST
- ✅ All contexts supported (expr, stmt, item)
- ✅ 50+ tests passing
- ✅ No regressions in existing parser
- ✅ Documentation complete

---

**Status**: ✅ **ANALYSIS COMPLETE**

**Next**: Task 2 - Design AST Changes
