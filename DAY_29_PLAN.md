# Day 29: Parser Integration for Macro Invocation

## 🎯 Objective

Integrate macro invocation into the parser so that macros can be invoked in D source code.

## 📋 Tasks

### Task 1: Analyze Current Parser Structure
- [ ] Review parser/mod.rs to understand current structure
- [ ] Identify where macro invocations should be parsed
- [ ] Document parser state machine
- [ ] Create parser integration plan

### Task 2: Define Macro Invocation Syntax
- [ ] Declarative macro invocation: `macro_name!(args)`
- [ ] Procedural macro invocation: `#[macro_name(args)]`
- [ ] Attribute macro invocation: `#[macro_name = value]`
- [ ] Function-like macro invocation: `macro_name!(args)`

### Task 3: Implement Macro Invocation Parsing
- [ ] Add MacroInvocation to AST
- [ ] Implement parse_macro_invocation()
- [ ] Handle different invocation contexts (expr, stmt, item)
- [ ] Add error handling for invalid invocations

### Task 4: Integrate with Expression Parser
- [ ] Add macro invocation to expression parsing
- [ ] Handle precedence correctly
- [ ] Support nested macro invocations
- [ ] Add tests for expression macros

### Task 5: Integrate with Statement Parser
- [ ] Add macro invocation to statement parsing
- [ ] Support statement-level macros
- [ ] Handle macro expansion in statements
- [ ] Add tests for statement macros

### Task 6: Integrate with Item Parser
- [ ] Add macro invocation to item parsing
- [ ] Support item-level macros (derive, attributes)
- [ ] Handle macro expansion in items
- [ ] Add tests for item macros

### Task 7: Create Comprehensive Tests
- [ ] Unit tests for macro parsing
- [ ] Integration tests with macro expansion
- [ ] Error handling tests
- [ ] Edge case tests

### Task 8: Documentation
- [ ] Update parser documentation
- [ ] Create macro invocation guide
- [ ] Add examples
- [ ] Document integration points

## 📊 Deliverables

| Item | Status |
|------|--------|
| Parser modifications | [ ] |
| Macro invocation AST | [ ] |
| Expression parsing | [ ] |
| Statement parsing | [ ] |
| Item parsing | [ ] |
| Tests (50+) | [ ] |
| Documentation | [ ] |

## 🔧 Implementation Details

### AST Changes
```rust
pub enum Expr {
    // ... existing variants
    MacroInvocation {
        name: String,
        args: Vec<TokenTree>,
        span: Span,
    },
}

pub enum Stmt {
    // ... existing variants
    MacroInvocation {
        name: String,
        args: Vec<TokenTree>,
        span: Span,
    },
}

pub enum Item {
    // ... existing variants
    MacroInvocation {
        name: String,
        args: Vec<TokenTree>,
        span: Span,
    },
}
```

### Parser Functions
```rust
fn parse_macro_invocation(&mut self) -> Result<Expr, ParseError>
fn parse_macro_invocation_stmt(&mut self) -> Result<Stmt, ParseError>
fn parse_macro_invocation_item(&mut self) -> Result<Item, ParseError>
fn parse_macro_args(&mut self) -> Result<Vec<TokenTree>, ParseError>
```

## 🧪 Testing Strategy

### Unit Tests
- Token recognition
- Argument parsing
- Error handling
- Edge cases

### Integration Tests
- Macro invocation in expressions
- Macro invocation in statements
- Macro invocation in items
- Nested invocations

### Example Programs
- Simple macro invocation
- Complex macro invocation
- Error cases

## 📚 Documentation

### Parser Integration Guide
- Overview of macro invocation
- Syntax examples
- Integration points
- Error handling

### API Reference
- Parser functions
- AST types
- Error types

### Examples
- Basic macro invocation
- Advanced macro invocation
- Error handling

## 🚀 Success Criteria

- ✅ Parser recognizes macro invocations
- ✅ Macro invocations parsed into AST
- ✅ All contexts supported (expr, stmt, item)
- ✅ 50+ tests passing
- ✅ Comprehensive documentation
- ✅ No regressions in existing parser

## 📈 Estimated Effort

| Task | Effort | Status |
|------|--------|--------|
| Analysis | 1 hour | [ ] |
| Syntax definition | 1 hour | [ ] |
| Implementation | 4 hours | [ ] |
| Testing | 2 hours | [ ] |
| Documentation | 1 hour | [ ] |
| **Total** | **9 hours** | [ ] |

## 🔗 Related Files

- `compiler/src/parser/mod.rs` — Parser implementation
- `compiler/src/ast/mod.rs` — AST definitions
- `compiler/src/macro_system/mod.rs` — Macro system
- `docs/MACRO_INTEGRATION.md` — Integration guide

## 🎓 Learning Resources

- Current parser implementation
- Macro system documentation
- AST structure
- Token stream handling

## ✅ Completion Checklist

- [ ] Parser modifications complete
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Code committed
- [ ] Code pushed
- [ ] Release tag created

---

**Status**: 🚀 **READY TO START**

**Next**: Day 29 - Parser Integration
