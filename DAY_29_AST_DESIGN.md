# Day 29: AST Design for Macro Invocation

## MacroInvocation Structure

### Core Definition
```rust
#[derive(Debug, Clone)]
pub struct MacroInvocation {
    pub id: NodeId,
    pub name: String,
    pub args: Vec<TokenTree>,
    pub span: Span,
}
```

### Fields
- **id**: Unique node identifier for tracking
- **name**: Macro name (e.g., "vec", "assert")
- **args**: Token trees representing macro arguments
- **span**: Source location for error reporting

## Integration with Expr Enum

### Current Expr Variants (partial)
```rust
pub enum Expr {
    Literal { ... },
    Path { ... },
    Binary { ... },
    Unary { ... },
    Call { ... },
    // ... other variants
}
```

### New Variant
```rust
pub enum Expr {
    // ... existing variants
    MacroInvocation(MacroInvocation),
}
```

### Usage Examples
```d
// Expression macro
let v = vec![1, 2, 3];
let m = matrix![[1, 2], [3, 4]];
let result = assert!(condition);
```

## Integration with Stmt Enum

### Current Stmt Variants
```rust
pub enum Stmt {
    Expression { ... },
    Item { ... },
    // ... other variants
}
```

### New Variant
```rust
pub enum Stmt {
    // ... existing variants
    MacroInvocation(MacroInvocation),
}
```

### Usage Examples
```d
// Statement macro
println!("Hello");
assert!(x > 0);
debug_assert!(condition);
```

## Integration with Item Enum

### Current Item Variants
```rust
pub enum Item {
    Function { ... },
    Struct { ... },
    Enum { ... },
    // ... other variants
}
```

### New Variant
```rust
pub enum Item {
    // ... existing variants
    MacroInvocation(MacroInvocation),
}
```

### Usage Examples
```d
// Item macro
#[derive(Debug, Clone)]
struct Point { x: i32, y: i32 }

macro_rules! vec { ... }

#[cfg(test)]
mod tests { ... }
```

## TokenTree Integration

### Import Statement
```rust
use crate::macro_system::token_tree::TokenTree;
```

### TokenTree Variants
```rust
pub enum TokenTree {
    Token(TokenWithCtx),
    Delimited(Delimiter, Vec<TokenTree>, Span),
}
```

### Macro Arguments Representation
```d
// Macro invocation: vec![1, 2, 3]
// Parsed as:
MacroInvocation {
    name: "vec",
    args: vec![
        TokenTree::Delimited(
            Delimiter::Bracket,
            vec![
                TokenTree::Token(1),
                TokenTree::Token(Comma),
                TokenTree::Token(2),
                TokenTree::Token(Comma),
                TokenTree::Token(3),
            ],
            span
        )
    ]
}
```

## Parser Integration Points

### 1. Expression Parsing
**Location**: `parse_primary()` function
**Trigger**: Identifier followed by `!`
**Action**: Call `parse_macro_invocation()`

### 2. Statement Parsing
**Location**: `parse_stmt()` function
**Trigger**: Identifier followed by `!` at statement level
**Action**: Call `parse_macro_invocation_stmt()`

### 3. Item Parsing
**Location**: `parse_item()` function
**Trigger**: Identifier followed by `!` at item level
**Action**: Call `parse_macro_invocation_item()`

## Parser Functions to Implement

### Main Function
```rust
fn parse_macro_invocation(&mut self) -> Result<MacroInvocation, ParseError> {
    let name = self.expect_ident()?;
    self.expect(TokenKind::Bang)?;
    let args = self.parse_macro_args()?;
    Ok(MacroInvocation {
        id: self.next_node_id(),
        name,
        args,
        span: /* ... */,
    })
}
```

### Argument Parsing
```rust
fn parse_macro_args(&mut self) -> Result<Vec<TokenTree>, ParseError> {
    match self.current().kind {
        TokenKind::LParen => self.parse_delimited_args(Delimiter::Parenthesis),
        TokenKind::LBracket => self.parse_delimited_args(Delimiter::Bracket),
        TokenKind::LBrace => self.parse_delimited_args(Delimiter::Brace),
        _ => Err(ParseError::expected("macro arguments")),
    }
}
```

### Delimited Argument Parsing
```rust
fn parse_delimited_args(&mut self, delim: Delimiter) 
    -> Result<Vec<TokenTree>, ParseError> 
{
    self.expect_open_delim(delim)?;
    let mut args = Vec::new();
    while !self.check_close_delim(delim) {
        args.push(self.parse_token_tree()?);
    }
    self.expect_close_delim(delim)?;
    Ok(args)
}
```

## Error Handling

### Error Types
```rust
pub enum ParseError {
    ExpectedMacroName { span: Span },
    ExpectedBang { span: Span },
    ExpectedMacroArgs { span: Span },
    UnclosedDelimiter { span: Span },
    InvalidTokenInMacro { span: Span },
}
```

### Error Messages
- "Expected macro name"
- "Expected '!' after macro name"
- "Expected macro arguments"
- "Unclosed delimiter in macro arguments"
- "Invalid token in macro arguments"

## Display Implementation

### For MacroInvocation
```rust
impl Display for MacroInvocation {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}!(", self.name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", arg)?;
        }
        write!(f, ")")
    }
}
```

## Testing Strategy

### Unit Tests
- Parse simple macro invocation
- Parse macro with arguments
- Parse nested macros
- Error cases

### Integration Tests
- Macro in expression context
- Macro in statement context
- Macro in item context
- Multiple macros

## Implementation Checklist

- [ ] Add MacroInvocation struct to ast/mod.rs
- [ ] Add MacroInvocation variant to Expr enum
- [ ] Add MacroInvocation variant to Stmt enum
- [ ] Add MacroInvocation variant to Item enum
- [ ] Implement Display for MacroInvocation
- [ ] Implement parse_macro_invocation()
- [ ] Implement parse_macro_args()
- [ ] Integrate with parse_primary()
- [ ] Integrate with parse_stmt()
- [ ] Integrate with parse_item()
- [ ] Add error types
- [ ] Create tests

---

**Status**: ✅ **DESIGN COMPLETE**

**Next**: Task 3 - Implement Parser Changes
