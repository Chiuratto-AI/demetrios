# Macro System API Reference

## Module: `macro_system::token_tree`

### Types

#### `TokenTree`
```rust
pub enum TokenTree {
    Token(TokenWithCtx),
    Delimited(Delimiter, Vec<TokenTree>, Span),
}
```

Represents a single token or delimited group in macro input/output.

#### `TokenWithCtx`
```rust
pub struct TokenWithCtx {
    pub token: Token,
    pub ctx: SyntaxContext,
}
```

A token with its hygiene context.

#### `SyntaxContext`
```rust
pub struct SyntaxContext(u64);

impl SyntaxContext {
    pub const ROOT: Self;
    pub fn fresh() -> Self;
}
```

Unique identifier for hygiene tracking.

#### `Delimiter`
```rust
pub enum Delimiter {
    Parenthesis,  // ()
    Bracket,      // []
    Brace,        // {}
    None,
}
```

### Errors

#### `MacroError`
```rust
pub enum MacroError {
    UnclosedDelimiter { span: Span },
    UnexpectedClosingDelimiter { span: Span },
    PatternMismatch { expected: String, found: String, span: Span },
    RecursionLimit { depth: usize, span: Span },
    UndefinedMetaVariable { name: String, span: Span },
    InvalidRepetition { span: Span },
    HygieneViolation { name: String, span: Span },
}
```

## Module: `macro_system::pattern`

### Types

#### `Pattern`
```rust
pub enum Pattern {
    Token(TokenKind),
    Literal(String),
    MetaVar { name: String, fragment: FragmentSpecifier },
    Group { delimiter: Delimiter, patterns: Vec<Pattern> },
    Repeat { patterns: Vec<Pattern>, separator: Option<Box<Pattern>>, kind: RepeatKind },
    Wildcard,
}
```

#### `FragmentSpecifier`
```rust
pub enum FragmentSpecifier {
    Ident, Ty, Expr, Stmt, Pat, Block, Item,
    Lifetime, Literal, Path, Tt, Vis, TokenTree,
    Effect, Unit,
}
```

#### `RepeatKind`
```rust
pub enum RepeatKind {
    ZeroOrMore,  // *
    OneOrMore,   // +
    Optional,    // ?
}
```

#### `Bindings`
```rust
pub struct Bindings {
    pub singles: HashMap<String, Capture>,
    pub repeats: HashMap<String, Vec<Bindings>>,
}

impl Bindings {
    pub fn get_single(&self, name: &str) -> Option<&Capture>;
    pub fn get_repeat(&self, name: &str) -> Option<&[Bindings]>;
    pub fn merge(&mut self, other: Bindings);
}
```

### Functions

#### `PatternMatcher::match_pattern`
```rust
pub fn match_pattern(
    &mut self,
    pattern: &Pattern,
    input: &[TokenTree],
) -> Result<(Bindings, usize), MacroError>;
```

Match a pattern against token trees, returning bindings and consumed count.

## Module: `macro_system::declarative`

### Types

#### `MacroDef`
```rust
pub struct MacroDef {
    pub name: String,
    pub arms: Vec<MacroArm>,
    pub is_pub: bool,
    pub is_exported: bool,
    pub doc: Option<String>,
    pub span: Span,
}
```

#### `MacroArm`
```rust
pub struct MacroArm {
    pub pattern: Vec<Pattern>,
    pub template: Vec<TemplateTree>,
    pub guard: Option<TemplateTree>,
}
```

#### `MacroExpander`
```rust
pub struct MacroExpander {
    pub macros: HashMap<String, MacroDef>,
}

impl MacroExpander {
    pub fn new() -> Self;
    pub fn define(&mut self, macro_def: MacroDef);
    pub fn expand(&mut self, name: &str, input: Vec<TokenTree>) 
        -> Result<Vec<TokenTree>, MacroError>;
}
```

## Module: `macro_system::proc_macro`

### Types

#### `TokenStream`
```rust
pub struct TokenStream {
    trees: Vec<TokenTree>,
}

impl TokenStream {
    pub fn new() -> Self;
    pub fn push(&mut self, tree: TokenTree);
    pub fn extend(&mut self, other: TokenStream);
    pub fn into_trees(self) -> Vec<TokenTree>;
}
```

#### `ProcMacroDef`
```rust
pub struct ProcMacroDef {
    pub name: String,
    pub kind: ProcMacroKind,
    pub implementation: ProcMacroImpl,
}
```

#### `ProcMacroKind`
```rust
pub enum ProcMacroKind {
    FunctionLike,
    Derive { trait_name: String, attributes: Vec<String> },
    Attribute { targets: Vec<AttributeTarget> },
}
```

#### `ProcMacroRegistry`
```rust
pub struct ProcMacroRegistry;

impl ProcMacroRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, macro_def: ProcMacroDef);
    pub fn invoke_function(&self, name: &str, input: TokenStream) 
        -> Result<TokenStream, ProcMacroError>;
    pub fn invoke_attribute(&self, name: &str, attr: TokenStream, item: TokenStream) 
        -> Result<TokenStream, ProcMacroError>;
    pub fn invoke_derive(&self, trait_name: &str, item: TokenStream) 
        -> Result<TokenStream, ProcMacroError>;
}
```

## Module: `macro_system::ctfe`

### Types

#### `ConstValue`
```rust
pub enum ConstValue {
    Unit, Bool(bool), Int(i128), Uint(u128), Float(f64),
    String(String), Char(char), Array(Vec<ConstValue>),
    Tuple(Vec<ConstValue>), Struct { ... }, Enum { ... },
    Function { ... }, Type(String), Error(String),
}
```

#### `CtfeContext`
```rust
pub struct CtfeContext;

impl CtfeContext {
    pub fn new() -> Self;
    pub fn lookup_var(&self, name: &str) -> Result<ConstValue, CtfeError>;
    pub fn set_var(&mut self, name: String, value: ConstValue);
    pub fn eval_binary_op(&self, op: &str, left: &ConstValue, right: &ConstValue) 
        -> Result<ConstValue, CtfeError>;
    pub fn eval_unary_op(&self, op: &str, operand: &ConstValue) 
        -> Result<ConstValue, CtfeError>;
}
```

#### `CtfeError`
```rust
pub struct CtfeError {
    pub message: String,
    pub span: Option<Span>,
    pub backtrace: Vec<String>,
}
```

## Module: `macro_system::scientific::units`

### Types

#### `Dimension`
```rust
pub struct Dimension {
    pub length: i8,
    pub mass: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminosity: i8,
}

impl Dimension {
    pub fn mul(&self, other: &Dimension) -> Dimension;
    pub fn div(&self, other: &Dimension) -> Dimension;
    pub fn pow(&self, n: i8) -> Dimension;
}
```

### Functions

#### `parse_unit`
```rust
pub fn parse_unit(name: &str) -> Option<Dimension>;
```

Parse a unit name (e.g., "mg", "L", "mM") to its dimension.

#### `expand_unit_macro`
```rust
pub fn expand_unit_macro(input: TokenStream) -> Result<TokenStream, ProcMacroError>;
```

Expand `unit!(value: unit_name)` macro.

## Module: `macro_system::scientific::autodiff`

### Types

#### `SymExpr`
```rust
pub enum SymExpr {
    Const(f64),
    Var(String),
    Binary(Box<SymExpr>, BinOp, Box<SymExpr>),
    Unary(UnOp, Box<SymExpr>),
    Call(String, Vec<SymExpr>),
}

impl SymExpr {
    pub fn diff(&self, var: &str) -> SymExpr;
    pub fn simplify(&self) -> SymExpr;
    pub fn to_tokens(&self) -> TokenStream;
}
```

#### `Gradient`
```rust
pub struct Gradient {
    pub partials: HashMap<String, SymExpr>,
}

impl Gradient {
    pub fn compute(expr: &SymExpr, vars: &[String]) -> Self;
}
```

## Error Handling

All macro operations return `Result<T, MacroError>` or `Result<T, ProcMacroError>`:

```d
match expander.expand("my_macro", input) {
    Ok(output) => { /* use output */ },
    Err(MacroError::PatternMismatch { expected, found, span }) => {
        eprintln!("Pattern mismatch at {:?}: expected {}, found {}", span, expected, found);
    }
    Err(e) => eprintln!("Macro error: {}", e),
}
```
