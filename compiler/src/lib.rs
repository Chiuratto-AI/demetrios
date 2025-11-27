//! Demetrios (D) Programming Language Compiler
//!
//! A novel L0 systems + scientific programming language with:
//! - Full algebraic effects with handlers
//! - Linear and affine types for safe resource management
//! - Units of measure with compile-time dimensional analysis
//! - Refinement types with SMT-backed verification
//! - GPU-native computation
//!
//! # Architecture
//!
//! ```text
//! Source → Lexer → Parser → AST → Type Checker → HIR → HLIR → Codegen
//! ```
//!
//! # Example
//!
//! ```d
//! module example
//!
//! let dose: mg = 500.0
//! let volume: mL = 10.0
//! let concentration: mg/mL = dose / volume
//!
//! fn simulate(params: PKParams) -> Vec<f64> with Prob, Alloc {
//!     let eta = sample(Normal(0.0, 0.3))
//!     // ...
//! }
//! ```

#![allow(dead_code)]
#![allow(unused_variables)]

pub mod ast;
pub mod check;
pub mod codegen;
pub mod common;
pub mod diagnostics;
pub mod effects;
pub mod hir;
pub mod hlir;
pub mod interp;
pub mod lexer;
pub mod mlir;
pub mod ownership;
pub mod parser;
pub mod repl;
pub mod resolve;
pub mod sourcemap;
pub mod types;

// Re-export diagnostics for convenience
pub use diagnostics::{CompileError, Reporter, SourceFile};

// Re-exports for convenience
pub use ast::Ast;
pub use hir::Hir;
pub use hlir::HlirModule;
pub use types::Type;

/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compile source code to an executable
pub fn compile(source: &str) -> miette::Result<Vec<u8>> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let hlir = hlir::lower(&hir);

    // TODO: Actual code generation
    Err(miette::miette!("Code generation not yet implemented"))
}

/// Type-check source code without compiling
pub fn typecheck(source: &str) -> miette::Result<Hir> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    check::check(&ast)
}

/// Parse source code to AST
pub fn parse(source: &str) -> miette::Result<Ast> {
    let tokens = lexer::lex(source)?;
    parser::parse(&tokens, source)
}

/// Interpret source code directly
pub fn interpret(source: &str) -> miette::Result<interp::Value> {
    let tokens = lexer::lex(source)?;
    let ast = parser::parse(&tokens, source)?;
    let hir = check::check(&ast)?;
    let mut interpreter = interp::Interpreter::new();
    interpreter.interpret(&hir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
