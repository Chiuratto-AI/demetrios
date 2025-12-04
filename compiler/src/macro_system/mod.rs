//! Macro system and compile-time metaprogramming
//!
//! Implements:
//! - Declarative macros (macro_rules!)
//! - Procedural macros (derive, attribute, function-like)
//! - Compile-time function execution (CTFE)
//! - Scientific domain-specific macros

pub mod token_tree;
pub mod pattern;
pub mod declarative;
pub mod proc_macro;
pub mod derive;
pub mod ctfe;
pub mod scientific;

pub use token_tree::{TokenTree, TokenWithCtx, SyntaxContext, Delimiter, MacroError};
pub use pattern::{Pattern, FragmentSpecifier, Bindings, PatternMatcher};
pub use declarative::{MacroDef, MacroArm, MacroExpander};
pub use proc_macro::{TokenStream, ProcMacroDef, ProcMacroKind, ProcMacroRegistry, ProcMacroError};
pub use derive::{DeriveInput, parse_derive_input};
pub use ctfe::{ConstValue, CtfeContext, CtfeError};

/// Macro expansion context
pub struct MacroContext {
    /// Declarative macro expander
    pub declarative: MacroExpander,
    
    /// Procedural macro registry
    pub proc_macros: ProcMacroRegistry,
    
    /// Compile-time evaluation context
    pub ctfe: CtfeContext,
}

impl MacroContext {
    pub fn new() -> Self {
        MacroContext {
            declarative: MacroExpander::new(),
            proc_macros: ProcMacroRegistry::new(),
            ctfe: CtfeContext::new(),
        }
    }
}

impl Default for MacroContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_context_creation() {
        let ctx = MacroContext::new();
        assert!(ctx.declarative.macros.is_empty());
    }
}
