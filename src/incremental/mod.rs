//! Incremental infrastructure (edits, lexer reuse, parser glue)
//!
//! These modules provide the building blocks for incremental compilation,
//! allowing small text changes to reuse previous work where possible.

pub mod edits;
pub mod lexer;
pub mod parser;

pub use edits::{EditSequence, TextChange, TextEdit};
pub use lexer::{CachedToken, IncrementalLexer, TokenCache};
pub use parser::{IncrementalAst, IncrementalNode, IncrementalParseResult, NodeId};
