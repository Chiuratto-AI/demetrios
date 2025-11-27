//! Diagnostic reporting with source locations
//!
//! This module provides rich error messages with source locations using miette.

use crate::common::Span;
use miette::{Diagnostic, NamedSource, SourceSpan};
use std::sync::Arc;
use thiserror::Error;

/// Source file for error reporting
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub content: Arc<str>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: Arc::from(content.into()),
        }
    }

    pub fn to_named_source(&self) -> NamedSource<String> {
        NamedSource::new(self.name.clone(), self.content.to_string())
    }
}

/// Convert our Span to miette's SourceSpan
impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        SourceSpan::new(span.start.into(), span.len())
    }
}

/// Compiler diagnostic
#[derive(Error, Debug, Diagnostic, Clone)]
pub enum CompileError {
    // === Parse Errors ===
    #[error("Unexpected token: expected {expected}, found {found}")]
    #[diagnostic(code(parse::unexpected_token))]
    UnexpectedToken {
        expected: String,
        found: String,
        #[label("unexpected token here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Unexpected end of file")]
    #[diagnostic(code(parse::unexpected_eof))]
    UnexpectedEof {
        #[label("expected more tokens")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Resolution Errors ===
    #[error("Undefined variable `{name}`")]
    #[diagnostic(
        code(resolve::undefined_var),
        help("did you mean to declare this variable with `let`?")
    )]
    UndefinedVariable {
        name: String,
        #[label("not found in this scope")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Undefined type `{name}`")]
    #[diagnostic(code(resolve::undefined_type))]
    UndefinedType {
        name: String,
        #[label("type not found")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Duplicate definition of `{name}`")]
    #[diagnostic(code(resolve::duplicate_def))]
    DuplicateDefinition {
        name: String,
        #[label("redefined here")]
        span: SourceSpan,
        #[label("first defined here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Type Errors ===
    #[error("Type mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(typecheck::mismatch))]
    TypeMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },

    #[error("Cannot unify `{t1}` with `{t2}`")]
    #[diagnostic(code(typecheck::unification_failed))]
    UnificationFailed {
        t1: String,
        t2: String,
        #[label("type mismatch here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Missing type annotation")]
    #[diagnostic(
        code(typecheck::annotation_required),
        help("add a type annotation: `let x: Type = ...`")
    )]
    AnnotationRequired {
        #[label("cannot infer type")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Effect Errors ===
    #[error("Unhandled effect `{effect}`")]
    #[diagnostic(
        code(effect::unhandled),
        help(
            "either handle this effect with `with handler {{ ... }}` or add it to the function signature"
        )
    )]
    UnhandledEffect {
        effect: String,
        #[label("effect `{effect}` escapes here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Effect `{effect}` not declared in function signature")]
    #[diagnostic(code(effect::undeclared))]
    UndeclaredEffect {
        effect: String,
        #[label("this operation has effect `{effect}`")]
        span: SourceSpan,
        #[label("function does not declare this effect")]
        fn_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot perform `{effect}` in pure context")]
    #[diagnostic(code(effect::pure_context))]
    EffectInPureContext {
        effect: String,
        #[label("effectful operation here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Ownership Errors ===
    #[error("Use of moved value `{name}`")]
    #[diagnostic(code(ownership::use_after_move))]
    UseAfterMove {
        name: String,
        #[label("value used here after move")]
        use_span: SourceSpan,
        #[label("value moved here")]
        move_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable because it is already borrowed")]
    #[diagnostic(code(ownership::already_borrowed))]
    AlreadyBorrowed {
        name: String,
        #[label("cannot borrow as mutable")]
        span: SourceSpan,
        #[label("previous borrow here")]
        prev_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable more than once")]
    #[diagnostic(code(ownership::double_mut_borrow))]
    DoubleMutBorrow {
        name: String,
        #[label("second mutable borrow here")]
        span: SourceSpan,
        #[label("first mutable borrow here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Linearity Errors ===
    #[error("Linear value `{name}` used more than once")]
    #[diagnostic(
        code(linear::multiple_use),
        help("linear values must be used exactly once")
    )]
    LinearMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Linear value `{name}` not consumed")]
    #[diagnostic(
        code(linear::not_consumed),
        help("linear values must be explicitly consumed before going out of scope")
    )]
    LinearNotConsumed {
        name: String,
        #[label("linear value declared here")]
        decl_span: SourceSpan,
        #[label("goes out of scope here without being consumed")]
        scope_end: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Affine value `{name}` used more than once")]
    #[diagnostic(code(affine::multiple_use))]
    AffineMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Unit Errors ===
    #[error("Unit mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(unit::mismatch))]
    UnitMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot add values with different units: `{u1}` and `{u2}`")]
    #[diagnostic(code(unit::incompatible_add))]
    IncompatibleUnits {
        u1: String,
        u2: String,
        #[label("incompatible units")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },
}

/// Error reporter that collects diagnostics
pub struct Reporter {
    source: SourceFile,
    errors: Vec<CompileError>,
    warnings: Vec<CompileError>,
}

impl Reporter {
    pub fn new(source: SourceFile) -> Self {
        Self {
            source,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn error(&mut self, error: CompileError) {
        self.errors.push(error);
    }

    pub fn warning(&mut self, warning: CompileError) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Create NamedSource for this file
    pub fn named_source(&self) -> NamedSource<String> {
        self.source.to_named_source()
    }

    /// Get the source file
    pub fn source(&self) -> &SourceFile {
        &self.source
    }

    /// Print all diagnostics
    pub fn emit_all(&self) {
        for warning in &self.warnings {
            eprintln!("{:?}", miette::Report::new(warning.clone()));
        }
        for error in &self.errors {
            eprintln!("{:?}", miette::Report::new(error.clone()));
        }
    }

    /// Consume and return errors
    pub fn into_errors(self) -> Vec<CompileError> {
        self.errors
    }

    /// Get errors by reference
    pub fn errors(&self) -> &[CompileError] {
        &self.errors
    }
}
