//! Token definitions for the Demetrios lexer

use crate::common::Span;
use logos::Logos;
use serde::{Deserialize, Serialize};

/// A token with its kind, span, and text
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

/// Token kinds recognized by the lexer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Logos, Serialize, Deserialize)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum TokenKind {
    // Keywords
    #[token("module")]
    Module,
    #[token("import")]
    Import,
    #[token("export")]
    Export,
    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
    #[token("const")]
    Const,
    #[token("type")]
    Type,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("trait")]
    Trait,
    #[token("impl")]
    Impl,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("in")]
    In,
    #[token("as")]
    As,
    #[token("where")]
    Where,
    #[token("pub")]
    Pub,
    #[token("self")]
    SelfLower,
    #[token("Self")]
    SelfUpper,

    // D-specific keywords
    #[token("effect")]
    Effect,
    #[token("handler")]
    Handler,
    #[token("handle")]
    Handle,
    #[token("with")]
    With,
    #[token("perform")]
    Perform,
    #[token("resume")]
    Resume,
    #[token("linear")]
    Linear,
    #[token("affine")]
    Affine,
    #[token("move")]
    Move,
    #[token("copy")]
    Copy,
    #[token("drop")]
    Drop,
    #[token("kernel")]
    Kernel,
    #[token("device")]
    Device,
    #[token("shared")]
    Shared,
    #[token("gpu")]
    Gpu,
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("spawn")]
    Spawn,
    #[token("sample")]
    Sample,
    #[token("observe")]
    Observe,
    #[token("infer")]
    Infer,
    #[token("proof")]
    Proof,
    #[token("invariant")]
    Invariant,
    #[token("requires")]
    Requires,
    #[token("ensures")]
    Ensures,
    #[token("assert")]
    Assert,
    #[token("assume")]
    Assume,
    #[token("unsafe")]
    Unsafe,
    #[token("extern")]
    Extern,

    // Boolean literals
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Literals
    #[regex(r"[0-9][0-9_]*", priority = 2)]
    IntLit,
    #[regex(r"0x[0-9a-fA-F][0-9a-fA-F_]*")]
    HexLit,
    #[regex(r"0b[01][01_]*")]
    BinLit,
    #[regex(r"0o[0-7][0-7_]*")]
    OctLit,
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?")]
    FloatLit,
    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLit,
    #[regex(r#"'([^'\\]|\\.)'"#)]
    CharLit,

    // Identifiers (priority 1 so _ token takes precedence)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 1)]
    Ident,

    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("^")]
    Caret,
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("~")]
    Tilde,
    #[token("!")]
    Bang,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    // Compound operators
    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AmpEq,
    #[token("|=")]
    PipeEq,
    #[token("^=")]
    CaretEq,
    #[token("<<=")]
    ShlEq,
    #[token(">>=")]
    ShrEq,

    // Arrows
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("<-")]
    LeftArrow,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // Punctuation
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token("::")]
    ColonColon,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token("...")]
    DotDotDot,
    #[token("..=")]
    DotDotEq,
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("$")]
    Dollar,
    #[token("?")]
    Question,
    #[token("_", priority = 2)]
    Underscore,

    // Special
    Eof,
}

impl TokenKind {
    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Module
                | TokenKind::Import
                | TokenKind::Export
                | TokenKind::Fn
                | TokenKind::Let
                | TokenKind::Mut
                | TokenKind::Const
                | TokenKind::Type
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Match
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::In
                | TokenKind::As
                | TokenKind::Where
                | TokenKind::Pub
                | TokenKind::SelfLower
                | TokenKind::SelfUpper
                | TokenKind::Effect
                | TokenKind::Handler
                | TokenKind::Handle
                | TokenKind::With
                | TokenKind::Perform
                | TokenKind::Resume
                | TokenKind::Linear
                | TokenKind::Affine
                | TokenKind::Move
                | TokenKind::Copy
                | TokenKind::Drop
                | TokenKind::Kernel
                | TokenKind::Device
                | TokenKind::Shared
                | TokenKind::Gpu
                | TokenKind::Async
                | TokenKind::Await
                | TokenKind::Spawn
                | TokenKind::Sample
                | TokenKind::Observe
                | TokenKind::Infer
                | TokenKind::Proof
                | TokenKind::Invariant
                | TokenKind::Requires
                | TokenKind::Ensures
                | TokenKind::Assert
                | TokenKind::Assume
                | TokenKind::Unsafe
                | TokenKind::Extern
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntLit
                | TokenKind::HexLit
                | TokenKind::BinLit
                | TokenKind::OctLit
                | TokenKind::FloatLit
                | TokenKind::StringLit
                | TokenKind::CharLit
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::Caret
                | TokenKind::Amp
                | TokenKind::Pipe
                | TokenKind::Tilde
                | TokenKind::Bang
                | TokenKind::Eq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::EqEq
                | TokenKind::Ne
                | TokenKind::Le
                | TokenKind::Ge
                | TokenKind::AmpAmp
                | TokenKind::PipePipe
                | TokenKind::Shl
                | TokenKind::Shr
        )
    }

    /// Get the string representation of the token
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenKind::Module => "module",
            TokenKind::Import => "import",
            TokenKind::Export => "export",
            TokenKind::Fn => "fn",
            TokenKind::Let => "let",
            TokenKind::Mut => "mut",
            TokenKind::Const => "const",
            TokenKind::Type => "type",
            TokenKind::Struct => "struct",
            TokenKind::Enum => "enum",
            TokenKind::Trait => "trait",
            TokenKind::Impl => "impl",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Match => "match",
            TokenKind::For => "for",
            TokenKind::While => "while",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Return => "return",
            TokenKind::In => "in",
            TokenKind::As => "as",
            TokenKind::Where => "where",
            TokenKind::Pub => "pub",
            TokenKind::SelfLower => "self",
            TokenKind::SelfUpper => "Self",
            TokenKind::Effect => "effect",
            TokenKind::Handler => "handler",
            TokenKind::Handle => "handle",
            TokenKind::With => "with",
            TokenKind::Perform => "perform",
            TokenKind::Resume => "resume",
            TokenKind::Linear => "linear",
            TokenKind::Affine => "affine",
            TokenKind::Move => "move",
            TokenKind::Copy => "copy",
            TokenKind::Drop => "drop",
            TokenKind::Kernel => "kernel",
            TokenKind::Device => "device",
            TokenKind::Shared => "shared",
            TokenKind::Gpu => "gpu",
            TokenKind::Async => "async",
            TokenKind::Await => "await",
            TokenKind::Spawn => "spawn",
            TokenKind::Sample => "sample",
            TokenKind::Observe => "observe",
            TokenKind::Infer => "infer",
            TokenKind::Proof => "proof",
            TokenKind::Invariant => "invariant",
            TokenKind::Requires => "requires",
            TokenKind::Ensures => "ensures",
            TokenKind::Assert => "assert",
            TokenKind::Assume => "assume",
            TokenKind::Unsafe => "unsafe",
            TokenKind::Extern => "extern",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::IntLit => "<int>",
            TokenKind::HexLit => "<hex>",
            TokenKind::BinLit => "<bin>",
            TokenKind::OctLit => "<oct>",
            TokenKind::FloatLit => "<float>",
            TokenKind::StringLit => "<string>",
            TokenKind::CharLit => "<char>",
            TokenKind::Ident => "<ident>",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Caret => "^",
            TokenKind::Amp => "&",
            TokenKind::Pipe => "|",
            TokenKind::Tilde => "~",
            TokenKind::Bang => "!",
            TokenKind::Eq => "=",
            TokenKind::Lt => "<",
            TokenKind::Gt => ">",
            TokenKind::EqEq => "==",
            TokenKind::Ne => "!=",
            TokenKind::Le => "<=",
            TokenKind::Ge => ">=",
            TokenKind::AmpAmp => "&&",
            TokenKind::PipePipe => "||",
            TokenKind::Shl => "<<",
            TokenKind::Shr => ">>",
            TokenKind::PlusEq => "+=",
            TokenKind::MinusEq => "-=",
            TokenKind::StarEq => "*=",
            TokenKind::SlashEq => "/=",
            TokenKind::PercentEq => "%=",
            TokenKind::AmpEq => "&=",
            TokenKind::PipeEq => "|=",
            TokenKind::CaretEq => "^=",
            TokenKind::ShlEq => "<<=",
            TokenKind::ShrEq => ">>=",
            TokenKind::Arrow => "->",
            TokenKind::FatArrow => "=>",
            TokenKind::LeftArrow => "<-",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Semi => ";",
            TokenKind::Colon => ":",
            TokenKind::ColonColon => "::",
            TokenKind::Dot => ".",
            TokenKind::DotDot => "..",
            TokenKind::DotDotDot => "...",
            TokenKind::DotDotEq => "..=",
            TokenKind::At => "@",
            TokenKind::Hash => "#",
            TokenKind::Dollar => "$",
            TokenKind::Question => "?",
            TokenKind::Underscore => "_",
            TokenKind::Eof => "<eof>",
        }
    }
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
