//! Semantic tokens provider for syntax highlighting
//!
//! Provides rich semantic tokens for enhanced syntax highlighting.

use tower_lsp::lsp_types::*;

use crate::lexer::{self, TokenKind};

/// Provider for semantic tokens
pub struct SemanticTokensProvider;

impl SemanticTokensProvider {
    /// Create a new semantic tokens provider
    pub fn new() -> Self {
        Self
    }

    /// Tokenize source for semantic highlighting
    pub fn tokenize(&self, source: &str) -> SemanticTokens {
        let mut data = Vec::new();

        let mut prev_line = 0u32;
        let mut prev_col = 0u32;

        // Lex the source
        if let Ok(tokens) = lexer::lex(source) {
            for token in &tokens {
                if let Some((token_type, modifiers)) = self.classify_token(&token.kind) {
                    // Calculate line and column from byte offset
                    let (line, col) = offset_to_line_col(source, token.span.start);
                    let line = line as u32;
                    let col = col as u32;
                    let length = (token.span.end - token.span.start) as u32;

                    // Delta encoding
                    let delta_line = line - prev_line;
                    let delta_col = if delta_line == 0 { col - prev_col } else { col };

                    // Push semantic token
                    data.push(SemanticToken {
                        delta_line,
                        delta_start: delta_col,
                        length,
                        token_type,
                        token_modifiers_bitset: modifiers,
                    });

                    prev_line = line;
                    prev_col = col;
                }
            }
        }

        SemanticTokens {
            result_id: None,
            data,
        }
    }

    /// Classify a token kind to semantic token type and modifiers
    fn classify_token(&self, kind: &TokenKind) -> Option<(u32, u32)> {
        match kind {
            // Keywords
            TokenKind::Fn
            | TokenKind::Let
            | TokenKind::Mut
            | TokenKind::Const
            | TokenKind::If
            | TokenKind::Else
            | TokenKind::While
            | TokenKind::For
            | TokenKind::Loop
            | TokenKind::Match
            | TokenKind::Return
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::In
            | TokenKind::As
            | TokenKind::Where
            | TokenKind::Pub
            | TokenKind::SelfLower
            | TokenKind::SelfUpper => Some((TOKEN_KEYWORD, 0)),

            // Type keywords
            TokenKind::Struct | TokenKind::Enum | TokenKind::Trait | TokenKind::Type => {
                Some((TOKEN_KEYWORD, 0))
            }

            // D-specific keywords
            TokenKind::Effect
            | TokenKind::Handler
            | TokenKind::Handle
            | TokenKind::With
            | TokenKind::Perform
            | TokenKind::Resume => Some((TOKEN_KEYWORD, 0)),

            TokenKind::Linear | TokenKind::Affine => Some((TOKEN_MODIFIER, 0)),

            TokenKind::Kernel | TokenKind::Device | TokenKind::Shared | TokenKind::Gpu => {
                Some((TOKEN_KEYWORD, MOD_ASYNC))
            }

            TokenKind::Async | TokenKind::Await | TokenKind::Spawn => {
                Some((TOKEN_KEYWORD, MOD_ASYNC))
            }

            TokenKind::Sample | TokenKind::Observe | TokenKind::Infer => Some((TOKEN_KEYWORD, 0)),

            TokenKind::Proof
            | TokenKind::Invariant
            | TokenKind::Requires
            | TokenKind::Ensures
            | TokenKind::Assert
            | TokenKind::Assume => Some((TOKEN_KEYWORD, 0)),

            TokenKind::Unsafe | TokenKind::Extern => Some((TOKEN_KEYWORD, MOD_UNSAFE)),

            TokenKind::Impl => Some((TOKEN_KEYWORD, 0)),

            TokenKind::Move | TokenKind::Copy | TokenKind::Drop => Some((TOKEN_KEYWORD, 0)),

            // Boolean literals
            TokenKind::True | TokenKind::False => Some((TOKEN_KEYWORD, 0)),

            // Numeric literals
            TokenKind::IntLit | TokenKind::HexLit | TokenKind::BinLit | TokenKind::OctLit => {
                Some((TOKEN_NUMBER, 0))
            }
            TokenKind::FloatLit => Some((TOKEN_NUMBER, 0)),

            // Unit literals - special highlighting
            TokenKind::IntUnitLit | TokenKind::FloatUnitLit => Some((TOKEN_UNIT, 0)),

            // String literals
            TokenKind::StringLit => Some((TOKEN_STRING, 0)),
            TokenKind::CharLit => Some((TOKEN_STRING, 0)),

            // Identifiers - would need semantic analysis for proper classification
            TokenKind::Ident => Some((TOKEN_VARIABLE, 0)),

            // Operators
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
            | TokenKind::Shr => Some((TOKEN_OPERATOR, 0)),

            // Assignment operators
            TokenKind::PlusEq
            | TokenKind::MinusEq
            | TokenKind::StarEq
            | TokenKind::SlashEq
            | TokenKind::PercentEq
            | TokenKind::AmpEq
            | TokenKind::PipeEq
            | TokenKind::CaretEq
            | TokenKind::ShlEq
            | TokenKind::ShrEq => Some((TOKEN_OPERATOR, 0)),

            // Arrows
            TokenKind::Arrow | TokenKind::FatArrow | TokenKind::LeftArrow => {
                Some((TOKEN_OPERATOR, 0))
            }

            // Module/Import
            TokenKind::Module | TokenKind::Import | TokenKind::Export => Some((TOKEN_KEYWORD, 0)),

            // Skip punctuation and delimiters
            TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::Comma
            | TokenKind::Semi
            | TokenKind::Colon
            | TokenKind::ColonColon
            | TokenKind::Dot
            | TokenKind::DotDot
            | TokenKind::DotDotDot
            | TokenKind::DotDotEq
            | TokenKind::At
            | TokenKind::Hash
            | TokenKind::Dollar
            | TokenKind::Question
            | TokenKind::Underscore
            | TokenKind::Eof => None,
        }
    }
}

impl Default for SemanticTokensProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert byte offset to line/column (0-indexed)
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let mut line = 0;
    let mut col = 0;

    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    (line, col)
}

// Token type indices (must match server capabilities order)
const TOKEN_NAMESPACE: u32 = 0;
const TOKEN_TYPE: u32 = 1;
const TOKEN_CLASS: u32 = 2;
const TOKEN_ENUM: u32 = 3;
const TOKEN_INTERFACE: u32 = 4;
const TOKEN_STRUCT: u32 = 5;
const TOKEN_TYPE_PARAMETER: u32 = 6;
const TOKEN_PARAMETER: u32 = 7;
const TOKEN_VARIABLE: u32 = 8;
const TOKEN_PROPERTY: u32 = 9;
const TOKEN_ENUM_MEMBER: u32 = 10;
const TOKEN_EVENT: u32 = 11;
const TOKEN_FUNCTION: u32 = 12;
const TOKEN_METHOD: u32 = 13;
const TOKEN_MACRO: u32 = 14;
const TOKEN_KEYWORD: u32 = 15;
const TOKEN_MODIFIER: u32 = 16;
const TOKEN_COMMENT: u32 = 17;
const TOKEN_STRING: u32 = 18;
const TOKEN_NUMBER: u32 = 19;
const TOKEN_REGEXP: u32 = 20;
const TOKEN_OPERATOR: u32 = 21;
const TOKEN_DECORATOR: u32 = 22;
// Custom types
const TOKEN_EFFECT: u32 = 23;
const TOKEN_UNIT: u32 = 24;
const TOKEN_REFINEMENT: u32 = 25;
const TOKEN_LIFETIME: u32 = 26;

// Token modifier bit flags (must match server capabilities order)
const MOD_DECLARATION: u32 = 1 << 0;
const MOD_DEFINITION: u32 = 1 << 1;
const MOD_READONLY: u32 = 1 << 2;
const MOD_STATIC: u32 = 1 << 3;
const MOD_DEPRECATED: u32 = 1 << 4;
const MOD_ABSTRACT: u32 = 1 << 5;
const MOD_ASYNC: u32 = 1 << 6;
const MOD_MODIFICATION: u32 = 1 << 7;
const MOD_DOCUMENTATION: u32 = 1 << 8;
const MOD_DEFAULT_LIBRARY: u32 = 1 << 9;
// Custom modifiers
const MOD_MUTABLE: u32 = 1 << 10;
const MOD_LINEAR: u32 = 1 << 11;
const MOD_AFFINE: u32 = 1 << 12;
const MOD_UNSAFE: u32 = 1 << 13;

// Suppress unused warnings for now (will be used with enhanced semantic analysis)
#[allow(dead_code)]
const _TOKENS: [u32; 12] = [
    TOKEN_NAMESPACE,
    TOKEN_TYPE,
    TOKEN_CLASS,
    TOKEN_ENUM,
    TOKEN_INTERFACE,
    TOKEN_STRUCT,
    TOKEN_TYPE_PARAMETER,
    TOKEN_PARAMETER,
    TOKEN_PROPERTY,
    TOKEN_ENUM_MEMBER,
    TOKEN_EVENT,
    TOKEN_MACRO,
];

#[allow(dead_code)]
const _CUSTOM_TOKENS: [u32; 3] = [TOKEN_EFFECT, TOKEN_REFINEMENT, TOKEN_LIFETIME];

#[allow(dead_code)]
const _MODIFIERS: [u32; 10] = [
    MOD_DECLARATION,
    MOD_DEFINITION,
    MOD_READONLY,
    MOD_STATIC,
    MOD_DEPRECATED,
    MOD_ABSTRACT,
    MOD_MODIFICATION,
    MOD_DOCUMENTATION,
    MOD_DEFAULT_LIBRARY,
    MOD_MUTABLE,
];

#[allow(dead_code)]
const _CUSTOM_MODIFIERS: [u32; 2] = [MOD_LINEAR, MOD_AFFINE];
