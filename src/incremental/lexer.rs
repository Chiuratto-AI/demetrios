//! Incremental lexer that reuses tokens from previous passes.
//!
//! The implementation is conservative: it invalidates tokens that overlap with
//! the edited range, re-lexes a small window, and then rebuilds the token list.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::common::Span;
use crate::incremental::edits::TextEdit;
use crate::lexer::{Token, TokenKind};
use logos::Logos;
use miette::Result;

/// A cached token with its byte range
#[derive(Debug, Clone)]
pub struct CachedToken {
    pub token: Token,
    pub range: Range<usize>,
}

/// Token cache for incremental lexing
#[derive(Debug, Clone)]
pub struct TokenCache {
    /// Tokens indexed by start position
    tokens: BTreeMap<usize, CachedToken>,

    /// Source text hash for validation
    source_hash: u64,
}

impl TokenCache {
    pub fn new() -> Self {
        TokenCache {
            tokens: BTreeMap::new(),
            source_hash: 0,
        }
    }

    /// Build cache from full lex
    pub fn from_tokens(tokens: Vec<Token>, source: &str) -> Self {
        let mut cache = TokenCache::new();
        cache.source_hash = hash_source(source);

        for token in tokens {
            let range = token.span.start..token.span.end;
            cache
                .tokens
                .insert(token.span.start, CachedToken { token, range });
        }

        cache
    }

    /// Get token at position
    pub fn get(&self, position: usize) -> Option<&CachedToken> {
        self.tokens.get(&position)
    }

    /// Get tokens in range
    pub fn tokens_in_range(&self, range: Range<usize>) -> Vec<&CachedToken> {
        self.tokens
            .range(range.start..range.end)
            .map(|(_, t)| t)
            .collect()
    }

    /// Invalidate tokens affected by edit
    pub fn invalidate(&mut self, edit: &TextEdit) {
        // Find tokens that overlap with the edit range
        let affected: Vec<usize> = self
            .tokens
            .iter()
            .filter(|(_, t)| ranges_overlap(&t.range, &edit.range))
            .map(|(k, _)| *k)
            .collect();

        for key in affected {
            self.tokens.remove(&key);
        }

        // Adjust positions of tokens after the edit
        let delta = edit.length_delta();
        if delta != 0 {
            let after_edit: Vec<(usize, CachedToken)> = self
                .tokens
                .range(edit.range.end..)
                .map(|(k, v)| (*k, v.clone()))
                .collect();

            for (old_pos, _) in &after_edit {
                self.tokens.remove(old_pos);
            }

            for (old_pos, mut cached) in after_edit {
                let new_pos = (old_pos as isize + delta) as usize;
                cached.range.start = (cached.range.start as isize + delta) as usize;
                cached.range.end = (cached.range.end as isize + delta) as usize;
                cached.token.span.start = (cached.token.span.start as isize + delta) as usize;
                cached.token.span.end = (cached.token.span.end as isize + delta) as usize;
                self.tokens.insert(new_pos, cached);
            }
        }
    }
}

/// Incremental lexer
pub struct IncrementalLexer {
    cache: TokenCache,
}

impl IncrementalLexer {
    pub fn new() -> Self {
        IncrementalLexer {
            cache: TokenCache::new(),
        }
    }

    /// Full lex (first time or cache miss)
    pub fn lex_full(&mut self, source: &str) -> Result<Vec<Token>> {
        let tokens = lex_with_offset(source, 0)?;
        self.cache = TokenCache::from_tokens(tokens.clone(), source);
        Ok(tokens)
    }

    /// Incremental lex after edit
    pub fn lex_incremental(&mut self, new_source: &str, edit: &TextEdit) -> Result<Vec<Token>> {
        // Invalidate affected tokens
        self.cache.invalidate(edit);

        // Find the range that needs re-lexing
        let relex_start = self.find_relex_start(edit.range.start);
        let relex_end = self.find_relex_end(edit.range.start + edit.new_text.len(), new_source);

        // Re-lex the affected region
        let region = &new_source[relex_start..relex_end];
        let new_tokens = lex_with_offset(region, relex_start)?;

        // Insert new tokens into cache
        for token in &new_tokens {
            let range = token.span.start..token.span.end;
            self.cache.tokens.insert(
                token.span.start,
                CachedToken {
                    token: token.clone(),
                    range,
                },
            );
        }

        // Rebuild full token list
        Ok(self
            .cache
            .tokens
            .values()
            .map(|ct| ct.token.clone())
            .collect())
    }

    /// Find safe position to start re-lexing
    fn find_relex_start(&self, edit_start: usize) -> usize {
        // Find the token containing or just before the edit
        if let Some((&pos, _)) = self.cache.tokens.range(..=edit_start).next_back() {
            pos
        } else {
            0
        }
    }

    /// Find safe position to end re-lexing
    fn find_relex_end(&self, edit_end: usize, source: &str) -> usize {
        // Continue until we reach a token boundary that matches the cache
        // For simplicity, re-lex to end of line or next cached token
        let line_end = source[edit_end..]
            .find('\n')
            .map(|i| edit_end + i + 1)
            .unwrap_or(source.len());

        if let Some((&pos, _)) = self.cache.tokens.range(line_end..).next() {
            pos
        } else {
            source.len()
        }
    }
}

fn ranges_overlap(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

fn hash_source(source: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

fn lex_with_offset(source: &str, offset: usize) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut lexer = TokenKind::lexer(source);

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let kind = match result {
            Ok(kind) => kind,
            Err(_) => {
                return Err(miette::miette!(
                    "Unexpected character at position {}: {:?}",
                    span.start + offset,
                    &source[span.clone()]
                ));
            }
        };

        tokens.push(Token {
            kind,
            span: Span::new(span.start + offset, span.end + offset),
            text: source[span].to_string(),
        });
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(source.len() + offset, source.len() + offset),
        text: String::new(),
    });

    Ok(tokens)
}
