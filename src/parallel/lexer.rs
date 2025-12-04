//! Parallel lexer using chunked tokenization

use crate::common::Span;
use crate::lexer::{Token, TokenKind};
use logos::Logos;
use rayon::prelude::*;

/// Configuration for parallel lexing
#[derive(Debug, Clone)]
pub struct ParallelLexConfig {
    /// Minimum chunk size in bytes
    pub min_chunk_size: usize,

    /// Maximum number of chunks
    pub max_chunks: usize,

    /// Number of threads to use
    pub num_threads: usize,
}

impl Default for ParallelLexConfig {
    fn default() -> Self {
        ParallelLexConfig {
            min_chunk_size: 4096,
            max_chunks: 64,
            num_threads: num_cpus::get(),
        }
    }
}

/// A chunk of source code for parallel lexing
#[derive(Debug)]
struct SourceChunk {
    /// Start offset in original source
    start: usize,

    /// End offset in original source
    end: usize,

    /// The chunk text
    text: String,

    /// Whether this chunk starts at a line boundary
    starts_at_line: bool,
}

/// Parallel lexer
pub struct ParallelLexer {
    config: ParallelLexConfig,
}

impl ParallelLexer {
    pub fn new(config: ParallelLexConfig) -> Self {
        ParallelLexer { config }
    }

    /// Lex source in parallel. Falls back to sequential lexing for small inputs.
    pub fn lex(&self, source: &str) -> Vec<Token> {
        // Small files don't benefit from parallelism
        if source.len() < self.config.min_chunk_size * 2 {
            return crate::lexer::lex(source).unwrap_or_else(|_| Vec::new());
        }

        // Split into chunks at line boundaries
        let chunks = self.split_into_chunks(source);

        // Lex chunks in parallel
        let chunk_tokens: Vec<Vec<Token>> = chunks
            .par_iter()
            .map(|chunk| self.lex_chunk(chunk))
            .collect();

        // Merge results
        let mut tokens = self.merge_tokens(chunk_tokens);

        // Ensure EOF token
        if let Some(last) = tokens.last() {
            if last.kind != TokenKind::Eof {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: Span::new(source.len(), source.len()),
                    text: String::new(),
                });
            }
        }

        tokens
    }

    /// Split source into chunks at line boundaries
    fn split_into_chunks(&self, source: &str) -> Vec<SourceChunk> {
        let target_chunk_size =
            (source.len() / self.config.max_chunks).max(self.config.min_chunk_size);

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < source.len() {
            // Find chunk end at a line boundary
            let mut end = (start + target_chunk_size).min(source.len());

            // Adjust to line boundary
            if end < source.len() {
                if let Some(newline_pos) = source[end..].find('\n') {
                    end += newline_pos + 1;
                } else {
                    end = source.len();
                }
            }

            chunks.push(SourceChunk {
                start,
                end,
                text: source[start..end].to_string(),
                starts_at_line: start == 0
                    || source.as_bytes().get(start.saturating_sub(1)) == Some(&b'\n'),
            });

            start = end;
        }

        chunks
    }

    /// Lex a single chunk
    fn lex_chunk(&self, chunk: &SourceChunk) -> Vec<Token> {
        let mut lexer = TokenKind::lexer(&chunk.text);
        let mut tokens: Vec<Token> = Vec::new();

        // Handle potential mid-token start (simplified: rely on line boundaries)
        if !chunk.starts_at_line {
            // Could scan forward to next whitespace or delimiter to sync, but
            // for now we trust the chunk boundary.
        }

        while let Some(result) = lexer.next() {
            let span = lexer.span();
            let kind = match result {
                Ok(kind) => kind,
                Err(_) => continue,
            };

            tokens.push(Token {
                kind,
                span: Span::new(span.start + chunk.start, span.end + chunk.start),
                text: chunk.text[span.clone()].to_string(),
            });
        }

        tokens
    }

    /// Merge token lists from chunks
    fn merge_tokens(&self, chunk_tokens: Vec<Vec<Token>>) -> Vec<Token> {
        let total_tokens: usize = chunk_tokens.iter().map(|c| c.len()).sum();
        let mut result = Vec::with_capacity(total_tokens + 1);

        for tokens in chunk_tokens {
            // Handle potential duplicate tokens at boundaries
            if let (Some(last), Some(first)) = (result.last(), tokens.first()) {
                if last.span.end > first.span.start
                    && last.kind == first.kind
                    && last.text == first.text
                {
                    // Overlapping - remove the duplicate
                    result.pop();
                }
            }

            result.extend(tokens);
        }

        result
    }
}

/// Statistics from parallel lexing
#[derive(Debug, Default)]
pub struct ParallelLexStats {
    pub total_bytes: usize,
    pub num_chunks: usize,
    pub total_tokens: usize,
    pub duration_ms: u64,
    pub throughput_mb_s: f64,
}
