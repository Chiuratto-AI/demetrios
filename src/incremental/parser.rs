//! Incremental parser with tree patching primitives.
//!
//! The current implementation is conservative: it re-parses the entire source
//! after an edit but keeps the APIs needed for finer-grained reuse.

use std::ops::Range;

use crate::ast::{Ast, Item, Path};
use crate::common::Span;
use crate::incremental::edits::TextEdit;
use miette::Report;

/// Node identifier for incremental updates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// A syntax node with identity for incremental updates
#[derive(Debug, Clone)]
pub struct IncrementalNode<T> {
    /// Unique node identifier
    pub id: NodeId,

    /// The actual AST node
    pub node: T,

    /// Byte range in source
    pub range: Range<usize>,

    /// Hash of the source text for this node
    pub source_hash: u64,
}

impl<T> IncrementalNode<T> {
    pub fn new(node: T, range: Range<usize>, source: &str) -> Self {
        IncrementalNode {
            id: NodeId::new(),
            node,
            range: range.clone(),
            source_hash: hash_range(source, &range),
        }
    }

    /// Check if this node is affected by an edit
    pub fn is_affected(&self, edit: &TextEdit) -> bool {
        // Node is affected if edit range overlaps with node range
        self.range.start < edit.range.end && edit.range.start < self.range.end
    }

    /// Check if node can be reused (source unchanged)
    pub fn can_reuse(&self, new_source: &str, offset_delta: isize) -> bool {
        let new_start = (self.range.start as isize + offset_delta) as usize;
        let new_end = (self.range.end as isize + offset_delta) as usize;

        if new_end > new_source.len() {
            return false;
        }

        let new_hash = hash_range(new_source, &(new_start..new_end));
        new_hash == self.source_hash
    }
}

/// Incremental AST with change tracking
#[derive(Debug, Clone)]
pub struct IncrementalAst {
    /// Optional module name
    pub module_name: Option<Path>,

    /// Root items
    pub items: Vec<IncrementalNode<Item>>,

    /// Source text
    pub source: String,

    /// Version counter
    pub version: u64,
}

impl IncrementalAst {
    /// Create from full parse
    pub fn from_ast(ast: Ast, source: &str) -> Self {
        let items = ast
            .items
            .into_iter()
            .map(|item| {
                let range = span_to_range(item_span(&item));
                IncrementalNode::new(item, range, source)
            })
            .collect();

        IncrementalAst {
            module_name: ast.module_name,
            items,
            source: source.to_string(),
            version: 1,
        }
    }

    /// Apply an edit conservatively by re-parsing the source
    pub fn apply_edit(&mut self, _edit: &TextEdit, new_source: &str) -> IncrementalParseResult {
        let mut result = IncrementalParseResult::new();

        let tokens = match crate::lexer::lex(new_source) {
            Ok(tokens) => tokens,
            Err(err) => {
                result.errors.push(err);
                return result;
            }
        };

        match crate::parser::parse(&tokens, new_source) {
            Ok(ast) => {
                let mut new_items = Vec::new();
                for item in ast.items {
                    let range = span_to_range(item_span(&item));
                    new_items.push(IncrementalNode::new(item, range, new_source));
                }

                result.reparsed_nodes = new_items.len();
                self.items = new_items;
                self.source = new_source.to_string();
                self.module_name = ast.module_name;
                self.version = self.version.saturating_add(1);
            }
            Err(err) => result.errors.push(err),
        }

        result
    }

    /// Convert back to regular AST
    pub fn to_ast(&self) -> Ast {
        Ast {
            module_name: self.module_name.clone(),
            items: self.items.iter().map(|n| n.node.clone()).collect(),
        }
    }
}

/// Result of incremental parse
#[derive(Debug, Default)]
pub struct IncrementalParseResult {
    /// Number of nodes reused from previous parse
    pub reused_nodes: usize,

    /// Number of nodes that were re-parsed
    pub reparsed_nodes: usize,

    /// Parse errors encountered
    pub errors: Vec<Report>,
}

impl IncrementalParseResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// Percentage of nodes that were reused
    pub fn reuse_percentage(&self) -> f64 {
        let total = self.reused_nodes + self.reparsed_nodes;
        if total == 0 {
            0.0
        } else {
            (self.reused_nodes as f64 / total as f64) * 100.0
        }
    }
}

fn hash_range(source: &str, range: &Range<usize>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source[range.clone()].hash(&mut hasher);
    hasher.finish()
}

fn span_to_range(span: Span) -> Range<usize> {
    span.start..span.end
}

fn item_span(item: &Item) -> Span {
    match item {
        Item::Function(f) => f.span,
        Item::Struct(s) => s.span,
        Item::Enum(e) => e.span,
        Item::Trait(t) => t.span,
        Item::Impl(i) => i.span,
        Item::TypeAlias(t) => t.span,
        Item::Effect(e) => e.span,
        Item::Handler(h) => h.span,
        Item::Import(i) => i.span,
        Item::Extern(e) => e.span,
        Item::Global(g) => g.span,
    }
}

fn adjust_item_span(item: &mut Item, delta: isize) {
    // Adjust only the outer span for now. Full traversal can be added if needed.
    match item {
        Item::Function(f) => shift_span(&mut f.span, delta),
        Item::Struct(s) => shift_span(&mut s.span, delta),
        Item::Enum(e) => shift_span(&mut e.span, delta),
        Item::Trait(t) => shift_span(&mut t.span, delta),
        Item::Impl(i) => shift_span(&mut i.span, delta),
        Item::TypeAlias(t) => shift_span(&mut t.span, delta),
        Item::Effect(e) => shift_span(&mut e.span, delta),
        Item::Handler(h) => shift_span(&mut h.span, delta),
        Item::Import(i) => shift_span(&mut i.span, delta),
        Item::Extern(e) => shift_span(&mut e.span, delta),
        Item::Global(g) => shift_span(&mut g.span, delta),
    }
}

fn shift_span(span: &mut Span, delta: isize) {
    span.start = (span.start as isize + delta) as usize;
    span.end = (span.end as isize + delta) as usize;
}
