//! Integration tests for Day 8: Source Maps
//!
//! Tests for:
//! - File database management
//! - Line/column position tracking
//! - Span mapping through compilation phases

use demetrios::sourcemap::{FileId, Located, SourceDb, SourceLocation, Span};
use std::path::PathBuf;

// ==================== SourceDb Tests ====================

#[test]
fn test_source_db_add_file() {
    let mut db = SourceDb::new();

    let source = r#"fn main() {
    let x = 42;
}"#;

    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());

    assert_eq!(file_id, FileId(0));
    assert!(db.get(file_id).is_some());
}

#[test]
fn test_source_db_multiple_files() {
    let mut db = SourceDb::new();

    let id1 = db.add_file(PathBuf::from("a.d"), "fn a() {}".to_string());
    let id2 = db.add_file(PathBuf::from("b.d"), "fn b() {}".to_string());
    let id3 = db.add_file(PathBuf::from("c.d"), "fn c() {}".to_string());

    assert_eq!(id1, FileId(0));
    assert_eq!(id2, FileId(1));
    assert_eq!(id3, FileId(2));

    assert!(db.get(id1).is_some());
    assert!(db.get(id2).is_some());
    assert!(db.get(id3).is_some());
}

#[test]
fn test_source_db_lookup_by_path() {
    let mut db = SourceDb::new();

    let path = PathBuf::from("test.d");
    let id = db.add_file(path.clone(), "fn main() {}".to_string());

    let found = db.get_by_path(&path);
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, id);
}

// ==================== SourceFile Tests ====================

#[test]
fn test_source_file_line_lookup() {
    let mut db = SourceDb::new();

    let source = "line 1\nline 2\nline 3";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    // Position 0 is line 1, column 1
    let (line, col) = file.line_col(0);
    assert_eq!(line, 1);
    assert_eq!(col, 1);

    // Position 7 is start of line 2 (after "line 1\n")
    let (line, col) = file.line_col(7);
    assert_eq!(line, 2);
    assert_eq!(col, 1);

    // Position 14 is start of line 3
    let (line, col) = file.line_col(14);
    assert_eq!(line, 3);
    assert_eq!(col, 1);
}

#[test]
fn test_source_file_column_lookup() {
    let mut db = SourceDb::new();

    let source = "fn main() {\n    let x = 42;\n}";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    // Position 3 is 'm' in 'main' (line 1, column 4)
    let (line, col) = file.line_col(3);
    assert_eq!(line, 1);
    assert_eq!(col, 4);

    // Position 16 is 'l' in 'let' (line 2, column 5 due to 4-space indent)
    let (line, col) = file.line_col(16);
    assert_eq!(line, 2);
    assert_eq!(col, 5);
}

#[test]
fn test_source_file_empty_lines() {
    let mut db = SourceDb::new();

    let source = "line 1\n\n\nline 4";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    // Line 2 starts at position 7 (empty)
    let (line, _) = file.line_col(7);
    assert_eq!(line, 2);

    // Line 3 starts at position 8 (empty)
    let (line, _) = file.line_col(8);
    assert_eq!(line, 3);

    // Line 4 starts at position 9
    let (line, _) = file.line_col(9);
    assert_eq!(line, 4);
}

// ==================== Span Tests ====================

#[test]
fn test_span_creation() {
    let span = Span::new(FileId(0), 10, 20);

    assert_eq!(span.file, FileId(0));
    assert_eq!(span.start, 10);
    assert_eq!(span.end, 20);
    assert_eq!(span.len(), 10);
}

#[test]
fn test_span_merge() {
    let span1 = Span::new(FileId(0), 5, 10);
    let span2 = Span::new(FileId(0), 15, 25);

    let merged = span1.merge(&span2);

    assert_eq!(merged.start, 5);
    assert_eq!(merged.end, 25);
}

#[test]
fn test_span_dummy() {
    let dummy = Span::dummy();
    assert!(dummy.is_dummy());

    let real = Span::new(FileId(0), 10, 20);
    assert!(!real.is_dummy());
}

#[test]
fn test_span_len() {
    let span = Span::new(FileId(0), 0, 100);
    assert_eq!(span.len(), 100);

    let empty = Span::new(FileId(0), 50, 50);
    assert!(empty.is_empty());
}

// ==================== Located Tests ====================

#[test]
fn test_located_wrapper() {
    let span = Span::new(FileId(0), 0, 5);
    let located = Located::new(42, span);

    assert_eq!(located.value, 42);
    assert_eq!(located.span.start, 0);
    assert_eq!(located.span.end, 5);
}

#[test]
fn test_located_map() {
    let span = Span::new(FileId(0), 0, 5);
    let located = Located::new(10, span);

    let mapped = located.map(|x| x * 2);

    assert_eq!(mapped.value, 20);
    assert_eq!(mapped.span.start, 0);
}

// ==================== SourceLocation Tests ====================

#[test]
fn test_source_location_creation() {
    let loc = SourceLocation::new(FileId(0), 10, 5);

    assert_eq!(loc.file, FileId(0));
    assert_eq!(loc.line, 10);
    assert_eq!(loc.column, 5);
}

#[test]
fn test_source_location_display() {
    let loc = SourceLocation::new(FileId(0), 10, 5);
    let s = format!("{}", loc);
    assert_eq!(s, "10:5");
}

// ==================== Integration Tests ====================

#[test]
fn test_span_to_location_roundtrip() {
    let mut db = SourceDb::new();

    let source = "fn foo() {\n    return 42;\n}";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    // Create a span for "return" (position 15-21)
    let span = Span::new(file_id, 15, 21);

    // Get start location
    let (start_line, start_col) = file.line_col(span.start as usize);
    assert_eq!(start_line, 2);
    assert_eq!(start_col, 5);

    // Get end location
    let (end_line, end_col) = file.line_col(span.end as usize);
    assert_eq!(end_line, 2);
    assert_eq!(end_col, 11);
}

#[test]
fn test_source_slice() {
    let mut db = SourceDb::new();

    let source = "fn main() { let x = 42; }";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    // Slice for "main"
    let slice = file.span_text(3, 7);
    assert_eq!(slice, "main");

    // Slice for "let x = 42"
    let slice = file.span_text(12, 22);
    assert_eq!(slice, "let x = 42");
}

#[test]
fn test_line_text() {
    let mut db = SourceDb::new();

    let source = "line one\nline two\nline three";
    let file_id = db.add_file(PathBuf::from("test.d"), source.to_string());
    let file = db.get(file_id).unwrap();

    assert_eq!(file.line_text(1), Some("line one"));
    assert_eq!(file.line_text(2), Some("line two"));
    assert_eq!(file.line_text(3), Some("line three"));
    assert_eq!(file.line_text(4), None);
}

#[test]
fn test_file_count() {
    let mut db = SourceDb::new();

    assert_eq!(db.file_count(), 0);

    db.add_file(PathBuf::from("a.d"), "a".to_string());
    assert_eq!(db.file_count(), 1);

    db.add_file(PathBuf::from("b.d"), "b".to_string());
    assert_eq!(db.file_count(), 2);
}

#[test]
fn test_virtual_file() {
    let mut db = SourceDb::new();

    let id = db.add_virtual("repl", "let x = 1".to_string());
    let file = db.get(id).unwrap();

    assert_eq!(file.name(), "<repl>");
}
