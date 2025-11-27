//! Integration tests for the full pipeline

use demetrios::lexer::lex;
use demetrios::parser::parse;

const MINIMAL: &str = r#"
fn main() -> i32 {
    let x: i32 = 42
    return x
}
"#;

#[test]
fn test_lex_minimal() {
    let tokens = lex(MINIMAL).expect("Lexing failed");

    // Should have tokens: fn, main, (, ), ->, i32, {, let, x, :, i32, =, 42, return, x, }, EOF
    assert!(
        tokens.len() > 10,
        "Expected multiple tokens, got {}",
        tokens.len()
    );

    // Lexing succeeded without errors (errors would cause lex() to return Err)
    println!("Lexed {} tokens", tokens.len());
}

#[test]
fn test_parse_minimal() {
    let tokens = lex(MINIMAL).expect("Lexing failed");
    let ast = parse(&tokens, MINIMAL).expect("Parse failed");

    // Should have one item (the function)
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_function() {
    let src = "fn add(a: i32, b: i32) -> i32 { return a }";
    let tokens = lex(src).expect("Lexing failed");
    let ast = parse(&tokens, src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_struct() {
    let src = "struct Point { x: f64, y: f64 }";
    let tokens = lex(src).expect("Lexing failed");
    let ast = parse(&tokens, src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_linear_struct() {
    let src = "linear struct FileHandle { fd: i32 }";
    let tokens = lex(src).expect("Lexing failed");
    let ast = parse(&tokens, src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_effect_annotation() {
    let src = "fn read_file(path: String) -> String with IO { return path }";
    let tokens = lex(src).expect("Lexing failed");
    let ast = parse(&tokens, src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_parse_let_with_type() {
    let src = "fn test() { let d: f64 = 500.0 }";
    let tokens = lex(src).expect("Lexing failed");
    let ast = parse(&tokens, src).expect("Parse failed");
    assert_eq!(ast.items.len(), 1);
}

#[test]
fn test_full_pipeline_minimal() {
    // Test full pipeline: lex -> parse -> typecheck
    let result = demetrios::typecheck(MINIMAL);
    assert!(result.is_ok(), "Type check failed: {:?}", result.err());
}

#[test]
fn test_full_pipeline_function_with_params() {
    let src = "fn add(a: i32, b: i32) -> i32 { return a }";
    let result = demetrios::typecheck(src);
    assert!(result.is_ok(), "Type check failed: {:?}", result.err());
}

#[test]
fn test_full_pipeline_struct() {
    let src = "struct Point { x: f64, y: f64 }";
    let result = demetrios::typecheck(src);
    assert!(result.is_ok(), "Type check failed: {:?}", result.err());
}
