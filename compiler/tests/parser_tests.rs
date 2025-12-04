//! Parser tests

use demetrios::lexer::lex;
use demetrios::parser::parse;
use demetrios::ast::*;

fn parse_source(source: &str) -> Ast {
    let tokens = lex(source).unwrap();
    parse(&tokens, source).unwrap()
}

#[test]
fn test_parse_empty_module() {
    let ast = parse_source("");
    assert!(ast.items.is_empty());
}

#[test]
fn test_parse_module_declaration() {
    let ast = parse_source("module foo");
    assert!(ast.module_name.is_some());
    assert_eq!(ast.module_name.unwrap().segments, vec!["foo"]);
}

#[test]
fn test_parse_simple_function() {
    let ast = parse_source("fn main() { }");
    assert_eq!(ast.items.len(), 1);

    if let Item::Function(f) = &ast.items[0] {
        assert_eq!(f.name, "main");
        assert!(f.params.is_empty());
        assert!(f.return_type.is_none());
    } else {
        panic!("Expected function");
    }
}

#[test]
fn test_parse_function_with_params() {
    let ast = parse_source("fn add(a: i32, b: i32) -> i32 { a + b }");

    if let Item::Function(f) = &ast.items[0] {
        assert_eq!(f.name, "add");
        assert_eq!(f.params.len(), 2);
        assert!(f.return_type.is_some());
    } else {
        panic!("Expected function");
    }
}

#[test]
fn test_parse_function_with_effects() {
    let ast = parse_source("fn greet(name: String) with IO { }");

    if let Item::Function(f) = &ast.items[0] {
        assert_eq!(f.effects.len(), 1);
        assert_eq!(f.effects[0].name.segments, vec!["IO"]);
    } else {
        panic!("Expected function");
    }
}

#[test]
fn test_parse_struct() {
    let ast = parse_source("struct Point { x: f64, y: f64 }");

    if let Item::Struct(s) = &ast.items[0] {
        assert_eq!(s.name, "Point");
        assert_eq!(s.fields.len(), 2);
        assert_eq!(s.fields[0].name, "x");
        assert_eq!(s.fields[1].name, "y");
    } else {
        panic!("Expected struct");
    }
}

#[test]
fn test_parse_linear_struct() {
    let ast = parse_source("linear struct FileHandle { fd: i32 }");

    if let Item::Struct(s) = &ast.items[0] {
        assert!(s.modifiers.linear);
    } else {
        panic!("Expected struct");
    }
}

#[test]
fn test_parse_enum() {
    let ast = parse_source("enum Option<T> { Some(T), None }");

    if let Item::Enum(e) = &ast.items[0] {
        assert_eq!(e.name, "Option");
        assert_eq!(e.variants.len(), 2);
        assert_eq!(e.variants[0].name, "Some");
        assert_eq!(e.variants[1].name, "None");
    } else {
        panic!("Expected enum");
    }
}

#[test]
fn test_parse_trait() {
    let ast = parse_source("trait Display { fn fmt(self) -> String; }");

    if let Item::Trait(t) = &ast.items[0] {
        assert_eq!(t.name, "Display");
        assert_eq!(t.items.len(), 1);
    } else {
        panic!("Expected trait");
    }
}

#[test]
fn test_parse_let_binding() {
    let ast = parse_source("fn main() { let x = 42 }");

    if let Item::Function(f) = &ast.items[0] {
        assert_eq!(f.body.stmts.len(), 1);
        if let Stmt::Let { pattern, value, .. } = &f.body.stmts[0] {
            if let Pattern::Binding { name, .. } = pattern {
                assert_eq!(name, "x");
            }
            assert!(value.is_some());
        } else {
            panic!("Expected let statement");
        }
    } else {
        panic!("Expected function");
    }
}

#[test]
fn test_parse_if_expression() {
    let ast = parse_source("fn main() { if true { 1 } else { 2 } }");

    if let Item::Function(f) = &ast.items[0] {
        if let Stmt::Expr { expr, .. } = &f.body.stmts[0] {
            assert!(matches!(expr, Expr::If { .. }));
        } else {
            panic!("Expected expression statement");
        }
    } else {
        panic!("Expected function");
    }
}

#[test]
fn test_parse_match_expression() {
    let ast = parse_source(r#"
        fn main() {
            match x {
                0 => "zero",
                _ => "other",
            }
        }
    "#);

    if let Item::Function(f) = &ast.items[0] {
        if let Stmt::Expr { expr, .. } = &f.body.stmts[0] {
            if let Expr::Match { arms, .. } = expr {
                assert_eq!(arms.len(), 2);
            } else {
                panic!("Expected match expression");
            }
        }
    }
}

#[test]
fn test_parse_binary_expressions() {
    let ast = parse_source("fn main() { 1 + 2 * 3 }");

    // Should parse as 1 + (2 * 3) due to precedence
    if let Item::Function(f) = &ast.items[0] {
        if let Stmt::Expr { expr, .. } = &f.body.stmts[0] {
            if let Expr::Binary { op, .. } = expr {
                assert_eq!(*op, BinaryOp::Add);
            } else {
                panic!("Expected binary expression");
            }
        }
    }
}

#[test]
fn test_parse_effect_definition() {
    let ast = parse_source(r#"
        effect IO {
            fn print(s: String);
            fn read_line() -> String;
        }
    "#);

    if let Item::Effect(e) = &ast.items[0] {
        assert_eq!(e.name, "IO");
        assert_eq!(e.operations.len(), 2);
    } else {
        panic!("Expected effect definition");
    }
}

#[test]
fn test_parse_kernel_function() {
    let ast = parse_source("kernel fn vector_add(a: &[f32], b: &[f32]) { }");

    if let Item::Function(f) = &ast.items[0] {
        assert!(f.modifiers.is_kernel);
    } else {
        panic!("Expected function");
    }
}
