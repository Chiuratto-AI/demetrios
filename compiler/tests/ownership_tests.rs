//! Ownership and borrow checking tests

use demetrios::diagnostics::SourceFile;
use demetrios::ownership::OwnershipChecker;
use demetrios::parser;
use demetrios::resolve;

fn check_ownership(src: &str) -> Result<(), String> {
    let tokens = demetrios::lexer::lex(src).map_err(|e| format!("{:?}", e))?;
    let ast = parser::parse(&tokens, src).map_err(|e| format!("{:?}", e))?;
    let resolved = resolve::resolve(ast).map_err(|e| format!("{:?}", e))?;
    let source = SourceFile::new("test.d", src);
    let mut checker = OwnershipChecker::new(&resolved.symbols, &source);
    checker
        .check_program(&resolved.ast)
        .map_err(|e| format!("{:?}", e))
}

#[test]
fn test_simple_binding() {
    let result = check_ownership(
        r#"
        fn main() -> i32 {
            let x: i32 = 42
            return x
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_multiple_bindings() {
    let result = check_ownership(
        r#"
        fn main() -> i32 {
            let x: i32 = 42
            let y: i32 = x
            return y
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_shared_borrow() {
    let result = check_ownership(
        r#"
        fn main() -> i32 {
            let x: i32 = 42
            let r1: &i32 = &x
            let r2: &i32 = &x
            return x
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_function_with_params() {
    let result = check_ownership(
        r#"
        fn add(a: i32, b: i32) -> i32 {
            return a + b
        }

        fn main() -> i32 {
            return add(1, 2)
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_nested_blocks() {
    let result = check_ownership(
        r#"
        fn main() -> i32 {
            let x: i32 = 1
            {
                let y: i32 = 2
                let z: i32 = x + y
            }
            return x
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_if_expression() {
    let result = check_ownership(
        r#"
        fn main() -> i32 {
            let x: i32 = 42
            if x > 0 {
                return x
            }
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_struct_construction() {
    let result = check_ownership(
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        fn main() -> i32 {
            let p: Point = Point { x: 1, y: 2 }
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}
