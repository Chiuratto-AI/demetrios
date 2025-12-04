//! Name resolution tests

use demetrios::lexer::lex;
use demetrios::parser::parse;
use demetrios::resolve::resolve;

fn resolve_source(src: &str) -> Result<demetrios::resolve::ResolvedAst, String> {
    let tokens = lex(src).map_err(|e| format!("{:?}", e))?;
    let ast = parse(&tokens, src).map_err(|e| format!("{:?}", e))?;
    resolve(ast).map_err(|e| format!("{:?}", e))
}

#[test]
fn test_resolve_function() {
    let src = "fn foo(x: i32) -> i32 { return x }";
    let resolved = resolve_source(src).expect("Resolution failed");

    // Check that 'foo' is defined
    assert!(
        resolved.symbols.lookup("foo").is_some(),
        "Function 'foo' should be defined"
    );
}

#[test]
fn test_resolve_variable() {
    let src = r#"
        fn main() -> i32 {
            let x: i32 = 42
            return x
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    assert!(
        resolved.symbols.lookup("main").is_some(),
        "Function 'main' should be defined"
    );
}

#[test]
fn test_undefined_variable() {
    let src = r#"
        fn main() -> i32 {
            return y
        }
    "#;
    let result = resolve_source(src);

    assert!(result.is_err(), "Should fail on undefined variable 'y'");
    let err = result.unwrap_err();
    assert!(
        err.contains("Undefined") || err.contains("undefined"),
        "Error should mention undefined: {}",
        err
    );
}

#[test]
fn test_shadowing() {
    let src = r#"
        fn main() -> i32 {
            let x: i32 = 1
            let x: i32 = 2
            return x
        }
    "#;
    let resolved = resolve_source(src);

    // Should succeed (shadowing is allowed in nested scopes)
    assert!(
        resolved.is_ok(),
        "Shadowing should be allowed: {:?}",
        resolved.err()
    );
}

#[test]
fn test_resolve_struct() {
    let src = r#"
        struct Point {
            x: f64,
            y: f64
        }

        fn main() -> i32 {
            return 0
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    // Check that 'Point' is defined as a type
    assert!(
        resolved.symbols.lookup_type("Point").is_some(),
        "Type 'Point' should be defined"
    );
}

#[test]
fn test_resolve_linear_struct() {
    let src = r#"
        linear struct FileHandle {
            fd: i32
        }

        fn main() -> i32 {
            return 0
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    assert!(
        resolved.symbols.lookup_type("FileHandle").is_some(),
        "Type 'FileHandle' should be defined"
    );
}

#[test]
fn test_resolve_enum() {
    let src = r#"
        enum Option {
            Some(i32),
            None
        }

        fn main() -> i32 {
            return 0
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    assert!(
        resolved.symbols.lookup_type("Option").is_some(),
        "Type 'Option' should be defined"
    );
}

#[test]
fn test_resolve_multiple_functions() {
    let src = r#"
        fn add(a: i32, b: i32) -> i32 {
            return a + b
        }

        fn sub(a: i32, b: i32) -> i32 {
            return a - b
        }

        fn main() -> i32 {
            return 0
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    assert!(resolved.symbols.lookup("add").is_some());
    assert!(resolved.symbols.lookup("sub").is_some());
    assert!(resolved.symbols.lookup("main").is_some());
}

#[test]
fn test_resolve_builtin_types() {
    let src = r#"
        fn test(a: i32, b: i64, c: f64, d: bool, e: String) -> i32 {
            return a
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    // Should succeed - all types are built-in
    assert!(resolved.symbols.lookup("test").is_some());
}

#[test]
fn test_undefined_type() {
    let src = r#"
        fn test(x: UnknownType) -> i32 {
            return 0
        }
    "#;
    let result = resolve_source(src);

    assert!(
        result.is_err(),
        "Should fail on undefined type 'UnknownType'"
    );
}

#[test]
fn test_resolve_parameter_scope() {
    let src = r#"
        fn test(x: i32) -> i32 {
            return x
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    // Parameter 'x' should be visible inside the function
    assert!(resolved.symbols.lookup("test").is_some());
}

#[test]
fn test_resolve_nested_blocks() {
    let src = r#"
        fn test() -> i32 {
            let a: i32 = 1
            if true {
                let b: i32 = 2
                return a + b
            }
            return a
        }
    "#;
    let resolved = resolve_source(src).expect("Resolution failed");

    assert!(resolved.symbols.lookup("test").is_some());
}
