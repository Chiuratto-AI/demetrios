//! Linear and affine type tests

use demetrios::diagnostics::SourceFile;
use demetrios::ownership::OwnershipChecker;
use demetrios::parser;
use demetrios::resolve;

fn check_linear(src: &str) -> Result<(), String> {
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
fn test_linear_struct_definition() {
    // Just defining a linear struct should work
    let result = check_linear(
        r#"
        linear struct Handle {
            id: i32,
        }

        fn main() -> i32 {
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_affine_struct_definition() {
    // Affine structs can be used at most once
    let result = check_linear(
        r#"
        affine struct TempBuffer {
            size: i32,
        }

        fn main() -> i32 {
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_regular_struct() {
    // Regular structs have no restrictions
    let result = check_linear(
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

#[test]
fn test_linear_struct_created() {
    // Linear struct created and let to go out of scope
    // The ownership checker tracks this but doesn't yet enforce must-consume
    let result = check_linear(
        r#"
        linear struct Handle {
            id: i32,
        }

        fn main() -> i32 {
            let h: Handle = Handle { id: 1 }
            return 0
        }
    "#,
    );
    // Currently passes - full linear enforcement will be added later
    assert!(result.is_ok());
}

#[test]
fn test_enum_definition() {
    let result = check_linear(
        r#"
        enum Option {
            Some(i32),
            None,
        }

        fn main() -> i32 {
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_linear_enum() {
    let result = check_linear(
        r#"
        linear enum Resource {
            File(i32),
            Socket(i32),
        }

        fn main() -> i32 {
            return 0
        }
    "#,
    );
    assert!(result.is_ok());
}
