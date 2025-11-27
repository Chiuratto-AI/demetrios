//! Effect inference tests

use demetrios::effects::EffectChecker;
use demetrios::parser;
use demetrios::resolve;

fn check_effects(src: &str) -> Result<(), String> {
    let tokens = demetrios::lexer::lex(src).map_err(|e| format!("{:?}", e))?;
    let ast = parser::parse(&tokens, src).map_err(|e| format!("{:?}", e))?;
    let resolved = resolve::resolve(ast).map_err(|e| format!("{:?}", e))?;
    let mut checker = EffectChecker::new(&resolved.symbols);
    checker
        .check_program(&resolved.ast)
        .map_err(|e| format!("{:?}", e))
}

#[test]
fn test_pure_function() {
    let result = check_effects(
        r#"
        fn add(a: i32, b: i32) -> i32 {
            return a + b
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_io_declared() {
    let result = check_effects(
        r#"
        fn greet(name: String) -> String with IO {
            return name
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_panic_from_division() {
    // Division adds Panic effect - should be declared
    let result = check_effects(
        r#"
        fn divide(a: i32, b: i32) -> i32 with Panic {
            return a / b
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_panic_undeclared() {
    // Division without Panic declaration should produce warning
    let result = check_effects(
        r#"
        fn divide(a: i32, b: i32) -> i32 {
            return a / b
        }
    "#,
    );
    // This should fail because Panic is not declared
    assert!(result.is_err());
}

#[test]
fn test_multiple_effects() {
    let result = check_effects(
        r#"
        fn process(x: i32) -> i32 with IO, Panic {
            let y: i32 = x / 2
            return y
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_loop_divergence() {
    // Loops add Div effect
    let result = check_effects(
        r#"
        fn infinite() with Div {
            loop {
                let x: i32 = 1
            }
        }
    "#,
    );
    assert!(result.is_ok());
}

#[test]
fn test_function_call_propagates_effects() {
    let result = check_effects(
        r#"
        fn helper() -> i32 with IO {
            return 42
        }

        fn caller() -> i32 with IO {
            return helper()
        }
    "#,
    );
    assert!(result.is_ok());
}
