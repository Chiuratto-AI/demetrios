//! HLIR and JIT tests

use demetrios::hlir::{self, FunctionBuilder, FunctionId, HlirType, ModuleBuilder};

#[test]
fn test_hlir_type_conversion() {
    use demetrios::hir::HirType;

    assert_eq!(HlirType::from_hir(&HirType::I32), HlirType::I32);
    assert_eq!(HlirType::from_hir(&HirType::I64), HlirType::I64);
    assert_eq!(HlirType::from_hir(&HirType::F64), HlirType::F64);
    assert_eq!(HlirType::from_hir(&HirType::Bool), HlirType::Bool);
    assert_eq!(HlirType::from_hir(&HirType::Unit), HlirType::Void);
}

#[test]
fn test_hlir_type_properties() {
    assert!(HlirType::I64.is_integer());
    assert!(HlirType::I32.is_integer());
    assert!(HlirType::U64.is_integer());

    assert!(HlirType::I64.is_signed());
    assert!(HlirType::I32.is_signed());
    assert!(!HlirType::U64.is_signed());

    assert!(HlirType::F64.is_float());
    assert!(HlirType::F32.is_float());
    assert!(!HlirType::I64.is_float());
}

#[test]
fn test_hlir_builder_simple_function() {
    let mut builder = FunctionBuilder::new(FunctionId(0), "test", HlirType::I64);
    let a = builder.add_param("a", HlirType::I64);
    let b = builder.add_param("b", HlirType::I64);

    let entry = builder.create_block("entry");
    builder.switch_to_block(entry);

    let sum = builder.build_add(a, b, HlirType::I64);
    builder.build_return(Some(sum));

    let func = builder.build();
    assert_eq!(func.name, "test");
    assert_eq!(func.params.len(), 2);
    assert_eq!(func.blocks.len(), 1);
}

#[test]
fn test_hlir_builder_conditional() {
    let mut builder = FunctionBuilder::new(FunctionId(0), "abs", HlirType::I64);
    let n = builder.add_param("n", HlirType::I64);

    let entry = builder.create_block("entry");
    let then_block = builder.create_block("then");
    let else_block = builder.create_block("else");

    builder.switch_to_block(entry);
    let zero = builder.build_i64(0);
    let is_neg = builder.build_slt(n, zero);
    builder.build_cond_branch(is_neg, then_block, else_block);

    builder.switch_to_block(then_block);
    let neg_n = builder.build_neg(n, HlirType::I64);
    builder.build_return(Some(neg_n));

    builder.switch_to_block(else_block);
    builder.build_return(Some(n));

    let func = builder.build();
    assert_eq!(func.blocks.len(), 3);
}

#[test]
fn test_hlir_module_builder() {
    let mut module_builder = ModuleBuilder::new("test_module");

    let mut func_builder = FunctionBuilder::new(module_builder.fresh_func_id(), "main", HlirType::I64);
    let entry = func_builder.create_block("entry");
    func_builder.switch_to_block(entry);
    let val = func_builder.build_i64(42);
    func_builder.build_return(Some(val));

    module_builder.add_function(func_builder.build());

    let module = module_builder.build();
    assert_eq!(module.name, "test_module");
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "main");
}

#[test]
fn test_hlir_lower_simple_function() {
    let source = "fn main() -> i64 { 42 }";
    let tokens = demetrios::lexer::lex(source).unwrap();
    let ast = demetrios::parser::parse(&tokens, source).unwrap();
    let hir = demetrios::check::check(&ast).unwrap();
    let hlir = hlir::lower(&hir);

    assert_eq!(hlir.functions.len(), 1);
    let func = &hlir.functions[0];
    assert_eq!(func.name, "main");
    assert_eq!(func.return_type, HlirType::I64);
}

#[test]
fn test_hlir_lower_binary_op() {
    let source = "fn main() -> i64 { 1 + 2 }";
    let tokens = demetrios::lexer::lex(source).unwrap();
    let ast = demetrios::parser::parse(&tokens, source).unwrap();
    let hir = demetrios::check::check(&ast).unwrap();
    let hlir = hlir::lower(&hir);

    assert_eq!(hlir.functions.len(), 1);
    let func = &hlir.functions[0];
    assert!(!func.blocks.is_empty());
    // The entry block should have instructions for the binary op
    assert!(!func.blocks[0].instructions.is_empty());
}

#[test]
fn test_hlir_lower_function_with_params() {
    let source = "fn add(a: i64, b: i64) -> i64 { a + b }";
    let tokens = demetrios::lexer::lex(source).unwrap();
    let ast = demetrios::parser::parse(&tokens, source).unwrap();
    let hir = demetrios::check::check(&ast).unwrap();
    let hlir = hlir::lower(&hir);

    assert_eq!(hlir.functions.len(), 1);
    let func = &hlir.functions[0];
    assert_eq!(func.name, "add");
    assert_eq!(func.params.len(), 2);
    assert_eq!(func.params[0].name, "a");
    assert_eq!(func.params[1].name, "b");
}

#[test]
fn test_hlir_lower_conditional() {
    let source = r#"
        fn abs(n: i64) -> i64 {
            if n < 0 {
                0 - n
            } else {
                n
            }
        }
    "#;
    let tokens = demetrios::lexer::lex(source).unwrap();
    let ast = demetrios::parser::parse(&tokens, source).unwrap();
    let hir = demetrios::check::check(&ast).unwrap();
    let hlir = hlir::lower(&hir);

    assert_eq!(hlir.functions.len(), 1);
    let func = &hlir.functions[0];
    // Should have multiple blocks for if/then/else/merge
    assert!(func.blocks.len() >= 3);
}

#[test]
fn test_hlir_lower_struct() {
    let source = r#"
        struct Point {
            x: i64,
            y: i64
        }
        fn main() -> i64 { 0 }
    "#;
    let tokens = demetrios::lexer::lex(source).unwrap();
    let ast = demetrios::parser::parse(&tokens, source).unwrap();
    let hir = demetrios::check::check(&ast).unwrap();
    let hlir = hlir::lower(&hir);

    assert_eq!(hlir.types.len(), 1);
    assert_eq!(hlir.types[0].name, "Point");
}

// JIT tests (only run with jit feature)
#[cfg(feature = "jit")]
mod jit_tests {
    use demetrios::codegen::cranelift::CraneliftJit;
    use demetrios::hlir;

    fn compile_and_run(source: &str) -> Result<i64, String> {
        let tokens = demetrios::lexer::lex(source).map_err(|e| format!("{}", e))?;
        let ast = demetrios::parser::parse(&tokens, source).map_err(|e| format!("{:?}", e))?;
        let hir = demetrios::check::check(&ast).map_err(|e| format!("{}", e))?;
        let hlir = hlir::lower(&hir);

        let jit = CraneliftJit::new();
        jit.compile_and_run(&hlir)
    }

    #[test]
    fn test_jit_simple_return() {
        let result = compile_and_run("fn main() -> i64 { 42 }");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_jit_arithmetic() {
        let result = compile_and_run("fn main() -> i64 { 1 + 2 * 3 }");
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn test_jit_subtraction() {
        let result = compile_and_run("fn main() -> i64 { 10 - 3 }");
        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn test_jit_division() {
        let result = compile_and_run("fn main() -> i64 { 20 / 4 }");
        assert_eq!(result.unwrap(), 5);
    }

    #[test]
    fn test_jit_negative() {
        let result = compile_and_run("fn main() -> i64 { 0 - 42 }");
        assert_eq!(result.unwrap(), -42);
    }

    #[test]
    fn test_jit_conditional() {
        let result = compile_and_run(r#"
            fn main() -> i64 {
                if true { 1 } else { 0 }
            }
        "#);
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_jit_comparison() {
        let result = compile_and_run(r#"
            fn main() -> i64 {
                if 5 > 3 { 1 } else { 0 }
            }
        "#);
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn test_jit_let_binding() {
        let result = compile_and_run(r#"
            fn main() -> i64 {
                let x = 10;
                let y = 20;
                x + y
            }
        "#);
        assert_eq!(result.unwrap(), 30);
    }

    #[test]
    fn test_jit_function_call() {
        let result = compile_and_run(r#"
            fn double(n: i64) -> i64 {
                n * 2
            }
            fn main() -> i64 {
                double(21)
            }
        "#);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_jit_with_optimization() {
        let source = "fn main() -> i64 { 1 + 2 + 3 + 4 + 5 }";
        let tokens = demetrios::lexer::lex(source).unwrap();
        let ast = demetrios::parser::parse(&tokens, source).unwrap();
        let hir = demetrios::check::check(&ast).unwrap();
        let hlir = hlir::lower(&hir);

        let jit = CraneliftJit::new().with_optimization();
        let result = jit.compile_and_run(&hlir);
        assert_eq!(result.unwrap(), 15);
    }
}
