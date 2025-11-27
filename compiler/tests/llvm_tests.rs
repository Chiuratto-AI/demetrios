//! LLVM Backend Integration Tests
//!
//! These tests verify the LLVM code generation backend.
//! Run with: cargo test --features llvm llvm_tests

#![cfg(feature = "llvm")]

use demetrios::codegen::llvm::{
    codegen::{LLVMCodegen, OptLevel},
    passes,
    target::{TargetConfig, create_native_target_machine, initialize_native_target},
    types::TypeConverter,
};
use demetrios::hlir::{HlirModule, HlirType};
use inkwell::context::Context;

/// Helper to compile source to HLIR
fn compile_to_hlir(source: &str) -> Result<HlirModule, String> {
    let tokens = demetrios::lexer::lex(source).map_err(|e| format!("{:?}", e))?;
    let ast = demetrios::parser::parse(&tokens, source).map_err(|e| format!("{:?}", e))?;
    let hir = demetrios::check::check(&ast).map_err(|e| format!("{:?}", e))?;
    Ok(demetrios::hlir::lower(&hir))
}

#[test]
fn test_llvm_context_creation() {
    let context = Context::create();
    let module = context.create_module("test");
    assert_eq!(module.get_name().to_str().unwrap(), "test");
}

#[test]
fn test_type_converter_basic_types() {
    let context = Context::create();
    let mut converter = TypeConverter::new(&context);

    // Integer types
    let i32_ty = converter.convert(&HlirType::I32);
    assert!(i32_ty.is_int_type());
    assert_eq!(i32_ty.into_int_type().get_bit_width(), 32);

    let i64_ty = converter.convert(&HlirType::I64);
    assert!(i64_ty.is_int_type());
    assert_eq!(i64_ty.into_int_type().get_bit_width(), 64);

    // Float types
    let f32_ty = converter.convert(&HlirType::F32);
    assert!(f32_ty.is_float_type());

    let f64_ty = converter.convert(&HlirType::F64);
    assert!(f64_ty.is_float_type());

    // Bool type
    let bool_ty = converter.convert(&HlirType::Bool);
    assert!(bool_ty.is_int_type());
    assert_eq!(bool_ty.into_int_type().get_bit_width(), 1);
}

#[test]
fn test_type_converter_pointer_types() {
    let context = Context::create();
    let mut converter = TypeConverter::new(&context);

    let ptr_ty = converter.convert(&HlirType::Ptr(Box::new(HlirType::I32)));
    assert!(ptr_ty.is_pointer_type());
}

#[test]
fn test_type_converter_array_types() {
    let context = Context::create();
    let mut converter = TypeConverter::new(&context);

    let arr_ty = converter.convert(&HlirType::Array(Box::new(HlirType::I32), 10));
    assert!(arr_ty.is_array_type());
    assert_eq!(arr_ty.into_array_type().len(), 10);
}

#[test]
fn test_type_converter_tuple_types() {
    let context = Context::create();
    let mut converter = TypeConverter::new(&context);

    let tuple_ty = converter.convert(&HlirType::Tuple(vec![HlirType::I32, HlirType::F64]));
    assert!(tuple_ty.is_struct_type());
    assert_eq!(tuple_ty.into_struct_type().count_fields(), 2);
}

#[test]
fn test_type_converter_function_types() {
    let context = Context::create();
    let mut converter = TypeConverter::new(&context);

    let fn_ty = converter.function_type(&[HlirType::I32, HlirType::I32], &HlirType::I64);
    assert_eq!(fn_ty.count_param_types(), 2);
}

#[test]
fn test_codegen_empty_module() {
    initialize_native_target();
    let context = Context::create();

    let hlir = HlirModule::new("test");
    let mut codegen = LLVMCodegen::new(&context, "test", OptLevel::O0, false);

    let module = codegen.compile(&hlir);
    assert!(codegen.verify().is_ok());

    let ir = codegen.print_ir();
    assert!(ir.contains("ModuleID"));
}

#[test]
fn test_codegen_simple_function() {
    let source = r#"
        fn add(a: i64, b: i64) -> i64 {
            return a + b
        }
    "#;

    let hlir = compile_to_hlir(source).expect("Failed to compile");

    initialize_native_target();
    let context = Context::create();
    let mut codegen = LLVMCodegen::new(&context, "test", OptLevel::O0, false);

    let module = codegen.compile(&hlir);
    let ir = codegen.print_ir();

    // Should contain the add function
    assert!(ir.contains("define") || ir.contains("@add"));
}

#[test]
fn test_codegen_with_optimization() {
    let source = r#"
        fn constant_fold() -> i64 {
            let x = 10
            let y = 20
            return x + y
        }
    "#;

    let hlir = compile_to_hlir(source).expect("Failed to compile");

    initialize_native_target();
    let context = Context::create();

    // Without optimization
    let mut codegen_o0 = LLVMCodegen::new(&context, "test_o0", OptLevel::O0, false);
    let _module_o0 = codegen_o0.compile(&hlir);
    let ir_o0 = codegen_o0.print_ir();

    // With optimization
    let mut codegen_o3 = LLVMCodegen::new(&context, "test_o3", OptLevel::O3, false);
    let module_o3 = codegen_o3.compile(&hlir);

    let target = create_native_target_machine(OptLevel::O3).expect("target");
    passes::optimize_module(module_o3, OptLevel::O3, &target);

    let ir_o3 = codegen_o3.print_ir();

    // Both should be valid
    assert!(ir_o0.contains("ModuleID"));
    assert!(ir_o3.contains("ModuleID"));
}

#[test]
fn test_target_config() {
    let config = TargetConfig::native();
    assert!(!config.triple.is_empty());
    assert_eq!(config.cpu, "generic");
}

#[test]
fn test_target_machine_creation() {
    initialize_native_target();
    let target = create_native_target_machine(OptLevel::O2);
    assert!(target.is_ok());
}

#[test]
fn test_opt_level_parsing() {
    assert_eq!(OptLevel::from_str("0"), Some(OptLevel::O0));
    assert_eq!(OptLevel::from_str("1"), Some(OptLevel::O1));
    assert_eq!(OptLevel::from_str("2"), Some(OptLevel::O2));
    assert_eq!(OptLevel::from_str("3"), Some(OptLevel::O3));
    assert_eq!(OptLevel::from_str("s"), Some(OptLevel::Os));
    assert_eq!(OptLevel::from_str("z"), Some(OptLevel::Oz));
    assert_eq!(OptLevel::from_str("O2"), Some(OptLevel::O2));
    assert_eq!(OptLevel::from_str("invalid"), None);
}

#[test]
fn test_module_verification() {
    initialize_native_target();
    let context = Context::create();

    let hlir = HlirModule::new("verify_test");
    let mut codegen = LLVMCodegen::new(&context, "verify_test", OptLevel::O0, false);

    codegen.compile(&hlir);

    // Empty module should verify successfully
    assert!(codegen.verify().is_ok());
}

#[test]
fn test_ir_output() {
    initialize_native_target();
    let context = Context::create();

    let hlir = HlirModule::new("ir_test");
    let mut codegen = LLVMCodegen::new(&context, "ir_test", OptLevel::O0, false);

    codegen.compile(&hlir);

    let ir = codegen.print_ir();

    // Should have module declaration
    assert!(ir.contains("ModuleID"));
    assert!(ir.contains("ir_test"));
}

#[test]
fn test_type_sizes() {
    let context = Context::create();
    let converter = TypeConverter::new(&context);

    assert_eq!(converter.size_bits(&HlirType::I8), 8);
    assert_eq!(converter.size_bits(&HlirType::I16), 16);
    assert_eq!(converter.size_bits(&HlirType::I32), 32);
    assert_eq!(converter.size_bits(&HlirType::I64), 64);
    assert_eq!(converter.size_bits(&HlirType::I128), 128);
    assert_eq!(converter.size_bits(&HlirType::F32), 32);
    assert_eq!(converter.size_bits(&HlirType::F64), 64);

    assert_eq!(converter.size_bytes(&HlirType::I32), 4);
    assert_eq!(converter.size_bytes(&HlirType::I64), 8);
}

#[test]
fn test_type_properties() {
    let context = Context::create();
    let converter = TypeConverter::new(&context);

    // Integer checks
    assert!(converter.is_integer_type(&HlirType::I32));
    assert!(converter.is_integer_type(&HlirType::U64));
    assert!(!converter.is_integer_type(&HlirType::F64));

    // Float checks
    assert!(converter.is_float_type(&HlirType::F32));
    assert!(converter.is_float_type(&HlirType::F64));
    assert!(!converter.is_float_type(&HlirType::I32));

    // Signed checks
    assert!(converter.is_signed(&HlirType::I32));
    assert!(!converter.is_signed(&HlirType::U32));
}

#[test]
fn test_multiple_functions() {
    let source = r#"
        fn foo() -> i64 {
            return 42
        }

        fn bar() -> i64 {
            return 100
        }

        fn main() -> i64 {
            return 0
        }
    "#;

    let hlir = compile_to_hlir(source).expect("Failed to compile");
    assert!(hlir.functions.len() >= 3);

    initialize_native_target();
    let context = Context::create();
    let mut codegen = LLVMCodegen::new(&context, "multi_func", OptLevel::O0, false);

    codegen.compile(&hlir);
    assert!(codegen.verify().is_ok());

    let ir = codegen.print_ir();
    // Should contain all functions
    assert!(ir.contains("define"));
}

#[test]
fn test_control_flow() {
    let source = r#"
        fn max(a: i64, b: i64) -> i64 {
            if a > b {
                return a
            } else {
                return b
            }
        }
    "#;

    let hlir = compile_to_hlir(source).expect("Failed to compile");

    initialize_native_target();
    let context = Context::create();
    let mut codegen = LLVMCodegen::new(&context, "control_flow", OptLevel::O0, false);

    codegen.compile(&hlir);
    assert!(codegen.verify().is_ok());

    let ir = codegen.print_ir();
    // Should have branch instructions
    assert!(ir.contains("br") || ir.contains("ret") || ir.contains("icmp"));
}

// Test for linker (without actually linking)
mod linker_tests {
    use demetrios::codegen::llvm::linker::Linker;
    use std::path::PathBuf;

    #[test]
    fn test_linker_creation() {
        let linker = Linker::new();
        assert!(!linker.get_command().is_empty());
    }

    #[test]
    fn test_linker_builder() {
        let linker = Linker::new()
            .command("clang")
            .lib_path("/usr/lib")
            .lib("m")
            .flag("-O2")
            .target("x86_64-unknown-linux-gnu")
            .pie(true)
            .strip(false);

        assert_eq!(linker.get_command(), "clang");
    }
}

// Target-related tests
mod target_tests {
    use demetrios::codegen::llvm::target::{
        TargetConfig, TargetInfo, executable_extension, object_extension, triples,
    };

    #[test]
    fn test_file_extensions() {
        assert_eq!(object_extension("x86_64-unknown-linux-gnu"), "o");
        assert_eq!(object_extension("x86_64-pc-windows-msvc"), "obj");
        assert_eq!(executable_extension("x86_64-unknown-linux-gnu"), "");
        assert_eq!(executable_extension("x86_64-pc-windows-msvc"), "exe");
    }

    #[test]
    fn test_target_triples() {
        assert_eq!(triples::X86_64_LINUX_GNU, "x86_64-unknown-linux-gnu");
        assert_eq!(triples::X86_64_MACOS, "x86_64-apple-darwin");
        assert_eq!(triples::AARCH64_MACOS, "aarch64-apple-darwin");
        assert_eq!(triples::X86_64_WINDOWS_MSVC, "x86_64-pc-windows-msvc");
    }

    #[test]
    fn test_target_info() {
        let info64 = TargetInfo::for_triple("x86_64-unknown-linux-gnu");
        assert_eq!(info64.pointer_size, 64);
        assert!(!info64.is_big_endian);

        let info32 = TargetInfo::for_triple("wasm32-unknown-unknown");
        assert_eq!(info32.pointer_size, 32);
    }

    #[test]
    fn test_host_triple() {
        let host = triples::host();
        assert!(!host.is_empty());
    }
}
