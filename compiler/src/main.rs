//! Demetrios (D) Compiler CLI
//!
//! Main entry point for the `dc` command.

use clap::{Parser, Subcommand};
use miette::Result;
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Parser)]
#[command(name = "dc")]
#[command(author = "Demetrios Chiuratto Agourakis, Dionisio Chiuratto Agourakis")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "The Demetrios (D) programming language compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a D source file
    Compile {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Emit intermediate representation
        #[arg(long, value_enum)]
        emit: Option<EmitType>,

        /// Optimization level (0-3)
        #[arg(short = 'O', default_value = "0")]
        opt_level: u8,
    },

    /// Build a D source file to native executable (requires --features llvm)
    Build {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Optimization level (0, 1, 2, 3, s, z)
        #[arg(short = 'O', default_value = "2")]
        opt_level: String,

        /// Generate debug information
        #[arg(short = 'g', long)]
        debug: bool,

        /// Emit LLVM IR instead of compiling
        #[arg(long)]
        emit_llvm: bool,

        /// Emit assembly instead of compiling
        #[arg(long)]
        emit_asm: bool,

        /// Target triple (e.g., x86_64-unknown-linux-gnu)
        #[arg(long)]
        target: Option<String>,

        /// Strip debug symbols from output
        #[arg(long)]
        strip: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Type-check a D source file without compiling
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Show the parsed AST
        #[arg(long)]
        show_ast: bool,

        /// Show resolved symbols
        #[arg(long)]
        show_resolved: bool,

        /// Show inferred types
        #[arg(long)]
        show_types: bool,

        /// Show inferred effects
        #[arg(long)]
        show_effects: bool,

        /// Skip ownership checking
        #[arg(long)]
        skip_ownership: bool,
    },

    /// Run a D program using the interpreter
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Run a D program using JIT compilation (requires --features jit)
    Jit {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Enable optimizations
        #[arg(short = 'O', long)]
        optimize: bool,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl {
        /// Use JIT compilation instead of interpreter
        #[arg(long)]
        jit: bool,
    },

    /// Benchmark interpreter vs JIT performance
    Bench {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: u32,
    },

    /// Format D source code
    Fmt {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },

    /// Generate documentation for a package
    Doc {
        /// Open documentation in browser after generation
        #[arg(long)]
        open: bool,

        /// Document private items
        #[arg(long)]
        document_private: bool,
    },

    /// Generate mdBook documentation
    DocBook {
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run documentation tests
    Doctest {
        /// Filter tests by name pattern
        #[arg(long)]
        filter: Option<String>,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show documentation coverage
    DocCoverage,

    /// Show information about the compiler
    Info,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum EmitType {
    /// Abstract Syntax Tree (JSON)
    Ast,
    /// High-level IR
    Hir,
    /// Low-level IR (SSA)
    Hlir,
    /// LLVM IR
    Llvm,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if cli.verbose {
        tracing::info!("Verbose mode enabled");
    }

    match cli.command {
        Commands::Compile {
            input,
            output,
            emit,
            opt_level,
        } => compile(&input, output.as_deref(), emit, opt_level),

        Commands::Build {
            input,
            output,
            opt_level,
            debug,
            emit_llvm,
            emit_asm,
            target,
            strip,
            verbose,
        } => build(
            &input,
            output.as_deref(),
            &opt_level,
            debug,
            emit_llvm,
            emit_asm,
            target.as_deref(),
            strip,
            verbose,
        ),

        Commands::Check {
            input,
            show_ast,
            show_resolved,
            show_types,
            show_effects,
            skip_ownership,
        } => check(
            &input,
            show_ast,
            show_resolved,
            show_types,
            show_effects,
            skip_ownership,
        ),

        Commands::Run { input, args } => run(&input, &args),

        Commands::Jit {
            input,
            optimize,
            args,
        } => jit_run(&input, optimize, &args),

        Commands::Repl { jit } => repl(jit),

        Commands::Bench { input, iterations } => bench(&input, iterations),

        Commands::Fmt { path, check } => format_code(&path, check),

        Commands::Doc {
            open,
            document_private,
        } => doc(open, document_private),

        Commands::DocBook { output } => doc_book(output),

        Commands::Doctest { filter, verbose } => doctest(filter, verbose),

        Commands::DocCoverage => doc_coverage(),

        Commands::Info => info(),
    }
}

/// Build a D source file to native executable using LLVM
#[allow(clippy::too_many_arguments)]
fn build(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    opt_level: &str,
    debug: bool,
    emit_llvm: bool,
    emit_asm: bool,
    target: Option<&str>,
    strip: bool,
    verbose: bool,
) -> Result<()> {
    #[cfg(feature = "llvm")]
    {
        use demetrios::codegen::llvm::{
            codegen::{LLVMCodegen, OptLevel},
            linker::Linker,
            passes,
            target::{
                compile_to_asm, compile_to_object, create_native_target_machine,
                create_target_machine, executable_extension, initialize_native_target,
                object_extension,
            },
        };
        use inkwell::context::Context;

        tracing::info!("Building {:?} with LLVM", input);

        // Parse optimization level
        let opt = match opt_level {
            "0" => OptLevel::O0,
            "1" => OptLevel::O1,
            "2" => OptLevel::O2,
            "3" => OptLevel::O3,
            "s" => OptLevel::Os,
            "z" => OptLevel::Oz,
            _ => {
                return Err(miette::miette!(
                    "Invalid optimization level: {}. Use 0, 1, 2, 3, s, or z",
                    opt_level
                ));
            }
        };

        // Read source file
        let source = std::fs::read_to_string(input)
            .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

        // Lex and parse
        let tokens = demetrios::lexer::lex(&source)?;
        let ast = demetrios::parser::parse(&tokens, &source)?;

        // Type check
        let hir = demetrios::check::check(&ast)?;

        // Lower to HLIR
        let hlir = demetrios::hlir::lower(&hir);

        if verbose {
            eprintln!(
                "Compiled {} items, {} functions",
                ast.items.len(),
                hlir.functions.len()
            );
        }

        // Initialize LLVM
        initialize_native_target();

        // Create LLVM context and codegen
        let context = Context::create();
        let module_name = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        let mut codegen = LLVMCodegen::new(&context, module_name, opt, debug);

        // Compile to LLVM IR
        let module = codegen.compile(&hlir);

        // Verify module
        if let Err(e) = codegen.verify() {
            return Err(miette::miette!("LLVM verification failed: {}", e));
        }

        // Get target machine
        let target_machine = if let Some(triple) = target {
            create_target_machine(triple, opt)
                .map_err(|e| miette::miette!("Failed to create target machine: {}", e))?
        } else {
            create_native_target_machine(opt)
                .map_err(|e| miette::miette!("Failed to create target machine: {}", e))?
        };

        // Run optimization passes
        passes::optimize_module(module, opt, &target_machine);

        // Handle emit options
        if emit_llvm {
            let ir = codegen.print_ir();
            if let Some(out_path) = output {
                std::fs::write(out_path, &ir)
                    .map_err(|e| miette::miette!("Failed to write LLVM IR: {}", e))?;
                println!("Wrote LLVM IR to {}", out_path.display());
            } else {
                println!("{}", ir);
            }
            return Ok(());
        }

        if emit_asm {
            let asm_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
                let mut p = input.to_path_buf();
                p.set_extension("s");
                p
            });

            compile_to_asm(module, &target_machine, &asm_path)
                .map_err(|e| miette::miette!("Failed to generate assembly: {}", e))?;

            println!("Wrote assembly to {}", asm_path.display());
            return Ok(());
        }

        // Compile to object file
        let triple = target.unwrap_or("native");
        let obj_ext = object_extension(triple);
        let obj_path = {
            let mut p = input.to_path_buf();
            p.set_extension(obj_ext);
            p
        };

        compile_to_object(module, &target_machine, &obj_path)
            .map_err(|e| miette::miette!("Failed to generate object file: {}", e))?;

        if verbose {
            eprintln!("Generated object file: {}", obj_path.display());
        }

        // Link to executable
        let exe_ext = executable_extension(triple);
        let exe_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            let mut p = input.to_path_buf();
            p.set_extension(exe_ext);
            if exe_ext.is_empty() {
                // Remove extension for Unix executables
                p.set_extension("");
            }
            p
        });

        let linker = Linker::new().strip(strip).verbose(verbose);

        linker
            .link_with_stdlib(&[obj_path.clone()], &exe_path)
            .map_err(|e| miette::miette!("Linking failed: {}", e))?;

        // Clean up object file
        if std::fs::remove_file(&obj_path).is_err() && verbose {
            eprintln!("Warning: could not remove temporary object file");
        }

        println!("Built: {}", exe_path.display());
        Ok(())
    }

    #[cfg(not(feature = "llvm"))]
    {
        let _ = (
            input, output, opt_level, debug, emit_llvm, emit_asm, target, strip, verbose,
        );
        Err(miette::miette!(
            "LLVM backend not enabled. Rebuild with: cargo build --features llvm"
        ))
    }
}

fn compile(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    emit: Option<EmitType>,
    opt_level: u8,
) -> Result<()> {
    tracing::info!(
        "Compiling {:?} with optimization level {}",
        input,
        opt_level
    );

    // Read source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    // Lex
    let tokens = demetrios::lexer::lex(&source)?;
    tracing::debug!("Lexed {} tokens", tokens.len());

    // Parse
    let ast = demetrios::parser::parse(&tokens, &source)?;
    tracing::debug!("Parsed {} items", ast.items.len());

    // Handle emit options
    if let Some(emit_type) = emit {
        match emit_type {
            EmitType::Ast => {
                let json = serde_json::to_string_pretty(&ast)
                    .map_err(|e| miette::miette!("Failed to serialize AST: {}", e))?;
                println!("{}", json);
                return Ok(());
            }
            EmitType::Hir => {
                let hir = demetrios::check::check(&ast)?;
                println!("{:#?}", hir);
                return Ok(());
            }
            EmitType::Hlir => {
                let hir = demetrios::check::check(&ast)?;
                let hlir = demetrios::hlir::lower(&hir);
                println!("{:#?}", hlir);
                return Ok(());
            }
            EmitType::Llvm => {
                return Err(miette::miette!("LLVM emit not yet implemented"));
            }
        }
    }

    // Type check
    let hir = demetrios::check::check(&ast)?;

    // Lower to HLIR
    let hlir = demetrios::hlir::lower(&hir);

    // Code generation
    let _output_path = output.unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        p.set_extension("");
        Box::leak(Box::new(p))
    });

    // TODO: Actual code generation
    tracing::info!("Compilation successful (codegen not yet implemented)");
    println!(
        "Compiled {} ({} items, {} functions)",
        input.display(),
        ast.items.len(),
        hlir.functions.len()
    );

    Ok(())
}

fn check(
    input: &std::path::Path,
    show_ast: bool,
    show_resolved: bool,
    show_types: bool,
    show_effects: bool,
    skip_ownership: bool,
) -> Result<()> {
    tracing::info!("Type-checking {:?}", input);

    let source_content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let source_file =
        demetrios::SourceFile::new(input.to_string_lossy().to_string(), source_content.clone());

    // 1. Lex
    let tokens = demetrios::lexer::lex(&source_content)?;

    // 2. Parse
    let ast = demetrios::parser::parse(&tokens, &source_content)?;

    if show_ast {
        println!("=== AST ===");
        let json = serde_json::to_string_pretty(&ast)
            .map_err(|e| miette::miette!("Failed to serialize AST: {}", e))?;
        println!("{}", json);
        println!();
    }

    // 3. Resolve names
    let resolved = demetrios::resolve::resolve(ast)?;

    if show_resolved {
        println!("=== Resolved Symbols ===");
        let mut count = 0;
        for sym in resolved.symbols.all_symbols() {
            println!("  {:?}: {} ({:?})", sym.def_id, sym.name, sym.kind);
            count += 1;
        }
        println!("Total: {} symbols", count);
        println!();
    }

    // 4. Type check
    let hir = demetrios::check::check(&resolved.ast)?;

    if show_types {
        println!("=== HIR (with types) ===");
        println!("{:#?}", hir);
        println!();
    }

    // 5. Effect inference
    let mut effect_checker = demetrios::effects::EffectChecker::new(&resolved.symbols);
    if let Err(errors) = effect_checker.check_program(&resolved.ast) {
        if show_effects {
            println!("=== Effect Errors ===");
            for e in &errors {
                println!("  {}", e);
            }
            println!();
        }
        // Effect errors are warnings for now, not fatal
        for e in &errors {
            eprintln!("Warning: {}", e);
        }
    } else if show_effects {
        println!("=== Effects ===");
        println!("  All effects properly declared");
        println!();
    }

    // 6. Ownership check
    if !skip_ownership {
        let mut ownership_checker =
            demetrios::ownership::OwnershipChecker::new(&resolved.symbols, &source_file);
        if let Err(errors) = ownership_checker.check_program(&resolved.ast) {
            for e in &errors {
                eprintln!("{:?}", miette::Report::new(e.clone()));
            }
            return Err(miette::miette!("{} ownership errors found", errors.len()));
        }
    }

    println!("All checks passed: {}", input.display());
    Ok(())
}

fn run(input: &std::path::Path, args: &[String]) -> Result<()> {
    tracing::info!("Running {:?} with args {:?}", input, args);

    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = demetrios::lexer::lex(&source)?;
    let ast = demetrios::parser::parse(&tokens, &source)?;
    let hir = demetrios::check::check(&ast)?;

    // Use tree-walking interpreter
    let mut interpreter = demetrios::interp::Interpreter::new();
    match interpreter.interpret(&hir) {
        Ok(result) => {
            // Only print non-unit results
            match &result {
                demetrios::interp::Value::Unit => {}
                _ => println!("{}", result),
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

fn jit_run(input: &std::path::Path, optimize: bool, _args: &[String]) -> Result<()> {
    #[cfg(feature = "jit")]
    {
        tracing::info!("JIT compiling {:?} (optimize={})", input, optimize);

        let source = std::fs::read_to_string(input)
            .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

        let tokens = demetrios::lexer::lex(&source)?;
        let ast = demetrios::parser::parse(&tokens, &source)?;
        let hir = demetrios::check::check(&ast)?;
        let hlir = demetrios::hlir::lower(&hir);

        let jit = if optimize {
            demetrios::codegen::cranelift::CraneliftJit::new().with_optimization()
        } else {
            demetrios::codegen::cranelift::CraneliftJit::new()
        };

        match jit.compile_and_run(&hlir) {
            Ok(result) => {
                println!("{}", result);
                Ok(())
            }
            Err(e) => Err(miette::miette!("JIT error: {}", e)),
        }
    }

    #[cfg(not(feature = "jit"))]
    {
        let _ = (input, optimize); // Suppress unused warnings
        Err(miette::miette!(
            "JIT backend not enabled. Recompile with --features jit"
        ))
    }
}

fn repl(use_jit: bool) -> Result<()> {
    let config = demetrios::repl::ReplConfig {
        use_jit,
        ..Default::default()
    };

    demetrios::repl::run_with_config(config).map_err(|e| miette::miette!("REPL error: {}", e))
}

fn bench(input: &std::path::Path, iterations: u32) -> Result<()> {
    use std::time::Instant;

    println!("Benchmarking {:?} ({} iterations)", input, iterations);
    println!();

    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = demetrios::lexer::lex(&source)?;
    let ast = demetrios::parser::parse(&tokens, &source)?;
    let hir = demetrios::check::check(&ast)?;

    // Warm up
    println!("Warming up...");
    let mut interpreter = demetrios::interp::Interpreter::new();
    let _ = interpreter.interpret(&hir);

    // Benchmark interpreter
    println!("Running interpreter benchmark...");
    let start = Instant::now();
    for _ in 0..iterations {
        let mut interpreter = demetrios::interp::Interpreter::new();
        let _ = interpreter.interpret(&hir);
    }
    let interp_time = start.elapsed();
    let interp_per_iter = interp_time / iterations;

    println!(
        "  Interpreter: {:?} total, {:?} per iteration",
        interp_time, interp_per_iter
    );

    // Benchmark JIT if available
    #[cfg(feature = "jit")]
    {
        let hlir = demetrios::hlir::lower(&hir);
        let jit = demetrios::codegen::cranelift::CraneliftJit::new();

        // Compile once
        println!("Compiling with JIT...");
        let compile_start = Instant::now();
        let compiled = jit
            .compile(&hlir)
            .map_err(|e| miette::miette!("JIT compile error: {}", e))?;
        let compile_time = compile_start.elapsed();
        println!("  JIT compile time: {:?}", compile_time);

        // Run benchmark
        println!("Running JIT benchmark...");
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = compiled.call_i64("main");
            }
        }
        let jit_time = start.elapsed();
        let jit_per_iter = jit_time / iterations;

        println!(
            "  JIT: {:?} total, {:?} per iteration",
            jit_time, jit_per_iter
        );
        println!();

        // Calculate speedup
        let speedup = interp_per_iter.as_nanos() as f64 / jit_per_iter.as_nanos() as f64;
        println!("JIT speedup: {:.2}x", speedup);

        // Break-even point
        let break_even = compile_time.as_nanos() as f64
            / (interp_per_iter.as_nanos() as f64 - jit_per_iter.as_nanos() as f64);
        println!("Break-even point: {:.0} iterations", break_even.max(0.0));
    }

    #[cfg(not(feature = "jit"))]
    {
        println!();
        println!("JIT backend not enabled. Recompile with --features jit for JIT benchmarks.");
    }

    Ok(())
}

fn format_code(path: &std::path::Path, check: bool) -> Result<()> {
    if check {
        println!("Checking formatting of {:?}", path);
    } else {
        println!("Formatting {:?}", path);
    }

    // TODO: Implement formatter
    Err(miette::miette!("Formatter not yet implemented"))
}

fn doc(open: bool, document_private: bool) -> Result<()> {
    demetrios::pkg::cli::cmd_doc(open, document_private)
        .map_err(|e| miette::miette!("Documentation generation failed: {}", e))
}

fn doc_book(output: Option<PathBuf>) -> Result<()> {
    demetrios::pkg::cli::cmd_doc_book(output)
        .map_err(|e| miette::miette!("Book generation failed: {}", e))
}

fn doctest(filter: Option<String>, verbose: bool) -> Result<()> {
    demetrios::pkg::cli::cmd_doctest(filter, verbose)
        .map_err(|e| miette::miette!("Doctest failed: {}", e))
}

fn doc_coverage() -> Result<()> {
    demetrios::pkg::cli::cmd_doc_coverage()
        .map_err(|e| miette::miette!("Coverage calculation failed: {}", e))
}

fn info() -> Result<()> {
    println!("Demetrios (D) Compiler");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Features:");
    println!("  - Algebraic effects with handlers");
    println!("  - Linear and affine types");
    println!("  - Units of measure");
    println!("  - Refinement types");
    println!("  - GPU-native computation");
    println!();
    println!("Backends:");
    #[cfg(feature = "llvm")]
    {
        println!("  - LLVM (enabled)");
        println!("    Use 'dc build' for AOT compilation");
    }
    #[cfg(not(feature = "llvm"))]
    println!("  - LLVM (disabled) - rebuild with --features llvm");
    #[cfg(feature = "jit")]
    {
        println!("  - Cranelift JIT (enabled)");
        println!("    Use 'dc jit' for JIT execution");
    }
    #[cfg(not(feature = "jit"))]
    println!("  - Cranelift JIT (disabled) - rebuild with --features jit");
    #[cfg(feature = "smt")]
    println!("  - SMT Solver (enabled) - refinement type verification");
    #[cfg(not(feature = "smt"))]
    println!("  - SMT Solver (disabled) - rebuild with --features smt");
    #[cfg(feature = "lsp")]
    println!("  - LSP Server (enabled) - IDE integration");
    #[cfg(not(feature = "lsp"))]
    println!("  - LSP Server (disabled) - rebuild with --features lsp");
    #[cfg(feature = "gpu")]
    println!("  - GPU codegen (enabled)");
    #[cfg(not(feature = "gpu"))]
    println!("  - GPU codegen (disabled) - rebuild with --features gpu");
    println!();
    println!("Build with all features: cargo build --features full");

    Ok(())
}
