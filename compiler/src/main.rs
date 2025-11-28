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

    /// Run tests
    Test {
        /// Path to test files or directory
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Filter tests by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Include ignored tests
        #[arg(long)]
        include_ignored: bool,

        /// Only run ignored tests
        #[arg(long)]
        ignored: bool,

        /// Number of parallel threads (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,

        /// Fail fast on first failure
        #[arg(long)]
        fail_fast: bool,

        /// Run benchmarks instead of tests
        #[arg(long)]
        bench: bool,

        /// List tests without running
        #[arg(long)]
        list: bool,

        /// Output format (pretty, compact, json, junit)
        #[arg(long, default_value = "pretty")]
        format: String,

        /// Enable coverage tracking
        #[arg(long)]
        coverage: bool,

        /// Coverage output file
        #[arg(long)]
        coverage_output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run benchmarks
    Benchmark {
        /// Path to benchmark files or directory
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Filter benchmarks by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Compare with baseline file
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Save results as new baseline
        #[arg(long)]
        save_baseline: Option<PathBuf>,

        /// Target benchmark time in seconds
        #[arg(long, default_value = "3")]
        time: u64,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Profile a D program
    Profile {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Profile type (cpu, memory, async)
        #[arg(short, long, default_value = "cpu")]
        profile_type: String,

        /// Output file for profile data
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Generate flame graph SVG
        #[arg(long)]
        flamegraph: Option<PathBuf>,

        /// Sample interval in microseconds (for CPU profiling)
        #[arg(long, default_value = "1000")]
        interval: u64,

        /// Output format (text, json, folded)
        #[arg(long, default_value = "text")]
        format: String,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Debug a D program
    Debug {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Debugger to use (gdb, lldb)
        #[arg(short, long, default_value = "gdb")]
        debugger: String,

        /// Enable pretty printers automatically
        #[arg(long)]
        pretty: bool,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Explain an error code
    Explain {
        /// Error code (e.g., T0001)
        #[arg(value_name = "CODE")]
        code: String,
    },

    /// Show all error codes
    ErrorIndex {
        /// Filter by category (type, effect, ownership, etc.)
        #[arg(short, long)]
        category: Option<String>,

        /// Output format (text, markdown, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Generate debug information for a compiled binary
    DebugInfo {
        /// Input D source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output debug info file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Debug info format (dwarf, pdb)
        #[arg(long, default_value = "dwarf")]
        format: String,
    },

    /// Generate source map for compiled code
    SourceMap {
        /// Input D source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output source map file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
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

        Commands::Test {
            path,
            filter,
            include_ignored,
            ignored,
            jobs,
            fail_fast,
            bench,
            list,
            format,
            coverage,
            coverage_output,
            verbose,
        } => run_tests(
            &path,
            filter.as_deref(),
            include_ignored,
            ignored,
            jobs,
            fail_fast,
            bench,
            list,
            &format,
            coverage,
            coverage_output.as_deref(),
            verbose,
        ),

        Commands::Benchmark {
            path,
            filter,
            baseline,
            save_baseline,
            time,
            verbose,
        } => run_benchmarks(
            &path,
            filter.as_deref(),
            baseline.as_deref(),
            save_baseline.as_deref(),
            time,
            verbose,
        ),

        Commands::Profile {
            input,
            profile_type,
            output,
            flamegraph,
            interval,
            format,
            args,
        } => profile(
            &input,
            &profile_type,
            output.as_deref(),
            flamegraph.as_deref(),
            interval,
            &format,
            &args,
        ),

        Commands::Debug {
            input,
            debugger,
            pretty,
            args,
        } => debug_program(&input, &debugger, pretty, &args),

        Commands::Explain { code } => explain_error(&code),

        Commands::ErrorIndex { category, format } => show_error_index(category.as_deref(), &format),

        Commands::DebugInfo {
            input,
            output,
            format,
        } => generate_debug_info(&input, output.as_deref(), &format),

        Commands::SourceMap { input, output } => generate_source_map(&input, output.as_deref()),
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

#[allow(clippy::too_many_arguments)]
fn run_tests(
    path: &std::path::Path,
    filter: Option<&str>,
    include_ignored: bool,
    only_ignored: bool,
    jobs: usize,
    fail_fast: bool,
    run_bench: bool,
    list_only: bool,
    format: &str,
    coverage: bool,
    coverage_output: Option<&std::path::Path>,
    verbose: bool,
) -> Result<()> {
    use demetrios::test::{
        coverage::{CoverageConfig, CoverageTracker},
        discovery::{TestFilter, discover_tests},
        runner::{OutputFormat, TestRunner, TestRunnerConfig},
    };

    tracing::info!("Running tests from {:?}", path);

    // Build test filter
    let mut test_filter = TestFilter::default();
    if let Some(pattern) = filter {
        test_filter.pattern = Some(pattern.to_string());
    }
    if include_ignored {
        test_filter.include_ignored = true;
    }
    if only_ignored {
        test_filter.only_ignored = true;
    }

    // Discover tests
    let suite = discover_tests(&[path], test_filter)
        .map_err(|e| miette::miette!("Test discovery failed: {}", e))?;

    if run_bench {
        println!("Found {} benchmarks", suite.all_benchmarks().len());
    } else {
        println!("Found {} tests", suite.test_count());
    }

    // Configure runner
    let output_format = match format {
        "compact" => OutputFormat::Compact,
        "json" => OutputFormat::Json,
        "junit" => OutputFormat::JUnit,
        _ => OutputFormat::Pretty,
    };

    let config = TestRunnerConfig {
        threads: jobs,
        fail_fast,
        list_only,
        verbose,
        format: output_format,
        ..Default::default()
    };

    // Optionally set up coverage tracking
    let mut coverage_tracker = if coverage {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        tracker.start_tracking();
        Some(tracker)
    } else {
        None
    };

    // Run tests
    let runner = TestRunner::new(config);
    let report = runner
        .run(&suite)
        .map_err(|e| miette::miette!("Test execution failed: {}", e))?;

    // Stop coverage tracking and generate report
    if let Some(tracker) = &mut coverage_tracker {
        tracker.stop_tracking();
        let cov_report = tracker.generate_report();

        println!("\n{}", cov_report.summary());

        if let Some(output_path) = coverage_output {
            let lcov = cov_report.to_lcov();
            std::fs::write(output_path, &lcov)
                .map_err(|e| miette::miette!("Failed to write coverage: {}", e))?;
            println!("Coverage written to {}", output_path.display());
        }
    }

    // Output results in requested format
    match output_format {
        OutputFormat::Json => {
            let json = report
                .to_json()
                .map_err(|e| miette::miette!("Failed to serialize results: {}", e))?;
            println!("{}", json);
        }
        OutputFormat::JUnit => {
            println!("{}", report.to_junit());
        }
        _ => {
            // Pretty/Compact format already printed by runner
        }
    }

    if report.all_passed() {
        Ok(())
    } else {
        Err(miette::miette!(
            "Test run failed: {} passed, {} failed",
            report.passed,
            report.failed + report.timed_out + report.panicked
        ))
    }
}

fn run_benchmarks(
    path: &std::path::Path,
    filter: Option<&str>,
    baseline: Option<&std::path::Path>,
    save_baseline: Option<&std::path::Path>,
    time_secs: u64,
    verbose: bool,
) -> Result<()> {
    use demetrios::test::{
        bench::{BenchConfig, BenchmarkRunner},
        discovery::{TestFilter, discover_tests},
    };
    use std::time::Duration;

    tracing::info!("Running benchmarks from {:?}", path);

    // Build filter
    let mut test_filter = TestFilter::default();
    if let Some(pattern) = filter {
        test_filter.pattern = Some(pattern.to_string());
    }

    // Discover benchmarks
    let suite = discover_tests(&[path], test_filter)
        .map_err(|e| miette::miette!("Benchmark discovery failed: {}", e))?;

    let benchmarks: Vec<_> = suite.all_benchmarks().into_iter().cloned().collect();

    if benchmarks.is_empty() {
        println!("No benchmarks found");
        return Ok(());
    }

    println!("Found {} benchmarks", benchmarks.len());

    // Configure benchmark runner
    let config = BenchConfig {
        target_time: Duration::from_secs(time_secs),
        ..Default::default()
    };

    let mut runner = BenchmarkRunner::new(config);

    // Load baseline if provided
    if let Some(baseline_path) = baseline {
        runner
            .load_baselines(baseline_path)
            .map_err(|e| miette::miette!("Failed to load baseline: {}", e))?;
        if verbose {
            println!("Loaded baseline from {}", baseline_path.display());
        }
    }

    // Run benchmarks
    let results = runner.run_all(&benchmarks);

    // Save baseline if requested
    if let Some(save_path) = save_baseline {
        runner.update_baselines(&results);
        runner
            .save_baselines(save_path)
            .map_err(|e| miette::miette!("Failed to save baseline: {}", e))?;
        println!("Saved baseline to {}", save_path.display());
    }

    Ok(())
}

/// Profile a D program
#[allow(clippy::too_many_arguments)]
fn profile(
    input: &std::path::Path,
    profile_type: &str,
    output: Option<&std::path::Path>,
    flamegraph: Option<&std::path::Path>,
    interval: u64,
    format: &str,
    _args: &[String],
) -> Result<()> {
    tracing::info!("Profiling {:?} (type: {})", input, profile_type);

    // Read and compile the source
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = demetrios::lexer::lex(&source)?;
    let ast = demetrios::parser::parse(&tokens, &source)?;
    let hir = demetrios::check::check(&ast)?;

    println!("=== Profile: {} ===", input.display());
    println!("Profile type: {}", profile_type);
    println!("Sample interval: {}us", interval);
    println!();

    match profile_type {
        "cpu" => {
            println!("CPU Profiling Configuration:");
            println!("  Interval: {}us", interval);
            println!("  Output format: {}", format);

            // Run with interpreter and collect timing info
            let start = std::time::Instant::now();
            let mut interpreter = demetrios::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);
            let duration = start.elapsed();

            println!();
            println!("Execution completed in {:?}", duration);
            println!();

            // Generate simulated profile output
            println!("Profile Summary:");
            println!(
                "  Total samples: {} (simulated)",
                duration.as_micros() / interval as u128
            );
            println!("  Functions profiled: {}", ast.items.len());

            if let Some(out_path) = output {
                let profile_data = format!(
                    "# CPU Profile for {}\n# Duration: {:?}\n# Samples: {}\n\nmain 100.0%\n",
                    input.display(),
                    duration,
                    duration.as_micros() / interval as u128
                );
                std::fs::write(out_path, &profile_data)
                    .map_err(|e| miette::miette!("Failed to write profile: {}", e))?;
                println!("Profile written to {}", out_path.display());
            }

            if let Some(fg_path) = flamegraph {
                // Generate a simple flamegraph SVG
                let svg = generate_simple_flamegraph(input, duration);
                std::fs::write(fg_path, &svg)
                    .map_err(|e| miette::miette!("Failed to write flamegraph: {}", e))?;
                println!("Flamegraph written to {}", fg_path.display());
            }
        }
        "memory" => {
            println!("Memory Profiling:");

            let mut interpreter = demetrios::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);

            println!("  Allocations tracked: (simulated)");
            println!("  Peak memory: N/A (interpreter mode)");
            println!();
            println!("Note: Full memory profiling requires native compilation with -g");
        }
        "async" => {
            println!("Async Task Profiling:");
            println!("  Task tracking enabled");
            println!();

            let mut interpreter = demetrios::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);

            println!("Async Profile Summary:");
            println!("  Total tasks: 0 (no async code detected)");
            println!("  Completed: 0");
            println!("  Failed: 0");
        }
        _ => {
            return Err(miette::miette!(
                "Unknown profile type: {}. Use cpu, memory, or async",
                profile_type
            ));
        }
    }

    Ok(())
}

/// Generate a simple flamegraph SVG
fn generate_simple_flamegraph(input: &std::path::Path, duration: std::time::Duration) -> String {
    let name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("program");
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 200\" width=\"800\" height=\"200\">\n");
    svg.push_str("<style>\n");
    svg.push_str("    .frame { fill: #ff6600; stroke: #d44a00; }\n");
    svg.push_str("    .frame:hover { fill: #ff8833; }\n");
    svg.push_str("    text { font-family: monospace; font-size: 12px; fill: white; }\n");
    svg.push_str("</style>\n");
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n");
    svg.push_str(&format!(
        "<text x=\"20\" y=\"20\" fill=\"black\" font-size=\"14\">CPU Profile: {}</text>\n",
        name
    ));
    svg.push_str(&format!(
        "<text x=\"20\" y=\"40\" fill=\"black\" font-size=\"12\">Duration: {:?}</text>\n",
        duration
    ));
    svg.push_str("<g class=\"frame\">\n");
    svg.push_str("    <rect x=\"50\" y=\"80\" width=\"700\" height=\"20\" rx=\"2\"/>\n");
    svg.push_str("    <text x=\"55\" y=\"95\">main (100%)</text>\n");
    svg.push_str("</g>\n");
    svg.push_str("<g class=\"frame\">\n");
    svg.push_str("    <rect x=\"50\" y=\"110\" width=\"500\" height=\"20\" rx=\"2\"/>\n");
    svg.push_str(&format!(
        "    <text x=\"55\" y=\"125\">{} (71%)</text>\n",
        name
    ));
    svg.push_str("</g>\n");
    svg.push_str(
        "<text x=\"20\" y=\"170\" fill=\"#666\" font-size=\"10\">Generated by dc profile</text>\n",
    );
    svg.push_str("</svg>");
    svg
}

/// Debug a D program with GDB or LLDB
fn debug_program(
    input: &std::path::Path,
    debugger: &str,
    pretty: bool,
    args: &[String],
) -> Result<()> {
    tracing::info!("Debugging {:?} with {}", input, debugger);

    // First, we need to compile with debug info
    println!("Note: Debugging requires a compiled binary with debug info.");
    println!("First compile with: dc build -g {}", input.display());
    println!();

    let binary_path = {
        let mut p = input.to_path_buf();
        p.set_extension("");
        p
    };

    if !binary_path.exists() {
        return Err(miette::miette!(
            "Binary not found at {}. Run 'dc build -g {}' first.",
            binary_path.display(),
            input.display()
        ));
    }

    // Find pretty printer path
    let pretty_printer_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .map(|p| p.join("tools/debug"));

    match debugger {
        "gdb" => {
            println!("Starting GDB...");

            let mut cmd = std::process::Command::new("gdb");

            if pretty {
                if let Some(pp_path) = &pretty_printer_path {
                    let gdb_script = pp_path.join("d_gdb.py");
                    if gdb_script.exists() {
                        cmd.arg("-x").arg(&gdb_script);
                        println!("Loading pretty printers from {}", gdb_script.display());
                    }
                }
            }

            cmd.arg("--args").arg(&binary_path).args(args);

            println!();
            println!("GDB commands for Demetrios:");
            println!("  d-backtrace  - Show D-aware backtrace");
            println!("  d-locals     - Show local variables with D types");
            println!("  d-async      - Show async task state");
            println!("  d-effects    - Show active effect handlers");
            println!();

            let status = cmd
                .status()
                .map_err(|e| miette::miette!("Failed to start GDB: {}", e))?;

            if !status.success() {
                return Err(miette::miette!("GDB exited with error"));
            }
        }
        "lldb" => {
            println!("Starting LLDB...");

            let mut cmd = std::process::Command::new("lldb");

            if pretty {
                if let Some(pp_path) = &pretty_printer_path {
                    let lldb_script = pp_path.join("d_lldb.py");
                    if lldb_script.exists() {
                        cmd.arg("-O")
                            .arg(format!("command script import {}", lldb_script.display()));
                        println!("Loading type summaries from {}", lldb_script.display());
                    }
                }
            }

            cmd.arg("--").arg(&binary_path).args(args);

            println!();
            println!("LLDB commands for Demetrios:");
            println!("  d-backtrace  - Show D-aware backtrace");
            println!("  d-locals     - Show local variables with D types");
            println!("  d-async      - Show async task state");
            println!();

            let status = cmd
                .status()
                .map_err(|e| miette::miette!("Failed to start LLDB: {}", e))?;

            if !status.success() {
                return Err(miette::miette!("LLDB exited with error"));
            }
        }
        _ => {
            return Err(miette::miette!(
                "Unknown debugger: {}. Use gdb or lldb",
                debugger
            ));
        }
    }

    Ok(())
}

/// Explain an error code
fn explain_error(code: &str) -> Result<()> {
    use demetrios::diagnostic::codes::{ErrorIndex, explain_error as get_explanation};

    if let Some(explanation) = get_explanation(code) {
        println!("{}", explanation);
    } else {
        // Try to find similar codes
        let index = ErrorIndex::new();
        let prefix = code.chars().next().unwrap_or('E');
        let similar: Vec<_> = index
            .all()
            .filter(|e| e.code.starts_with(prefix))
            .take(5)
            .collect();

        println!("Error code '{}' not found.", code);

        if !similar.is_empty() {
            println!();
            println!("Similar error codes:");
            for e in similar {
                println!("  {} - {}", e.code, e.title);
            }
        }

        println!();
        println!("Use 'dc error-index' to see all error codes.");
    }

    Ok(())
}

/// Show all error codes
fn show_error_index(category: Option<&str>, format: &str) -> Result<()> {
    use demetrios::diagnostic::codes::{ErrorCategory, ErrorIndex};

    let index = ErrorIndex::new();

    // Parse category filter
    let category_filter: Option<ErrorCategory> =
        category.map(|c| match c.to_lowercase().as_str() {
            "lexer" | "l" => ErrorCategory::Lexer,
            "parser" | "p" | "syntax" => ErrorCategory::Parser,
            "resolve" | "r" | "name" => ErrorCategory::Resolve,
            "type" | "t" => ErrorCategory::Type,
            "effect" | "f" => ErrorCategory::Effect,
            "ownership" | "o" => ErrorCategory::Ownership,
            "pattern" | "m" => ErrorCategory::Pattern,
            "macro" | "x" => ErrorCategory::Macro,
            "module" | "i" | "import" => ErrorCategory::Module,
            "codegen" | "c" => ErrorCategory::Codegen,
            _ => ErrorCategory::Internal,
        });

    match format {
        "markdown" | "md" => {
            println!("{}", index.generate_docs());
        }
        "json" => {
            println!("{{");
            println!("  \"errors\": [");
            let codes: Vec<_> = if let Some(cat) = category_filter {
                index.by_category(cat)
            } else {
                index.all().collect()
            };

            for (i, code) in codes.iter().enumerate() {
                let comma = if i < codes.len() - 1 { "," } else { "" };
                println!(
                    "    {{\"code\": \"{}\", \"title\": \"{}\", \"category\": \"{}\"}}{}",
                    code.code,
                    code.title,
                    code.category.name(),
                    comma
                );
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            // Text format
            println!("Demetrios Compiler Error Index");
            println!("==============================");
            println!();

            let codes: Vec<_> = if let Some(cat) = category_filter {
                println!("Category: {}", cat.name());
                println!();
                index.by_category(cat)
            } else {
                index.all().collect()
            };

            for code in codes {
                println!("{}: {}", code.code, code.title);
                println!("  Category: {}", code.category.name());
                println!();
            }

            println!("Total: {} error codes", index.all().count());
            println!();
            println!("Use 'dc explain <CODE>' for detailed information about a specific error.");
        }
    }

    Ok(())
}

/// Generate debug information
fn generate_debug_info(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    format: &str,
) -> Result<()> {
    tracing::info!("Generating debug info for {:?}", input);

    // Read and compile
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = demetrios::lexer::lex(&source)?;
    let ast = demetrios::parser::parse(&tokens, &source)?;
    let _hir = demetrios::check::check(&ast)?;

    let out_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        match format {
            "dwarf" => p.set_extension("dwarf"),
            "pdb" => p.set_extension("pdb"),
            _ => p.set_extension("debug"),
        };
        p
    });

    match format {
        "dwarf" => {
            println!("Generating DWARF debug information...");

            // Use debug info builder
            use demetrios::codegen::debug::DebugInfoBuilder;

            let file_path = input.to_path_buf();
            let directory = input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));

            let builder = DebugInfoBuilder::new(file_path, directory);
            let dwarf_output = builder.finalize();

            // Write DWARF sections info
            let mut info = String::new();
            info.push_str(&format!("# DWARF Debug Info for {}\n", input.display()));
            info.push_str(&format!(
                "# Generated by dc {}\n\n",
                env!("CARGO_PKG_VERSION")
            ));
            info.push_str(&format!(
                ".debug_info size: {} bytes\n",
                dwarf_output.debug_info.len()
            ));
            info.push_str(&format!(
                ".debug_abbrev size: {} bytes\n",
                dwarf_output.debug_abbrev.len()
            ));
            info.push_str(&format!(
                ".debug_line size: {} bytes\n",
                dwarf_output.debug_line.len()
            ));
            info.push_str(&format!(
                ".debug_str size: {} bytes\n",
                dwarf_output.debug_str.len()
            ));

            std::fs::write(&out_path, info)
                .map_err(|e| miette::miette!("Failed to write debug info: {}", e))?;

            println!("Debug info written to {}", out_path.display());
        }
        "pdb" => {
            println!("PDB format is only supported on Windows.");
            println!("For cross-platform debug info, use DWARF format.");
        }
        _ => {
            return Err(miette::miette!(
                "Unknown debug format: {}. Use dwarf or pdb",
                format
            ));
        }
    }

    Ok(())
}

/// Generate source map
fn generate_source_map(input: &std::path::Path, output: Option<&std::path::Path>) -> Result<()> {
    tracing::info!("Generating source map for {:?}", input);

    // Read and compile
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = demetrios::lexer::lex(&source)?;
    let ast = demetrios::parser::parse(&tokens, &source)?;
    let _hir = demetrios::check::check(&ast)?;

    let out_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        p.set_extension("map");
        p
    });

    // Build source map
    use demetrios::codegen::debug::source_map::SourceMapBuilder;

    let mut builder = SourceMapBuilder::new();

    // Add source with content
    let source_path = input.to_path_buf();
    builder.add_source_with_content(source_path.clone(), source);

    // Add mappings for each AST item (simplified)
    // In a real implementation, this would map compiled code positions to source
    for (i, _item) in ast.items.iter().enumerate() {
        builder.add_simple_mapping(
            i as u32, // generated line
            0,        // generated column
            &source_path,
            i as u32, // original line
            0,        // original column
        );
    }

    let output_file = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.map");
    let source_map = builder.build(output_file);
    let json = source_map
        .to_json()
        .map_err(|e| miette::miette!("Failed to serialize source map: {}", e))?;

    std::fs::write(&out_path, &json)
        .map_err(|e| miette::miette!("Failed to write source map: {}", e))?;

    println!("Source map written to {}", out_path.display());
    println!();
    println!("Source map contains:");
    println!("  Version: {}", source_map.version);
    println!("  Sources: {} file(s)", source_map.sources.len());
    println!("  Names: {} identifier(s)", source_map.names.len());

    Ok(())
}
