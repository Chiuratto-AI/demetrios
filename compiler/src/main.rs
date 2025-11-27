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

    /// Run a D program using JIT compilation
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl,

    /// Format D source code
    Fmt {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },

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

        Commands::Repl => repl(),

        Commands::Fmt { path, check } => format_code(&path, check),

        Commands::Info => info(),
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

fn repl() -> Result<()> {
    println!("Demetrios REPL v{}", env!("CARGO_PKG_VERSION"));
    println!("Type :help for help, :quit to exit");
    println!();

    let stdin = std::io::stdin();
    let mut line = String::new();

    loop {
        print!("d> ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        line.clear();
        if stdin.read_line(&mut line).is_err() {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            ":quit" | ":q" => break,
            ":help" | ":h" => {
                println!("Commands:");
                println!("  :help, :h    Show this help");
                println!("  :quit, :q    Exit the REPL");
                println!("  :type <expr> Show the type of an expression");
                println!();
            }
            _ if trimmed.starts_with(":type ") => {
                let expr = &trimmed[6..];
                println!("Type inference for '{}' not yet implemented", expr);
            }
            _ => {
                // Try to parse and evaluate
                match demetrios::lexer::lex(trimmed) {
                    Ok(tokens) => {
                        println!(
                            "Tokens: {:?}",
                            tokens.iter().map(|t| &t.kind).collect::<Vec<_>>()
                        );
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
        }
    }

    println!("Goodbye!");
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
    println!("  - LLVM (enabled)");
    #[cfg(not(feature = "llvm"))]
    println!("  - LLVM (disabled)");
    #[cfg(feature = "jit")]
    println!("  - Cranelift JIT (enabled)");
    #[cfg(not(feature = "jit"))]
    println!("  - Cranelift JIT (disabled)");
    #[cfg(feature = "gpu")]
    println!("  - GPU codegen (enabled)");
    #[cfg(not(feature = "gpu"))]
    println!("  - GPU codegen (disabled)");

    Ok(())
}
