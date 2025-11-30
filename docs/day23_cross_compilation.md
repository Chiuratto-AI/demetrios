# Day 23: Cross-Compilation & Target Specification

## Overview

Day 23 implements a comprehensive cross-compilation system for the Demetrios compiler. This enables building programs for different architectures, operating systems, and environments from a single host machine.

## Features Implemented

### 1. Target Specification System (`src/target/spec.rs`)

The target specification system provides a complete description of compilation targets:

```rust
// Parse a target triple
let spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu")?;

// Access target properties
println!("Architecture: {}", spec.arch.arch);  // x86_64
println!("OS: {}", spec.os.os);                 // linux
println!("Environment: {}", spec.env.env);     // gnu
println!("Pointer width: {}", spec.pointer_width());  // 64
```

#### Target Triple Format

Targets follow the standard triple format: `<arch>-<vendor>-<os>[-<env>]`

- **Architecture**: `x86_64`, `aarch64`, `riscv64`, `wasm32`, `arm`, etc.
- **Vendor**: `unknown`, `pc`, `apple`, `nvidia`, etc.
- **Operating System**: `linux`, `windows`, `macos`, `none` (bare metal), etc.
- **Environment**: `gnu`, `musl`, `msvc`, `eabi`, etc.

#### Target Components

```rust
pub struct TargetSpec {
    pub triple: TargetTriple,    // Parsed triple
    pub arch: ArchSpec,          // Architecture details
    pub os: OsSpec,              // Operating system details
    pub env: EnvSpec,            // Environment/ABI details
    pub linker: LinkerSpec,      // Linker configuration
    pub codegen: CodegenSpec,    // Code generation options
    pub options: TargetOptions,  // Additional options
}
```

### 2. Sysroot Management (`src/target/sysroot.rs`)

Sysroots contain platform-specific libraries and headers needed for cross-compilation:

```rust
let manager = SysrootManager::new();

// Discover existing sysroots
let sysroots = manager.discover_sysroots()?;

// Get sysroot for a specific target
if let Some(sysroot) = manager.get_sysroot(&target_spec)? {
    println!("Sysroot path: {}", sysroot.path.display());
    println!("Components: {:?}", sysroot.components);
}

// Build a new sysroot
let builder = SysrootBuilder::new(target_spec);
builder.build(&output_dir)?;
```

#### Sysroot Components

- **Core**: Core runtime library
- **Std**: Standard library
- **Alloc**: Allocation support
- **Test**: Testing framework
- **Proc_macro**: Procedural macros

### 3. Conditional Compilation (`src/target/cfg.rs`)

The cfg system enables platform-specific code:

```d
// In Demetrios source code
#[cfg(target_os = "linux")]
fn platform_init() { ... }

#[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
fn optimized_hash() { ... }

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn desktop_only() { ... }

#[cfg(not(target_env = "musl"))]
fn glibc_specific() { ... }
```

#### Cfg Predicates

```rust
// Evaluate cfg predicates
let ctx = CfgContext::from_target(&target_spec);

let pred = CfgPredicate::parse("all(target_os = \"linux\", target_arch = \"x86_64\")")?;
if ctx.evaluate(&pred) {
    // Include this code
}
```

Built-in cfg keys:
- `target_arch`: Architecture (x86_64, aarch64, etc.)
- `target_os`: Operating system (linux, windows, etc.)
- `target_env`: Environment (gnu, musl, msvc)
- `target_vendor`: Vendor (unknown, apple, etc.)
- `target_family`: OS family (unix, windows)
- `target_pointer_width`: Pointer size (32, 64)
- `target_endian`: Byte order (little, big)
- `target_feature`: CPU features (sse4.2, neon, etc.)

### 4. Linker Integration (`src/target/linker.rs`)

Platform-specific linker support:

```rust
let linker = Linker::new(&target_spec);

// Link object files
let output = linker.link(
    &[PathBuf::from("main.o"), PathBuf::from("lib.o")],
    &PathBuf::from("output"),
    OutputType::Executable,
)?;
```

#### Linker Flavors

- **Gnu**: GNU ld (Linux default)
- **Gcc**: GCC as linker driver
- **Lld**: LLVM's lld
- **MsvcLink**: Microsoft link.exe
- **Darwin**: macOS linker
- **WasmLd**: WebAssembly linker
- **Em**: Emscripten

#### Linker Scripts

For embedded/kernel targets:

```rust
let builder = LinkerScriptBuilder::new(&target_spec);
builder
    .entry_point("_start")
    .add_memory_region("FLASH", 0x08000000, 512 * 1024, "rx")
    .add_memory_region("RAM", 0x20000000, 128 * 1024, "rwx")
    .add_section(".text", "FLASH", SectionType::Code)
    .add_section(".data", "RAM", SectionType::Data);

let script = builder.build()?;
```

### 5. Target Registry (`src/target/registry.rs`)

Built-in target definitions:

```rust
// Get the global registry
let registry = global_registry();

// List all targets
for name in registry.list() {
    println!("{}", name);
}

// Get a specific target
let spec = registry.get("x86_64-unknown-linux-gnu")?;

// Register a custom target
let mut registry = TargetRegistry::with_builtins();
registry.register(custom_spec);
```

#### Built-in Targets

**Linux:**
- `x86_64-unknown-linux-gnu`
- `x86_64-unknown-linux-musl`
- `i686-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `arm-unknown-linux-gnueabihf`
- `riscv64gc-unknown-linux-gnu`

**macOS:**
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`

**Windows:**
- `x86_64-pc-windows-msvc`
- `x86_64-pc-windows-gnu`
- `i686-pc-windows-msvc`
- `i686-pc-windows-gnu`
- `aarch64-pc-windows-msvc`

**WebAssembly:**
- `wasm32-unknown-unknown`
- `wasm32-wasi`
- `wasm32-unknown-emscripten`

**Embedded/Bare Metal:**
- `thumbv7em-none-eabihf` (ARM Cortex-M4/M7)
- `riscv32imac-unknown-none-elf`
- `riscv64gc-unknown-none-elf`
- `x86_64-unknown-none` (Kernel development)

**Other:**
- `nvptx64-nvidia-cuda` (NVIDIA GPU)

## CLI Commands

### Target Commands

```bash
# List all available targets
dc target list

# Show host target information
dc target host

# Show detailed info for a target
dc target info x86_64-unknown-linux-gnu

# Add custom target from JSON file
dc target add my-target.json

# Create a new target specification interactively
dc target create --name my-embedded --base thumbv7em-none-eabihf

# Evaluate cfg predicates for a target
dc target cfg x86_64-unknown-linux-gnu 'target_os = "linux"'
dc target cfg aarch64-apple-darwin 'any(target_os = "macos", target_os = "ios")'
```

### Sysroot Commands

```bash
# List installed sysroots
dc sysroot list

# Show sysroot for a target
dc sysroot show x86_64-unknown-linux-gnu

# Install sysroot for a target
dc sysroot install aarch64-unknown-linux-gnu

# Remove a sysroot
dc sysroot remove aarch64-unknown-linux-gnu

# Clean all cached sysroots
dc sysroot clean
```

### Cross-Compilation

```bash
# Build for a different target
dc build --target aarch64-unknown-linux-gnu main.d

# Build for WebAssembly
dc build --target wasm32-wasi main.d

# Build for embedded ARM
dc build --target thumbv7em-none-eabihf --release embedded.d
```

## Target JSON Format

Custom targets can be defined in JSON:

```json
{
  "triple": {
    "arch": "myarch",
    "vendor": "myvendor",
    "os": "myos",
    "env": "myenv"
  },
  "arch": {
    "arch": "x86_64",
    "cpu": "generic",
    "features": ["sse2", "sse4.2"],
    "data_model": "LP64"
  },
  "os": {
    "os": "Linux",
    "family": "Unix",
    "min_version": "4.0",
    "requires_pie": true,
    "panic_strategy": "Unwind"
  },
  "env": {
    "env": "Gnu",
    "crt": "Glibc",
    "relocation_model": "Pic",
    "code_model": "Small"
  },
  "linker": {
    "flavor": "Gnu",
    "path": "/usr/bin/ld"
  },
  "options": {
    "is_builtin": false,
    "description": "My custom target"
  }
}
```

## Architecture

```
src/target/
├── mod.rs          # Module exports
├── spec.rs         # Target specification (TargetSpec, TargetTriple, etc.)
├── sysroot.rs      # Sysroot management
├── cfg.rs          # Conditional compilation
├── linker.rs       # Linker integration
└── registry.rs     # Target registry with built-in targets
```

## Usage Examples

### Cross-Compiling for ARM Linux

```bash
# Install the sysroot
dc sysroot install aarch64-unknown-linux-gnu

# Build for ARM64 Linux
dc build --target aarch64-unknown-linux-gnu --release app.d

# The output will be in target/aarch64-unknown-linux-gnu/release/
```

### Building for WebAssembly

```bash
# Build for WASI (WebAssembly System Interface)
dc build --target wasm32-wasi app.d

# Build for browser (no WASI)
dc build --target wasm32-unknown-unknown --release web_app.d
```

### Embedded Development

```bash
# Build for ARM Cortex-M4
dc build --target thumbv7em-none-eabihf --release firmware.d

# Generate linker script for custom memory layout
dc target create --name my-mcu --base thumbv7em-none-eabihf \
    --flash-start 0x08000000 --flash-size 512K \
    --ram-start 0x20000000 --ram-size 128K
```

### Kernel Development

```bash
# Build for bare-metal x86_64 (kernel)
dc build --target x86_64-unknown-none kernel.d
```

## Dependencies Added

- `which = "6"` - Finding linker executables
- `dirs = "5"` - Platform-specific cache directories
- `chrono = "0.4"` - Timestamp formatting for sysroot metadata

## Testing

All 405 tests pass:

```bash
cargo test
# test result: ok. 405 passed; 0 failed
```

New tests added:
- `target::spec::tests::*` - Target specification parsing
- `target::cfg::tests::*` - Cfg predicate evaluation
- `target::registry::tests::*` - Target registry operations
- `target::sysroot::tests::*` - Sysroot management
- `target::linker::tests::*` - Linker integration

## Future Enhancements

1. **Remote sysroot downloading** - Download pre-built sysroots from a server
2. **Target feature detection** - Auto-detect CPU features at compile time
3. **Cross-compilation toolchain management** - Install/manage cross-compilers
4. **Container-based cross-compilation** - Use Docker for hermetic builds
5. **iOS/Android targets** - Mobile platform support

## Summary

Day 23 adds comprehensive cross-compilation support to the Demetrios compiler:

- **25+ built-in targets** covering Linux, macOS, Windows, WebAssembly, and embedded platforms
- **Target specification system** with full control over architecture, OS, and ABI settings
- **Sysroot management** for cross-compilation dependencies
- **Conditional compilation** with cfg predicates
- **Linker integration** supporting GNU, LLVM, MSVC, and platform-specific linkers
- **CLI commands** for target discovery, sysroot management, and cross-compilation
