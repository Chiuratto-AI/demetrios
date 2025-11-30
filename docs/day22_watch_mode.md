# Day 22: Watch Mode, Hot Reload & Build Hooks

## Overview

Day 22 implements a comprehensive development experience system for the Demetrios compiler, including file watching with automatic rebuilds, hot reload capabilities, build hooks, and a development server with live reload.

## Features Implemented

### 1. File System Watcher (`watch/watcher.rs`)

A cross-platform file system watcher with:

- **Polling backend** for maximum compatibility
- **Event debouncing** to coalesce rapid file changes
- **Glob pattern filtering** for include/exclude rules
- **Recursive directory watching**

```rust
use demetrios::watch::{Watcher, WatchConfig};
use std::time::Duration;

let config = WatchConfig {
    paths: vec!["src".into()],
    recursive: true,
    debounce: Duration::from_millis(100),
    include_patterns: vec!["**/*.d".to_string()],
    ignore_patterns: vec!["**/target/**".to_string()],
    ..Default::default()
};

let mut watcher = Watcher::new(config)?;
watcher.start()?;

// Process events
while let Ok(event) = watcher.recv() {
    println!("File changed: {:?}", event.path);
}
```

### 2. Watch Mode Controller (`watch/mode.rs`)

Continuous build mode with:

- **Automatic rebuilds** on file changes
- **Keyboard controls** (quit, rebuild, pause/resume)
- **Status display** with ANSI colors
- **Build statistics** tracking
- **Optional test running** after builds
- **Command execution** on successful builds

```bash
# Start watch mode
dc watch src --clear --test

# Watch with custom command
dc watch src -x "dc run examples/main.d"

# Watch with debounce control
dc watch src --debounce 200 --ignore "*.tmp"
```

### 3. Hot Reload Engine (`watch/hotreload.rs`)

Runtime code patching system with:

- **TCP server** for client connections
- **JSON protocol** for reload messages
- **Function patching** with relocations
- **State preservation** across reloads
- **Rollback capability** on errors

```rust
use demetrios::watch::{HotReloadEngine, HotReloadConfig};

let config = HotReloadConfig {
    port: 9090,
    preserve_state: true,
    ..Default::default()
};

let mut engine = HotReloadEngine::new(config)?;
engine.start()?;

// Push updates to connected clients
engine.broadcast_update(function_updates)?;
```

#### Reload Protocol

```json
// Server -> Client: Hello
{"type": "Hello", "version": 1, "client_id": 12345}

// Server -> Client: Update available
{"type": "Update", "version": 2, "functions": [...]}

// Client -> Server: Apply update
{"type": "Apply", "version": 2}

// Server -> Client: Rollback on error
{"type": "Rollback", "version": 1}
```

### 4. Build Script System (`watch/buildscript.rs`)

Support for `build.d` build scripts (similar to Cargo's `build.rs`):

- **Instruction parsing** with `cargo:` format
- **Build API** for generated files
- **FFI binding generation**
- **Conditional compilation**

```d
// build.d
fn main() with IO {
    // Tell the build system to rerun if build.d changes
    println("cargo:rerun-if-changed=build.d")
    
    // Link a native library
    println("cargo:rustc-link-lib=sqlite3")
    
    // Set a cfg flag
    println("cargo:rustc-cfg=feature=\"native\"")
    
    // Generate bindings
    generate_bindings("include/api.h", "src/bindings.d")
}
```

#### Supported Instructions

| Instruction | Description |
|-------------|-------------|
| `cargo:rerun-if-changed=PATH` | Rerun if file changes |
| `cargo:rerun-if-env-changed=VAR` | Rerun if env var changes |
| `cargo:rustc-link-lib=NAME` | Link a library |
| `cargo:rustc-link-search=PATH` | Add library search path |
| `cargo:rustc-cfg=CFG` | Set compilation flag |
| `cargo:warning=MSG` | Emit a warning |
| `cargo:error=MSG` | Emit an error |

### 5. Build Hooks Manager (`watch/hooks.rs`)

Lifecycle hooks for build events:

- **Multiple hook points** (pre-build, post-build, etc.)
- **Shell commands** and D scripts
- **Built-in actions** (echo, touch, copy)
- **Conditional execution**
- **Priority ordering**
- **Hook chaining**

```toml
# d.toml
[[hooks]]
name = "format-check"
points = ["pre-build"]
command = "dc"
args = ["fmt", "--check"]
priority = 10

[[hooks]]
name = "notify-success"
points = ["post-build"]
command = "notify-send"
args = ["Build", "Compilation successful!"]
continue_on_failure = true
```

#### Hook Points

| Point | Description |
|-------|-------------|
| `pre-build` | Before compilation starts |
| `post-build` | After successful compilation |
| `on-build-error` | After compilation failure |
| `pre-test` | Before tests run |
| `post-test` | After tests complete |
| `watch-start` | When watch mode starts |
| `watch-stop` | When watch mode stops |
| `on-file-change` | When files change |
| `pre-reload` | Before hot reload |
| `post-reload` | After hot reload |

### 6. Development Server (`watch/devserver.rs`)

Lightweight HTTP server for development:

- **Static file serving**
- **Live reload injection** (WebSocket)
- **CORS configuration**
- **Directory listing**
- **SPA fallback** support
- **Custom headers**

```bash
# Start development server
dc serve ./dist --port 8080 --open

# With directory listing
dc serve . --directory-listing

# SPA mode
dc serve ./dist --spa index.html
```

### 7. CLI Integration

New commands added to `dc`:

```bash
# Watch mode
dc watch [PATHS...] [OPTIONS]
  -c, --clear           Clear screen before each rebuild
  -t, --test            Run tests after successful build
  -x, --exec <CMD>      Command to run after build
      --debounce <MS>   Debounce delay (default: 100)
      --ignore <GLOB>   Patterns to ignore
  -v, --verbose         Verbose output

# Development server
dc serve [DIR] [OPTIONS]
  -p, --port <PORT>     Port to listen on (default: 3000)
      --host <HOST>     Host to bind to (default: 127.0.0.1)
      --no-reload       Disable live reload
  -o, --open            Open browser automatically
      --directory-listing  Enable directory listing
      --spa <FILE>      SPA fallback file
  -v, --verbose         Verbose output

# Run build hooks
dc hook <POINT> [OPTIONS]
  -p, --project <DIR>   Project root directory
  -v, --verbose         Verbose output
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Watch Mode System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐        │
│  │  Watcher │──>│ Debouncer│──>│ Watch Controller │        │
│  └──────────┘   └──────────┘   └────────┬─────────┘        │
│                                          │                  │
│  ┌──────────────────────────────────────┴───────────┐      │
│  │              Build Pipeline                       │      │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────────────┐ │      │
│  │  │ Pre-Hook│──>│  Build  │──>│ Post-Hook       │ │      │
│  │  └─────────┘   └─────────┘   └────────┬────────┘ │      │
│  └───────────────────────────────────────┼──────────┘      │
│                                          │                  │
│  ┌───────────────────────────────────────┴───────────┐     │
│  │              Hot Reload / Dev Server              │     │
│  │  ┌──────────────┐   ┌────────────────────────┐   │     │
│  │  │ Hot Reload   │   │ Development Server     │   │     │
│  │  │ Engine       │   │ (Static + Live Reload) │   │     │
│  │  └──────────────┘   └────────────────────────┘   │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/watch/
├── mod.rs           # Module exports and convenience functions
├── watcher.rs       # File system watcher (~500 lines)
├── mode.rs          # Watch mode controller (~450 lines)
├── hotreload.rs     # Hot reload engine (~500 lines)
├── buildscript.rs   # Build script support (~400 lines)
├── hooks.rs         # Build hooks manager (~550 lines)
└── devserver.rs     # Development server (~600 lines)
```

## Dependencies Added

```toml
[dependencies]
ctrlc = "3.4"  # Signal handling for graceful shutdown
```

## Test Results

All watch module tests pass:

```
test watch::devserver::tests::test_content_type_guessing ... ok
test watch::devserver::tests::test_cors_config ... ok
test watch::devserver::tests::test_dev_server_config_default ... ok
test watch::devserver::tests::test_http_response ... ok
test watch::devserver::tests::test_url_decode ... ok
test watch::devserver::tests::test_websocket_frame_encoding ... ok
test watch::hooks::tests::test_builtin_echo ... ok
test watch::hooks::tests::test_hook_condition_env ... ok
test watch::hooks::tests::test_hook_condition_logic ... ok
test watch::hooks::tests::test_hook_context_env ... ok
test watch::hooks::tests::test_hook_manager_register ... ok
test watch::hooks::tests::test_hook_point_roundtrip ... ok
test watch::hooks::tests::test_hook_priority ... ok
test watch::hotreload::tests::test_function_update_serialization ... ok
test watch::hotreload::tests::test_hot_reload_config_default ... ok
test watch::hotreload::tests::test_hot_reload_runtime ... ok
test watch::hotreload::tests::test_reload_message_serialization ... ok
test watch::mode::tests::test_build_result_default ... ok
test watch::mode::tests::test_watch_mode_config_default ... ok
test watch::mode::tests::test_watch_state_display ... ok
test watch::mode::tests::test_watch_stats ... ok
test watch::tests::test_hook_point_values ... ok
test watch::tests::test_module_exports ... ok
test watch::watcher::tests::test_debouncer ... ok
test watch::watcher::tests::test_poll_backend_file_detection ... ok
test watch::watcher::tests::test_should_include ... ok
test watch::watcher::tests::test_watch_config_builder ... ok
test watch::watcher::tests::test_watch_config_default ... ok
test watch::watcher::tests::test_watcher_creation ... ok
```

## Usage Examples

### Basic Watch Mode

```bash
# Watch src directory and rebuild on changes
dc watch src

# Watch with automatic test running
dc watch src --test --clear

# Watch and run specific command on success
dc watch src -x "./target/debug/myapp"
```

### Development Server

```bash
# Serve current directory
dc serve

# Serve with custom port and auto-open browser
dc serve ./public -p 8080 --open

# Serve SPA with fallback routing
dc serve ./dist --spa index.html
```

### Build Hooks

```bash
# Run pre-build hooks
dc hook pre-build -v

# Run post-test hooks
dc hook post-test --project ./my-project
```

## Future Enhancements

1. **Native file system events** - Use `notify` crate for efficient watching
2. **Incremental hot reload** - Only patch changed functions
3. **Source map support** - Map hot-reloaded code to source
4. **Remote debugging** - Hot reload over network
5. **Plugin hooks** - Custom hook actions in D

## Summary

Day 22 adds ~3,000 lines of code implementing:

- File system watching with debouncing
- Continuous build mode with keyboard controls
- Hot reload protocol and runtime
- Build script support (build.d)
- Lifecycle hooks system
- Development server with live reload
- Full CLI integration

The watch system provides a modern development experience with instant feedback on code changes, essential for productive development workflows.
