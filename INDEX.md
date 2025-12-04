# Demetrios (D) — Master Index

## Quick Start

1. **Configure Claude Code** — Copy `CLAUDE.md` to `/mnt/e/workspace/demetrios/CLAUDE.md`
2. **Set System Prompt** — Use `SYSTEM_PROMPT_DEMETRIOS.md` in Zed settings
3. **Follow Daily Prompts** — Execute Day 2-5 prompts in sequence

---

## Files Index

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| [CLAUDE.md](computer:///mnt/user-data/outputs/CLAUDE.md) | Claude Code rules | Copy to repo root |
| [SYSTEM_PROMPT_DEMETRIOS.md](computer:///mnt/user-data/outputs/SYSTEM_PROMPT_DEMETRIOS.md) | Full system prompt | Zed settings |

### Daily Development Prompts

| Day | Focus | File |
|-----|-------|------|
| 1 | Scaffold | [PROMPT_CLAUDE_CODE_DEMETRIOS.md](computer:///mnt/user-data/outputs/PROMPT_CLAUDE_CODE_DEMETRIOS.md) |
| 2 | Stub Files | [PROMPT_DEMETRIOS_DAY2.md](computer:///mnt/user-data/outputs/PROMPT_DEMETRIOS_DAY2.md) |
| 3 | First Pipeline | [PROMPT_DEMETRIOS_DAY3.md](computer:///mnt/user-data/outputs/PROMPT_DEMETRIOS_DAY3.md) |
| 4 | Name Resolution | [PROMPT_DEMETRIOS_DAY4.md](computer:///mnt/user-data/outputs/PROMPT_DEMETRIOS_DAY4.md) |
| 5 | Effects & Ownership | [PROMPT_DEMETRIOS_DAY5.md](computer:///mnt/user-data/outputs/PROMPT_DEMETRIOS_DAY5.md) |

### Documentation

| File | Purpose |
|------|---------|
| [LANGUAGE_SPECIFICATION.md](computer:///mnt/user-data/outputs/docs/LANGUAGE_SPECIFICATION.md) | Complete language spec (Q1 quality) |
| [DOCUMENTATION_TEMPLATE.md](computer:///mnt/user-data/outputs/docs/DOCUMENTATION_TEMPLATE.md) | How to update docs after each day |

### Source Code

| File | Purpose |
|------|---------|
| [demetrios-scaffold.zip](computer:///mnt/user-data/outputs/demetrios-scaffold.zip) | Initial compiler scaffold |

---

## Setup Instructions

### 1. Configure Zed for Claude Code

Edit `~/.config/zed/settings.json`:

```json
{
  "assistant": {
    "default_model": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514"
    },
    "version": "2"
  },
  "language_models": {
    "anthropic": {
      "api_url": "https://api.anthropic.com"
    }
  }
}
```

### 2. Add Rules File

Copy `CLAUDE.md` to your repo:

```bash
cp CLAUDE.md /mnt/e/workspace/demetrios/CLAUDE.md
```

### 3. Configure MCP Filesystem

Edit `~/.config/claude/mcp_servers.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/mnt/e/workspace"
      ]
    }
  }
}
```

---

## Development Workflow

### Daily Session

1. **Start**
   ```bash
   cd /mnt/e/workspace/demetrios
   cargo build
   cargo test
   ```

2. **Open Prompt**
   - Copy the day's prompt to Claude Code
   - Follow instructions

3. **End**
   - Run `cargo test`
   - Update documentation
   - Commit changes

### Documentation Update

After each day, update:
- `docs/LANGUAGE_SPECIFICATION.md`
- `docs/IMPLEMENTATION_STATUS.md`
- `docs/CHANGELOG.md`

See `DOCUMENTATION_TEMPLATE.md` for detailed checklist.

---

## Project Structure

```
demetrios/
├── CLAUDE.md               ← Claude Code rules
├── compiler/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         ← CLI
│       ├── lib.rs          ← Library root
│       ├── lexer/          ← Tokenization
│       ├── parser/         ← Parsing
│       ├── ast/            ← Syntax tree
│       ├── resolve/        ← Name resolution
│       ├── check/          ← Type checker
│       ├── types/          ← Type system
│       ├── effects/        ← Effect system
│       ├── ownership/      ← Borrow checker
│       ├── hir/            ← High-level IR
│       ├── hlir/           ← SSA IR
│       ├── mlir/           ← MLIR integration
│       └── codegen/        ← Backends
├── stdlib/                 ← Standard library
├── docs/                   ← Documentation
├── examples/               ← Example programs
└── tests/                  ← Integration tests
```

---

## Key Language Features

| Feature | Syntax | Status |
|---------|--------|--------|
| Effects | `fn foo() with IO` | ✅ Day 5 |
| Linear Types | `linear struct X {}` | ✅ Day 5 |
| Exclusive Ref | `&!T` | ✅ Day 4 |
| Units | `500.0_mg` | 🟡 Partial |
| Refinements | `{ x: T \| pred }` | 🔴 Planned |
| GPU | `kernel fn` | 🔴 Planned |

---

## Quick Reference

### D Syntax Cheatsheet

```d
// Variables
let x: int = 42              // Immutable
var y: f64 = 0.0             // Mutable
const PI: f64 = 3.14159      // Compile-time

// Functions
fn add(a: int, b: int) -> int {
    return a + b
}

// Effects
fn read_file(path: string) -> string with IO, Panic {
    // ...
}

// Structs
linear struct FileHandle {
    fd: int,
}

// References
let r: &int = &x             // Shared borrow
let m: &!int = &!y           // Exclusive borrow

// Generics
fn identity<T>(x: T) -> T {
    return x
}
```

---

## Contact

**Language Creator**: Demetrios Chiuratto Agourakis  
**Project**: Demetrios Programming Language  
**Repository**: `/mnt/e/workspace/demetrios`
