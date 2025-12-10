# Installing Demetrios on Windows

## Quick Install (One-Line)

Open PowerShell and run:

`powershell
iwr -useb https://raw.githubusercontent.com/Chiuratto-AI/demetrios/main/install.ps1 | iex
`

## Manual Installation

### Prerequisites

1. **Rust** (automatically installed if missing)
2. **Visual Studio Build Tools** (C++ workload)

### Steps

1. Clone the repository:
`powershell
git clone https://github.com/Chiuratto-AI/demetrios.git
cd demetrios
`

2. Run the installer:
`powershell
.\install-windows.ps1
`

### Installer Options

| Option | Description |
|--------|-------------|
| -InstallPath <path> | Custom install location (default: %LOCALAPPDATA%\Demetrios) |
| -Features <list> | Enable features: lsp, jit, gpu, smt, ull |
| -SkipRust | Skip Rust installation check |
| -Uninstall | Remove Demetrios |
| -Help | Show help |

### Examples

`powershell
# Basic install
.\install-windows.ps1

# With LSP support (for VS Code)
.\install-windows.ps1 -Features "lsp"

# Custom location with JIT
.\install-windows.ps1 -InstallPath "C:\Demetrios" -Features "jit"

# Full installation (all features)
.\install-windows.ps1 -Features "full"

# Uninstall
.\install-windows.ps1 -Uninstall
`

## After Installation

Restart your terminal, then:

`powershell
dc --version          # Check version
dc --help             # Show help
dc new myproject      # Create new project
cd myproject
dc build              # Build project
dc run                # Run project
`

## Features

| Feature | Description | Requirements |
|---------|-------------|--------------|
| lsp | Language Server Protocol (IDE support) | None |
| jit | Cranelift JIT compilation | None |
| gpu | GPU codegen (PTX, SPIR-V) | None |
| smt | Z3 SMT solver for refinement types | Z3 library |
| ull | All features | All above |

## Troubleshooting

### Rust not found after install
Restart your terminal or run:
`powershell
$env:PATH = \"$env:USERPROFILE\.cargo\bin;$env:PATH\"
`

### Build fails with linker error
Install Visual Studio Build Tools:
`powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override \"--add Microsoft.VisualStudio.Workload.VCTools\"
`

### Permission denied
Run PowerShell as Administrator, or change execution policy:
`powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
`

## Uninstalling

`powershell
.\install-windows.ps1 -Uninstall
`

Or manually:
1. Delete %LOCALAPPDATA%\Demetrios
2. Remove from PATH in System Properties > Environment Variables