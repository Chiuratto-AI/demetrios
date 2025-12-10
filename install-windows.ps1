#Requires -Version 5.1
param(
    [string]$InstallPath = "$env:LOCALAPPDATA\Demetrios",
    [string]$Features = "",
    [switch]$AddToPath = $true,
    [switch]$SkipRust = $false,
    [switch]$Uninstall = $false,
    [switch]$Help = $false
)
$ErrorActionPreference = "Stop"
function Write-Banner {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  DEMETRIOS - Programming Language Installer" -ForegroundColor White
    Write-Host "  Version: 0.58.0" -ForegroundColor Gray
    Write-Host "================================================================" -ForegroundColor Cyan
}
function Write-Step { param([string]$M) Write-Host "[+] $M" -ForegroundColor Green }
function Write-Info { param([string]$M) Write-Host "[i] $M" -ForegroundColor Cyan }
function Write-Err { param([string]$M) Write-Host "[x] $M" -ForegroundColor Red }
function Test-RustInstalled { try { $null -ne (Get-Command rustc -EA 0) -and $null -ne (Get-Command cargo -EA 0) } catch { $false } }
function Test-VSBuildTools { Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools" }
function Install-Rust {
    Write-Step "Installing Rust..."
    $ri = "$env:TEMP\rustup-init.exe"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $ri -UseBasicParsing
    Start-Process -FilePath $ri -ArgumentList "-y","--default-toolchain","stable","--profile","minimal" -Wait -NoNewWindow
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    Test-RustInstalled
}
function Install-VSBuildTools {
    if (Test-VSBuildTools) { Write-Info "Already installed"; return }
    Write-Step "Installing VS Build Tools..."
    $vi = "$env:TEMP\vs_buildtools.exe"
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" -OutFile $vi -UseBasicParsing
    Start-Process $vi -ArgumentList "--passive","--wait","--add","Microsoft.VisualStudio.Workload.VCTools","--includeRecommended" -Wait -Verb RunAs
}
function Build-Demetrios {
    param([string]$Src,[string]$Feat)
    Write-Step "Building Demetrios..."
    $cp = Join-Path $Src "compiler"
    Push-Location $cp
    try {
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
        $a = "build --release"; if ($Feat) { $a += " --features $Feat" }
        Write-Info "cargo $a (3-5 min)..."
        Invoke-Expression "cargo $a"
        $dc = Join-Path $cp "target\release\dc.exe"
        if (Test-Path $dc) { return $dc }
    } finally { Pop-Location }
}
function Install-Files {
    param([string]$Dc,[string]$Dst)
    $bin = Join-Path $Dst "bin"
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    Copy-Item $Dc (Join-Path $bin "dc.exe") -Force
    $std = Join-Path (Split-Path (Split-Path $Dc)) "..\stdlib"
    $lib = Join-Path $Dst "lib\stdlib"
    New-Item -ItemType Directory -Force -Path $lib | Out-Null
    if (Test-Path $std) { Copy-Item "$std\*" $lib -Recurse -Force }
    return $bin
}
function Add-ToUserPath { param([string]$B)
    $p = [Environment]::GetEnvironmentVariable("Path","User")
    if ($p -notlike "*$B*") { [Environment]::SetEnvironmentVariable("Path","$B;$p","User"); Write-Info "Restart terminal" }
}
if ($Help) { Write-Banner; Write-Host "  -InstallPath -Features -SkipRust -Uninstall -Help"; exit 0 }
Write-Banner
if ($Uninstall) { if (Test-Path $InstallPath) { Remove-Item $InstallPath -Recurse -Force }; exit 0 }
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $SkipRust -and -not (Test-RustInstalled)) { if ((Read-Host "Install Rust? (Y/n)") -match "^[Yy]?$") { Install-Rust } else { exit 1 } }
if (-not (Test-VSBuildTools)) { if ((Read-Host "Install VS Build Tools? (Y/n)") -match "^[Yy]?$") { Install-VSBuildTools } }
$dc = Build-Demetrios -Src $ScriptDir -Feat $Features
if (-not $dc) { Write-Err "Build failed!"; exit 1 }
$bin = Install-Files -Dc $dc -Dst $InstallPath
if ($AddToPath) { Add-ToUserPath -B $bin }
Write-Host "Installation Complete! Run: dc --help" -ForegroundColor Green