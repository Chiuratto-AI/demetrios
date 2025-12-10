# Demetrios One-Line Installer for Windows
# Usage: iwr -useb https://raw.githubusercontent.com/Chiuratto-AI/demetrios/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
Write-Host "Demetrios Installer" -ForegroundColor Cyan

# Create temp directory
$tmp = Join-Path $env:TEMP "demetrios-install"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null

# Clone or download
$hasGit = $null -ne (Get-Command git -EA 0)
if ($hasGit) {
    Write-Host "[+] Cloning repository..." -ForegroundColor Green
    git clone --depth 1 https://github.com/Chiuratto-AI/demetrios.git $tmp
} else {
    Write-Host "[+] Downloading..." -ForegroundColor Green
    $zip = Join-Path $tmp "demetrios.zip"
    Invoke-WebRequest -Uri "https://github.com/Chiuratto-AI/demetrios/archive/refs/heads/main.zip" -OutFile $zip
    Expand-Archive $zip $tmp -Force
    Move-Item (Join-Path $tmp "demetrios-main\*") $tmp -Force
}

# Run installer
$installer = Join-Path $tmp "install-windows.ps1"
if (Test-Path $installer) {
    & $installer
} else {
    Write-Host "[x] Installer not found" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item $tmp -Recurse -Force -EA 0