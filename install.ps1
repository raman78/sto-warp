<#
.SYNOPSIS
Installs STO-WARP via pipx on Windows.

.DESCRIPTION
This script checks if Python and pipx are installed.
If not, it attempts to install them via winget.
Then it installs or upgrades sto-warp using pipx.
#>

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "   STO-WARP Installer (Windows)            " -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check for Python
$pythonInstalled = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $pythonInstalled) {
    Write-Host "Python not found. Attempting to install via winget..." -ForegroundColor Yellow
    $wingetInstalled = Get-Command "winget" -ErrorAction SilentlyContinue
    if (-not $wingetInstalled) {
        Write-Host "Error: winget is not installed. Please install Python manually from python.org" -ForegroundColor Red
        exit 1
    }
    # Install Python 3.14 (sto-warp requires >=3.14)
    winget install --id Python.Python.3.14 --source winget --accept-package-agreements --accept-source-agreements
    
    # Reload environment variables so Python is available in current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 2. Check for pipx
$pipxInstalled = Get-Command "pipx" -ErrorAction SilentlyContinue
if (-not $pipxInstalled) {
    Write-Host "pipx not found. Installing pipx..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install --user pipx
    python -m pipx ensurepath
    
    # Reload environment variables to include pipx path
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 3. Install or Upgrade sto-warp
Write-Host ""
Write-Host "Installing/Updating sto-warp via pipx..." -ForegroundColor Cyan

# We run it via cmd to avoid powershell stopping on non-terminating errors from pipx
cmd.exe /c "pipx install sto-warp || pipx upgrade sto-warp"

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "You can now run WARP by typing:"
Write-Host "    sto-warp" -ForegroundColor Yellow
Write-Host "If the command is not recognized, please restart your terminal."
Write-Host "===========================================" -ForegroundColor Cyan
