@echo off
:: sto_warp.bat — Windows entry point for sto-warp.
::
:: First run: creates a local .venv next to the script and installs the
:: project in editable mode. Subsequent runs reuse the venv and just
:: launch the sto-warp console script.

setlocal
cd /d "%~dp0"
if errorlevel 1 (
    echo [sto-warp] Cannot change to script directory: %~dp0
    exit /b 1
)

set "VENV_DIR=%~dp0.venv"
set "VENV_SCRIPTS=%VENV_DIR%\Scripts"
set "STO_WARP=%VENV_SCRIPTS%\sto-warp.exe"

if exist "%STO_WARP%" (
    "%STO_WARP%" %*
    exit /b %errorlevel%
)

:: --- First run: locate a Python 3.14+ launcher ---
set "PYTHON="

py -3.14 --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON=py -3.14" & goto :found )

py -3.15 --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON=py -3.15" & goto :found )

py --version >nul 2>&1
if not errorlevel 1 (
    py -c "import sys; sys.exit(0 if sys.version_info >= (3,14) else 1)" >nul 2>&1
    if not errorlevel 1 ( set "PYTHON=py" & goto :found )
)

python --version >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,14) else 1)" >nul 2>&1
    if not errorlevel 1 ( set "PYTHON=python" & goto :found )
)

echo [sto-warp] No Python 3.14+ found on PATH.
echo            Install it from https://www.python.org/downloads/ and re-run.
pause
exit /b 1

:found
echo [sto-warp] First run -- creating venv in %VENV_DIR% (one-time, takes a few minutes)...
%PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [sto-warp] venv creation failed
    pause
    exit /b 1
)

"%VENV_SCRIPTS%\python.exe" -m pip install --upgrade pip wheel >nul
"%VENV_SCRIPTS%\python.exe" -m pip install -e "%~dp0"
if errorlevel 1 (
    echo [sto-warp] pip install failed -- see output above
    pause
    exit /b 1
)

if not exist "%STO_WARP%" (
    echo [sto-warp] install completed but %STO_WARP% is missing -- aborting.
    pause
    exit /b 1
)

echo [sto-warp] Setup done. Launching...
"%STO_WARP%" %*
exit /b %errorlevel%
