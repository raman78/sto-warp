#!/bin/sh
set -e

echo "==========================================="
echo "   STO-WARP Installer (Linux / macOS)      "
echo "==========================================="
echo ""

# Check if Python is installed
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed. Please install Python 3.14 or newer."
    exit 1
fi

# Check if pipx is installed, if not, try to install it
if ! command -v pipx >/dev/null 2>&1; then
    echo "pipx is not installed. Attempting to install pipx..."
    
    if command -v apt >/dev/null 2>&1; then
        echo "Detected Debian/Ubuntu (apt). Running: sudo apt update && sudo apt install -y pipx"
        sudo apt update && sudo apt install -y pipx
    elif command -v dnf >/dev/null 2>&1; then
        echo "Detected Fedora/RHEL (dnf). Running: sudo dnf install -y pipx"
        sudo dnf install -y pipx
    elif command -v pacman >/dev/null 2>&1; then
        echo "Detected Arch Linux (pacman). Running: sudo pacman -Sy --noconfirm python-pipx"
        sudo pacman -Sy --noconfirm python-pipx
    elif command -v brew >/dev/null 2>&1; then
        echo "Detected macOS/Homebrew (brew). Running: brew install pipx"
        brew install pipx
    else
        echo "Error: Could not determine package manager to install pipx."
        echo "Please install pipx manually: https://pipx.pypa.io/stable/installation/"
        exit 1
    fi
    
    # Ensure pipx path is set for the current shell session
    pipx ensurepath
    export PATH="$PATH:$HOME/.local/bin"
fi

echo ""
echo "Installing/Updating sto-warp via pipx..."
if pipx list | grep -q "sto-warp"; then
    pipx upgrade sto-warp
else
    pipx install sto-warp
fi

echo ""
echo "==========================================="
echo "Installation complete!"
echo "You can now run WARP by typing:"
echo "    sto-warp"
echo "==========================================="
