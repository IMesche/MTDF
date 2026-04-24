#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# MTDF Repository Setup Script
# Creates a virtual environment and installs all Python dependencies.

set -e

echo "=== MTDF Environment Setup ==="
echo ""

# Check Python version
PYTHON=${PYTHON:-python3}
if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: $PYTHON not found. Please install Python 3.8+."
    exit 1
fi

PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYVER"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies from requirements.txt
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install gravity sector dependencies
echo ""
echo "Installing gravity sector dependencies..."
pip install matplotlib scipy

# Install GPU validation dependencies (optional)
echo ""
echo "Installing GPU validation dependencies (optional, some require CUDA)..."
pip install astropy h5py healpy || echo "WARNING: Some GPU validation dependencies failed (may need system packages)."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To reproduce the validation dashboard:"
echo "  cd validation/code && python run_validate.py --workbook ../data/DB_Workbook_STRICT_V18.xlsx --out ../output/My_Dashboard.html"
echo ""
echo "To download external data:"
echo "  bash scripts/download_data.sh"
echo ""
echo "For MCMC reproduction (requires cobaya + Planck likelihood):"
echo "  bash scripts/install_cobaya.sh"
