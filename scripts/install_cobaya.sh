#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Install cobaya and the Planck 2018 likelihood for MCMC reproduction.
# This is only needed for Phase 5 MCMC runs.
set -e

echo "=== Installing Cobaya + Planck Likelihood ==="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected."
    echo "It is recommended to activate a virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        exit 0
    fi
fi

# Install cobaya
echo "Installing cobaya..."
pip install cobaya

# Install Planck 2018 likelihood
echo ""
echo "Installing Planck 2018 plik likelihood..."
echo "This downloads ~2 GB of data from the Planck Legacy Archive."
echo ""
cobaya-install cosmo --packages-path cobaya_packages -f

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To run MCMC chains:"
echo "  cobaya-run mcmc_results/lcdm_mcmc.input.yaml"
echo "  cobaya-run mcmc_results/mtdf_mcmc.input.yaml"
echo ""
echo "Make sure class_mtdf is built first:"
echo "  cd class_mtdf && make clean && make -j4"
