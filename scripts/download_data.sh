#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Master download script for all external MTDF data.
# Downloads ~19 GB of external scientific data from public archives.
#
# Usage: bash scripts/download_data.sh [--skip-kids]
#   --skip-kids: Skip the 17 GB KiDS-1000 download (useful for quick setup)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SKIP_KIDS=false
for arg in "$@"; do
    case $arg in
        --skip-kids) SKIP_KIDS=true ;;
    esac
done

echo "=============================================="
echo "  MTDF External Data Download"
echo "=============================================="
echo ""
echo "Repository root: $REPO_ROOT"
echo ""

# Small datasets first
echo "--- Downloading Pantheon+ (33 MB) ---"
bash "$SCRIPT_DIR/download_pantheonplus.sh"
echo ""

echo "--- Downloading BAO + CC + growth + BOSS voids (~60 MB) ---"
bash "$SCRIPT_DIR/download_bao.sh"
echo ""

echo "--- Downloading Pittordis+2023 wide binaries (164 MB) ---"
bash "$SCRIPT_DIR/download_pittordis.sh"
echo ""

echo "--- Downloading Planck PR4 lensing (482 MB) ---"
bash "$SCRIPT_DIR/download_planck.sh"
echo ""

echo "--- Downloading DESI VAST voids (1.2 GB) ---"
bash "$SCRIPT_DIR/download_desi_voids.sh"
echo ""

echo "--- Downloading ZTF DR2 cosmology sample (~5 MB) ---"
bash "$SCRIPT_DIR/download_ztf_dr2.sh"
echo ""

echo "--- Downloading Foundation DR1 (~30 MB) ---"
bash "$SCRIPT_DIR/download_foundation_dr1.sh"
echo ""

if [ "$SKIP_KIDS" = false ]; then
    echo "--- Downloading KiDS-1000 WL catalogue (17 GB) ---"
    echo "    (Use --skip-kids to skip this large download)"
    bash "$SCRIPT_DIR/download_kids.sh"
    echo ""
else
    echo "--- Skipping KiDS-1000 (--skip-kids flag set) ---"
    echo ""
fi

echo "=============================================="
echo "  All downloads complete."
echo ""
echo "  Run 'bash scripts/verify_checksums.sh' to verify integrity."
echo "=============================================="
