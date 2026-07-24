#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V75
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
echo "  Download pass complete. Summary:"
echo "=============================================="
echo ""

EXT_DIR="$REPO_ROOT/validation/data/External"

report() {
    # report <label> <marker-file> [manual-hint]
    local label="$1"
    local marker="$2"
    local hint="$3"
    if [ -e "$marker" ]; then
        echo "  FETCHED   $label"
    else
        if [ -n "$hint" ]; then
            echo "  MISSING   $label  (MANUAL: $hint)"
        else
            echo "  MISSING   $label"
        fi
    fi
}

report "Pantheon+ SNe"            "$EXT_DIR/pantheonplus/Pantheon+SH0ES.dat"
report "Cosmic chronometers H(z)" "$EXT_DIR/hz_cc/HzTable_MM_BC03.dat"
report "Growth f-sigma8 (DR16)"   "$EXT_DIR/growth_fsig8/sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat"
report "BOSS DR12 voids"          "$EXT_DIR/boss_voids/table1.dat"
report "Pittordis+2023 binaries"  "$REPO_ROOT/gravity/data/pittordis2023/pittordis2023_wb.csv"
report "Planck PR4 lensing"       "$EXT_DIR/planck_lensing/PR4_variations/PR42018like_klm_dat_MV.fits"
report "Planck 2018 CMB distance prior" "$EXT_DIR/cmb_planck2018/planck2018_distance_means.txt"
report "ZTF DR2 cosmology sample" "$EXT_DIR/ztf_dr2/ztf_dr2_cosmology.csv"
report "Foundation DR1 SNe"       "$EXT_DIR/foundation_dr1/Foundation_DR1.FITRES.TEXT"
if [ "$SKIP_KIDS" = false ]; then
    report "KiDS-1000 WL catalogue" "$EXT_DIR/kids_1000/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
else
    echo "  SKIPPED   KiDS-1000 WL catalogue  (--skip-kids flag set)"
fi
report "DESI BAO (Y1 Gaussian)"   "$EXT_DIR/bao_desi/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt" \
       "provider does not allow scripted download; run 'bash scripts/download_bao.sh' for the exact filenames and instructions"
report "DESI VAST void catalogues" "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits" \
       "provider does not allow scripted download; see scripts/download_desi_voids.sh"

echo ""
echo "  NOTE: DESI BAO and the DESI VAST void catalogues always require manual"
echo "  placement (their providers do not allow scripted download). Without the"
echo "  DESI BAO files the combined strict chi2/nu = 1.17 headline cannot be"
echo "  regenerated; the shipped Validation_Dashboard_V75.html contains the"
echo "  full result for inspection."
echo ""
echo "  Run 'bash scripts/verify_checksums.sh' to verify integrity of the"
echo "  fetched datasets (coverage listed in the script)."
echo "=============================================="
