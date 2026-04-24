#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download Planck PR4 CMB lensing convergence maps (Carron+2022)
# Source: https://github.com/carronj/planck_PR4_lensing
# Size: ~482 MB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/planck_lensing/PR4_variations"
mkdir -p "$TARGET_DIR"

echo "Downloading Planck PR4 lensing maps..."

# Clone the PR4 lensing repository (contains the FITS files)
if [ ! -f "$TARGET_DIR/PR42018like_klm_dat_MV.fits" ]; then
    TMPDIR=$(mktemp -d)
    echo "  Cloning carronj/planck_PR4_lensing..."
    git clone --depth 1 https://github.com/carronj/planck_PR4_lensing.git "$TMPDIR/pr4" 2>&1 | tail -3

    # Copy the relevant files
    echo "  Copying MV/PP/TT data and mean-field alms..."
    for tag in dat mf; do
        for est in MV PP TT; do
            SRC="$TMPDIR/pr4/PR42018like_klm_${tag}_${est}.fits"
            if [ -f "$SRC" ]; then
                cp "$SRC" "$TARGET_DIR/"
            else
                echo "  WARNING: $SRC not found in repo"
            fi
        done
    done

    # Copy mask
    if [ -f "$TMPDIR/pr4/mask.fits.gz" ]; then
        cp "$TMPDIR/pr4/mask.fits.gz" "$TARGET_DIR/"
    fi

    rm -rf "$TMPDIR"
else
    echo "  PR4 files already exist, skipping."
fi

# Also download the Planck 2018 CMB distance prior (small, used by Phase 1)
CMB_DIR="$REPO_ROOT/validation/data/External/cmb_planck2018"
mkdir -p "$CMB_DIR"

if [ ! -f "$CMB_DIR/planck2018_distance_means.txt" ]; then
    echo "  Creating Planck 2018 distance prior (from arXiv:1807.06209 Table 1)..."
    # These are the compressed CMB distance priors from Planck 2018 TT,TE,EE+lowE+lensing
    cat > "$CMB_DIR/planck2018_distance_means.txt" << 'ENDDATA'
# Planck 2018 compressed CMB distance prior (TT,TE,EE+lowE+lensing)
# Source: Planck Collaboration VI (2020), arXiv:1807.06209, Table 1
# Columns: R (shift parameter), lA (acoustic scale), omega_b*h^2
1.7502  301.471  0.02236
ENDDATA

    cat > "$CMB_DIR/planck2018_distance_cov.txt" << 'ENDDATA'
# 3x3 covariance matrix for (R, lA, omega_b*h^2)
# Source: Planck Collaboration VI (2020), arXiv:1807.06209
 2.4774e-04  1.0899e-02  5.7882e-08
 1.0899e-02  8.7101e-01  3.8042e-06
 5.7882e-08  3.8042e-06  2.2500e-10
ENDDATA
else
    echo "  Planck 2018 distance prior already exists, skipping."
fi

echo "Planck download complete."
echo "  Lensing: $TARGET_DIR/"
echo "  Distance prior: $CMB_DIR/"
echo "  Reference: Carron et al. (2022), Planck Collaboration (2020)"
