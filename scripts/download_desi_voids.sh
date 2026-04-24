#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download DESIVAST BGS void catalogues (Douglass+2024)
# Source: DESI data release / VAST void catalogues
# Reference: Douglass et al. (2024, ApJS 275, 38)
# Size: ~1.2 GB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/desivast_voids"
mkdir -p "$TARGET_DIR"

echo "Downloading DESIVAST BGS void catalogues..."
echo ""
echo "NOTE: The DESIVAST catalogues are distributed via the DESI data release."
echo "If automatic download fails, please download manually from:"
echo "  https://data.desi.lbl.gov/public/"
echo "  or https://github.com/DESI-UR/VAST"
echo ""

# Expected files (DESIVAST BGS VOLLIM V2)
FILES=(
    "DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits"
    "DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits"
    "DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits"
    "DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits"
    "DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits"
    "DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits"
)

MISSING=0
for f in "${FILES[@]}"; do
    if [ ! -f "$TARGET_DIR/$f" ]; then
        echo "  MISSING: $f"
        MISSING=$((MISSING + 1))
    else
        echo "  OK: $f"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  $MISSING files missing."
    echo ""
    echo "  To download, visit the DESI data portal or VAST GitHub:"
    echo "    https://github.com/DESI-UR/VAST"
    echo "    https://data.desi.lbl.gov/public/"
    echo ""
    echo "  Place the .fits files in:"
    echo "    $TARGET_DIR/"
    echo ""
    echo "  Expected checksums:"
    echo "    REVOLVER_NGC: 08d94b3e1740e154bde865b470c1f1a8d05d6e30ea4800fae2df2f5adec317b8"
    echo "    REVOLVER_SGC: 297e51d92b7955d81d68c674af78f43f2bc6c04247f7510c0224d1f81a984ad2"
    echo "    VIDE_NGC:     41b9073805b6ee11ab3855d0c21c255b7ba309186755e7d6e17c06eb1fac8ce3"
    echo "    VIDE_SGC:     9ae33edfe0c13ed6b0c189d2258edef4ec48171d27153b6b3ef8cd30c2ecb45c"
    echo "    VoidFinder_NGC: c69f2f7b2b1fed4554527475dd96584169b1ead5bbcd0c152164200e6a2f34c8"
    echo "    VoidFinder_SGC: 47c43b9b446f4bcb47cbc023115ac5297f9afb0045aea96d853649a80d7219c1"
else
    echo ""
    echo "  All DESIVAST files present."
fi

echo ""
echo "Reference: Douglass et al. (2024, ApJS 275, 38)"
echo "DOI: 10.3847/1538-4365/ad85de"
