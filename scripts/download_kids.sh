#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download KiDS-1000 (DR4) weak lensing shape catalogue
# Source: https://kids.strw.leidenuniv.nl/DR4/
# Size: ~17 GB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/kids_1000"
mkdir -p "$TARGET_DIR"

TARGET_FILE="$TARGET_DIR/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
URL="https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"

echo "Downloading KiDS-1000 DR4 weak lensing catalogue..."
echo "  URL: $URL"
echo "  Size: ~17 GB"
echo "  This may take a while depending on your connection."
echo ""

if [ -f "$TARGET_FILE" ]; then
    echo "  File already exists. Checking size..."
    SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || stat -f%z "$TARGET_FILE" 2>/dev/null)
    if [ "$SIZE" -gt 17000000000 ]; then
        echo "  Size OK ($SIZE bytes). Skipping download."
        echo "  SHA256: dcc2bf039190d0c53542f6f08ac6ce27e749dd31999cd4ab049edd4d41fbef32"
        exit 0
    else
        echo "  File appears incomplete ($SIZE bytes). Re-downloading..."
    fi
fi

wget -c --show-progress -O "$TARGET_FILE" "$URL"

echo ""
echo "KiDS-1000 download complete."
echo "  File: $TARGET_FILE"
echo "  Expected SHA256: dcc2bf039190d0c53542f6f08ac6ce27e749dd31999cd4ab049edd4d41fbef32"
echo "  Reference: KiDS DR4 (Kuijken+2019), science data: https://kids.strw.leidenuniv.nl/sciencedata.php"
