#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download Pantheon+ SN Ia data (Brout+2022, Scolnic+2022)
# Source: https://github.com/PantheonPlusSH0ES/DataRelease
# Size: ~33 MB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/pantheonplus"
mkdir -p "$TARGET_DIR"

BASE_URL="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"

echo "Downloading Pantheon+ data..."

# Data file
if [ ! -f "$TARGET_DIR/Pantheon+SH0ES.dat" ]; then
    wget -q --show-progress -O "$TARGET_DIR/Pantheon+SH0ES.dat" \
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat" \
        || { echo "Direct download failed. Trying git clone...";
             TMPDIR=$(mktemp -d)
             git clone --depth 1 https://github.com/PantheonPlusSH0ES/DataRelease.git "$TMPDIR/pp"
             cp "$TMPDIR/pp/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat" "$TARGET_DIR/"
             cp "$TMPDIR/pp/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov" "$TARGET_DIR/"
             rm -rf "$TMPDIR"
           }
else
    echo "  Pantheon+SH0ES.dat already exists, skipping."
fi

# Covariance matrix
if [ ! -f "$TARGET_DIR/Pantheon+SH0ES_STAT+SYS.cov" ]; then
    wget -q --show-progress -O "$TARGET_DIR/Pantheon+SH0ES_STAT+SYS.cov" \
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov" \
        || echo "WARNING: Direct download of covariance failed. May need manual download."
else
    echo "  Pantheon+SH0ES_STAT+SYS.cov already exists, skipping."
fi

echo "Pantheon+ download complete."
echo "  Files: $TARGET_DIR/"
echo "  Reference: Brout et al. (2022), Scolnic et al. (2022)"
echo "  DOI: 10.3847/1538-4357/ac8e04"
