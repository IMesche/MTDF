#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download Foundation Supernova Survey DR1 (Foley+2018, Jones+2019)
# Source: https://github.com/djones1040/Foundation_DR1
# Reference: Jones et al. (2019, ApJ, 881, 19), arXiv:1811.09286
# Size: ~30 MB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/foundation_dr1"
mkdir -p "$TARGET_DIR"

FITRES_FILE="$TARGET_DIR/Foundation_DR1.FITRES.TEXT"
LC_DIR="$TARGET_DIR/Foundation_DR1"

if [ -f "$FITRES_FILE" ] && [ -d "$LC_DIR" ]; then
    echo "  Foundation DR1 already exists, skipping."
    echo "  FITRES: $FITRES_FILE"
    echo "  Light curves: $LC_DIR/"
else
    echo "Downloading Foundation DR1 from GitHub..."
    TMPDIR=$(mktemp -d)

    git clone --depth 1 https://github.com/djones1040/Foundation_DR1.git "$TMPDIR/fdn" || {
        echo "ERROR: Failed to clone Foundation DR1 repository."
        echo "Please clone manually:"
        echo "  git clone https://github.com/djones1040/Foundation_DR1.git"
        echo "  cp Foundation_DR1/Foundation_DR1.FITRES.TEXT $TARGET_DIR/"
        echo "  cp -r Foundation_DR1/Foundation_DR1/ $TARGET_DIR/"
        rm -rf "$TMPDIR"
        exit 1
    }

    # Copy FITRES file
    cp "$TMPDIR/fdn/Foundation_DR1.FITRES.TEXT" "$TARGET_DIR/"

    # Copy light curve directory
    cp -r "$TMPDIR/fdn/Foundation_DR1/" "$TARGET_DIR/"

    rm -rf "$TMPDIR"
    echo "  Foundation DR1 downloaded successfully."
fi

echo ""
echo "Foundation DR1 download complete."
echo "  FITRES: $FITRES_FILE"
echo "  Light curves: $LC_DIR/"
echo "  Reference: Foley et al. (2018, MNRAS 475, 193)"
echo "  Reference: Jones et al. (2019, ApJ 881, 19), arXiv:1811.09286"
