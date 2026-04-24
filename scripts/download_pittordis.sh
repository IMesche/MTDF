#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download Pittordis & Sutherland (2023) wide binary catalogue
# Source: https://zenodo.org/records/7629240
# Size: ~164 MB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/gravity/data/pittordis2023"
mkdir -p "$TARGET_DIR"

TARGET_FILE="$TARGET_DIR/pittordis2023_wb.csv"
# Upstream filename on Zenodo was renamed from "pittordis2023_wb.csv" to the
# descriptive form below (confirmed via Zenodo API 2026-04-24). The local
# target filename is preserved for downstream-pipeline compatibility.
ZENODO_FILENAME="CleanedWB_EDR3_Prlx300pc_Gmag20_20230111_Size73087_ZenodoSample.csv"
ZENODO_URL="https://zenodo.org/records/7629240/files/${ZENODO_FILENAME}"

echo "Downloading Pittordis+2023 wide binary data..."

if [ -f "$TARGET_FILE" ]; then
    echo "  File already exists, skipping."
else
    echo "  Trying Zenodo direct download..."
    wget -q --show-progress -O "$TARGET_FILE" "$ZENODO_URL" 2>/dev/null \
        || {
            echo "  Direct download failed."
            echo ""
            echo "  Please download manually from:"
            echo "    https://zenodo.org/records/7629240"
            echo "  and save the file"
            echo "    ${ZENODO_FILENAME}"
            echo "  as:"
            echo "    $TARGET_FILE"
            echo ""
            echo "  Alternatively, search CDS/VizieR for:"
            echo "    Pittordis & Sutherland 2023 (2023MNRAS.527.4573P)"
            echo "    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/527/4573"
        }
fi

echo ""
# The upstream filename on Zenodo changed (2026-04-24 verification) but the
# file contents are byte-identical to the prior release; the SHA-256 below
# matches both the old and new filenames on record 7629240.
echo "  Expected SHA256: 826cbb2c73768515b883a4ff56588ab148592d5f5e11fb577045faba1d9bb8d9"
echo "  Reference: Pittordis & Sutherland (2023, MNRAS 527, 4573)"
echo "  DOI: 10.1093/mnras/stad3474"
echo "  arXiv: 2205.02846"
