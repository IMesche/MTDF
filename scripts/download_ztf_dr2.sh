#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download ZTF DR2 cosmology sample (Rigault+2024, Amenouche+2024)
# Source: ztfcosmo Python package (https://github.com/ZwickyTransientFacility/ztfcosmo)
# Reference: Rigault et al. (2024), arXiv:2409.04346
# Size: ~5 MB (CSV export from Python package)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$REPO_ROOT/validation/data/External/ztf_dr2"
mkdir -p "$TARGET_DIR"

OUTPUT_FILE="$TARGET_DIR/ztf_dr2_cosmology.csv"

if [ -f "$OUTPUT_FILE" ]; then
    echo "  ZTF DR2 cosmology sample already exists, skipping."
    echo "  File: $OUTPUT_FILE"
else
    echo "Downloading ZTF DR2 cosmology sample via ztfcosmo package..."
    echo ""
    echo "This requires the ztfcosmo Python package."
    echo "Installing if not present..."

    pip install ztfcosmo 2>/dev/null || pip install --user ztfcosmo 2>/dev/null || {
        echo "ERROR: Could not install ztfcosmo. Please install manually:"
        echo "  pip install ztfcosmo"
        exit 1
    }

    python3 -c "
import ztfcosmo
import pandas as pd

print('Loading ZTF DR2 cosmology sample...')
sample = ztfcosmo.get_cosmological_sample()
df = sample.data if hasattr(sample, 'data') else pd.DataFrame(sample)
df.to_csv('$OUTPUT_FILE', index=False)
print(f'Saved {len(df)} SNe to $OUTPUT_FILE')
" || {
        echo ""
        echo "ERROR: Failed to export ZTF DR2 data."
        echo "Please run manually in Python:"
        echo "  import ztfcosmo"
        echo "  sample = ztfcosmo.get_cosmological_sample()"
        echo "  sample.data.to_csv('$OUTPUT_FILE', index=False)"
        exit 1
    }
fi

echo ""
echo "ZTF DR2 download complete."
echo "  File: $OUTPUT_FILE"
echo "  Reference: Rigault et al. (2024), arXiv:2409.04346"
echo "  Reference: Amenouche et al. (2024), arXiv:2409.04344"
