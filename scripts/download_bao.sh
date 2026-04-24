#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Download BAO, cosmic chronometer, growth rate, and BOSS void data
# Multiple sources, relatively small files (~60 MB total)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXT_DIR="$REPO_ROOT/validation/data/External"

# -------------------------------------------------------
# safe_fetch: wget + reject empty / HTML error pages
#   usage: safe_fetch <url> <target-path>
#   returns 0 on a valid non-empty, non-HTML response;
#   removes the target on any failure so callers can retry.
# -------------------------------------------------------
safe_fetch() {
    local url="$1"
    local out="$2"
    rm -f "$out"
    wget -q -O "$out" "$url" 2>/dev/null || { rm -f "$out"; return 1; }
    if [ ! -s "$out" ]; then
        rm -f "$out"
        return 1
    fi
    # Crude HTML-error-page detector: many servers (GitHub, Zenodo,
    # GitLab, CDS) return a 200-looking HTML error doc for missing paths.
    if head -c 512 "$out" | grep -qiE '<html|<!doctype html|<title>'; then
        rm -f "$out"
        return 1
    fi
    return 0
}

echo "Downloading BAO + CC + growth + BOSS void data..."

# -------------------------------------------------------
# 1. DESI Y1 BAO (Gaussian approximation)
# Reference: DESI Collaboration (2024), arXiv:2404.03000
# -------------------------------------------------------
BAO_DIR="$EXT_DIR/bao_desi"
mkdir -p "$BAO_DIR"

if [ ! -f "$BAO_DIR/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt" ]; then
    echo "  Downloading DESI Y1 BAO data..."
    echo "  NOTE: If automatic download fails, get files from:"
    echo "    https://github.com/cosmodesi/desilike"
    echo "    or the DESI data release: https://data.desi.lbl.gov/public/"
    echo ""
    echo "  Place these files in: $BAO_DIR/"
    echo "    - desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
    echo "    - desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
else
    echo "  DESI BAO data already exists."
fi

# -------------------------------------------------------
# 2. Cosmic Chronometers H(z) (Moresco et al.)
# Reference: Moresco (2022), Living Reviews in Relativity
# Primary source (verified 2026-04-24): Michele Moresco's GitLab repository
#   https://gitlab.com/mmoresco/CCcovariance
# Historical source (now 404): github.com/music-group/CC_covariance was the
# original distribution point; that repository has been deleted or moved.
# The GitLab repository above is the current canonical public source and
# ships the same data files under data/.
# -------------------------------------------------------
CC_DIR="$EXT_DIR/hz_cc"
mkdir -p "$CC_DIR"

CC_BASE="https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data"
for fname in HzTable_MM_BC03.dat HzTable_MM_M11.dat data_MM20.dat; do
    target="$CC_DIR/$fname"
    if [ -s "$target" ]; then
        echo "  CC: $fname already present."
    else
        echo "  CC: fetching $fname ..."
        if safe_fetch "$CC_BASE/$fname" "$target"; then
            echo "  CC: $fname OK ($(wc -c < "$target") bytes)."
        else
            echo "  WARNING: CC $fname download failed or returned an error page."
        fi
    fi
done

# -------------------------------------------------------
# 3. Growth rate f*sigma8 (SDSS DR16 / eBOSS)
# Reference: eBOSS DR16 (arXiv:2007.08991)
# Primary source (verified 2026-04-24): CobayaSampler/bao_data on GitHub
#   https://github.com/CobayaSampler/bao_data
# Historical source: cmbant/CosmoMC used to expose these paths directly; the
# CobayaSampler/bao_data repository now mirrors the eBOSS DR16 BAOplus files
# originally distributed with CosmoMC and is the current canonical public
# source for this analysis.
# -------------------------------------------------------
GROWTH_DIR="$EXT_DIR/growth_fsig8"
mkdir -p "$GROWTH_DIR"

GROWTH_BASE="https://raw.githubusercontent.com/CobayaSampler/bao_data/master"
for fname in \
    sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat \
    sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt \
    sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat \
    sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt \
; do
    target="$GROWTH_DIR/$fname"
    if [ -s "$target" ]; then
        echo "  Growth: $fname already present."
    else
        echo "  Growth: fetching $fname ..."
        if safe_fetch "$GROWTH_BASE/$fname" "$target"; then
            echo "  Growth: $fname OK ($(wc -c < "$target") bytes)."
        else
            echo "  WARNING: Growth $fname download failed or returned an error page."
        fi
    fi
done

# -------------------------------------------------------
# 4. BOSS DR12 void catalogue (Mao+2017)
# Reference: Mao et al. (2017, ApJ 835, 161)
# Source: VizieR J/ApJ/835/161
# -------------------------------------------------------
BOSS_DIR="$EXT_DIR/boss_voids"
mkdir -p "$BOSS_DIR"

BOSS_TARGET="$BOSS_DIR/table1.dat"
if [ -s "$BOSS_TARGET" ]; then
    echo "  BOSS void data already exists."
else
    echo "  Downloading BOSS DR12 void catalogue (Mao+2017)..."
    # Primary source is CDS/VizieR (verified reachable 2026-04-24).
    # The Vanderbilt mirror (lss.phy.vanderbilt.edu) was unreachable at
    # verification time; the fallback is retained in case it returns.
    CDS_URL="https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?J/ApJ/835/161/table1.dat"
    VAND_URL="http://lss.phy.vanderbilt.edu/voids/ZOBOV_voids_BOSS_DR12/table1.dat"
    if safe_fetch "$CDS_URL" "$BOSS_TARGET"; then
        echo "  BOSS void data OK via CDS ($(wc -c < "$BOSS_TARGET") bytes)."
    elif safe_fetch "$VAND_URL" "$BOSS_TARGET"; then
        echo "  BOSS void data OK via Vanderbilt ($(wc -c < "$BOSS_TARGET") bytes)."
    else
        echo "  WARNING: BOSS void download failed. Get data from:"
        echo "    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/835/161"
        echo "    or http://lss.phy.vanderbilt.edu/voids/  (unreachable 2026-04-24)"
    fi
fi

# -------------------------------------------------------
# 5. Planck plik-lite binned data (Phase 2 GPU)
# Source: Planck Legacy Archive
# -------------------------------------------------------
PLIK_DIR="$REPO_ROOT/gpu_validation/phase2/data/planck"
if [ ! -d "$PLIK_DIR" ] || [ ! -f "$PLIK_DIR/cl_cmb_plik_v22.dat" ]; then
    echo "  NOTE: Planck plik-lite data should be in gpu_validation/phase2/data/planck/"
    echo "  If missing, download from: https://pla.esac.esa.int/pla/#cosmology"
fi

# -------------------------------------------------------
# 6. GAMA group catalogue (Phase 6)
# Source: http://www.gama-survey.org/dr3/
# -------------------------------------------------------
GAMA_DIR="$EXT_DIR/gama"
mkdir -p "$GAMA_DIR"
if [ ! -f "$GAMA_DIR/G3CGalv10.fits" ]; then
    echo ""
    echo "  NOTE: GAMA G3C catalogue not found."
    echo "  Download from: http://www.gama-survey.org/dr3/"
    echo "  Place G3CGalv10.fits and G3CFoFGroupv10.fits in: $GAMA_DIR/"
fi

echo ""
echo "BAO + CC + growth + BOSS void downloads complete."
