#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
# Verify SHA256 checksums for all downloaded external data.
# Run after download_data.sh to confirm data integrity.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXT_DIR="$REPO_ROOT/validation/data/External"
GRAV_DIR="$REPO_ROOT/gravity/data"

PASS=0
FAIL=0
SKIP=0

check_file() {
    local filepath="$1"
    local expected="$2"
    local label="$3"

    if [ ! -f "$filepath" ]; then
        echo "  SKIP  $label (not downloaded)"
        SKIP=$((SKIP + 1))
        return
    fi

    actual=$(sha256sum "$filepath" | cut -d' ' -f1)
    if [ "$actual" = "$expected" ]; then
        echo "  PASS  $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $label"
        echo "        Expected: $expected"
        echo "        Got:      $actual"
        FAIL=$((FAIL + 1))
    fi
}

echo "=============================================="
echo "  MTDF Data Integrity Verification"
echo "=============================================="
echo ""

echo "--- Pantheon+ ---"
check_file "$EXT_DIR/pantheonplus/Pantheon+SH0ES.dat" \
    "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8" \
    "Pantheon+SH0ES.dat"
check_file "$EXT_DIR/pantheonplus/Pantheon+SH0ES_STAT+SYS.cov" \
    "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc" \
    "Pantheon+SH0ES_STAT+SYS.cov"

echo ""
echo "--- DESI VAST Voids ---"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits" \
    "08d94b3e1740e154bde865b470c1f1a8d05d6e30ea4800fae2df2f5adec317b8" \
    "REVOLVER_NGC"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits" \
    "297e51d92b7955d81d68c674af78f43f2bc6c04247f7510c0224d1f81a984ad2" \
    "REVOLVER_SGC"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits" \
    "41b9073805b6ee11ab3855d0c21c255b7ba309186755e7d6e17c06eb1fac8ce3" \
    "VIDE_NGC"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits" \
    "9ae33edfe0c13ed6b0c189d2258edef4ec48171d27153b6b3ef8cd30c2ecb45c" \
    "VIDE_SGC"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits" \
    "c69f2f7b2b1fed4554527475dd96584169b1ead5bbcd0c152164200e6a2f34c8" \
    "VoidFinder_NGC"
check_file "$EXT_DIR/desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits" \
    "47c43b9b446f4bcb47cbc023115ac5297f9afb0045aea96d853649a80d7219c1" \
    "VoidFinder_SGC"

echo ""
echo "--- Planck PR4 Lensing ---"
check_file "$EXT_DIR/planck_lensing/PR4_variations/PR42018like_klm_dat_MV.fits" \
    "00edb4c2ec67f15396a6d0be896158f4bd20923000bdaab67ada7d4aa52f0f22" \
    "PR4 klm_dat_MV"
check_file "$EXT_DIR/planck_lensing/PR4_variations/mask.fits.gz" \
    "efa07353ff637e3b21c12b9b9a4cc3ae03d5c0b703b442fd5b263e304bb49fe6" \
    "PR4 mask"

echo ""
echo "--- KiDS-1000 ---"
check_file "$EXT_DIR/kids_1000/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits" \
    "dcc2bf039190d0c53542f6f08ac6ce27e749dd31999cd4ab049edd4d41fbef32" \
    "KiDS DR4.1 WL catalogue"

echo ""
echo "--- Pittordis+2023 ---"
check_file "$GRAV_DIR/pittordis2023/pittordis2023_wb.csv" \
    "826cbb2c73768515b883a4ff56588ab148592d5f5e11fb577045faba1d9bb8d9" \
    "Wide binary catalogue"

echo ""
echo "--- BOSS DR12 Voids ---"
check_file "$EXT_DIR/boss_voids/table1.dat" \
    "d579cfc1dc0bcb8886dab0645221c3788e8b784ba7bea1d2be5f096545971883" \
    "Mao+2017 table1.dat"

echo ""
echo "--- Cosmic Chronometers ---"
check_file "$EXT_DIR/hz_cc/HzTable_MM_BC03.dat" \
    "32ce92caf251cb60a7a837c71f1856bea2b44fa5c1041f85410d11cb8164da98" \
    "CC BC03"

echo ""
echo "--- Growth f*sigma8 ---"
check_file "$EXT_DIR/growth_fsig8/sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat" \
    "a098ea4df320ac1c18a9404237a75ae26953e16403a20294beb1d9573be33c56" \
    "eBOSS DR16 LRG"

echo ""
echo "--- Planck 2018 Distance Prior ---"
check_file "$EXT_DIR/cmb_planck2018/planck2018_distance_means.txt" \
    "9757a4b2b9663458de2c2b8e2d4fe54d0a16df836205845081649389655fe30a" \
    "Distance means"

echo ""
echo "=============================================="
echo "  Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "=============================================="

if [ $FAIL -gt 0 ]; then
    echo "  WARNING: Some checksums do not match!"
    exit 1
fi
