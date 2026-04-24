#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

#
# MTDF Gravity - Run All Steps
#
# Reproduces all output artefacts (JSON, plots) for Steps 1-17, 19-22, 22b-d.
# Each step is independent and can be run individually.
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Data files required:
#   data/brouwer2021/      - Brouwer+2021 KiDS x GAMA ESD profiles
#   data/mandelbaum2016/   - Mandelbaum+2016 SDSS Red LBG ESD profiles
#   data/sparc/            - SPARC rotation curve data (Lelli+2016)
#   data/pittordis2023/    - Pittordis & Sutherland (2023) wide binaries (Zenodo 7629240)
#
# Usage:
#   bash run_all_steps.sh          # run all steps
#   bash run_all_steps.sh 12 15    # run only steps 12 and 15

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"

# Map step numbers to script filenames
declare -A SCRIPTS=(
    [1]="step1_ggl_comparison.py"
    [2]="step2_algebraic_target.py"
    [3]="step3_nonlinear_source.py"
    [4]="step4_shear_comparison.py"
    [5]="step5_density_inversion.py"
    [6]="step6_nonlinear_equation.py"
    [7]="step7_constitutive_ode.py"
    [8]="step8_compression_hypothesis.py"
    [9]="step9_4pi_alpha_test.py"
    [10]="step10_vref_closure.py"
    [11]="step11_jbar_identification.py"
    [12]="step12_delta_sigma_comparison.py"
    [13]="step13_solar_system_sanity.py"
    [14]="step14_constitutive_law.py"
    [15]="step15_cross_dataset_lensing.py"
    [16]="step16_robustness_suite.py"
    [17]="step17_baryon_completion.py"
    [19]="step19_relativistic_completion.py"
    [20]="step20_strong_lensing.py"
    [21]="step21_cluster_baryons.py"
    [22]="step22_elliptical_dispersions.py"
    [22b]="step22b_satellite_kinematics.py"
    [22c]="step22c_rotation_curves_rar.py"
    [22d]="step22d_wide_binaries.py"
)

# Determine which steps to run
if [ $# -eq 0 ]; then
    STEPS="$(seq 1 17) 19 20 21 22 22b 22c 22d"
else
    STEPS="$@"
fi

echo "=============================================="
echo "MTDF Gravity - Reproducibility Run"
echo "=============================================="
echo ""

FAILED=0
PASSED=0

for STEP in $STEPS; do
    SCRIPT="${SCRIPTS[$STEP]}"
    if [ -z "$SCRIPT" ]; then
        echo "WARNING: No script for step $STEP, skipping."
        continue
    fi

    SCRIPT_PATH="${CODE_DIR}/${SCRIPT}"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "ERROR: Script not found: $SCRIPT_PATH"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "--- Step $STEP: $SCRIPT ---"
    if python3 "$SCRIPT_PATH" > /dev/null 2>&1; then
        echo "  PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED (exit code $?)"
        echo "  Re-running with output for diagnostics:"
        python3 "$SCRIPT_PATH" 2>&1 | tail -20
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=============================================="
echo "Results: $PASSED passed, $FAILED failed"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
