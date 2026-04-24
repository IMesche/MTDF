#!/bin/bash
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

# Launch/resume MCMC with crash protection
# Usage: ./launch_mcmc.sh <lcdm|mtdf> [--resume]
#
# Enables: faulthandler (Python backtrace on segfault),
#          ulimit -c unlimited (core dumps if pattern allows),
#          stderr capture for post-mortem analysis.

set -euo pipefail

MODEL="${1:?Usage: launch_mcmc.sh <lcdm|mtdf> [--resume]}"
RESUME="${2:-}"

RESULTS_DIR="../../mcmc_results"
LOG_DIR="${RESULTS_DIR}/crash_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STDERR_LOG="${LOG_DIR}/${MODEL}_mcmc_${TIMESTAMP}.stderr"

# Enable core dumps
ulimit -c unlimited 2>/dev/null || true

# Activate venv
source ../../venv/bin/activate

# Enable Python faulthandler via env var (backup to code-level enable)
export PYTHONFAULTHANDLER=1

# Build command
CMD="python -u -m mtdf_validation.phase5_plik.run_phase5 --stage mcmc --model ${MODEL}"
if [ "${RESUME}" = "--resume" ]; then
    CMD="${CMD} --resume"
fi

echo "========================================"
echo "  Launching ${MODEL} MCMC"
echo "  Resume: ${RESUME:-no}"
echo "  Stderr log: ${STDERR_LOG}"
echo "  Core limit: $(ulimit -c)"
echo "  Checkpoint: ${RESULTS_DIR}/${MODEL}_mcmc.checkpoint"
echo "  PID: $$"
echo "  Time: $(date)"
echo "========================================"

# Run with stderr captured separately for crash analysis
${CMD} 2> >(tee "${STDERR_LOG}" >&2)

EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "  CRASH DETECTED: exit code ${EXIT_CODE}"
    echo "  Stderr saved to: ${STDERR_LOG}"
    echo "  Checkpoint: ${RESULTS_DIR}/${MODEL}_mcmc.checkpoint"
    echo "  Time: $(date)"
    echo "========================================"
fi

exit ${EXIT_CODE}
