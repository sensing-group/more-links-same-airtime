#!/usr/bin/env bash
# Scripted-trial analyses (about 1 GB of data). Writes into work/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/data}"
WORK="${WORK:-$ROOT/work}"
WORKERS="${WORKERS:-2}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TRIALS_DIR="$DATA/trials"
mkdir -p "$WORK/decision_windows/unmatched_none" "$WORK/rate_decimated" "$WORK/features"
cd "$ROOT"

# Scripted trials (residence and laboratory): §5, §6, Figs. 2 and 6-9
# The script appends _presence, _tx or _hgb to OUT_PATH for its variants.
OUT_PATH="$WORK/scripted_trial_decision_windows.json" python3 scripts/scripted_trial_decision_windows.py
TASKS=presence OUT_PATH="$WORK/scripted_trial_decision_windows.json" python3 scripts/scripted_trial_decision_windows.py
OUT_PATH="$WORK/scripted_trial_decision_windows.json" python3 scripts/scripted_trial_decision_windows.py --budget-unit tx
OUT_PATH="$WORK/scripted_trial_decision_windows.json" python3 scripts/scripted_trial_decision_windows.py --classifier hgb
OUT_PATH="$WORK/sounding_correlation.json" python3 scripts/sounding_correlation_analysis.py
OUT_PATH="$WORK/radio_noise_floor.json" python3 scripts/radio_noise_floor_analysis.py
OUT_PATH="$WORK/trial_window_counts.json" python3 scripts/trial_window_counts.py
OUT_PATH="$WORK/trial_phase_order_check.json" python3 scripts/trial_phase_order_check.py
LR_OCCUPANCY_PATH="$WORK/scripted_trial_decision_windows.json" LR_PRESENCE_PATH="$WORK/scripted_trial_decision_windows_presence.json" OUT_PATH="$WORK/neural_budget_check.json" python3 scripts/neural_budget_check.py
LR_OCCUPANCY_PATH="$WORK/scripted_trial_decision_windows.json" LR_PRESENCE_PATH="$WORK/scripted_trial_decision_windows_presence.json" INPUT_MODE=dynamic OUT_PATH="$WORK/neural_budget_check_dynamic.json" python3 scripts/neural_budget_check.py
OUT_PATH="$WORK/calibration_reference.json" python3 scripts/calibration_reference_analysis.py
IN_PATH="$WORK/calibration_reference.json" OUT_PATH="$WORK/calibration_model.json" python3 scripts/calibration_model.py
OUT_PATH="$WORK/trial_sounding_rates.json" python3 scripts/trial_sounding_rates.py
echo "Trial analyses written to $WORK"
