#!/usr/bin/env bash
# Runs every analysis of the paper in order, writing into work/.
# Expects the dataset under data/ (data/office/sessions, data/office/features, data/trials).
# The office extractions take hours; each step can be run on its own.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${DATA:-$ROOT/data}"
WORK="${WORK:-$ROOT/work}"
WORKERS="${WORKERS:-2}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TRIALS_DIR="$DATA/trials"
mkdir -p "$WORK/decision_windows/unmatched_none" "$WORK/rate_decimated" "$WORK/features"
cd "$ROOT"

# Scripted trials (residence and laboratory): §5, §6, Figs. 2 and 6-9
OUT_PATH="$WORK/scripted_trial_decision_windows.json" python3 scripts/scripted_trial_decision_windows.py
TASKS=presence OUT_PATH="$WORK/scripted_trial_decision_windows_presence.json" python3 scripts/scripted_trial_decision_windows.py
OUT_PATH="$WORK/scripted_trial_decision_windows_tx.json" python3 scripts/scripted_trial_decision_windows.py --budget-unit tx
OUT_PATH="$WORK/scripted_trial_decision_windows_hgb.json" python3 scripts/scripted_trial_decision_windows.py --classifier hgb
OUT_PATH="$WORK/sounding_correlation.json" python3 scripts/sounding_correlation_analysis.py
OUT_PATH="$WORK/radio_noise_floor.json" python3 scripts/radio_noise_floor_analysis.py
OUT_PATH="$WORK/trial_window_counts.json" python3 scripts/trial_window_counts.py
OUT_PATH="$WORK/trial_phase_order_check.json" python3 scripts/trial_phase_order_check.py
LR_OCCUPANCY_PATH="$WORK/scripted_trial_decision_windows.json" LR_PRESENCE_PATH="$WORK/scripted_trial_decision_windows_presence.json" OUT_PATH="$WORK/neural_budget_check.json" python3 scripts/neural_budget_check.py
LR_OCCUPANCY_PATH="$WORK/scripted_trial_decision_windows.json" LR_PRESENCE_PATH="$WORK/scripted_trial_decision_windows_presence.json" INPUT_MODE=dynamic OUT_PATH="$WORK/neural_budget_check_dynamic.json" python3 scripts/neural_budget_check.py
OUT_PATH="$WORK/calibration_reference.json" python3 scripts/calibration_reference_analysis.py
IN_PATH="$WORK/calibration_reference.json" OUT_PATH="$WORK/calibration_model.json" python3 scripts/calibration_model.py

# Office, decision windows on a fixed budget: Table 3, Figs. 2, 5 and 6
DATASET_DIR="$DATA/office" OUT_DIR="$WORK/decision_windows" BASE_RATE_HZ=1.0 RATE_FRACTIONS=1.0,0.5,0.25,0.125 \
  DECISION_WINDOWS_S=5,10,20,30,60 LINK_WHITELIST="$ROOT/results/decision_window_link_whitelist.json" \
  python3 scripts/extract_rate_decimated_features.py --workers "$WORKERS"
for W in 5 10 20 30 60; do
  DW_DIR="$WORK/decision_windows" python3 scripts/decision_window_ablation.py --window "w${W}s"
  # the same without time-of-day matching: the flawed column of Table 4's last row
  DW_DIR="$WORK/decision_windows" OUT_PATH="$WORK/decision_windows/unmatched_none/decision_window_ablation_w${W}s.json" \
    python3 scripts/decision_window_ablation.py --window "w${W}s" --empty-match none
done

# Office, 60 s windows: schedule leak (§7, Fig. 10) and practices (§8, Table 4)
DATASET_DIR="$DATA/office" OUT_DIR="$WORK/rate_decimated" python3 scripts/extract_rate_decimated_features.py --workers "$WORKERS"
IN_DIR="$WORK/rate_decimated" python3 scripts/rate_decimation_ablation.py
IN_DIR="$WORK/rate_decimated" OUT_PATH="$WORK/schedule_leak.json" python3 scripts/schedule_leak_analysis.py

# Office, per-session features of all 27 sessions: threshold practices (§8, Fig. 11)
DATASET_DIR="$DATA/office" OUT_DIR="$WORK/features" python3 scripts/extract_dataset_release_features.py
EWSN_DIR="$WORK/features" OUT_PATH="$WORK/threshold_practices.json" python3 scripts/threshold_practices_analysis.py
IN_DIR="$WORK/rate_decimated" EWSN_DIR="$WORK/features" OUT_PATH="$WORK/office_session_stats.json" python3 scripts/office_session_stats.py

echo "Analyses written to $WORK; point ANALYSIS_DIR=$WORK at the paper/ scripts to redraw the figures."
