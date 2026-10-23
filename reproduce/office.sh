#!/usr/bin/env bash
# Office analyses (19 GB of sessions; several hours). Writes into work/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/data}"
WORK="${WORK:-$ROOT/work}"
WORKERS="${WORKERS:-2}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TRIALS_DIR="$DATA/trials"
mkdir -p "$WORK/decision_windows/unmatched_none" "$WORK/rate_decimated" "$WORK/features" "$WORK/history"
cd "$ROOT"

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

# Earlier analyses quoted in Section 8: the earlier version's threshold detector (J 0.55 best pair,
# 0.01 all pairs) and the first 20-session office analysis (time of day alone at AUC 0.61)
EWSN_DIR="$DATA/office/features" OUT_PATH="$WORK/history/dilution_analysis_results.json" python3 history/dilution_analysis.py
WORK="$WORK" history/first_office/run.sh
echo "Office analyses written to $WORK"
