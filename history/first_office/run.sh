#!/usr/bin/env bash
# Reproduces the first office analysis quoted in Section 8 (20 sessions, time of day alone predicts
# occupancy at AUC 0.61). It used the analysis code of that time (scripts/, without time-of-day
# matching) and two sessions that were still incomplete then; their window files are in
# partial_windows/. The other 18 window files are identical to those office.sh writes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-$ROOT/work}"
DW="$WORK/history/first_office"
# 5 s windows choose the 16 pairs, 20 s windows are scored
for W in w5s w20s; do
  mkdir -p "$DW/$W"
  while read -r f; do
    if [ -f "$HERE/partial_windows/$W/$f" ]; then cp "$HERE/partial_windows/$W/$f" "$DW/$W/$f"
    else cp "$WORK/decision_windows/$W/$f" "$DW/$W/$f"; fi
  done < "$HERE/sessions.txt"
done
PYTHONPATH="$ROOT" DW_DIR="$DW" OUT_PATH="$WORK/history/first_office_w20s.json" \
  python3 "$HERE/scripts/decision_window_ablation.py" --window w20s
