#!/usr/bin/env bash
# Redraws every plotted figure and Tables 3 and 4 into figures/ (or FIG_DIR).
# By default from the shipped outputs in results/; ANALYSIS_DIR=work uses your own rerun.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export FIG_DIR="${FIG_DIR:-$ROOT/figures}"
export ANALYSIS_DIR="$(cd "$ROOT" && realpath "${ANALYSIS_DIR:-results}")"
mkdir -p "$ROOT/work"
# Numbers derived from other outputs (bootstrap intervals, cover cost, correlation crossing)
ANALYSIS_DIR="$ANALYSIS_DIR" OUT_PATH="$ROOT/work/derived_numbers.json" python3 "$ROOT/scripts/derived_numbers.py" > /dev/null
cd "$ROOT/paper"
for s in fig_*.py make_tab_*.py; do python3 "$s" > /dev/null; done
echo "Figures and tables written to $FIG_DIR from $ANALYSIS_DIR"
