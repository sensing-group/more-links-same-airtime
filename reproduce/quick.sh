#!/usr/bin/env bash
# Short path (about 3.5 hours on one CPU core, see ARTIFACT.md): redraw every figure and table from
# the shipped outputs, rerun all scripted-trial analyses from the 0.9 GB of trial data, and compare
# them with the shipped outputs. Needs data/trials (reproduce/fetch_data.sh, or trials.tar alone).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
./reproduce/figures.sh
./reproduce/trials.sh
python3 reproduce/compare.py --only scripted_trial sounding_correlation radio_noise_floor trial_ calibration_ neural_budget derived_numbers "$@"
