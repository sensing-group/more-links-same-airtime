#!/usr/bin/env bash
# The dropped apartment site (Sections 6 and 10): time of day alone predicts its labels at AUC 0.82,
# while no CSI detector does. CSI: the preprint's Zenodo record 10.5281/zenodo.18486250 (46 GB).
# Labels: data/apartment/labels/{confirmed_empty_periods.json,living_room_occupancy_labels.csv}.
# These two files record when a private home was occupied and are NOT published; the result this
# step produces is shipped as results/apartment/decision_window_ablation_w60s.json, and
# reproduce/compare.py treats the step as optional.
# The extraction streams one hour file at a time; it takes several hours.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/data}"
WORK="${WORK:-$ROOT/work}"
WORKERS="${WORKERS:-2}"
APT="$DATA/apartment"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for f in confirmed_empty_periods.json living_room_occupancy_labels.csv; do
  [ -s "$APT/labels/$f" ] || { echo "missing $APT/labels/$f"; exit 1; }
done
mkdir -p "$APT/csi_data" && cd "$APT"
for t in csi_week1_jan01-07.tar.gz csi_week2_jan08-13.tar.gz; do
  [ -s "$t" ] || curl -fL --retry 5 -C - -o "$t" "https://zenodo.org/records/18486250/files/$t?download=1"
  tar xzf "$t" -C csi_data
done
cd "$ROOT"
DATASET_DIR="$APT" OUT_DIR="$WORK/apartment/windows" BASE_RATE_HZ=0.25 DECISION_WINDOWS_S=60 \
  python3 scripts/apartment_budget_features.py --workers "$WORKERS"
DW_DIR="$WORK/apartment/windows" N_COMPLETE_LINKS=72 N_TRIALS=10 \
  OUT_PATH="$WORK/apartment/decision_window_ablation_w60s.json" \
  python3 scripts/decision_window_ablation.py --window w60s --empty-match none
echo "Apartment analysis written to $WORK/apartment"
