#!/usr/bin/env bash
# Runs every analysis of the paper in order, writing into work/.
# Expects the dataset under data/ (see reproduce/fetch_data.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/reproduce/trials.sh"
"$ROOT/reproduce/office.sh"
echo "All analyses written; compare with: python3 reproduce/compare.py"
