#!/usr/bin/env bash
# Downloads the dataset (21 GB) from Zenodo into data/, checks it and unpacks it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/data}"
RECORD="${RECORD:-23574119}"
FILES="SHA256SUMS DATASHEET.md LICENSE trials.tar office-features-labels.tar office-sessions-1.tar office-sessions-2.tar office-sessions-3.tar"
mkdir -p "$DATA" && cd "$DATA"
for f in $FILES; do
  [ -s "$f" ] || curl -fL --retry 5 -C - -o "$f" "https://zenodo.org/records/$RECORD/files/$f?download=1"
done
sha256sum -c SHA256SUMS
for t in *.tar; do tar xf "$t"; done
(cd office && sha256sum -c --quiet SHA256SUMS.sessions)
sha256sum -c --quiet SHA256SUMS.trials
echo "Dataset ready in $DATA"
