#!/usr/bin/env python3
"""
Per-link packet-decimation features for the two-factor (K x per-link rate) ablation.

temporal_spatial_decomposition.py "reduces temporal resolution" by dropping
whole 60 s windows; each kept window still holds every packet, so that Factor B
measures training-set size, not per-link sampling rate. Its input features
(extract_dataset_release_features.py) also zero-fill every link-window whose
250 ms sub-windows hold fewer than two frames, and under adaptive TDMA that
zero pattern alone predicts occupancy.

This extractor removes both effects:
  1. Frames are deduplicated by content (dataset windows are 250 ms long on a
     100 ms stride, so each frame appears in up to three windows).
  2. A link-window is ELIGIBLE when it holds at least BASE_RATE_HZ * WINDOW_S
     unique frames. Every eligible link-window is equalized to exactly that many
     evenly spaced frames, so a scheduler that upsampled a link carries no
     information through rate or frame count.
  3. Each eligible link-window is then decimated to RATE_FRACTIONS of the base
     rate. The eligible set is fixed at the base rate, so every rate sees the
     same links and the same windows; only frames per link-window change.

Output per session (OUT_DIR/rate_<session>.npz):
  link_ids (L,), labels (W,), timestamps (W,), native_frames (L, W) int32,
  features (R, L, W, F) float32 with NaN for ineligible link-windows,
  rate_fractions (R,), feature_names (F,)

Memory: one session per worker; a worker holds one 60 s window of amplitudes
(<= ~40 MB) plus its output array (~70 MB for a 24 h session).
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from memlimit.memlimit import get_available_ram_bytes, safe_pool_map, set_vm_limit  # noqa: E402

DATASET_DIR = Path(os.environ.get("DATASET_DIR", str(REPO_ROOT / "dataset-release")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(REPO_ROOT / "data" / "csi-sense-trials" / "rate_decimated")))

WINDOW_S = 60.0                     # outer aggregation window, as in the paper
# decision windows cut from each outer window in the same pass; each must divide WINDOW_S
DECISION_WINDOWS_S = tuple(float(w) for w in os.environ.get("DECISION_WINDOWS_S", "60").split(","))
BASE_RATE_HZ = float(os.environ.get("BASE_RATE_HZ", "1.4"))   # paper's per-link rate
RATE_FRACTIONS = tuple(float(f) for f in os.environ.get("RATE_FRACTIONS", "1.0,0.5,0.2,0.1").split(","))
MIN_DECIMATED_FRAMES = 2            # fewer frames than this: no decision possible, features stay NaN
# optional JSON list of link ids; frames of other links are skipped
LINK_WHITELIST = os.environ.get("LINK_WHITELIST")
N_SUBCARRIERS = 52
DEDUP_HISTORY = 64                  # recent frame hashes kept per link for dedup
PRESENCE_MAJORITY = 0.5             # window label = majority of dataset-window GT
MIN_SESSION_WINDOWS = 100

# Measured on 20260213_211853 (6000 windows): ~1.6 ms per dataset window parse.
EST_BYTES_PER_WORKER = 1 * 1024 ** 3
RSS_WARN_FRACTION = 0.5

FEATURE_NAMES = (
    "amp_mean", "amp_std", "amp_min", "amp_max", "amp_var", "diff_std", "nbvi",
    "diff_abs_mean", "diff_std_dup", "zcr", "energy_mean", "amp_var_subcarrier",
)
N_FEATURES = len(FEATURE_NAMES)

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

log = logging.getLogger("rate_decimation")


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def link_window_features(amp: np.ndarray) -> np.ndarray:
    """12 features from an (n_frames, 52) amplitude matrix ordered in time."""
    s = amp.mean(axis=1)
    d = np.diff(s)
    mean = float(s.mean())
    std = float(s.std())
    centred = d - d.mean()
    zcr = float(np.sum(np.abs(np.diff(np.sign(centred))))) / (2 * len(d))
    return np.array([
        mean, std, float(s.min()), float(s.max()), float(s.var()), float(d.std()),
        std / mean if mean > 0 else 0.0,
        float(np.abs(d).mean()), float(d.std()), zcr, float(np.mean(s ** 2)),
        float(amp.var(axis=0).mean()),
    ], dtype=np.float32)


def decimated_indices(n_frames: int, n_keep: int) -> np.ndarray:
    """Evenly spaced frame indices; equalizes and decimates in one step."""
    return np.unique(np.linspace(0, n_frames - 1, n_keep).round().astype(np.int64))


def frames_per_rate(base_frames: int) -> List[int]:
    return [int(round(base_frames * f)) for f in RATE_FRACTIONS]


def stream_samples(session_dir: Path):
    for fpath in sorted(session_dir.glob("samples_*.jsonl.gz")):
        with gzip.open(fpath, "rb") as f:
            try:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = _loads(line)
                    except ValueError:
                        continue
                    if "csi_data" in obj:
                        yield obj
            except EOFError:
                # a file cut while it was being written; keep what precedes the cut
                log.warning("%s truncated; using the samples before the cut", fpath)


def process_session(session_dir: Path) -> Optional[Dict]:
    """Stream one session; write one npz per decision window; return a summary dict."""
    t0 = time.time()
    sid = session_dir.name
    whitelist = set(json.loads(Path(LINK_WHITELIST).read_text())) if LINK_WHITELIST else None
    plan = {}   # decision window -> (sub-windows per outer window, base frames, frames per rate)
    for w in DECISION_WINDOWS_S:
        n_sub = int(round(WINDOW_S / w))
        if abs(n_sub * w - WINDOW_S) > 1e-6:
            raise ValueError(f"decision window {w}s does not divide {WINDOW_S}s")
        base = int(round(BASE_RATE_HZ * w))
        plan[w] = (n_sub, base, frames_per_rate(base))

    recent: Dict[str, deque] = defaultdict(lambda: deque(maxlen=DEDUP_HISTORY))
    recent_set: Dict[str, set] = defaultdict(set)
    cur_frames: Dict[str, List[tuple]] = defaultdict(list)    # link -> [(ts, frame)]
    cur_votes: List[tuple] = []                                  # [(ts, label)]
    # per decision window: one {link: (native_count, (R, F) features or None)} per closed sub-window
    closed: Dict[float, List[Dict[str, tuple]]] = {w: [] for w in plan}
    labels: Dict[float, List[int]] = {w: [] for w in plan}
    starts: Dict[float, List[float]] = {w: [] for w in plan}
    first_ts = None
    cur_idx = 0
    n_samples = 0

    def close_window():
        t_outer = first_ts + cur_idx * WINDOW_S
        for w, (n_sub, base, n_keep) in plan.items():
            for j in range(n_sub):
                lo, hi = t_outer + j * w, t_outer + (j + 1) * w
                per_link = {}
                for lid, frames in cur_frames.items():
                    sub = [fr for ts, fr in frames if lo <= ts < hi]
                    n = len(sub)
                    if n < max(base, MIN_DECIMATED_FRAMES):
                        per_link[lid] = (n, None)
                        continue
                    arr = np.asarray(sub, dtype=np.float32)
                    amp = np.sqrt(arr[:, 0::2] ** 2 + arr[:, 1::2] ** 2)
                    feats = np.full((len(n_keep), N_FEATURES), np.nan, dtype=np.float32)
                    for r, k in enumerate(n_keep):
                        if k >= MIN_DECIMATED_FRAMES:
                            feats[r] = link_window_features(amp[decimated_indices(n, k)])
                    per_link[lid] = (n, feats)
                votes = [v for ts, v in cur_votes if lo <= ts < hi]
                closed[w].append(per_link)
                labels[w].append(-1 if not votes else int(np.mean(votes) > PRESENCE_MAJORITY))
                starts[w].append(lo)
        cur_frames.clear()
        cur_votes.clear()

    for obj in stream_samples(session_dir):
        ts = obj.get("timestamp", 0.0)
        if first_ts is None:
            first_ts = ts
        idx = int((ts - first_ts) / WINDOW_S)
        while idx > cur_idx:
            close_window()
            cur_idx += 1
        n_samples += 1
        cur_votes.append((ts, 1 if obj.get("ground_truth", {}).get("presence", False) else 0))
        for link in obj["csi_data"].get("links", []):
            lid = f"{link.get('tx', '')}->{link.get('rx', '')}"
            if whitelist is not None and lid not in whitelist:
                continue
            for fr in link.get("csi_frames", []):
                if len(fr) != 2 * N_SUBCARRIERS:
                    continue
                h = hash(tuple(fr))
                rs = recent_set[lid]
                if h in rs:
                    continue
                dq = recent[lid]
                if len(dq) == dq.maxlen:
                    rs.discard(dq[0])
                dq.append(h)
                rs.add(h)
                cur_frames[lid].append((ts, fr))
        if n_samples % 100_000 == 0:
            rss = _rss_mb()
            log.info("%s: %d samples, RSS=%.0f MB", sid, n_samples, rss)
            if rss * 1024 ** 2 > RSS_WARN_FRACTION * EST_BYTES_PER_WORKER:
                log.warning("%s: RSS %.0f MB above budget", sid, rss)
    if first_ts is None or n_samples < MIN_SESSION_WINDOWS:
        return None
    close_window()

    summary = {"session_id": sid, "n_samples": n_samples, "windows": {}}
    for w in plan:
        link_ids = sorted({lid for sw in closed[w] for lid in sw})
        li = {lid: i for i, lid in enumerate(link_ids)}
        n_l, n_w = len(link_ids), len(closed[w])
        feats = np.full((len(RATE_FRACTIONS), n_l, n_w, N_FEATURES), np.nan, dtype=np.float32)
        native = np.zeros((n_l, n_w), dtype=np.int32)
        for i, per_link in enumerate(closed[w]):
            for lid, (n, f) in per_link.items():
                native[li[lid], i] = n
                if f is not None:
                    feats[:, li[lid], i, :] = f
        out_dir = OUT_DIR if len(plan) == 1 else OUT_DIR / f"w{w:g}s"
        out_dir.mkdir(parents=True, exist_ok=True)
        lab = np.array(labels[w], dtype=np.int8)
        np.savez_compressed(
            out_dir / f"rate_{sid}.npz", link_ids=np.array(link_ids), labels=lab,
            timestamps=np.array(starts[w]), native_frames=native, features=feats,
            rate_fractions=np.array(RATE_FRACTIONS), feature_names=np.array(FEATURE_NAMES),
            base_rate_hz=BASE_RATE_HZ, window_s=w, session_id=sid,
        )
        summary["windows"][f"{w:g}"] = {
            "n_links": n_l, "n_windows": n_w, "n_pos": int((lab == 1).sum()),
            "eligible_fraction": float(np.isfinite(feats[0, :, :, 0]).mean()) if n_l else 0.0,
        }
        del feats, native
    del closed
    gc.collect()
    summary.update({"seconds": round(time.time() - t0, 1), "peak_rss_mb": round(_rss_mb(), 0)})
    log.info("%s done: %s", sid, summary)
    return summary


def discover(dataset_dir: Path) -> List[Path]:
    sessions = [d for d in (dataset_dir / "sessions").iterdir() if d.is_dir() and any(d.glob("samples_*"))]
    # largest first so the long sessions do not finish last on one worker
    return sorted(sessions, key=lambda d: -sum(f.stat().st_size for f in d.glob("samples_*")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="estimate memory and list sessions, load nothing")
    ap.add_argument("--only", nargs="*", help="session ids to process")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    set_vm_limit()

    sessions = discover(DATASET_DIR)
    if args.only:
        sessions = [s for s in sessions if s.name in set(args.only)]
    gz_bytes = sum(f.stat().st_size for s in sessions for f in s.glob("samples_*"))
    log.info("%d sessions, %.1f GB compressed, base %.2f Hz, decision windows %s s, frames per rate %s",
             len(sessions), gz_bytes / 1024 ** 3, BASE_RATE_HZ, DECISION_WINDOWS_S,
             {w: frames_per_rate(int(round(BASE_RATE_HZ * w))) for w in DECISION_WINDOWS_S})
    log.info("budget %d workers x %.1f GB; available RAM %.1f GB",
             args.workers, EST_BYTES_PER_WORKER / 1024 ** 3, get_available_ram_bytes() / 1024 ** 3)
    if args.dry_run:
        return

    summaries = [s for s in safe_pool_map(process_session, sessions, args.workers, EST_BYTES_PER_WORKER) if s]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "extraction_summary.json"
    merged = {}
    if summary_path.exists():   # an --only run adds to earlier sessions instead of replacing them
        merged = {s["session_id"]: s for s in json.loads(summary_path.read_text())["sessions"]}
    merged.update({s["session_id"]: s for s in summaries})
    with open(summary_path, "w") as f:
        json.dump({"base_rate_hz": BASE_RATE_HZ, "rate_fractions": RATE_FRACTIONS, "window_s": WINDOW_S,
                   "decision_windows_s": DECISION_WINDOWS_S, "link_whitelist": LINK_WHITELIST,
                   "sessions": [merged[k] for k in sorted(merged)]}, f, indent=2)
    log.info("wrote %d sessions to %s", len(summaries), OUT_DIR)


if __name__ == "__main__":
    main()
