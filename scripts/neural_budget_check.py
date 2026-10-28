#!/usr/bin/env python3
"""
More pairs or faster pairs at equal airtime, with a learned model on raw CSI amplitude.

Reviewer check on the scripted 8-node trials: does a neural network on the raw
thinned amplitude sequences show the trend that logistic regression and
gradient-boosted trees show, namely that at a fixed budget of BUDGET_HZ soundings
per second, spreading the budget over many pairs beats spending it on few fast
pairs?

Windows, labels, calibration span, the complete-pair rule and the thinning are
those of scripted_trial_decision_windows.py (imported), at one decision window of
DECISION_WINDOW_S seconds. Per window the input is, for each of the K pairs, the
thinned (f * W, 52) amplitude sequence, standardized per pair and subcarrier
against the trial's calibration windows. One encoder is shared across pairs
(two temporal convolutions over the 52 subcarriers, mean and max over time, a
linear embedding); the pair embeddings are pooled by mean and max (pair-order
free) and a linear head scores the window. Training is leave-one-trial-out with
fixed epochs, Adam and N_SEEDS fixed seeds per fold (AUC averaged over them); nothing is
tuned on the held-out trial. Because every trial follows the same script, the held-out AUC
is also reported against the opening and against the final empty phase alone: a model that
read elapsed time instead of the room would score the final empty phase as occupied.
For K below the smallest complete-pair count, N_SUBSETS random pair subsets per
trial are averaged, as the logistic-regression cells average N_TRIALS subsets.

The logistic-regression AUC of the same "K|rate" cells is read from the existing
JSON outputs and written side by side.

INPUT_MODE=dynamic also removes each pair's per-window, per-subcarrier mean, so
the network sees only the temporal fluctuation and not the shift of the static
profile away from the calibration windows (output suffix _dynamic).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # before numpy is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from memlimit.memlimit import get_available_ram_bytes, set_vm_limit  # noqa: E402
import scripted_trial_decision_windows as st  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "csi-sense-trials"
# INPUT_MODE "calibrated": sequences standardized against calibration (keeps the static profile shift);
# "dynamic": each pair's per-window, per-subcarrier mean is also removed, leaving only temporal fluctuation
INPUT_MODES = ("calibrated", "dynamic")
INPUT_MODE = os.environ.get("INPUT_MODE", INPUT_MODES[0])
OUT_PATH = Path(os.environ.get("OUT_PATH", str(DATA_DIR / ("neural_budget_check.json" if INPUT_MODE == INPUT_MODES[0]
                                                           else f"neural_budget_check_{INPUT_MODE}.json"))))
LR_OCCUPANCY_PATH = Path(os.environ.get("LR_OCCUPANCY_PATH", str(DATA_DIR / "scripted_trial_decision_windows.json")))
LR_PRESENCE_PATH = Path(os.environ.get("LR_PRESENCE_PATH", str(DATA_DIR / "scripted_trial_decision_windows_presence.json")))

DECISION_WINDOW_S = 5.0
BUDGET_HZ = st.BASE_RATE_HZ                       # soundings per second, K pairs x f Hz
SPLITS = ((1, 8.0), (2, 4.0), (4, 2.0), (8, 1.0), (16, 0.5))   # (K pairs, Hz per pair)
TASKS = ("occupancy", "presence")
N_SUBSETS = int(os.environ.get("N_SUBSETS", "5"))
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))   # networks per fold; one seed is unstable for one pair
RNG_SEED = 42

# standardization of the inputs against the calibration windows
SD_MIN = 1e-6                     # a subcarrier with smaller calibration sd is left unscaled
INPUT_CLIP = 10.0                 # |z| cap, keeps rare huge deviations from dominating a batch

# network and training (fixed, not tuned on any held-out trial)
CONV_CHANNELS = 32
CONV_KERNEL = 3
EMBED_DIM = 32
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "2"))

# memory budget
RSS_BUDGET_BYTES = 4 * 1024 ** 3
RSS_WARN_FRACTION = 0.5           # warn when peak RSS exceeds this share of available RAM
EST_PAIRS_PER_TRIAL = 56          # dry-run estimate: 8 nodes, all ordered pairs
FLOAT_BYTES = 4
MIN_PHASE_WINDOWS = 3             # empty windows a phase needs for its own AUC


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024     # Linux reports kB


def log_rss(tag: str, avail: int) -> None:
    peak = peak_rss_bytes()
    warn = "  WARNING: above budget" if peak > min(RSS_BUDGET_BYTES, RSS_WARN_FRACTION * avail) else ""
    print(f"  [rss] {tag}: peak {peak / 1024 ** 3:.2f} GB{warn}", flush=True)


def frames_per_split(rate: float) -> int:
    return int(round(rate * DECISION_WINDOW_S))      # same rounding as the logistic-regression cells


def trial_sequences(tdir: Path):
    """Per split rate: standardized (pairs, windows, frames, 52) inputs; window labels; calibration mask."""
    links, phases = st.read_trial(tdir)
    F, labels, cal, lids = st.trial_features(links, phases, DECISION_WINDOW_S)
    del F
    w = DECISION_WINDOW_S
    t0 = phases[0][0]                                 # window grid exactly as in trial_features
    t_end = max(ts[-1] for ts, _ in links.values())
    starts = np.arange(t0, t_end - w, w)
    lab_all = np.array([st.window_label(s, s + w, phases) or "" for s in starts])
    cal_all = (starts + w <= t0 + st.CAL_S) & (lab_all == "empty_room")
    use = cal_all | ((starts >= t0 + st.CAL_S) & (lab_all != ""))
    starts = starts[use]
    assert np.array_equal(lab_all[use], labels) and np.array_equal(cal_all[use], cal)

    base = int(round(st.BASE_RATE_HZ * w))
    counts = np.zeros((len(lids), len(starts)), np.int64)
    bounds = {}
    for li, lid in enumerate(lids):
        ts, _ = links[lid]
        lo, hi = np.searchsorted(ts, starts), np.searchsorted(ts, starts + w)
        counts[li], bounds[lid] = hi - lo, lo
    ok = counts >= base                               # = finite base-rate features in the parent script
    sel = np.where(ok.mean(axis=1) >= st.COMPLETE_LINK_MIN_FRACTION)[0]
    wins = np.where(ok[sel].all(axis=0))[0]
    cal_w = cal[wins]
    seqs = {}
    for _, rate in SPLITS:
        k = frames_per_split(rate)
        X = np.empty((len(sel), len(wins), k, st.N_SUBCARRIERS), np.float32)
        for pi, li in enumerate(sel):
            _, amp = links[lids[li]]
            lo = bounds[lids[li]]
            for j, wi in enumerate(wins):
                X[pi, j] = amp[lo[wi]:lo[wi] + counts[li, wi]][st.decimated_indices(counts[li, wi], k)]
        mu = X[:, cal_w].mean(axis=(1, 2), keepdims=True)
        sd = X[:, cal_w].std(axis=(1, 2), keepdims=True)
        sd = np.where(sd > SD_MIN, sd, 1.0)
        X = np.clip((X - mu) / sd, -INPUT_CLIP, INPUT_CLIP)
        if INPUT_MODE == INPUT_MODES[1]:
            X -= X.mean(axis=2, keepdims=True)
        seqs[rate] = X.astype(np.float32, copy=False)
    del links
    return {"trial": tdir.name, "X": seqs, "labels": labels[wins], "cal": cal_w, "n_pairs": len(sel)}


def empty_phases(labels) -> dict:
    """Masks of the opening and the final empty phase; labels are in time order."""
    occupied = np.where(np.isin(labels, st.OCCUPIED))[0]
    idx, empty = np.arange(len(labels)), labels == "empty_room"
    return {"opening_empty": empty & (idx < occupied.min()), "final_empty": empty & (idx > occupied.max())}


def build_model(torch):
    nn = torch.nn

    class PairPoolNet(nn.Module):
        """Shared per-pair encoder over (time, subcarrier); mean+max pooling over pairs; linear head."""

        def __init__(self):
            super().__init__()
            pad = CONV_KERNEL // 2
            self.conv = nn.Sequential(
                nn.Conv1d(st.N_SUBCARRIERS, CONV_CHANNELS, CONV_KERNEL, padding=pad), nn.ReLU(),
                nn.Conv1d(CONV_CHANNELS, CONV_CHANNELS, CONV_KERNEL, padding=pad), nn.ReLU())
            self.embed = nn.Sequential(nn.Linear(2 * CONV_CHANNELS, EMBED_DIM), nn.ReLU())
            self.head = nn.Linear(2 * EMBED_DIM, 1)

        def forward(self, x):                         # x: (B, K, T, 52)
            b, k, t, c = x.shape
            h = self.conv(x.reshape(b * k, t, c).transpose(1, 2))          # (B*K, C, T)
            e = self.embed(torch.cat([h.mean(2), h.amax(2)], 1)).reshape(b, k, -1)
            return self.head(torch.cat([e.mean(1), e.amax(1)], 1)).squeeze(1)

    return PairPoolNet()


def train_and_score(torch, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = build_model(torch)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pos = float(ytr.mean())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(Xte)).numpy()


def lr_cells() -> dict:
    out = {}
    for task, path in (("occupancy", LR_OCCUPANCY_PATH), ("presence", LR_PRESENCE_PATH)):
        res = json.loads(path.read_text())["results"].get(f"{task}|{DECISION_WINDOW_S:g}", {})
        for k, rate in SPLITS:
            out[(task, k, rate)] = res.get(f"{k}|{rate:g}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="list trials and estimate memory, read nothing")
    args = ap.parse_args()
    if INPUT_MODE not in INPUT_MODES:
        ap.error(f"INPUT_MODE must be one of {INPUT_MODES}")
    set_vm_limit()
    avail = get_available_ram_bytes()
    t_start = time.time()
    trials = [st.TRIALS_DIR / t for t in st.TRIALS if (st.TRIALS_DIR / t / "raw" / "csi_raw.jsonl.zst").exists()]
    frames = sum(frames_per_split(r) for _, r in SPLITS)
    est = 0
    for tdir in trials:
        ph = [json.loads(l) for l in open(tdir / "annotations.jsonl")]
        dur = max(p["end_ts"] for p in ph) - min(p["start_ts"] for p in ph)
        est += int(dur / DECISION_WINDOW_S) * EST_PAIRS_PER_TRIAL * frames * st.N_SUBCARRIERS * FLOAT_BYTES
    print(f"input {INPUT_MODE}, {len(trials)} trials, W={DECISION_WINDOW_S:g} s, splits {SPLITS}, tasks {TASKS}, "
          f"{N_SUBSETS} subsets, {EPOCHS} epochs; held inputs ~{est / 1024 ** 3:.2f} GB "
          f"(+ one trial's raw read at a time); available RAM {avail / 1024 ** 3:.1f} GB", flush=True)
    if args.dry_run:
        return

    import torch
    torch.set_num_threads(TORCH_THREADS)
    n_params = sum(p.numel() for p in build_model(torch).parameters())
    print(f"torch {torch.__version__}, {n_params} parameters, {TORCH_THREADS} threads", flush=True)

    per = []
    for tdir in trials:                               # one trial in memory at a time
        per.append(trial_sequences(tdir))
        print(f"  read {tdir.name}: {per[-1]['n_pairs']} complete pairs, {len(per[-1]['labels'])} windows "
              f"({time.time() - t_start:.0f}s)", flush=True)
        log_rss(tdir.name, avail)
    n_min = min(p["n_pairs"] for p in per)
    lr = lr_cells()
    rng = np.random.default_rng(RNG_SEED)

    results = {}
    for task in TASKS:
        for si, (k, rate) in enumerate(SPLITS):
            n_sub = N_SUBSETS if k < n_min else 1
            by_trial, aucs, phase_aucs = defaultdict(list), [], {"opening_empty": [], "final_empty": []}
            for s in range(n_sub):
                Xs, ys, phases = [], [], []
                for p in per:
                    sub = np.sort(rng.choice(p["n_pairs"], k, replace=False))
                    y = st.task_labels(p["labels"], task)
                    test = (~p["cal"]) & (y >= 0)
                    Xs.append(np.ascontiguousarray(p["X"][rate][sub][:, test].transpose(1, 0, 2, 3)))
                    ys.append(y[test])
                    phases.append({key: m[test] for key, m in empty_phases(p["labels"]).items()})
                for i in range(len(per)):             # leave one trial out
                    if len(set(ys[i])) < 2:
                        continue
                    Xtr = np.concatenate([Xs[j] for j in range(len(per)) if j != i])
                    ytr = np.concatenate([ys[j] for j in range(len(per)) if j != i])
                    for r in range(N_SEEDS):
                        seed = RNG_SEED + 10000 * r + 1000 * si + 100 * s + i
                        score = train_and_score(torch, Xtr, ytr, Xs[i], seed)
                        aucs.append(float(roc_auc_score(ys[i], score)))
                        by_trial[per[i]["trial"]].append(aucs[-1])
                        for key, m in phases[i].items():
                            sel = (ys[i] == 1) | m
                            if m.sum() >= MIN_PHASE_WINDOWS:
                                phase_aucs[key].append(float(roc_auc_score(ys[i][sel], score[sel])))
                    del Xtr, ytr
                del Xs, ys
            cell = lr[(task, k, rate)]
            results[f"{task}|{k}|{rate:g}"] = {
                "task": task, "pairs": k, "rate_hz": rate, "frames_per_pair": frames_per_split(rate),
                "n_subsets": n_sub, "neural_auc": round(float(np.mean(aucs)), 4),
                "neural_auc_sd_over_folds_subsets": round(float(np.std(aucs)), 4),
                "neural_auc_by_trial": {t: round(float(np.mean(v)), 4) for t, v in by_trial.items()},
                "neural_auc_vs_opening_empty": round(float(np.mean(phase_aucs["opening_empty"])), 4),
                "neural_auc_vs_final_empty": round(float(np.mean(phase_aucs["final_empty"])), 4),
                "lr_auc": cell["auc"] if cell else None,
                "lr_auc_by_trial": cell["auc_by_trial"] if cell else None}
            print(f"[{task:9s}] K={k:2d} @ {rate:g} Hz: neural {np.mean(aucs):.3f}  "
                  f"LR {cell['auc'] if cell else float('nan'):.3f}  ({time.time() - t_start:.0f}s)", flush=True)
            log_rss(f"{task} K={k}", avail)

    wins = {}
    (k_lo, r_lo), (k_hi, r_hi) = SPLITS[0], SPLITS[-1]
    for task in TASKS:
        lo, hi = results[f"{task}|{k_lo}|{r_lo:g}"], results[f"{task}|{k_hi}|{r_hi:g}"]
        for model in ("neural", "lr"):
            a, b = lo[f"{model}_auc_by_trial"], hi[f"{model}_auc_by_trial"]
            if a and b:
                wins[f"{task}|{model}"] = {"many_slow_beats_one_fast": sum(b[t] > a[t] for t in a),
                                           "n_trials": len(a)}
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": [p["trial"] for p in per],
           "input_mode": INPUT_MODE, "decision_window_s": DECISION_WINDOW_S, "budget_hz": BUDGET_HZ, "splits": SPLITS,
           "complete_pairs_per_trial": {p["trial"]: p["n_pairs"] for p in per},
           "model": {"type": "shared conv1d pair encoder, mean+max pair pooling, linear head",
                     "n_parameters": n_params, "conv_channels": CONV_CHANNELS, "conv_kernel": CONV_KERNEL,
                     "embed_dim": EMBED_DIM, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY, "input_clip": INPUT_CLIP, "n_subsets": N_SUBSETS, "n_seeds": N_SEEDS,
                     "seed": RNG_SEED, "torch_version": torch.__version__, "torch_threads": TORCH_THREADS},
           "lr_sources": [str(LR_OCCUPANCY_PATH.relative_to(REPO_ROOT)) if LR_OCCUPANCY_PATH.is_relative_to(REPO_ROOT)
                          else str(LR_OCCUPANCY_PATH),
                          str(LR_PRESENCE_PATH.relative_to(REPO_ROOT)) if LR_PRESENCE_PATH.is_relative_to(REPO_ROOT)
                          else str(LR_PRESENCE_PATH)],
           "results": results, "wins_16_over_1": wins,
           "runtime_s": round(time.time() - t_start, 1), "peak_rss_gb": round(peak_rss_bytes() / 1024 ** 3, 3)}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wins 16 vs 1 pairs: {wins}", flush=True)
    print(f"wrote {OUT_PATH} ({out['runtime_s']:.0f}s, peak RSS {out['peak_rss_gb']:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
