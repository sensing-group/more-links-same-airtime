# More Links, Same Airtime

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23574119.svg)](https://doi.org/10.5281/zenodo.23574119)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23574129.svg)](https://doi.org/10.5281/zenodo.23574129)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10823-b31b1b.svg)](https://arxiv.org/abs/2602.10823)

Code for *More Links, Same Airtime: Spending a Shared Channel's Sounding Budget on WiFi Sensing*
(Karim Khamaisi and Bruno Rodrigues, Sensing Group, University of St. Gallen; submitted to
ACM MobiSys 2027).

Under IEEE 802.11bf, every pair of devices that senses people draws its soundings from the airtime
of one shared channel. The paper measures how a fixed sounding budget should be split: a budget
detects people best when spread over as many pairs as leave each pair about two soundings per
decision, and receivers that overhear a sounding add evidence at no airtime cost. Spreading costs
calibration (label-free detectors need a longer empty-room reference) and privacy (an adaptive
schedule reveals occupancy).

**Dataset:** CSI from a 14-node office mesh (27 sessions) and six scripted trials in a living room
and a laboratory, on Zenodo: https://doi.org/10.5281/zenodo.23574119. This code is archived at
https://doi.org/10.5281/zenodo.23574129.

## Quick start

```
git clone https://github.com/sensing-group/more-links-same-airtime
cd more-links-same-airtime
pip install -r requirements.txt
cd paper && for s in fig_*.py make_tab_*.py; do python3 $s; done   # figures and tables into ../figures/
```

This redraws every plotted figure and Tables 3 and 4 from the shipped analysis outputs in
`results/`, without the dataset. To rerun the analyses themselves, fetch the dataset (21 GB) into
`data/` and run `run_all.sh`:

```
mkdir -p data && cd data
for f in office-sessions-1.tar office-sessions-2.tar office-sessions-3.tar office-features-labels.tar trials.tar SHA256SUMS; do
  curl -L -o $f "https://zenodo.org/records/23574119/files/$f?download=1"
done
sha256sum -c SHA256SUMS && for t in *.tar; do tar xf $t; done
cd .. && ./run_all.sh
```

## Earlier version

This repository first accompanied the preprint
[*Less is More: The Dilution Effect in Multi-Link Wireless Sensing*](https://arxiv.org/abs/2602.10823),
which reported that a single well-placed link outperformed the full mesh. The paper above revisits
that question at equal airtime and with sound evaluation, and finds the opposite. Its §8 measures
the evaluation practices that produce an apparent dilution effect, and its §10 explains why the
preprint's apartment recording is not used. The preprint's recorder and analysis are kept in
`tdma-mesh/` and at tag `v1-dilution-preprint`.

License: MIT.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Feature extraction and every analysis behind the paper's numbers |
| `paper/` | Figure and table scripts; they read only `results/` |
| `results/` | The analysis outputs the paper reports (JSON) |
| `capture/` | The recorder and ESP32 firmware that captured the scripted trials, at commit `618c876`, with WiFi credentials replaced by placeholders |
| `ghost_eval/` | Memory limit helper used by the scripts |
| `data/` | Place the dataset here: `data/office/sessions/`, `data/office/features/` and `data/trials/` |
| `tdma-mesh/` | The earlier TDMA recorder and the preprint's analysis (see its README) |

## Reproducing the figures and tables

```
pip install -r requirements.txt
cd paper
for s in fig_*.py make_tab_*.py; do python3 $s; done   # writes ../figures/
```

The scripts regenerate the plots of Figs. 2 and 5-11 and Tables 3 and 4 of the paper from
`results/`, identical to the submitted figures. Figs. 1, 3 and 4 and the left half of Fig. 2 are
photographs and diagrams.

## Reproducing the analyses

`run_all.sh` runs every step in order; each step writes one file or folder under `work/`.
The office steps stream one session at a time and cap their memory (`ghost_eval/memlimit.py`).
Plan on a machine with 16 GB of RAM, and set `OPENBLAS_NUM_THREADS=1` when running steps in
parallel.

| Paper item | Script | Output in `results/` |
|---|---|---|
| Table 3 (trial rows), Figs. 2 and 6, §5 | `scripted_trial_decision_windows.py` (default, `TASKS=presence`, `--budget-unit tx`, `--classifier hgb`) | `scripted_trial_decision_windows*.json` |
| Table 3 (office rows), Figs. 2, 5 and 6, time-only control | `extract_rate_decimated_features.py` (decision windows), then `decision_window_ablation.py --window W` | `decision_windows/decision_window_ablation_w<W>s.json` |
| Fig. 7 | `sounding_correlation_analysis.py` | `sounding_correlation.json` |
| Fig. 8 | `scripted_trial_decision_windows.py` and `--budget-unit tx` | `scripted_trial_decision_windows.json`, `scripted_trial_decision_windows_tx.json` |
| Fig. 9, §6 | `calibration_reference_analysis.py`, `calibration_model.py` | `calibration_reference.json`, `calibration_model.json` |
| Fig. 10, §7 | `extract_rate_decimated_features.py` (60 s), `rate_decimation_ablation.py`, `schedule_leak_analysis.py` | `rate_decimated/rate_decimation_ablation.json`, `schedule_leak.json` |
| Table 4, Fig. 11, §8 | `extract_dataset_release_features.py`, `threshold_practices_analysis.py`, `rate_decimation_ablation.py` | `threshold_practices.json`, `rate_decimated/rate_decimation_ablation.json` |
| §4 counts and phase-order control | `office_session_stats.py`, `trial_window_counts.py`, `trial_phase_order_check.py` | `office_session_stats.json`, `trial_window_counts.json`, `trial_phase_order_check.json` |
| §5, neural network | `neural_budget_check.py` (`INPUT_MODE=calibrated` and `dynamic`) | `neural_budget_check*.json` |
| §10, radio floor | `radio_noise_floor_analysis.py` | `radio_noise_floor.json` |
| §10 (dropped apartment site) | `apartment_budget_features.py` on Zenodo 10.5281/zenodo.18486249 | not shipped |

The office analyses pair every occupied window with an empty window of the same fold and the
same quarter hour of the day, over 10 draws, so the time of day cannot separate the classes
(`--empty-match time`, the default; `--empty-match none` gives the unmatched protocol).

The office decision-window extraction keeps the 44 links in
`results/decision_window_link_whitelist.json`; `decision_window_ablation.py` then fixes, without
labels, the 16 of them most often available in 5 s windows.

## Citation

See `CITATION.cff`. The preprint can be cited as arXiv:2602.10823.
