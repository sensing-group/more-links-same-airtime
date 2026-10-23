# Artifact: More Links, Same Airtime

Paper: *More Links, Same Airtime: Spending a Shared Channel's Sounding Budget on WiFi Sensing*
(Karim Khamaisi and Bruno Rodrigues). Code: https://github.com/sensing-group/more-links-same-airtime,
archived at https://doi.org/10.5281/zenodo.23574129. Dataset: https://doi.org/10.5281/zenodo.23574119.

Badges sought: Artifacts Available, Artifacts Evaluated (Functional), Results Reproduced.

## Requirements

- Docker (recommended), or Python 3.12 with `pip install -r requirements.lock`
  (add `--extra-index-url https://download.pytorch.org/whl/cpu` for the CPU build of PyTorch).
- CPU only. 4 GB of RAM with the default two workers (peak 1.3 GB per process).
- Disk: 0.9 GB for the trials path; 43 GB for the full path (21 GB of archives, 20 GB unpacked;
  the archives can be deleted after unpacking), plus 1.1 GB of outputs.
- Network access to zenodo.org for the dataset.

## Running

Build the image once (1.8 GB):

```
git clone https://github.com/sensing-group/more-links-same-airtime && cd more-links-same-airtime
docker build -t mlsa .
```

The image holds the code; data and outputs stay on the host. Each command below runs in

```
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e FIG_DIR=/artifact/work/figures \
  -v "$PWD/data:/artifact/data" -v "$PWD/work:/artifact/work" mlsa <command>
```

| Step | Command | Time |
|---|---|---|
| Figures and tables from the shipped outputs | `./reproduce/figures.sh` | 15 s |
| Fetch the trials only (0.9 GB) | `env TRIALS_ONLY=1 ./reproduce/fetch_data.sh` | minutes |
| Trials path: trials, then compare | `./reproduce/quick.sh` | 3.5 h |
| Fetch the full dataset (21 GB) | `./reproduce/fetch_data.sh` | network bound |
| Full path: every analysis | `./run_all.sh` | 8 h |
| Figures and tables from your rerun | `env ANALYSIS_DIR=work ./reproduce/figures.sh` | 15 s |
| Compare every output | `python3 reproduce/compare.py` | seconds |
| Check every number in the paper | `python3 reproduce/check_claims.py --results work` | seconds |

Times were measured in the container on an 8-core Intel Xeon W-1290 with `WORKERS=2`; `WORKERS`
sets the parallel extraction processes of the office path. Of the full path, the office feature
extractions take 4.7 h and the boosted-tree trial run 1.6 h.

## Checking the results

`reproduce/compare.py` matches every JSON under `results/` (the outputs behind the paper) with the
file of the same name under `work/` and reports each file as OK, DIFFERS or MISSING. Values match
within 1e-6; `--paper` accepts differences up to 0.005, the rounding of the paper's two decimals.
Wall time and memory fields are not compared.

`reproduce/check_claims.py` recomputes 206 numbers the paper prints (abstract, Sections 1 and 4 to 10,
Table 3) from the outputs, rounds them as printed and compares.
`reproduce/claims.csv` maps each number to its sentence, output file and key. Run it on `results/`
(the default) to check the shipped outputs, or with `--results work` to check your rerun.

`reproduce/figures.sh` redraws the plots of Figs. 2 and 5 to 11 and writes Tables 3 and 4 as LaTeX.

Expected outcome of the full path, as measured on the run above: every output file OK, all nine
plotted figures pixel-identical to the paper's, Tables 3 and 4 byte-identical, and 205 of 206
claims matching with one skipped (the apartment, below).

## What maps to what

| Paper | Script (in `scripts/` unless noted) | Output (in `results/`) | Path |
|---|---|---|---|
| §5, Figs. 2 and 6, Table 3 (trials) | `scripted_trial_decision_windows.py` (also `_presence`, `_hgb`) | `scripted_trial_decision_windows*.json` | trials |
| §5, Fig. 8 (receivers) | `scripted_trial_decision_windows.py --budget-unit tx` | `scripted_trial_decision_windows_tx.json` | trials |
| §5, neural network | `neural_budget_check.py` | `neural_budget_check*.json` | trials |
| §5, Fig. 7, noise floor | `sounding_correlation_analysis.py`, `radio_noise_floor_analysis.py` | `sounding_correlation.json`, `radio_noise_floor.json` | trials |
| §4, trial windows, rates, phase order | `trial_window_counts.py`, `trial_sounding_rates.py`, `trial_phase_order_check.py` | `trial_*.json` | trials |
| §6, Fig. 9 | `calibration_reference_analysis.py`, `calibration_model.py` | `calibration_*.json` | trials |
| §4 and §5, Fig. 5, Table 3 (office) | `extract_rate_decimated_features.py`, `decision_window_ablation.py` | `decision_windows/` | full |
| §7, Fig. 10 | `rate_decimation_ablation.py`, `schedule_leak_analysis.py` | `rate_decimated/`, `schedule_leak.json` | full |
| §8, Fig. 11, Table 4 | `extract_dataset_release_features.py`, `threshold_practices_analysis.py`, `office_session_stats.py` | `threshold_practices.json`, `office_session_stats.json` | full |
| §8, earlier analyses | `history/dilution_analysis.py`, `history/first_office/run.sh` | `history/` | full |
| Intervals, cover cost, crossing | `derived_numbers.py` (run by `figures.sh`) | `derived_numbers.json` | any |
| §6 and §10, apartment | `reproduce/apartment.sh` | `apartment/` | optional |

## Known limits

- **Apartment labels are private.** The apartment result (time of day alone predicts the labels at
  AUC 0.82) needs two label files that record when a private home was occupied; they are not
  published. Its output is shipped, `compare.py` and `check_claims.py` skip it when absent, and
  `reproduce/apartment.sh` reruns it for anyone holding the labels. The apartment CSI is public
  (https://doi.org/10.5281/zenodo.18486250).
- **The office link list is an input.** `results/decision_window_link_whitelist.json` (44 pairs)
  was fixed from the first 20 office sessions, two of them still incomplete at the time.
  `scripts/build_link_whitelist.py` rebuilds it exactly from that extraction (usage in its header).
  Built from all 27 sessions instead, the list keeps 35 of the 44 pairs and swaps one transmitter
  for another, so the paper's list is not what the final sessions alone would select.
- **The first office analysis used partial sessions.** Its AUC 0.61 for time of day alone came
  from two sessions that were still recording; their window files are shipped in
  `history/first_office/partial_windows/` and the code of that time in `history/first_office/scripts/`.
- **Not regenerated by the scripts:** the photos of Fig. 1 (`paper/photos/`), the diagrams of
  Figs. 2 (left), 3 and 4 (`paper/diagrams/`) and the hand-written Tables 1, 2 and 5 to 8.
- The trial phase-order check prints AUC 0.939 where Table 3 prints 0.94 for the same cell from the
  main run (0.937): the two runs draw different pair subsets.
