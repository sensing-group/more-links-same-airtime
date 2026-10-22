#!/usr/bin/env python3
"""Practices table: evaluation practices and their measured effect on the office data,
read from the analysis outputs (ANALYSIS_DIR); writes figures/tab_practices.tex."""
import json
import os
from pathlib import Path

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
OUT = Path(__file__).resolve().parent.parent / "figures" / "tab_practices.tex"
K_ALL, K_CLASSIFIER = "226", "100"
TIME_WINDOW = "w20s"         # decision window of the time-of-day row


def main() -> None:
    thr = json.loads((ANALYSIS_DIR / "threshold_practices.json").read_text())["by_k"]
    abl = json.loads((ANALYSIS_DIR / "rate_decimated" / "rate_decimation_ablation.json").read_text())["results"]
    strat, sess = abl["stratified"][K_CLASSIFIER], abl["session_grouped"][K_CLASSIFIER]
    dw = f"decision_window_ablation_{TIME_WINDOW}.json"
    any_hour = json.loads((ANALYSIS_DIR / "decision_windows" / "unmatched_none" / dw).read_text())["windows"][TIME_WINDOW]
    matched = json.loads((ANALYSIS_DIR / "decision_windows" / dw).read_text())["windows"][TIME_WINDOW]
    rows = [
        ("Raw variance, threshold set on the scored data", "$J$, 226 pairs",
         f"{thr[K_ALL]['paper']:.2f}", f"{thr[K_ALL]['z_max_cv']:.2f} (max rule), {thr[K_ALL]['vote_cv']:.2f} (vote rule)"),
        ("Folds share windows of a session", "AUC, 100 pairs",
         f"{strat['lr_auc_mean'][0]:.2f}", f"{sess['lr_auc_mean'][0]:.2f}"),
        ("Pair availability left in the features", "AUC of availability alone",
         f"{strat['mask_only_auc']:.2f}", "removed"),
        ("Empty windows from any time of day", "AUC of time of day alone",
         f"{any_hour['time_only_auc']:.2f}", f"{matched['time_only_auc']:.2f}"),
    ]
    lines = [r"\begin{table*}[t]",
             r"\caption{Evaluation practices on the office data: flawed and sound versions of the same measurement.}",
             r"\label{tab:practices}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcl@{}}",
             r"\toprule",
             r"Practice & Measure & Flawed & Sound \\",
             r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
