#!/usr/bin/env python3
"""Table 2: few pairs sounded fast against many pairs sounded slowly at the same
airtime, read from the analysis outputs (ANALYSIS_DIR); writes figures/tab_budget.tex.

The last column counts how consistently many slow pairs win: on the office mesh
against each random few-pair subset, in the trials in each held-out trial.
"""
import json
import os
from pathlib import Path

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
OUT = Path(os.environ.get("FIG_DIR", Path(__file__).resolve().parent.parent / "figures")) / "tab_budget.tex"
OFFICE = "decision_windows/decision_window_ablation_w{w}s.json"
TRIALS = {"occupancy": "scripted_trial_decision_windows.json", "motion": "scripted_trial_decision_windows.json",
          "presence": "scripted_trial_decision_windows_presence.json"}
# (site label, task, window s, few-fast cell "K|rate", many-slow cell "K|rate", source)
ROWS = [
    ("Office", "occupancy", 10, "4|1", "16|0.25", "office"),
    ("Office", "occupancy", 20, "2|1", "16|0.125", "office"),
    ("Office", "occupancy", 60, "2|1", "16|0.125", "office"),
    ("Trials", "occupancy", 2, "1|8", "8|1", "trials"),
    ("Trials", "occupancy", 5, "1|8", "16|0.5", "trials"),
    ("Trials", "presence", 5, "1|8", "16|0.5", "trials"),
    ("Trials", "presence", 10, "1|8", "16|0.5", "trials"),
    ("Trials", "motion", 1, "1|8", "4|2", "trials"),
    ("Trials", "motion", 5, "1|8", "16|0.5", "trials"),
]


def cells(source: str, task: str, w: int) -> dict:
    if source == "office":
        return json.loads((ANALYSIS_DIR / OFFICE.format(w=w)).read_text())["windows"][f"w{w}s"]["cells"]
    return json.loads((ANALYSIS_DIR / TRIALS[task]).read_text())["results"][f"{task}|{w}"]


def wins(source: str, few: dict, many: dict) -> str:
    if source == "office":
        subsets = few["auc_subsets"]
        return f"{sum(many['auc'] > s for s in subsets)}/{len(subsets)}"
    trials = sorted(few["auc_by_trial"])
    return f"{sum(many['auc_by_trial'][t] > few['auc_by_trial'][t] for t in trials)}/{len(trials)}"


def num(v: float) -> str:
    """Two decimals with a typeset minus; a value that rounds to zero prints as 0.00."""
    s = f"{v:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "$-$")


def bold(v: float, other: float) -> str:
    """The larger AUC of a row in bold."""
    return f"\\textbf{{{num(v)}}}" if round(v, 2) > round(other, 2) else num(v)


def main() -> None:
    lines = [r"\begin{table}[t]",
             r"\caption{Few fast against many slow pairs at one budget (office 2 soundings/s, 4 at 10\,s; trials 8/s): classifier AUC and vote-rule $J$.}",
             r"\label{tab:budget}",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrccccc@{}}",
             r"\toprule",
             r" & & & \multicolumn{2}{c}{Few, fast} & \multicolumn{2}{c}{Many, slow} & \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Site & Task & $W$ & AUC & $J$ & AUC & $J$ & Wins \\",
             r"\midrule"]
    for site, task, w, few_key, many_key, src in ROWS:
        c = cells(src, task, w)
        few, many = c[few_key], c[many_key]
        lines.append(f"{site} & {task.capitalize()} & {w}\\,s & {bold(few['auc'], many['auc'])} & {num(few['j_vote'])} & "
                     f"{bold(many['auc'], few['auc'])} & {num(many['j_vote'])} & {wins(src, few, many)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
