#!/usr/bin/env python3
"""False alarms of a label-free threshold against the number of pairs, for
several lengths of the empty-room reference (scripted trials, solid) and the
finite-reference Gaussian model (dashed)."""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style as ps

ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(Path(__file__).resolve().parent.parent / "results")))
FIG_H_IN = 1.8
WINDOWS_S = (5.0, 10.0)
COLOURS = ("vermilion", "orange", "green", "blue")       # shortest to longest reference


def main() -> None:
    ps.apply()
    d = json.loads((ANALYSIS_DIR / "calibration_model.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(ps.COLUMN_W_IN, FIG_H_IN), sharey=True, layout="constrained")
    for ax, w, tag in zip(axes, WINDOWS_S, "ab"):
        cells = [c for c in d["cells"] if c["window_s"] == w]
        refs = sorted({c["reference_s"] for c in cells})
        for ref, col in zip(refs, COLOURS):
            pts = sorted((c["pairs"], c["false_alarm"], c["reference_windows"], c["model_false_alarm"])
                         for c in cells if c["reference_s"] == ref)
            k, fa, nref, model = zip(*pts)
            ax.plot(k, fa, marker="o", color=ps.OKABE_ITO[col], label=f"{ref:g} s")
            ax.plot(k, model, linestyle="--", linewidth=0.8, color=ps.OKABE_ITO[col])
        ax.axhline(d["alpha"], color=ps.OKABE_ITO["grey"], linestyle=":", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
        ax.set_xlabel("Pairs")
        ax.set_title(f"({tag}) {w:g} s decisions", loc="left")
    axes[0].set_ylabel("False-alarm rate")
    axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), title="Empty-room reference", loc="outside upper center",
               ncol=len(COLOURS), handlelength=1.4, columnspacing=1.0)
    ps.save(fig, "fig_calibration")


if __name__ == "__main__":
    main()
