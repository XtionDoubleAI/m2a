"""Paper figures for FORGE, generated directly from experiment archives.

Design rules (print/CVD-safe, per the dataviz method adapted to academic PDF):
  - Okabe-Ito palette, fixed assignment per SYSTEM IDENTITY (never per rank);
  - identity never color-alone: hatching + direct value labels + legend;
  - one axis; thin marks; recessive grid; text in ink color, not series color;
  - serif font to match the paper, 8.5pt; column width 3.35in, text width 6.5in.

Outputs PDFs to ../Pdev/writing/figures/ for \includegraphics.

Usage: python -m experiments.plots   (no LLM, no GPU; consumes results/*.jsonl)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
FIGS = HERE.parent / "Pdev" / "writing" / "figures"

# Okabe-Ito, fixed by system identity
C_FORGE = "#0072B2"      # blue -- FORGE (final)
C_FORGE_ABL = "#56B4E9"  # light blue -- FORGE ablation variants
C_BASE = "#E69F00"       # orange -- baselines (passive / A-Mem)
C_FLOOR = "#999999"      # grey -- floor / context rows
H_FORGE, H_ABL, H_BASE = "", "//", "xx"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "pdf.fonttype": 42,
})


def _save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


def fig_ablation():
    """Gain decomposition from floor to oracle; each bar is one ablation row."""
    rows = [
        ("No retrieval\n(paper)", 10.0, C_FLOOR, ""),
        ("fact store\nonly", 16.23, C_FORGE_ABL, H_ABL),
        ("hybrid store\n+ guard", 27.72, C_FORGE_ABL, H_ABL),
        ("hybrid store\n(final)", 33.88, C_FORGE, H_FORGE),
        ("oracle evidence\n(paper)", 53.8, C_FLOOR, ""),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    xs = range(len(rows))
    for x, (label, v, c, h) in zip(xs, rows):
        ax.bar(x, v, width=0.62, color=c, hatch=h, edgecolor="white", linewidth=0.8)
        ax.text(x, v + 1.0, f"{v:.1f}", ha="center", fontsize=7.5)
    for x, (label, v, c, h) in zip(list(xs)[1:4], rows[1:4]):
        pass
    # annotate deltas between consecutive FORGE rows
    for (x0, r0), (x1, r1) in zip(list(zip(xs, rows))[1:3], list(zip(xs, rows))[2:4]):
        ax.annotate("", xy=(x1, r1[1] + 4.5), xytext=(x0, r0[1] + 4.5),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))
        ax.text((x0 + x1) / 2, (r0[1] + r1[1]) / 2 + 5.5,
                f"+{r1[1]-r0[1]:.1f}", ha="center", fontsize=7, color="#333333")
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=7)
    ax.set_ylabel("F1 (400 tasks)")
    ax.set_ylim(0, 60)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "ablation.pdf")


def fig_funnel():
    """Coverage funnel: where gold values die, fact-only vs hybrid store.

    Three stages (all gold arguments -> present in store -> answered correctly),
    two systems as grouped horizontal bars. Lossless-by-construction coverage of
    the hybrid store vs 48.7% for the distilled-only store is the figure's point.
    """
    stages = ["all gold\narguments", "present in\nstore", "answered\ncorrectly"]
    fact_only = [100.0, 100 * 402 / 825, 100 * 83 / 825]
    hybrid = [100.0, 100.0, 100 * 216 / 825]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ys = range(len(stages))
    h = 0.34
    ax.barh([y + h / 2 + 0.02 for y in ys], fact_only, height=h, color=C_BASE,
            hatch=H_BASE, edgecolor="white", label="fact store only (16.23 F1)")
    ax.barh([y - h / 2 - 0.02 for y in ys], hybrid, height=h, color=C_FORGE,
            hatch=H_FORGE, edgecolor="white", label="hybrid store (33.88 F1)")
    for y, v in zip(ys, fact_only):
        ax.text(v + 1.5, y + h / 2 + 0.02, f"{v:.0f}%", va="center", fontsize=7)
    for y, v in zip(ys, hybrid):
        ax.text(v + 1.5, y - h / 2 - 0.02, f"{v:.0f}%", va="center", fontsize=7)
    ax.set_yticks(list(ys), stages, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of 825 gold arguments (%)")
    ax.xaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "funnel.pdf")


def fig_levels():
    """Per-difficulty-level F1: FORGE vs passive baseline."""
    def per_level(path, pred_key="pred_args"):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        inter = {json.loads(l)["qa_id"]: json.loads(l)
                 for l in open(str(path).replace(".jsonl", ".intermediates.jsonl"),
                               encoding="utf-8")}
        from forge.eval.metrics import score_sample
        out = {}
        for r in rows:
            m = inter.get(r["qa_id"])
            args = m["model_args"] if m else r[pred_key]
            res = score_sample(r["qa_id"], r["gold_tool"], r["gold_args"], r["gold_tool"], args)
            out.setdefault(r["level"], []).append(res.f1)
        return {k: 100 * sum(v) / len(v) for k, v in sorted(out.items())}

    forge = per_level(RESULTS / "m2a-hybrid-store.jsonl")
    base_rows = [json.loads(l) for l in open(RESULTS / "ltmemory_hybrid5_full.jsonl", encoding="utf-8")]
    base = {}
    for r in base_rows:
        base.setdefault(r.get("level", "?"), []).append(r["f1"])
    base = {k: 100 * sum(v) / len(v) for k, v in sorted(base.items())}

    levels = ["L1", "L2", "L3", "L4"]
    xs = range(len(levels))
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    ax.bar([x - 0.19 for x in xs], [base.get(l, 0) for l in levels], width=0.34,
           color=C_BASE, hatch=H_BASE, edgecolor="white", label="passive hybrid@5")
    ax.bar([x + 0.19 for x in xs], [forge.get(l, 0) for l in levels], width=0.34,
           color=C_FORGE, hatch=H_FORGE, edgecolor="white", label="FORGE")
    for x in xs:
        ax.text(x + 0.19, forge.get(levels[x], 0) + 1, f"{forge.get(levels[x],0):.0f}",
                ha="center", fontsize=7)
    ax.set_xticks(list(xs), ["L1\nexplicit", "L2\ninferred", "L3\nmulti-src", "L4\nconflict"], fontsize=7)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 50)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "levels.pdf")


if __name__ == "__main__":
    fig_ablation()
    fig_funnel()
    fig_levels()
