"""Paper figures for FORGE, generated directly from experiment archives.

Data sources (all offline, no LLM/GPU):
  - results/runs.jsonl  : registered aggregates per run (value-only F1)
  - results/ci_table.py : paired bootstrap diffs vs forge-chunks-full
  - results/*.jsonl     : per-sample files recomputed via forge.eval.metrics

Design rules (print/CVD-safe, per the dataviz method adapted to academic PDF):
  - Okabe-Ito palette, fixed assignment per SYSTEM IDENTITY (never per rank);
  - identity never color-alone: hatching + direct value labels + legend;
  - one axis; thin marks; recessive grid; text in ink color, not series color;
  - serif font to match the paper, 8.5pt; column width 3.35in, text width 6.5in.

Outputs PDFs to ../Pdev/writing/figures/ for \\includegraphics.

Usage: python -m experiments.plots
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
C_BASE = "#E69F00"       # orange -- passive baseline
C_FLOOR = "#999999"      # grey -- floor / context rows
C_MEM0 = "#E69F00"       # orange -- Mem0
C_AMEM = "#009E73"       # green -- A-Mem
H_FORGE, H_ABL, H_BASE = "", "//", "xx"
H_MEM0, H_AMEM = "\\\\", ".."

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "pdf.fonttype": 42,
})


def runs() -> dict:
    """name -> aggregates, from the run registry."""
    out = {}
    for line in open(RESULTS / "runs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        out[d["name"]] = d["aggregates"]
    return out


def mean_f1(path: Path) -> float:
    from forge.eval.metrics import score_sample
    xs = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        xs.append(score_sample(d["qa_id"], d["gold_tool"], d["gold_args"],
                               d["pred_tool"], d["pred_args"]).f1)
    return 100 * sum(xs) / len(xs)


def _save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


def fig_store():
    """Storage-chain ladder: distillation -> lossless supply (the paper's main
    ablation story; paper floor and oracle drawn as context rows)."""
    r = runs()
    rows = [
        ("no retrieval\n(paper)", 10.0, C_FLOOR, ""),
        ("distilled\nfacts only", mean_f1(RESULTS / "m2a_v6_full.jsonl"), C_FORGE_ABL, H_ABL),
        ("hybrid store\n+ guard", r["m2a-hybrid-store"]["f1"], C_FORGE_ABL, H_ABL),
        ("hybrid store", r["m2a-hybrid-store-nooverride"]["f1"], C_FORGE_ABL, H_ABL),
        ("lossless\nchunks (final)", r["forge-chunks-full"]["f1"], C_FORGE, H_FORGE),
        ("oracle evidence\n(paper)", 53.8, C_FLOOR, ""),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    xs = range(len(rows))
    for x, (label, v, c, h) in zip(xs, rows):
        ax.bar(x, v, width=0.62, color=c, hatch=h, edgecolor="white", linewidth=0.8)
        ax.text(x, v + 1.0, f"{v:.1f}", ha="center", fontsize=7.5)
    for (x0, r0), (x1, r1) in zip(list(zip(xs, rows))[1:4], list(zip(xs, rows))[2:5]):
        ax.annotate("", xy=(x1, r1[1] + 4.5), xytext=(x0, r0[1] + 4.5),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))
        ax.text((x0 + x1) / 2, (r0[1] + r1[1]) / 2 + 5.5,
                f"+{r1[1]-r0[1]:.1f}", ha="center", fontsize=7, color="#333333")
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=6.6)
    ax.set_ylabel("F1 (400 tasks)")
    ax.set_ylim(0, 60)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "store.pdf")


def fig_components():
    """Mechanism splits + component stack vs the full configuration, with 95%
    paired-bootstrap intervals (all intervals cross zero: the +6.2 total edge
    is architectural, not attributable to any single add-on). Dot plot -- the
    differences are small, so bars from a truncated axis would exaggerate."""
    ci = json.loads((RESULTS / "ci_table.json").read_text(encoding="utf-8"))
    r = runs()
    ref = r["forge-chunks-full"]["f1"]
    rows = [
        ("full (reference)", r["forge-chunks-full"]["f1"], None),
        ("no per-param questions", r["forge-chunks-nodemand"]["f1"],
         ci["forge-chunks-nodemand"]),
        ("flat render", r["forge-chunks-flat"]["f1"],
         ci["forge-chunks-flat"]),
        ("+ rerank", r["forge-chunks-r1"]["f1"], ci["forge-chunks-r1"]),
        ("+ candidate ordering", r["forge-chunks-r1r2"]["f1"], ci["forge-chunks-r1r2"]),
        ("+ verbatim anchor", r["forge-chunks-r1r2r3"]["f1"], ci["forge-chunks-r1r2r3"]),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    ys = range(len(rows))
    for y, (label, v, c) in zip(ys, rows):
        if c:  # interval of the true mean = reference + bootstrap diff interval
            ax.plot([ref + c["ci_low_pts"], ref + c["ci_high_pts"]], [y, y],
                    color="#444444", lw=1.0, solid_capstyle="butt", zorder=2)
        col = C_FORGE if c is None else C_FORGE_ABL
        ax.scatter([v], [y], s=22, color=col, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.text(v + 0.12, y, f"{v:.1f}", ha="left", va="center",
                fontsize=6.8, color="#333333")
    ax.axvline(ref, color="#0072B2", lw=0.6, ls="--", alpha=0.6)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (400 tasks)\nwhiskers: 95% paired bootstrap")
    ax.set_xlim(33, 39.5)
    ax.xaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "components.pdf")


def fig_funnel():
    """Coverage funnel: where gold values die, fact-only vs hybrid store.

    Lossless-by-construction coverage of the chunk/hybrid store vs 48.7% for
    the distilled-only store is the figure's point.
    """
    stages = ["all gold\narguments", "present in\nstore", "answered\ncorrectly"]
    fact_only = [100.0, 100 * 402 / 825, 100 * 83 / 825]
    hybrid = [100.0, 100.0, 100 * 216 / 825]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ys = range(len(stages))
    h = 0.34
    ax.barh([y + h / 2 + 0.02 for y in ys], fact_only, height=h, color=C_BASE,
            hatch=H_BASE, edgecolor="white", label="distilled facts only (16.23 F1)")
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
    """Per-difficulty-level F1: FORGE (lossless config) vs passive baseline,
    both recomputed from pred/gold args under the value-only metric."""
    from forge.eval.metrics import score_sample

    def per_level(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            res = score_sample(d["qa_id"], d["gold_tool"], d["gold_args"],
                               d["pred_tool"], d["pred_args"])
            out.setdefault(d.get("level", "?"), []).append(res.f1)
        return {k: 100 * sum(v) / len(v) for k, v in sorted(out.items())}

    forge = per_level(RESULTS / "forge-chunks-full.jsonl")
    base = per_level(RESULTS / "ltmemory_hybrid5_full.jsonl")

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


def fig_multidim():
    """Four-system multi-dimensional profile: error taxonomy (share of
    errors), schema validity, traceability. Complex-value fidelity is 0 for
    every system and omitted (stated in the caption): a shared failure."""
    rows = json.loads((RESULTS / "multimetrics.json").read_text(encoding="utf-8"))
    metrics = [
        ("fabrication\n(% of errors)", "fabrication_pct_of_errors"),
        ("corruption\n(% of errors)", "corruption_pct_of_errors"),
        ("miss\n(% of errors)", "miss_pct_of_errors"),
        ("schema\nvalidity (%)", "schema_validity_pct"),
        ("traceability\n(%)", "traceability_pct"),
    ]
    systems = [  # (label, color, hatch) in fixed identity order
        ("LTMemory", C_FLOOR, ""),
        ("FORGE", C_FORGE, H_FORGE),
        ("Mem0", C_MEM0, H_MEM0),
        ("A-Mem", C_AMEM, H_AMEM),
    ]
    by_run = {r["run"]: r for r in rows}
    runs = {"LTMemory": "ltmemory_hybrid5_full", "FORGE": "forge-chunks-full",
            "Mem0": "mem0-full", "A-Mem": "amem-full"}
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ys = range(len(metrics))
    w = 0.19
    for si, (label, col, h) in enumerate(systems):
        vals = [by_run[runs[label]][key] for _, key in metrics]
        offs = [y + (si - 1.5) * (w + 0.015) for y in ys]
        ax.barh(offs, vals, height=w, color=col, hatch=h,
                edgecolor="white", linewidth=0.6, label=label)
    ax.set_yticks(list(ys), [m[0] for m in metrics], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.set_xlabel("share (%)")
    ax.legend(frameon=False, fontsize=6.8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.14))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "multidim.pdf")


if __name__ == "__main__":
    fig_store()
    fig_components()
    fig_funnel()
    fig_levels()
    fig_multidim()
