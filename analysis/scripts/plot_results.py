"""Reproduce the paper's statistical figures from frozen analysis excerpts.

Usage: python scripts/plot_results.py
Optional source refresh: python scripts/plot_results.py --extract-from ../i16-self-evidence
No inference, output parsing, bootstrap, or experiment is run by this script.
The optional refresh copies exact metrics from the frozen summary JSON files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_data.json"
FIG = ROOT / "figures"
COLORS = {"self": "#2265C3", "other": "#D99924", "compatible": "#258F96",
          "normative": "#68717D", "ink": "#101B36", "grid": "#E6EBF0",
          "truncated": "#D99924", "invalid": "#B74A60", "valid": "#258F96"}


def find(rows, **criteria):
    matches = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}, got {len(matches)}")
    return matches[0]


def extract(source_root: Path):
    paths = {"R1": "results/formal-4gpu-recovery/analysis/summary.json",
             "R2": "results/mechanism-discrimination/analysis/summary.json",
             "R3": "results/validation-8gpu-recovery/analysis/summary.json"}
    sources, summaries = {}, {}
    for key, rel in paths.items():
        raw = (source_root / rel).read_bytes()
        summaries[key] = json.loads(raw.decode("utf-8-sig"))
        sources[key] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()}

    def contrast(round_id, model, name, metric, template=None):
        w = find(summaries[round_id]["wording"], model=model)
        a = find(w["analyses"], name=name)
        section = f"wording[model={model}]/analyses[name={name}]"
        if template:
            a = a["by_template"][template]
            section += f"/by_template/{template}"
        return {"round": round_id, "model": model, "source_section": section,
                "metric": metric, "counts": a["counts"],
                "point": a["metrics"][metric],
                "bounds": a["missing_probability_bounds"][metric]}

    r1 = []
    core = find(summaries["R1"]["core"], model="qwen3-8b-thinking")
    for reminder in ("explicit", "minimal"):
        arm = find(core["arms"], reminder=reminder)
        for condition in ("attribution_same_all", "attribution_independent_agree", "attribution_independent_conflict"):
            name = f"qwen3-8b-thinking:formal:{reminder}:{condition}"
            a = find(arm["analyses"], name=name)
            r1.append({"round": "R1", "reminder": reminder, "condition": condition,
                       "source_section": f"core[model=qwen3-8b-thinking]/arms[reminder={reminder}]/analyses[name={name}]",
                       "counts": a["counts"], "point": a["metrics"]["signed_directional_effect"]})
    decomposition = find(summaries["R3"]["attribution_error_decomposition"],
                         model="qwen3-8b-thinking", protocol="independent",
                         visibility="signal_visible", reminder="legacy",
                         output_instruction="restrictive", template_id="likelihood_v3", relation="agree")
    qwen = "qwen3-8b-thinking"
    qwen35 = "qwen35-9b-thinking"
    hidden_updates = []
    visible_updates = []
    mechanism = find(summaries["R3"]["mechanism"], model=qwen)
    for template in ("frequency_v3", "likelihood_v3"):
        for protocol in ("positive", "zero", "negative"):
            row = find(summaries["R3"]["hidden_updates"], model=qwen, protocol=protocol,
                       attribution="self", template_id=template)
            hidden_updates.append({"round": "R3", "model": qwen, "protocol": protocol,
                "attribution": "self", "template_id": template,
                "source_section": f"hidden_updates[model={qwen},protocol={protocol},attribution=self,template_id={template}]",
                "counts": {"candidate_sets": row["n_planned"], "complete_valid_sets": row["n_valid"]},
                "point": row["metrics"]["directional_update_from_prior"],
                "normative_same_sets": row["metrics"]["normative_directional_update"],
                "residual": row["metrics"]["update_residual"],
                "bounds": row["missing_probability_bounds"]["directional_update_from_prior"]})
        for protocol in ("positive", "zero", "negative", "independent"):
            name = f"{protocol}:self:answer_visible_minus_hidden"
            row = find(mechanism["answer_visibility_increments"], name=name)["by_template"][template]
            visible_updates.append({"round": "R3", "model": qwen, "protocol": protocol,
                "attribution": "self", "template_id": template,
                "source_section": f"mechanism[model={qwen}]/answer_visibility_increments[name={name}]/by_template/{template}",
                "counts": row["counts"], "point": row["metrics"]["signed_directional_effect"],
                "normative_same_sets": row["metrics"]["normative_directional_effect"],
                "residual": row["metrics"]["directional_residual"],
                "bounds": row["missing_probability_bounds"]["signed_directional_effect"]})
    metrics = {
        "instruction_mae": [contrast(r, qwen, "independent:instruction_main", "mae_contrast") for r in ("R2", "R3")],
        "instruction_mae_templates": [contrast("R3", qwen, "independent:instruction_main", "mae_contrast", t)
                                      for t in ("frequency_v3", "likelihood_v3")],
        "instruction_attribution": [contrast(r, qwen, "independent:instruction_by_attribution", "signed_directional_effect") for r in ("R2", "R3")],
        "qwen35_bounds": [contrast("R3", qwen35, "independent:attribution:scoped:compatible", "signed_directional_effect"),
                          contrast("R3", qwen35, "independent:instruction_main", "mae_contrast")],
    }
    data = {"description": "Exact excerpts of frozen analyses for plotting, with source hashes and section locators.",
            "sources": sources, "r1_attribution": r1, "r3_decomposition": decomposition, **metrics,
            "information_updates": {"selection": "R3 Qwen3-8B, self attribution, scoped reminder, compatible output, both templates. Template strata are copied without reaggregation.",
                                    "signal_hidden": hidden_updates, "signal_visible": visible_updates},
            "r3_coverage": [{k: model[k] for k in ("model", "n_planned", "n_observed", "n_valid", "valid_rate", "status_counts", "mae")}
                            for model in summaries["R3"]["models"]],
            "interpretation": {
                "point": "Family equally weighted estimate over complete valid matched sets.",
                "bounds": "Full planned finite-grid missing-probability bounds, not confidence intervals. Each distinct missing record has one probability in [0,1], with shared coefficients combined first. The complete-set point and full-grid bounds have different target sets.",
                "r1_interval": "Frozen 95% percentile bootstrap interval over numerical-family means, used as design-composition sensitivity, not population inference.",
                "counts": "Complete matched sets / planned matched sets. Instruction contrasts require all eight factorial cells. Attribution contrasts require two cells.",
                "units": "Probabilities and probability errors on the 0 to 1 scale.",
                "information_updates": "Signal-hidden points are directional changes from the prior among valid outputs. Signal-visible points compare A visible with A hidden on complete pairs. Normative points use exactly the same valid outputs or complete pairs. The displayed bounds are for the directional update itself, not the update residual, over the full planned template stratum."}}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def setup():
    # Explicitly fail when the requested font is absent rather than silently substitute.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9.5, "axes.linewidth": .65,
                         "axes.edgecolor": COLORS["normative"], "text.color": COLORS["ink"],
                         "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["ink"],
                         "ytick.color": COLORS["ink"], "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white", "axes.unicode_minus": True,
                         "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                         "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold"})
    FIG.mkdir(parents=True, exist_ok=True)


def clean(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.6)
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=.5, zorder=0)
    ax.set_axisbelow(True)


def panel(fig, ax, label, y=.035):
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, y, label, ha="center", va="bottom", fontsize=10.5, fontweight="bold")


def export(fig, stem):
    # Fixed 6.3-inch width matches the manuscript text area, with live SVG text.
    for ext in ("pdf", "svg"):
        target = FIG / f"{stem}.{ext}"
        fig.savefig(target)
        if ext == "svg":
            target.write_text("\n".join(line.rstrip() for line in
                              target.read_text(encoding="utf-8").splitlines()) + "\n",
                              encoding="utf-8")
    preview_dir = ROOT / "build" / "figure-previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir / f"{stem}.png", dpi=200)
    plt.close(fig)


def attribution(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.7), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=.10, right=.98, top=.74, bottom=.25, wspace=.49)
    for reminder, dx, marker, fill in [("explicit", -.13, "o", True), ("minimal", .13, "D", False)]:
        rows = [r for r in data["r1_attribution"] if r["reminder"] == reminder]
        for x, row in enumerate(rows):
            point = row["point"]
            value, (low, high) = point["estimate"], point["ci95"]
            a.errorbar(x+dx, value, yerr=[[value-low], [high-value]], fmt=marker, ms=5.5,
                       color=COLORS["ink"], mfc=COLORS["ink"] if fill else "white",
                       mew=.8, capsize=2, elinewidth=.8, label=reminder if x == 0 else None, zorder=3)
            count = row["counts"]
            a.text(x+dx, high+.008, f"{count['complete_valid_sets']}/{count['candidate_sets']}",
                   ha="center", va="bottom", fontsize=9.5, rotation=90)
    a.axhline(0, color=COLORS["normative"], lw=.7)
    a.set(xlim=(-.45, 2.5), ylim=(-.026, .17), xticks=[0, 1, 2],
          xticklabels=["Redundant", "Independent\nagreement", "Independent\nconflict"], ylabel="Attribution gap")
    a.set_yticks([0, .05, .10, .15])
    a.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(-.02, 1.22), ncol=2, handletextpad=.35, columnspacing=1)
    clean(a, "y")
    decomp = data["r3_decomposition"]
    vals = [decomp["metrics"][f"{who}_directional_error"]["estimate"] for who in ("self", "other")]
    for y, who, val in zip([1, 0], ["self", "other"], vals):
        b.barh(y, val, height=.43, color=COLORS[who], alpha=.9)
        b.text(val+.008, y, f"{val:+.4f}", ha="left", va="center", fontsize=10, color="white")
    b.axvline(0, color=COLORS["normative"], lw=.8, ls="--")
    b.set(xlim=(-.19, .006), ylim=(-.58, 1.7), yticks=[1, 0], yticklabels=["self", "other"],
          xlabel="Mean directional normative error\n(equal family weights)")
    b.set_xticks([-.2, -.1, 0])
    b.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    gap = decomp["metrics"]["attribution_gap"]["estimate"]
    b.text(.5, 1.42, "R3 · Qwen3-8B · independent\nS visible · agreement\nlegacy · restrictive · likelihood",
           transform=b.transAxes, ha="center", va="top", fontsize=9.5, linespacing=1.15)
    b.text(.5, 1.035, f"Gap {gap:+.4f} · {decomp['complete_pairs']}/{decomp['candidate_pairs']} pairs",
           transform=b.transAxes, ha="center", va="bottom", fontsize=10)
    clean(b)
    panel(fig, a, "(a) R1: evidence dependence")
    panel(fig, b, "(b) R3: error decomposition")
    export(fig, "attribution_diagnosis")


def bound_rows(ax, rows, labels, color, xlim, xlabel, label_counts=True):
    for i, row in enumerate(rows):
        y = len(rows)-1-i
        point = row["point"]["estimate"]
        low, high = row["bounds"]["lower"], row["bounds"]["upper"]
        # Offset rows intentionally separate the complete-set point from the full-grid bound.
        ax.plot(point, y+.12, "D", mfc="white", mec=color, mew=1.1, ms=5.5, zorder=4)
        ax.plot([low, high], [y-.10, y-.10], color=color, lw=3, solid_capstyle="butt", zorder=3)
        ax.plot([low, high], [y-.10, y-.10], "|", color=color, ms=6, mew=.8, zorder=3)
        if label_counts:
            c = row["counts"]
            ax.text(xlim[1], y+.12, f"{c['complete_valid_sets']}/{c['candidate_sets']}",
                    ha="right", va="center", fontsize=9.5)
    ax.axvline(0, color=COLORS["normative"], lw=.75, ls="--")
    ax.set(xlim=xlim, ylim=(-.55, len(rows)-.45), yticks=list(range(len(rows)-1, -1, -1)),
           yticklabels=labels, xlabel=xlabel)
    clean(ax)


def bound_legend(fig, y=.97):
    handles = [Line2D([], [], color=COLORS["ink"], marker="D", mfc="white", mew=1, lw=0, ms=5.5),
               Line2D([], [], color=COLORS["ink"], lw=3)]
    fig.legend(handles, ["Complete matched sets", "Full design bounds"],
               loc="upper center", bbox_to_anchor=(.5, y), frameon=False, ncol=2, columnspacing=2,
               handlelength=1.5, handletextpad=.6)


def instruction(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.75), gridspec_kw={"width_ratios": [1.2, .85]})
    fig.subplots_adjust(left=.175, right=.985, top=.74, bottom=.24, wspace=.51)
    rows = data["instruction_mae"] + data["instruction_mae_templates"]
    bound_rows(a, rows, ["R2 overall", "R3 overall", "R3 frequency", "R3 likelihood"],
               COLORS["compatible"], (-.222, .05), "MAE contrast")
    a.set_xticks([-.2, -.1, 0])
    a.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    bound_rows(b, data["instruction_attribution"], ["R2", "R3"], COLORS["self"], (-.065, .03),
               "Instruction × attribution")
    b.set_xticks([-.06, -.03, 0])
    b.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.text(.5, .985, "Qwen3-8B · independent evidence · compatible − restrictive",
             ha="center", va="top", fontsize=10)
    bound_legend(fig, y=.945)
    a.text(.5, 1.065, "Mean over reminders\nand attributions", transform=a.transAxes,
           ha="center", va="bottom", fontsize=9.5, linespacing=1.1)
    b.text(.5, 1.065, "Mean over reminders", transform=b.transAxes,
           ha="center", va="bottom", fontsize=9.5)
    panel(fig, a, "(a) Numerical accuracy")
    panel(fig, b, "(b) Attribution interaction")
    export(fig, "instruction_effects")


def coverage(data):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 3.7), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.13, right=.985, top=.72, bottom=.25, wspace=.70)
    models = data["r3_coverage"]
    for y, row in zip([1, 0], models):
        left = 0
        for status, color in [("ok", COLORS["valid"]), ("truncated", COLORS["truncated"]), ("invalid", COLORS["invalid"])]:
            share = row["status_counts"].get(status, 0) / row["n_planned"]
            a.barh(y, share, left=left, height=.40, color=color)
            left += share
        a.text(.025, y, f"{row['n_valid']}/{row['n_planned']}", color="white", va="center", fontsize=10)
    a.set(xlim=(0, 1), ylim=(-.6, 1.6), yticks=[1, 0], yticklabels=["Qwen3", "Qwen3.5"], xlabel="Fraction of terminal calls")
    a.set_xticks([0, .5, 1], ["0", "0.5", "1.0"])
    clean(a)
    from matplotlib.patches import Patch
    a.legend([Patch(color=COLORS[k]) for k in ("valid", "truncated", "invalid")],
             ["Valid", "Truncated", "Invalid"], frameon=False, ncol=2, loc="upper left",
             bbox_to_anchor=(-.12, 1.51), handlelength=.8, columnspacing=.7, handletextpad=.3)
    bound_rows(b, data["qwen35_bounds"], ["Attribution\ngap", "MAE\ncontrast"],
               COLORS["self"], (-.15, .18), "Contrast on probability scale")
    b.set_xticks([-.1, 0, .1], ["−0.10", "0", "0.10"])
    b.set_ylim(-.6, 1.75)
    b.text(.5, 1.07, "R3 · Qwen3.5 · independent", transform=b.transAxes,
           ha="center", va="bottom", fontsize=9.5)
    b.text(.5, 1.45, "scoped · compatible", transform=b.get_yaxis_transform(),
           ha="center", va="bottom", fontsize=9.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": .7})
    b.text(.5, .40, "compatible − restrictive\nmean over reminders + attributions", transform=b.get_yaxis_transform(),
           ha="center", va="bottom", fontsize=9.5, linespacing=1.1,
           bbox={"facecolor": "white", "edgecolor": "none", "pad": .7})
    # A separate compact legend makes the two estimands explicit in this panel.
    b.legend([Line2D([], [], color=COLORS["self"], marker="D", mfc="white", lw=0, ms=5.5),
              Line2D([], [], color=COLORS["self"], lw=3)],
             ["Complete sets", "Full design bounds"], frameon=False, loc="upper left",
             bbox_to_anchor=(-.05, 1.51), handlelength=1.3, labelspacing=.3)
    panel(fig, a, "(a) R3: output coverage")
    panel(fig, b, "(b) Qwen3.5: full design bounds")
    export(fig, "coverage_bounds")


def information(data):
    """Show the existing template strata without pooling or recomputing metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8))
    fig.subplots_adjust(left=.145, right=.985, top=.825, bottom=.145, wspace=.53, hspace=.88)
    templates = ("frequency_v3", "likelihood_v3")
    for i, visibility in enumerate(("signal_hidden", "signal_visible")):
        for j, template in enumerate(templates):
            ax = axes[i, j]
            rows = [r for r in data["information_updates"][visibility] if r["template_id"] == template]
            for index, row in enumerate(rows):
                y = len(rows)-1-index
                observed = row["point"]["estimate"]
                normative = row["normative_same_sets"]["estimate"]
                low, high = row["bounds"]["lower"], row["bounds"]["upper"]
                # Three separate vertical locations prevent a bound from appearing to be
                # an uncertainty interval around either point's different target set.
                ax.plot(observed, y+.23, "o", color=COLORS["self"], ms=4.5, zorder=4)
                ax.plot(normative, y, "D", mfc="white", mec=COLORS["normative"], mew=.9, ms=4.5, zorder=4)
                ax.plot([low, high], [y-.23, y-.23], color=COLORS["self"], lw=2.5, solid_capstyle="butt", zorder=3)
                ax.plot([low, high], [y-.23, y-.23], "|", color=COLORS["self"], ms=5.5, mew=.9, zorder=3)
                counts = row["counts"]
                ax.text(.43, y+.05, f"{counts['complete_valid_sets']}/{counts['candidate_sets']}",
                        ha="right", va="center", fontsize=9.5)
            ax.axvline(0, color=COLORS["normative"], lw=.65, ls="--")
            ax.set(xlim=(-.30, .43), ylim=(-.5, len(rows)-.5),
                   yticks=list(range(len(rows)-1, -1, -1)),
                   yticklabels=[r["protocol"] for r in rows], xlabel="Directional update")
            ax.set_xticks([-.25, 0, .25], ["−0.25", "0", "0.25"])
            clean(ax)
            panel(fig, ax, f"({chr(97+i*2+j)}) S {'hidden' if i == 0 else 'visible'}: {template}",
                  y=.473 if i == 0 else .025)
    handles = [Line2D([], [], marker="o", color=COLORS["self"], lw=0, ms=4.5),
               Line2D([], [], marker="D", color=COLORS["normative"], mfc="white", lw=0, ms=4.5),
               Line2D([], [], color=COLORS["self"], lw=2.5)]
    fig.text(.5, .982, "Qwen3-8B · R3 · self · scoped + compatible", ha="center", va="top", fontsize=10)
    fig.legend(handles, ["Observed", "Normative", "Full design update bounds"], loc="upper center",
               bbox_to_anchor=(.5, .943), frameon=False, ncol=3, columnspacing=1.5,
               handlelength=1.25, handletextpad=.4)
    export(fig, "information_updates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-from", type=Path, help="Original evidence root, for an exact excerpt refresh")
    args = parser.parse_args()
    if args.extract_from:
        extract(args.extract_from.resolve())
    data = json.loads(DATA.read_text(encoding="utf-8"))
    setup()
    attribution(data)
    information(data)
    instruction(data)
    coverage(data)
    print("Generated four 6.3-inch figures as PDF, editable SVG, and PNG previews.")


if __name__ == "__main__":
    main()
