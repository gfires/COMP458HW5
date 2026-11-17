"""Histogram of the final QAOA output distribution, grouped by cut value.

Reads outputs/results.json (written by qaoa_maxcut.py) and plots, for each distinct
cut value (original weights), the fraction of the 5000 shots landing on that value:
the selected-p QAOA circuit vs. a uniform random sampler. A second plot ranks the
partitions QAOA produced. Only sampled (shot-based) outputs are plotted.

Run:  python plot.py [--results outputs/results.json] [--out outputs/histogram.png] [--show]
"""

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt

QAOA_COLOR = "#2a78d6"     # categorical slot 1
UNIFORM_COLOR = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def group_by_cut(rows):
    """Sum shot fractions per distinct cut value."""
    shots = sum(r["shot_count"] for r in rows)
    freq = defaultdict(float)
    for r in rows:
        freq[round(r["cut_value"], 6)] += r["shot_count"] / shots
    return freq


def plot_histogram(results_path="outputs/results.json", out_path="outputs/histogram.png",
                   show=False):
    with open(results_path) as f:
        res = json.load(f)
    inst, final, unif = res["instance"], res["final"], res["uniform_baseline"]
    p = final["p"]
    q_freq = group_by_cut(final["bitstrings"])
    u_freq = group_by_cut(unif["bitstrings"])
    cuts = sorted(set(q_freq) | set(u_freq))

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.18
    ax.bar([c - w / 2 for c in cuts], [u_freq[c] for c in cuts], width=w,
           color=UNIFORM_COLOR, edgecolor=SURFACE, linewidth=0.6,
           label=f"Uniform random (5000 shots)", zorder=2)
    ax.bar([c + w / 2 for c in cuts], [q_freq[c] for c in cuts], width=w,
           color=QAOA_COLOR, edgecolor=SURFACE, linewidth=0.6,
           label=f"QAOA p={p} (5000 shots)", zorder=3)

    # reference lines: random-assignment mean and optimum
    for x, txt in ((inst["mean_cut"], f"mean cut {inst['mean_cut']:.2f}"),
                   (inst["c_max"], f"$C_{{max}}$ = {inst['c_max']:.1f}")):
        ax.axvline(x, color=INK_2, linestyle=(0, (3, 3)), linewidth=1, zorder=1)
        ax.annotate(txt, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=9,
                    color=INK_2)

    # direct label on the optimal bar
    s = final["summary"]
    ax.annotate(f"$P_{{opt}}$ = {s['empirical_P_opt']:.3f}",
                (inst["c_max"] + w / 2, q_freq[round(inst["c_max"], 6)]),
                xytext=(-10, -2), textcoords="offset points", ha="right", va="top",
                fontsize=9, color=INK)

    ax.set_xlabel("Cut value (original weights)", color=INK)
    ax.set_ylabel("Probability", color=INK)
    ax.set_title(f"Output distribution by cut value: QAOA p={p} vs. uniform  "
                 f"(<C> = {s['empirical_mean_cut']:.2f} ± {s['empirical_std_err']:.2f}, "
                 f"NI = {s['empirical_NI']:.3f})", color=INK, fontsize=11, loc="left")
    ax.set_xlim(min(cuts) - 1, max(cuts) + 1)
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left",  # QAOA first, then uniform
              frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def partition_label(bs):
    """'0101001' -> '{0,2,4,5} | {1,3,6}' (side containing node 0 first)."""
    a = [str(i) for i, c in enumerate(bs) if c == bs[0]]
    b = [str(i) for i, c in enumerate(bs) if c != bs[0]]
    return "{" + ",".join(a) + "} | {" + ",".join(b) + "}"


def group_by_partition(rows):
    """Merge each bitstring with its mirror image (same partition, same cut)."""
    shots = sum(r["shot_count"] for r in rows)
    parts = {}
    for r in rows:
        bs = r["bitstring"]
        key = min(bs, bs.translate(str.maketrans("01", "10")))
        d = parts.setdefault(key, {"label": partition_label(key), "cut": r["cut_value"],
                                   "optimal": r["is_optimal"], "freq": 0.0})
        d["freq"] += r["shot_count"] / shots
    return list(parts.values())


def plot_partitions(results_path="outputs/results.json", out_path="outputs/partitions.png",
                    top_n=15, show=False):
    """Ranked probability of each partition (cut) sampled by the final QAOA circuit."""
    with open(results_path) as f:
        res = json.load(f)
    final = res["final"]
    p = final["p"]
    parts = sorted(group_by_partition(final["bitstrings"]),
                   key=lambda d: (-d["freq"], -d["cut"]))
    sampled = [d for d in parts if d["freq"] > 0]
    top, rest = sampled[:top_n], sampled[top_n:]
    rows = [(f"{d['label']}   cut {d['cut']:.1f}" + ("  (optimal)" if d["optimal"] else ""),
             d["freq"]) for d in top]
    if rest:
        rows.append((f"Other ({len(rest)} sampled partitions)",
                     sum(d["freq"] for d in rest)))

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))[::-1]   # largest at the top
    ax.barh(y, [r[1] for r in rows], height=0.7, color=QAOA_COLOR, edgecolor=SURFACE,
            linewidth=0.6, label=f"QAOA p={p} (5000 shots)", zorder=3)
    for yi, (_, fr) in zip(y, rows):
        ax.annotate(f"{fr:.3f}", (fr, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK, family="monospace")
    ax.set_xlabel("Probability", color=INK)
    n_unsampled = len(parts) - len(sampled)
    fig.suptitle(f"Partitions produced by QAOA p={p}: probability over 5000 shots "
                 "(mirror bitstrings merged)\n"
                 f"{len(sampled)} of {len(parts)} possible partitions sampled in 5000 shots; "
                 f"{n_unsampled} never seen", color=INK, fontsize=11, x=0.01, ha="left")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Wrote {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.json")
    ap.add_argument("--out", default="outputs/histogram.png")
    ap.add_argument("--partitions-out", default="outputs/partitions.png")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    plot_histogram(a.results, a.out, a.show)
    plot_partitions(a.results, a.partitions_out, a.top_n, a.show)
