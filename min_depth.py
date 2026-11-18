"""Minimum-depth scheduling of the commuting ZZ cost gates via line-graph coloring.

All 13 cost-layer gates e^{-i c_ij gamma Z_i Z_j} commute, so they can be reordered
freely. Two gates can run in the same layer iff they share no qubit. Build the line
graph C' = T(C) (one vertex per edge of C, adjacent iff the edges share a node);
then the chromatic number of C' equals the minimum circuit depth, and each color
class is one parallel layer.

Run:  python min_depth.py [--out outputs/min_depth_circuit.png]
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from qaoa_maxcut import EDGES, N_NODES


def convert_graph(graph: dict) -> tuple[dict, list]:
    """Line graph T(G) of an undirected adjacency list (each edge listed both ways).

    Returns (new_graph, edges): new_graph maps edge index -> set of edge indices
    sharing a vertex with it; edges[i] = (u, v) with u < v is the edge for index i.
    Edges are numbered in sorted order, matching the default circuit ordering.
    """
    edges = sorted({(min(u, v), max(u, v)) for u in graph for v in graph[u]})
    index = {e: i for i, e in enumerate(edges)}
    new_graph = {i: set() for i in range(len(edges))}
    for u, nbrs in graph.items():
        incident = [index[(min(u, v), max(u, v))] for v in nbrs]
        # every pair of edges meeting at u is adjacent in T(G)
        for a in incident:
            new_graph[a].update(b for b in incident if b != a)
    return new_graph, edges


def chromatic_number(graph: dict) -> tuple[int, dict]:
    """Exact chromatic number by recursive backtracking.

    Tries k = 1, 2, ... and returns the first k admitting a proper coloring, plus
    that coloring {vertex: color in 0..k-1}. Vertices are colored in descending
    degree order; a vertex may only open color `used` (one past the highest color
    used so far), which removes color-permutation symmetry from the search.
    """
    order = sorted(graph, key=lambda v: -len(graph[v]))
    n = len(order)

    def backtrack(i, colors, k, used):
        if i == n:
            return True
        v = order[i]
        forbidden = {colors[u] for u in graph[v] if u in colors}
        for c in range(min(used + 1, k)):
            if c not in forbidden:
                colors[v] = c
                if backtrack(i + 1, colors, k, max(used, c + 1)):
                    return True
                del colors[v]
        return False

    for k in range(1 if n else 0, n + 1):
        colors = {}
        if backtrack(0, colors, k, 0):
            return k, colors
    return 0, {}


def draw_circuit(edges, coloring, n_qubits, out_path, title=None):
    """Draw the cost layer as parallel layers (one per color), in the two-box ZZ style.

    Gates in the same layer share no qubit, but their connecting lines could overlap
    on paper, so overlapping spans are offset into sub-columns within the layer band.
    """
    k = max(coloring.values()) + 1
    layers = [sorted((edges[i] for i in coloring if coloring[i] == c),
                     key=lambda e: (e[0], e[1])) for c in range(k)]

    # place gates: x positions, with sub-columns where vertical spans would overlap
    placed, bands, x = [], [], 0.0
    for c, gates in enumerate(layers):
        cols = []   # each sub-column: list of (lo, hi) spans
        for u, v in gates:
            for col_i, col in enumerate(cols):
                if all(v < lo or u > hi for lo, hi in col):
                    col.append((u, v))
                    break
            else:
                col_i = len(cols)
                cols.append([(u, v)])
            placed.append((x + col_i, u, v))
        bands.append((x - 0.5, x + len(cols) - 0.5, c))
        x += len(cols) + 0.6

    width = x - 0.6
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * (width + 1.5), 0.55 * n_qubits + 1.2),
                           facecolor="white")
    box = 0.5
    for x0, x1, c in bands:
        if c % 2 == 0:
            ax.add_patch(Rectangle((x0, -0.6), x1 - x0, n_qubits + 0.2,
                                   facecolor="#f0efec", edgecolor="none", zorder=0))
        ax.text((x0 + x1) / 2, -0.85, f"Layer {c + 1}", ha="center", va="bottom",
                fontsize=10, color="#52514e")
    for q in range(n_qubits):
        ax.plot([-1.0, width], [q, q], color="black", lw=1.6, zorder=1)
        ax.text(-1.25, q, f"$q_{q}$", ha="right", va="center", fontsize=15)
    for gx, u, v in placed:
        ax.plot([gx, gx], [u, v], color="black", lw=1.6, zorder=2)
        for q in (u, v):
            ax.add_patch(Rectangle((gx - box / 2, q - box / 2), box, box, facecolor="white",
                                   edgecolor="black", lw=1.8, zorder=3))
    ax.set_xlim(-1.9, width + 0.3)
    ax.set_ylim(n_qubits - 0.3, -1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return layers


def pennylane_depth(layers, n_qubits):
    """Depth reported by PennyLane for the ZZ gates applied in the given layer order."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circ():
        for gates in layers:
            for u, v in gates:
                qml.IsingZZ(0.1, wires=[u, v])
        return qml.state()

    return qml.specs(circ)()["resources"].depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/min_depth_circuit.png")
    args = ap.parse_args()

    graph = {i: set() for i in range(N_NODES)}
    for u, v, _ in EDGES:
        graph[u].add(v)
        graph[v].add(u)

    line_graph, edges = convert_graph(graph)
    assert len(edges) == len(EDGES) == len(line_graph) == 13
    k, coloring = chromatic_number(line_graph)

    # proper coloring <=> no two same-layer gates share a qubit
    assert all(coloring[a] != coloring[b] for a in line_graph for b in line_graph[a])
    max_deg = max(len(nbrs) for nbrs in graph.values())
    max_matching = N_NODES // 2
    lower = max(max_deg, -(-len(edges) // max_matching))
    print(f"Line graph C': {len(line_graph)} vertices, "
          f"{sum(map(len, line_graph.values())) // 2} edges")
    print(f"Lower bound: max degree {max_deg}, ceil(13 edges / {max_matching} per layer) = "
          f"{-(-len(edges) // max_matching)}  ->  depth >= {lower}")
    print(f"Chromatic number of C' (backtracking) = {k}")
    assert k == lower, "backtracking result disagrees with the counting lower bound"

    layers = draw_circuit(edges, coloring, N_NODES, args.out,
                          title=f"Minimum-depth cost layer: depth {k} (default order: 13)")
    for c, gates in enumerate(layers):
        print(f"  Layer {c + 1}: " + "  ".join(f"ZZ{u}{v}" for u, v in gates))
    depth = pennylane_depth(layers, N_NODES)
    print(f"PennyLane qml.specs depth of the layered circuit = {depth}")
    assert depth == k
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
