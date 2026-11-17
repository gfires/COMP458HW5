"""Weighted Max-Cut on a 7-node graph with QAOA (PennyLane default.qubit).

Run:  python qaoa_maxcut.py [--ni-threshold 0.90] [--out outputs/results.json]

Formulation: one qubit per node (node i -> wire i), z_i = 1 - 2 x_i.
H_C = sum_edges 0.5 * w~_ij (Z_i Z_j - I), w~ = w / 7.2, so diag(H_C) = -cut/7.2
and we minimize <H_C>. Circuit: H^{(x)7}, then p x [cost_layer(gamma_k), mixer_layer(beta_k)]
with the X mixer. Params packed [gamma_1..gamma_p, beta_1..beta_p]. Cuts are always
scored with ORIGINAL weights. Optimizer: L-BFGS-B on exact <H_C> with backprop
gradients, layerwise INTERP warm start (p=1 seeded by a 40x40 grid search).

Bitstring convention everywhere: node 0 / wire 0 is the LEFTMOST character, and the
basis-state index is int(bitstring, 2) (PennyLane's wire-0-most-significant order).

outputs/results.json schema (single file, all JSON-native types):
  instance:
    edges              list of {u, v, weight}            original weights
    norm_const         float                              7.2 (max weight)
    W                  float                              total weight
    mean_cut           float                              W/2 (uniform-distribution <C>)
    c_max              float                              brute-force optimum
    optimal_bitstrings list[str]                          all bitstrings achieving c_max
  config:
    seed, random_start_seed, shots_seed, uniform_seed   ints
    bounds             {gamma: [lo, hi], beta: [lo, hi]}
    optimizer          {method, maxiter, ftol, gtol}
    grid_size          [n_gamma, n_beta]
    n_random_starts, random_start_p_max                  ints
    shots              int
    ni_threshold       float     threshold actually used for selection
    selection_rule     str
    selected_p         int
    selected_p_by_threshold  {"0.9": int|null, "0.895": int|null}
  sweep: list over p=1..10 of
    {p, energy, NI, P_opt, iterations, grad_evals, gamma: list, beta: list}
    (energy = <H_C> with normalized weights; NI, P_opt exact from the statevector)
  random_start_check: list over p=1..4 of
    {p, interp_energy, best_random_energy, frac_within_1e-3, n_runs}
  p1_landscape:
    gamma_grid list[float], beta_grid list[float],
    energy     2D list, energy[i][j] = E(gamma_grid[i], beta_grid[j])
  final:
    p          int (selected p)
    gamma, beta  lists
    bitstrings list of 128 rows {bitstring, cut_value, is_optimal, exact_prob, shot_count}
    summary    {exact_mean_cut, empirical_mean_cut, empirical_std_err,
                exact_NI, empirical_NI, exact_P_opt, empirical_P_opt,
                best_sampled_cut, best_sampled_bitstring, shots}
  uniform_baseline:
    bitstrings same 128-row layout (exact_prob = 1/128, shot_count from a
               separate seeded 5000-shot uniform sample)
    summary    {exact_mean_cut, empirical_mean_cut, empirical_std_err,
                exact_P_opt, empirical_P_opt, best_sampled_cut, shots}

Histograms by cut value: group final.bitstrings / uniform_baseline.bitstrings by
cut_value and sum exact_prob or shot_count.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Instance and fixed configuration
# ----------------------------------------------------------------------------
N_NODES = 7
EDGES = [
    (0, 1, 6.0), (0, 4, 0.8), (0, 5, 1.0), (5, 6, 7.2), (1, 5, 3.6),
    (3, 5, 4.2), (4, 6, 5.4), (2, 4, 1.7), (2, 6, 2.9), (3, 6, 4.5),
    (1, 3, 0.5), (1, 2, 2.1), (2, 3, 3.2),
]
NORM = 7.2

SEED = 458
RANDOM_START_SEED = SEED
SHOTS_SEED = SEED + 1
UNIFORM_SEED = SEED + 2

GAMMA_BOUNDS = (0.0, np.pi)
BETA_BOUNDS = (-np.pi / 4, np.pi / 4)
OPT_OPTIONS = {"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6}
GRID_N = 40
P_MAX = 10
N_RANDOM = 30
RANDOM_P_MAX = 4
SHOTS = 5000
TOL = 5e-3

# Reference values (Validation item 5)
REFERENCE = {
    1: (-3.9422, 0.524, 0.085),
    2: (-4.2517, 0.694, 0.144),
    3: (-4.3922, 0.772, 0.218),
    4: (-4.4920, 0.827, 0.346),
    5: (-4.5591, 0.864, 0.479),
    6: (-4.6172, 0.896, 0.612),
    8: (-4.6905, 0.937, 0.743),
    10: (-4.7459, 0.967, 0.879),
}
REFERENCE_NI_P7 = 0.921
# Original spec said (0.942, -0.341), but that point is not stationary (|grad| ~ 0.03) and
# has E = -3.94208, above the reference energy -3.9422. Verified optimum: (0.95385, -0.34233).
REFERENCE_P1_PARAMS = (0.9538, -0.3423)


class ValidationError(RuntimeError):
    pass


def check(cond, msg):
    if not cond:
        raise ValidationError(msg)
    print(f"  [PASS] {msg}")


# ----------------------------------------------------------------------------
# Problem helpers
# ----------------------------------------------------------------------------
def bitstrings(n):
    """All n-bit strings in PennyLane basis-index order (wire 0 leftmost)."""
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def cut_value(bs, edges):
    return float(sum(w for u, v, w in edges if bs[u] != bs[v]))


def brute_force(n, edges):
    bss = bitstrings(n)
    cuts = np.array([cut_value(b, edges) for b in bss])
    c_max = cuts.max()
    opt = [b for b, c in zip(bss, cuts) if np.isclose(c, c_max, atol=1e-9)]
    return bss, cuts, float(c_max), opt


def build_cost_hamiltonian(edges, norm):
    """H_C = sum 0.5 * w~ (Z_i Z_j - I), built by hand (qml.qaoa.maxcut ignores weights)."""
    coeffs, ops = [], []
    for u, v, w in edges:
        wt = w / norm
        coeffs += [0.5 * wt, -0.5 * wt]
        ops += [qml.Z(u) @ qml.Z(v), qml.Identity(u)]
    return qml.dot(coeffs, ops)


class QAOA:
    """Exact-expectation QAOA circuits for a given graph."""

    def __init__(self, n, edges, norm):
        self.n = n
        self.edges = edges
        self.norm = norm
        self.H_C = build_cost_hamiltonian(edges, norm)
        self.H_M = qml.qaoa.x_mixer(range(n))
        self.dev = qml.device("default.qubit", wires=n)
        self.energy = qml.QNode(self._energy_circuit, self.dev,
                                diff_method="backprop", interface="autograd")
        self.probs = qml.QNode(self._probs_circuit, self.dev)
        self._grad = qml.grad(self.energy)
        self.n_evals = 0

    def _layers(self, params):
        p = len(params) // 2
        gammas, betas = params[:p], params[p:]
        for w in range(self.n):
            qml.Hadamard(wires=w)
        for k in range(p):
            qml.qaoa.cost_layer(gammas[k], self.H_C)
            qml.qaoa.mixer_layer(betas[k], self.H_M)

    def _energy_circuit(self, params):
        self._layers(params)
        return qml.expval(self.H_C)

    def _probs_circuit(self, params):
        self._layers(params)
        return qml.probs(wires=range(self.n))

    def energy_and_grad(self, x):
        """Value and gradient from one backprop pass (for scipy jac=True)."""
        self.n_evals += 1
        params = pnp.array(x, requires_grad=True)
        g = self._grad(params)
        return float(self._grad.forward), np.asarray(g, dtype=float)

    def energy_value(self, x):
        return float(self.energy(pnp.array(x, requires_grad=False)))

    def sample_circuit(self, params, shots, seed):
        dev = qml.device("default.qubit", wires=self.n, seed=seed)

        @qml.qnode(dev, shots=shots)
        def circ(prm):
            self._layers(prm)
            return qml.sample(wires=range(self.n))

        return np.asarray(circ(np.asarray(params)), dtype=int)


def bounds_for(p):
    return [GAMMA_BOUNDS] * p + [BETA_BOUNDS] * p


def run_lbfgsb(q, x0):
    p = len(x0) // 2
    q.n_evals = 0
    res = minimize(q.energy_and_grad, np.asarray(x0, dtype=float), jac=True,
                   method="L-BFGS-B", bounds=bounds_for(p), options=OPT_OPTIONS)
    return res, q.n_evals


def interp(a):
    """INTERP p -> p+1: a'_i = ((i-1)/p) A[i-1] + ((p-i+1)/p) A[i], A = [0, a, 0]."""
    p = len(a)
    A = np.concatenate([[0.0], a, [0.0]])
    return np.array([((i - 1) / p) * A[i - 1] + ((p - i + 1) / p) * A[i]
                     for i in range(1, p + 2)])


def metrics_from_probs(probs, cuts, opt_mask, mean_cut, c_max):
    exp_c = float(np.dot(probs, cuts))
    ni = (exp_c - mean_cut) / (c_max - mean_cut)
    p_opt = float(probs[opt_mask].sum())
    return exp_c, ni, p_opt


def grid_search_p1(q):
    g_grid = np.linspace(*GAMMA_BOUNDS, GRID_N)
    b_grid = np.linspace(*BETA_BOUNDS, GRID_N)
    E = np.array([[q.energy_value([g, b]) for b in b_grid] for g in g_grid])
    i, j = np.unravel_index(np.argmin(E), E.shape)
    return g_grid, b_grid, E, np.array([g_grid[i], b_grid[j]])


def select_p(sweep, threshold):
    for row in sweep:
        if row["NI"] >= threshold:
            return row["p"]
    return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ni-threshold", type=float, default=0.90,
                    help="select smallest p with NI >= threshold (default 0.90)")
    ap.add_argument("--out", default=os.path.join("outputs", "results.json"))
    args = ap.parse_args()
    np.random.seed(SEED)
    t0 = time.time()

    print(f"PennyLane {qml.__version__}; API check: "
          f"cost_layer={hasattr(qml.qaoa, 'cost_layer')}, "
          f"mixer_layer={hasattr(qml.qaoa, 'mixer_layer')}, "
          f"x_mixer={hasattr(qml.qaoa, 'x_mixer')}, dot={hasattr(qml, 'dot')}")

    # ---- Brute force ground truth (original weights) ----
    bss, cuts, c_max, opt_bs = brute_force(N_NODES, EDGES)
    W = float(sum(w for _, _, w in EDGES))
    mean_cut = float(cuts.mean())
    opt_mask = np.array([b in opt_bs for b in bss])

    print("\nValidation 1: instance / brute force")
    check(len(EDGES) == 13, f"13 edges (got {len(EDGES)})")
    check(abs(W - 43.1) < TOL, f"W = 43.1 (got {W:.6f})")
    check(abs(mean_cut - 21.55) < TOL and abs(mean_cut - W / 2) < 1e-9,
          f"mean cut = W/2 = 21.55 (got {mean_cut:.6f})")
    check(abs(c_max - 34.6) < TOL, f"C_max = 34.6 (got {c_max:.6f})")
    check(sorted(opt_bs) == ["0101001", "1010110"],
          f"optimal bitstrings = {{0101001, 1010110}} (got {opt_bs})")
    check(abs(max(w for *_, w in EDGES) - NORM) < 1e-12, "normalization = max weight = 7.2")

    q = QAOA(N_NODES, EDGES, NORM)

    print("\nValidation 2: H_C diagonal vs -(cut/7.2)")
    M = qml.matrix(q.H_C, wire_order=range(N_NODES))
    diag = np.real(np.diag(M))
    offdiag = np.abs(M - np.diag(np.diag(M))).max()
    check(np.max(np.abs(diag + cuts / NORM)) < 1e-9,
          f"diag(H_C) == -cut/7.2 for all 128 states (max err {np.max(np.abs(diag + cuts / NORM)):.1e})")
    check(offdiag < 1e-12, "H_C is diagonal")

    print("\nValidation 3: gamma=0 / beta=0 give the uniform energy (NI=0)")
    e_uniform = -mean_cut / NORM
    rng_v = np.random.default_rng(SEED)
    for p in (1, 3):
        g = rng_v.uniform(*GAMMA_BOUNDS, p)
        b = rng_v.uniform(*BETA_BOUNDS, p)
        for label, x in (("gamma=0", np.concatenate([np.zeros(p), b])),
                         ("beta=0", np.concatenate([g, np.zeros(p)]))):
            e = q.energy_value(x)
            check(abs(e - e_uniform) < 1e-9,
                  f"p={p}, {label}: E={e:.9f} == uniform {e_uniform:.9f}")

    print("\nValidation 4: single-edge graph, p=1 reaches <C> = w (NI = 1)")
    w_e = 2.5
    q2 = QAOA(2, [(0, 1, w_e)], w_e)
    _, _, _, x0 = grid_search_p1(q2)
    res2, _ = run_lbfgsb(q2, x0)
    c2 = -res2.fun * w_e
    ni2 = (c2 - w_e / 2) / (w_e - w_e / 2)
    check(abs(c2 - w_e) < TOL and abs(ni2 - 1) < TOL,
          f"<C> = {c2:.6f} (target {w_e}), NI = {ni2:.6f}, "
          f"gamma={res2.x[0]:.4f}, beta={res2.x[1]:.4f}")

    # ---- p = 1..10 sweep with INTERP ----
    print("\nOptimization sweep (INTERP warm start, L-BFGS-B)")
    g_grid, b_grid, E_grid, x0 = grid_search_p1(q)
    print(f"  p=1 grid seed: gamma={x0[0]:.4f}, beta={x0[1]:.4f}, E={E_grid.min():.5f}")
    sweep, gam, bet = [], None, None
    for p in range(1, P_MAX + 1):
        if p > 1:
            g0 = np.clip(interp(gam), *GAMMA_BOUNDS)
            b0 = np.clip(interp(bet), *BETA_BOUNDS)
            x0 = np.concatenate([g0, b0])
        res, nev = run_lbfgsb(q, x0)
        gam, bet = res.x[:p].copy(), res.x[p:].copy()
        probs = np.asarray(q.probs(res.x))
        exp_c, ni, p_opt = metrics_from_probs(probs, cuts, opt_mask, mean_cut, c_max)
        if abs(-exp_c / NORM - res.fun) > 1e-9:
            raise ValidationError(f"p={p}: energy {res.fun} inconsistent with probs-based <C> {exp_c}")
        sweep.append({"p": p, "energy": float(res.fun), "NI": float(ni), "P_opt": p_opt,
                      "iterations": int(res.nit), "grad_evals": int(nev),
                      "gamma": gam.tolist(), "beta": bet.tolist(),
                      "_status": res.message if isinstance(res.message, str) else str(res.message)})
        print(f"  p={p:2d}  E={res.fun:.5f}  NI={ni:.4f}  P_opt={p_opt:.4f}  "
              f"it={res.nit:3d}  evals={nev:3d}  ({time.time() - t0:.0f}s)  {sweep[-1]['_status']}")

    # ---- Random-start cross-check ----
    print(f"\nRandom-start cross-check ({N_RANDOM} runs per p, seed {RANDOM_START_SEED})")
    rng = np.random.default_rng(RANDOM_START_SEED)
    rs_check = []
    for p in range(1, RANDOM_P_MAX + 1):
        lo = np.array([b[0] for b in bounds_for(p)])
        hi = np.array([b[1] for b in bounds_for(p)])
        energies = np.array([run_lbfgsb(q, rng.uniform(lo, hi))[0].fun for _ in range(N_RANDOM)])
        best = float(min(energies.min(), sweep[p - 1]["energy"]))
        frac = float(np.mean(energies <= best + 1e-3))
        rs_check.append({"p": p, "interp_energy": sweep[p - 1]["energy"],
                         "best_random_energy": float(energies.min()),
                         "frac_within_1e-3": frac, "n_runs": N_RANDOM})
        print(f"  p={p}: INTERP E={sweep[p - 1]['energy']:.6f}  best random E={energies.min():.6f}  "
              f"frac within 1e-3 of best={frac:.2f}  ({time.time() - t0:.0f}s)")

    # ---- Validation 5 & 6 ----
    print("\nValidation 5: sweep vs reference values")
    for p, (e_ref, ni_ref, popt_ref) in REFERENCE.items():
        r = sweep[p - 1]
        check(abs(r["energy"] - e_ref) < TOL and abs(r["NI"] - ni_ref) < TOL
              and abs(r["P_opt"] - popt_ref) < TOL,
              f"p={p}: E={r['energy']:.4f} (ref {e_ref}), NI={r['NI']:.3f} (ref {ni_ref}), "
              f"P_opt={r['P_opt']:.3f} (ref {popt_ref})")
    check(abs(sweep[6]["NI"] - REFERENCE_NI_P7) < TOL,
          f"p=7: NI={sweep[6]['NI']:.3f} (ref {REFERENCE_NI_P7})")
    check(abs(sweep[0]["gamma"][0] - REFERENCE_P1_PARAMS[0]) < TOL
          and abs(sweep[0]["beta"][0] - REFERENCE_P1_PARAMS[1]) < TOL,
          f"p=1 params gamma={sweep[0]['gamma'][0]:.4f}, beta={sweep[0]['beta'][0]:.4f} "
          f"(ref {REFERENCE_P1_PARAMS})")
    es = [r["energy"] for r in sweep]
    check(all(es[i + 1] <= es[i] + 1e-9 for i in range(len(es) - 1)),
          "energy non-increasing in p")
    for r in rs_check:
        check(r["interp_energy"] <= r["best_random_energy"] + 1e-4,
              f"p={r['p']}: INTERP matches best of {N_RANDOM} random starts "
              f"({r['interp_energy']:.6f} vs {r['best_random_energy']:.6f})")

    print("\nValidation 6: parameter schedules look annealing-like")
    all_g = np.concatenate([r["gamma"] for r in sweep])
    check(np.all(all_g < np.pi - 1e-6), f"no gamma at upper bound pi (max gamma {all_g.max():.4f})")
    for r in sweep[1:]:
        g, b = np.array(r["gamma"]), np.array(r["beta"])
        check(g[-1] > g[0] and abs(b[-1]) < abs(b[0]),
              f"p={r['p']}: gamma_1={g[0]:.3f} -> gamma_p={g[-1]:.3f}, "
              f"|beta_1|={abs(b[0]):.3f} -> |beta_p|={abs(b[-1]):.3f}")
        mono = bool(np.all(np.diff(g) >= -1e-3))
        if not mono:
            print(f"  [NOTE] p={r['p']}: gamma not strictly monotone: {np.round(g, 3).tolist()}")

    # ---- Depth selection ----
    sel_090 = select_p(sweep, 0.90)
    sel_0895 = select_p(sweep, 0.895)
    p_sel = select_p(sweep, args.ni_threshold)
    print("\nDepth selection (smallest p with NI >= threshold)")
    print(f"  threshold 0.90  -> p = {sel_090}")
    print(f"  threshold 0.895 -> p = {sel_0895}")
    print(f"  NI(p=6) = {sweep[5]['NI']:.4f}, NI(p=7) = {sweep[6]['NI']:.4f}")
    if p_sel is None:
        raise ValidationError(f"no p in 1..{P_MAX} reaches NI >= {args.ni_threshold}")
    print(f"  USING threshold {args.ni_threshold} -> selected p = {p_sel}")
    if sel_090 != sel_0895:
        print("  *** NOTE: selection is threshold-sensitive; write-up states p=6, rule at 0.90 gives "
              f"p={sel_090}. Reconcile the write-up. ***")

    # ---- Final evaluation at selected p ----
    r = sweep[p_sel - 1]
    x_sel = np.array(r["gamma"] + r["beta"])
    probs = np.asarray(q.probs(x_sel))
    exact_c, exact_ni, exact_popt = metrics_from_probs(probs, cuts, opt_mask, mean_cut, c_max)
    samples = q.sample_circuit(x_sel, SHOTS, SHOTS_SEED)
    assert samples.shape == (SHOTS, N_NODES)
    sample_bs = ["".join(map(str, s)) for s in samples]   # column k = wire k = node k
    sample_cuts = np.array([cut_value(s, EDGES) for s in sample_bs])
    idx = np.array([int(s, 2) for s in sample_bs])
    counts = np.bincount(idx, minlength=2 ** N_NODES)
    emp_c = float(sample_cuts.mean())
    emp_se = float(sample_cuts.std(ddof=1) / np.sqrt(SHOTS))
    emp_ni = (emp_c - mean_cut) / (c_max - mean_cut)
    emp_popt = float(counts[opt_mask].sum() / SHOTS)
    best_i = int(np.argmax(sample_cuts))
    best_cut, best_bs = float(sample_cuts[best_i]), sample_bs[best_i]
    # sanity: sampled bitstrings consistent with exact distribution (wire ordering)
    popt_sigma = np.sqrt(exact_popt * (1 - exact_popt) / SHOTS)
    if abs(emp_popt - exact_popt) > 5 * popt_sigma or abs(emp_c - exact_c) > 5 * emp_se:
        raise ValidationError("sampled distribution inconsistent with exact distribution "
                              "(check bit ordering)")

    final_rows = [{"bitstring": b, "cut_value": float(c), "is_optimal": bool(m),
                   "exact_prob": float(pr), "shot_count": int(n)}
                  for b, c, m, pr, n in zip(bss, cuts, opt_mask, probs, counts)]

    # ---- Uniform baseline ----
    urng = np.random.default_rng(UNIFORM_SEED)
    u_idx = urng.integers(0, 2 ** N_NODES, SHOTS)
    u_counts = np.bincount(u_idx, minlength=2 ** N_NODES)
    u_cuts = cuts[u_idx]
    u_rows = [{"bitstring": b, "cut_value": float(c), "is_optimal": bool(m),
               "exact_prob": 1.0 / 2 ** N_NODES, "shot_count": int(n)}
              for b, c, m, n in zip(bss, cuts, opt_mask, u_counts)]
    u_summary = {"exact_mean_cut": mean_cut, "empirical_mean_cut": float(u_cuts.mean()),
                 "empirical_std_err": float(u_cuts.std(ddof=1) / np.sqrt(SHOTS)),
                 "exact_P_opt": 2 / 2 ** N_NODES,
                 "empirical_P_opt": float(u_counts[opt_mask].sum() / SHOTS),
                 "best_sampled_cut": float(u_cuts.max()), "shots": SHOTS}

    print(f"\nFinal evaluation at p = {p_sel} ({SHOTS} shots, seed {SHOTS_SEED})")
    print(f"  {'':22s}{'exact':>10s}{'empirical':>12s}{'uniform':>10s}")
    print(f"  {'<C> (orig. weights)':22s}{exact_c:10.3f}{emp_c:12.3f}{mean_cut:10.3f}")
    print(f"  {'std. error of <C>':22s}{'':10s}{emp_se:12.3f}")
    print(f"  {'NI':22s}{exact_ni:10.3f}{emp_ni:12.3f}{0.0:10.3f}")
    print(f"  {'P_opt':22s}{exact_popt:10.3f}{emp_popt:12.3f}{2 / 128:10.3f}")
    print(f"  best sampled cut: {best_cut} with {best_bs} "
          f"(S={{{','.join(str(i) for i, c in enumerate(best_bs) if c == '0')}}} | "
          f"{{{','.join(str(i) for i, c in enumerate(best_bs) if c == '1')}}})  "
          "[not a success metric]")
    for b in opt_bs:
        print(f"  brute-force optimum: {b}  cut={c_max}  partition "
              f"{{{','.join(str(i) for i, c in enumerate(b) if c == '0')}}} | "
              f"{{{','.join(str(i) for i, c in enumerate(b) if c == '1')}}}")

    # ---- Write results ----
    for row in sweep:
        row.pop("_status", None)
    results = {
        "instance": {
            "edges": [{"u": u, "v": v, "weight": w} for u, v, w in EDGES],
            "norm_const": NORM, "W": W, "mean_cut": mean_cut, "c_max": c_max,
            "optimal_bitstrings": opt_bs,
        },
        "config": {
            "seed": SEED, "random_start_seed": RANDOM_START_SEED,
            "shots_seed": SHOTS_SEED, "uniform_seed": UNIFORM_SEED,
            "bounds": {"gamma": list(GAMMA_BOUNDS), "beta": list(BETA_BOUNDS)},
            "optimizer": {"method": "L-BFGS-B", **OPT_OPTIONS},
            "grid_size": [GRID_N, GRID_N],
            "n_random_starts": N_RANDOM, "random_start_p_max": RANDOM_P_MAX,
            "shots": SHOTS, "ni_threshold": args.ni_threshold,
            "selection_rule": "smallest p in 1..10 with exact NI >= ni_threshold",
            "selected_p": p_sel,
            "selected_p_by_threshold": {"0.9": sel_090, "0.895": sel_0895},
            "pennylane_version": qml.__version__,
        },
        "sweep": sweep,
        "random_start_check": rs_check,
        "p1_landscape": {"gamma_grid": g_grid.tolist(), "beta_grid": b_grid.tolist(),
                         "energy": E_grid.tolist()},
        "final": {
            "p": p_sel, "gamma": r["gamma"], "beta": r["beta"],
            "bitstrings": final_rows,
            "summary": {"exact_mean_cut": exact_c, "empirical_mean_cut": emp_c,
                        "empirical_std_err": emp_se, "exact_NI": exact_ni,
                        "empirical_NI": emp_ni, "exact_P_opt": exact_popt,
                        "empirical_P_opt": emp_popt, "best_sampled_cut": best_cut,
                        "best_sampled_bitstring": best_bs, "shots": SHOTS},
        },
        "uniform_baseline": {"bitstrings": u_rows, "summary": u_summary},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {args.out}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        print(f"\n[FAIL] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
