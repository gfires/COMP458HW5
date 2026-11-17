# Weighted Max-Cut with QAOA (PennyLane, 7 nodes)

`qaoa_maxcut.py` solves a 7-node, 13-edge weighted Max-Cut instance with QAOA on
PennyLane's `default.qubit` simulator. It brute-forces the ground truth, sweeps
depth p = 1..10, selects a depth, runs a 5000-shot final evaluation, and writes
everything to `outputs/results.json`. The schema is documented in the script's
docstring.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python qaoa_maxcut.py                      # threshold 0.90 (default)
.venv/bin/python qaoa_maxcut.py --ni-threshold 0.895 # alternative threshold
```

A run takes about 90 s on a laptop. Tested with PennyLane 0.45.1, Python 3.12.
The script asserts every validation item and exits with status 1, printing
`[FAIL] ...`, on the first one that fails.

## Method (fixed design)

- One qubit per node; node i is wire i, and wire 0 is the **leftmost** bit.
- H_C = sum over edges of 0.5 * (w/7.2) * (Z_i Z_j - I), built by hand with `qml.dot`.
  `qml.qaoa.maxcut` is not used because it ignores edge weights. diag(H_C) = -cut/7.2.
- Layers use `qml.qaoa.cost_layer`, `mixer_layer` and `x_mixer`. Parameters are
  packed as [gamma_1..gamma_p, beta_1..beta_p].
- The objective is the exact <H_C>, with backprop/autograd gradients, minimized by
  L-BFGS-B (maxiter=200, ftol=1e-9, gtol=1e-6). Bounds: gamma in [0, pi],
  beta in [-pi/4, pi/4].
- Initialization: p=1 starts from a 40x40 grid search. For p -> p+1, INTERP is
  applied to gamma and beta separately, then clipped to the bounds.
- Cuts are always scored with the original weights. NI = (<C> - 21.55) / (34.6 - 21.55).

## Results (exact statevector)

| p | E = <H_C> | NI | P_opt | iters | grad evals |
|---|---|---|---|---|---|
| 1 | -3.94223 | 0.524 | 0.085 | 4 | 6 |
| 2 | -4.25166 | 0.694 | 0.144 | 12 | 16 |
| 3 | -4.39219 | 0.772 | 0.218 | 16 | 18 |
| 4 | -4.49201 | 0.827 | 0.346 | 23 | 26 |
| 5 | -4.55912 | 0.864 | 0.479 | 25 | 31 |
| 6 | -4.61723 | **0.896** | 0.612 | 30 | 35 |
| 7 | -4.66236 | **0.921** | 0.693 | 34 | 39 |
| 8 | -4.69052 | 0.937 | 0.743 | 46 | 53 |
| 9 | -4.71721 | 0.951 | 0.805 | 45 | 50 |
| 10 | -4.74593 | 0.967 | 0.879 | 52 | 58 |

Uniform baseline: NI = 0, P_opt = 2/128 = 0.016. The optimal cut is 34.6, from
`0101001` / `1010110`, i.e. the partition {0,2,4,5} | {1,3,6}.

For p = 1..4, the best of 30 random-start runs matches the INTERP energy exactly.
The fraction of random runs that reach it drops from 0.57 at p=1 to 0.13–0.23 at
p=3–4, which is why INTERP is worth using.

Final evaluation (5000 shots, seed 459):

| selected p | exact <C> | empirical <C> ± SE | exact NI | emp. NI | exact P_opt | emp. P_opt |
|---|---|---|---|---|---|---|
| 7 (threshold 0.90) | 33.569 | 33.552 ± 0.030 | 0.921 | 0.920 | 0.693 | 0.686 |
| 6 (threshold 0.895) | 33.244 | 33.217 ± 0.035 | 0.896 | 0.894 | 0.612 | 0.605 |

The uniform 5000-shot sample also finds the 34.6 optimum. That is why "best
sampled cut" is printed but not used as a success metric.

## ⚠️ Issues for the author to reconcile

### 1. Depth selection: the rule gives p = 7, the write-up says p = 6

NI(p=6) = 0.8961 is below 0.90; it only rounds to 0.90. The rule "smallest p with
NI >= 0.90" therefore selects **p = 7** (NI = 0.921). With a threshold of 0.895 it
selects p = 6. The script prints both selections on every run and records them in
`config.selected_p_by_threshold`.

**Decision: keep the 0.90 threshold, so p = 7 is selected.** This applies the
stated rule to the exact values, without rounding. The write-up should say p = 7.
(`--ni-threshold 0.895` reproduces the p = 6 numbers if needed.)

### 2. The reference p=1 parameters were corrected

The original spec gave gamma ≈ 0.942, beta ≈ -0.341 at p = 1. The script finds
**gamma = 0.9538, beta = -0.3423**, and the gamma difference (0.012) failed the
5e-3 tolerance. The validation reference was changed to the verified optimum
(0.9538, -0.3423).

Evidence that the script's value is correct and the original reference was off:

| point | E | gradient |
|---|---|---|
| (0.9538, -0.3423) — this script | -3.942232 | (-1e-4, 5e-4), stationary |
| (0.942, -0.341) — reference | -3.942085 | (-0.021, 0.032), not stationary |

- A tighter polish (ftol=1e-15, gtol=1e-12) converges to (0.95385, -0.34233).
- The reference's own energy, -3.9422, matches this point. At gamma = 0.942 the
  energy rounds to -3.9421.
- All 30 random starts at p=1 agree on the same minimum.

The original gamma looks like a typo or a coarse-grid value. Every other
reference number (E, NI, P_opt at p = 1..10, NI at p=7) matches within tolerance.

### 3. Note: gamma schedules are not strictly monotone at p >= 8

At every p, gamma rises from about 0.5 to about 1.9, |beta| shrinks from about
0.55 to about 0.1, and no gamma reaches pi. At p = 8, 9 and 10, though, there is a
small dip in a middle layer (up to about 0.05; for example, p=10 has
...1.568, 1.534, 1.552...). The script prints this as `[NOTE]`. It checks
annealing-like behaviour by comparing the first and last layers, not strict
monotonicity.
