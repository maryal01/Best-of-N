"""
Synthetic test of the LEVEL LAYOUT: which way of cutting a continuous score into levels works when the peak of
Best-of-N sits at different places in the upper tail? Compared with the direct empirical method.

World (continuous score, described by the RANK QUANTILE u in (0,1) of an answer's score; u -> 1 is the top):
  m(u) = accuracy of an answer at quantile u. It rises linearly from M_LOW to M_PEAK up to u = 1 - t, then FALLS
  to M_TOP over the top fraction t (the over-optimized tail). The tail width t sets where the peak of Best-of-N is.
  Best-of-N with N answers keeps the one with the highest quantile, whose density is N u^(N-1), so exactly
      V(N) = integral of m(u) N u^(N-1) du      (computed on a fine grid)

Calibration data: n labeled answers (quantile u_i, correct c_i ~ Bernoulli(m(u_i))). Only the order of the answers
matters: we sort them by u and keep the correctness c_(1..n), exactly like the rank bins used on the real tables.

Methods (all end in N_hat = argmax; regret = V(N*) - V(N_hat) on the TRUE curve):
  empirical   resample the n answers with replacement, keep the top one: acc_hat(N) = sum_i c_(i) ((i/n)^N - ((i-1)/n)^N)
  levels      cut the ranks into K levels of given masses p_j (the layout), m_hat_j = mean correctness in level j,
              V_hat(N) = sum_j m_hat_j (F_j^N - F_{j-1}^N)

Run from the repo root:  python experiments/synthetic/layout_simulation.py     (needs numpy, scipy, matplotlib)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_data"))
from discretization_check import level_weights          # weights that turn sorted correctness into level means

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
M_LOW, M_PEAK, M_TOP = 0.2, 0.7, 0.3
TAIL_WIDTHS = {"wide tail (top 20%)": 0.20, "medium tail (top 5%)": 0.05, "narrow tail (top 1%)": 0.01}

LAYOUTS = {                                              # level masses p_j, lowest score first
    "K=3 equal": [1 / 3, 1 / 3, 1 / 3],
    "K=5 equal": [0.2] * 5,
    "K=3 top 5%": [0.50, 0.45, 0.05],
    "K=6 dyadic": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125],
}
EMPIRICAL = "Empirical (direct)"

N_MAX = 300
N_VALUES = np.arange(1, N_MAX + 1)
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000]           # number of labeled answers
N_TRIALS = 1000


# ---------------------------------------------------------------------------
# 1. The world and the exact Best-of-N curve
# ---------------------------------------------------------------------------
def accuracy_at_quantile(u, t):
    """m(u): rises to M_PEAK at u = 1 - t, then falls to M_TOP at u = 1."""
    rising = M_LOW + (M_PEAK - M_LOW) * u / (1 - t)
    falling = M_PEAK + (M_TOP - M_PEAK) * (u - (1 - t)) / t
    return np.where(u <= 1 - t, rising, falling)


def true_curve(t):
    """V(N) = integral of m(u) N u^(N-1) du, N = 1..N_MAX (trapezoid rule on a fine grid, vectorized over N)."""
    u = np.linspace(0, 1, 200_001)
    density = N_VALUES[:, None] * u[None, :] ** (N_VALUES[:, None] - 1)        # (N_MAX, grid)
    return np.trapezoid(density * accuracy_at_quantile(u, t)[None, :], u, axis=1)


def best_N(curves):
    """argmax over N of each row; ties go to the largest N (a flat end of an increasing curve is best at N_MAX)."""
    return N_MAX - np.argmax(curves[:, ::-1], axis=1)


# ---------------------------------------------------------------------------
# 2. Calibration data and the two kinds of estimators
# ---------------------------------------------------------------------------
def draw_sorted_correctness(t, n):
    """N_TRIALS calibration sets: n answers sorted by quantile, with their correctness (0/1). Shape (N_TRIALS, n)."""
    u = np.sort(rng.random((N_TRIALS, n)), axis=1)
    return (rng.random((N_TRIALS, n)) < accuracy_at_quantile(u, t)).astype(float)


def empirical_curves(c_sorted):
    """acc_hat(N) = sum_i c_(i) ((i/n)^N - ((i-1)/n)^N): the best of N answers drawn with replacement. Shape (trials, N_MAX)."""
    n = c_sorted.shape[1]
    i = np.arange(1, n + 1)[:, None]
    win_prob = (i / n) ** N_VALUES[None, :] - ((i - 1) / n) ** N_VALUES[None, :]       # (n, N_MAX)
    return c_sorted @ win_prob


def level_curves(c_sorted, p):
    """V_hat(N) = sum_j m_hat_j (F_j^N - F_{j-1}^N) with m_hat_j = mean correctness of the answers in level j."""
    p = np.asarray(p)
    m_hat = c_sorted @ level_weights(c_sorted.shape[1], p)                             # (trials, K)
    F = np.cumsum(p)
    F_prev = np.concatenate([[0.0], F[:-1]])
    win_prob = F[None, :] ** N_VALUES[:, None] - F_prev[None, :] ** N_VALUES[:, None]  # (N_MAX, K)
    return m_hat @ win_prob.T


# ---------------------------------------------------------------------------
# 3. Run: one world (tail width), one sample size -> regret of every method
# ---------------------------------------------------------------------------
def run_one_size(t, curve, n):
    c_sorted = draw_sorted_correctness(t, n)
    all_curves = {EMPIRICAL: empirical_curves(c_sorted)} | {name: level_curves(c_sorted, p) for name, p in LAYOUTS.items()}
    result = {}
    for name, estimated in all_curves.items():                   # one loop over the methods
        N_hat = best_N(estimated)
        result[name] = {"N_hat": N_hat, "regret": curve.max() - curve[N_hat - 1]}
    return result


def run_world(t):
    curve = true_curve(t)
    return {"curve": curve, "true_N": int(best_N(curve[None])[0]), "by_size": {n: run_one_size(t, curve, n) for n in SAMPLE_SIZES}}


worlds = {name: run_world(t) for name, t in TAIL_WIDTHS.items()}

# floor of each layout: regret of the best N of the layout's own model with unlimited data (levels filled with the true m_j)
def layout_floor(t, p):
    p = np.asarray(p)
    u = np.linspace(0, 1, 400_001)
    edges = np.concatenate([[0.0], np.cumsum(p)])
    level_of = np.minimum(np.searchsorted(edges, u, side="right") - 1, len(p) - 1)
    m = accuracy_at_quantile(u, t)
    m_j = np.bincount(level_of, weights=m) / np.bincount(level_of)                    # true mean accuracy per level
    F = np.cumsum(p)
    F_prev = np.concatenate([[0.0], F[:-1]])
    model = (F[None, :] ** N_VALUES[:, None] - F_prev[None, :] ** N_VALUES[:, None]) @ m_j
    curve = true_curve(t)
    return curve.max() - curve[int(best_N(model[None])[0]) - 1]


floors = {wname: {lname: layout_floor(t, p) for lname, p in LAYOUTS.items()} for wname, t in TAIL_WIDTHS.items()}

# ---------------------------------------------------------------------------
# 4. Print tables
# ---------------------------------------------------------------------------
methods = [EMPIRICAL] + list(LAYOUTS)
header = " | ".join(f"{'n=' + str(n):>7}" for n in SAMPLE_SIZES)
for wname, w in worlds.items():
    c = w["curve"]
    print(f"\n=== {wname}: true N* = {w['true_N']}  (V* = {c.max():.3f}, V(1) = {c[0]:.3f}, V({N_MAX}) = {c[-1]:.3f})")
    print(f"  {'method':<20} | {header} | {'floor':>6}")
    for m in methods:
        row = " | ".join(f"{w['by_size'][n][m]['regret'].mean():>7.4f}" for n in SAMPLE_SIZES)
        print(f"  {m:<20} | {row} | {floors[wname].get(m, 0.0):>6.4f}")

# ---------------------------------------------------------------------------
# 5. Plot: one panel per world
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(worlds), figsize=(5.5 * len(worlds), 4.3))
for ax, (wname, w) in zip(axes, worlds.items()):
    for m in methods:
        mean = [w["by_size"][n][m]["regret"].mean() for n in SAMPLE_SIZES]
        err = [w["by_size"][n][m]["regret"].std() / np.sqrt(N_TRIALS) for n in SAMPLE_SIZES]
        style = {"color": "black", "linestyle": "--"} if m == EMPIRICAL else {}
        ax.errorbar(SAMPLE_SIZES, mean, yerr=err, marker="o", capsize=3, label=m, **style)
    ax.set_xscale("log")
    ax.set_xlabel("calibration answers n")
    ax.set_ylabel("mean regret")
    ax.set_title(f"{wname}: N* = {w['true_N']}")
    ax.legend(fontsize=8)
plt.tight_layout()
figure_path = Path(__file__).resolve().parents[2] / "docs" / "layout_simulation.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
print(f"\nsaved {figure_path}")
