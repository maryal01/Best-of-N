"""
Synthetic simulation for K-level verifiers (K = 3, 5, 10, 20): how do the number of levels and the amount of
calibration data affect the regret of the estimated best N?

Generalization of simulation.py (which has K = 3 only).

World with K levels (index 0 .. K-1, lowest score first):
  p_j : the top level is rare (P_TOP); the other K-1 levels share the rest equally.
  m_j : rises linearly from M_LOW to M_PEAK over levels 0 .. K-2, then DROPS at the top level (M_TOP < M_PEAK).
        For K = 3 this is exactly m = (0.2, 0.7, 0.3). The top level is the "misleading" one.
  U | level j ~ Bernoulli(m_j).

Part A: the estimator knows the K levels. Regret vs calibration size n, for each K.
        (Note: with discrete levels the direct empirical method "resample the n samples, keep the top one"
        has the same expectation as the plug-in formula, so there is no separate baseline here. That comparison
        needs continuous scores and is done on real data in experiments/real_data/calibration_sweep.py.)
Part B: the world has K = 20 levels but the estimator only sees COARSER levels (adjacent levels merged into
        K_coarse groups). Fewer parameters to estimate, but merging hides the misleading top level: a bias
        that more data cannot remove. This is the resolution-vs-estimation tradeoff.

Regret = V(N*) - V(N_hat), always on the TRUE K-level curve.
Run from the repo root:  python experiments/synthetic/k_level_simulation.py   (needs numpy, matplotlib)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
P_TOP = 0.03                       # probability of the top (misleading) level
M_LOW, M_PEAK, M_TOP = 0.2, 0.7, 0.3
K_VALUES = [3, 5, 10, 20]          # Part A: true number of levels
K_TRUE_B = 20                      # Part B: true number of levels
K_COARSE = [3, 5, 10, 20]          # Part B: number of merged groups used for estimation

N_MAX = 300                        # we look for the best N in 1..N_MAX
N_VALUES = np.arange(1, N_MAX + 1)
SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000]
N_TRIALS = 1000                    # random calibration sets per setting


# ---------------------------------------------------------------------------
# 1. The world and the exact Best-of-N curve
# ---------------------------------------------------------------------------
def make_world(K):
    """Level probabilities p (K,) and level mean utilities m (K,)."""
    p = np.full(K, (1 - P_TOP) / (K - 1))
    p[-1] = P_TOP                                           # rare top level
    m = np.empty(K)
    m[:-1] = np.linspace(M_LOW, M_PEAK, K - 1)              # rises over the lower K-1 levels
    m[-1] = M_TOP                                           # drops at the top level
    return p, m


def expected_utility(p, m):
    """
    V(N) = sum_j m_j (F_j^N - F_{j-1}^N) for N = 1..N_MAX, where F_j = p_1 + ... + p_j.
    (F_j^N - F_{j-1}^N is the probability that the best of N answers is at level j.)
    p and m have shape (..., K), so a whole batch of estimated worlds is handled at once. Output: (..., N_MAX).
    """
    F = np.minimum(np.cumsum(p, axis=-1), 1.0)
    F[..., -1] = 1.0                                        # guard against rounding above 1
    F_prev = np.concatenate([np.zeros(F.shape[:-1] + (1,)), F[..., :-1]], axis=-1)

    # (..., N, K): probability that the winner of N answers is at each level
    win_prob = F[..., None, :] ** N_VALUES[:, None] - F_prev[..., None, :] ** N_VALUES[:, None]
    return (win_prob * m[..., None, :]).sum(axis=-1)


# ---------------------------------------------------------------------------
# 2. Calibration data and plug-in estimation
# ---------------------------------------------------------------------------
def draw_calibration_sets(p, m, n):
    """N_TRIALS datasets of n labeled responses: level (0..K-1) and utility (0 or 1). Shapes (N_TRIALS, n)."""
    level = rng.choice(len(p), size=(N_TRIALS, n), p=p)
    utility = (rng.random((N_TRIALS, n)) < m[level]).astype(float)
    return level, utility


def estimate_parameters(level, utility, K_est):
    """
    Plug-in estimates over K_est levels: p_hat_j = fraction of responses at level j,
    m_hat_j = their average utility (0 if the level never appears: it is never selected, so it drops out).
    Returns p_hat, m_hat of shape (N_TRIALS, K_est).
    """
    trials, n = level.shape
    flat = (np.arange(trials)[:, None] * K_est + level).ravel()           # one bin per (trial, level)
    count = np.bincount(flat, minlength=trials * K_est).reshape(trials, K_est)
    utility_sum = np.bincount(flat, weights=utility.ravel(), minlength=trials * K_est).reshape(trials, K_est)
    return count / n, utility_sum / np.maximum(count, 1)


def best_N(curves):
    """
    argmax over N of each curve (rows). Ties go to the LARGEST N: an increasing curve that has become flat
    (a probability underflowed to 0) is best at N_MAX, and plain argmax would stop at the start of the flat part.
    """
    return N_MAX - np.argmax(curves[:, ::-1], axis=1)


def run_setting(p_true, m_true, true_curve, n, K_est, group_of_level):
    """
    Estimate N* from n samples using K_est (possibly merged) levels; return N_hat and regret for each trial.
    group_of_level maps each true level to the level the estimator sees (identity in Part A).
    """
    level, utility = draw_calibration_sets(p_true, m_true, n)
    p_hat, m_hat = estimate_parameters(group_of_level[level], utility, K_est)
    N_hat = best_N(expected_utility(p_hat, m_hat))
    return {"N_hat": N_hat, "regret": true_curve.max() - true_curve[N_hat - 1]}


def merge_levels(K, K_coarse):
    """Map each of K levels to one of K_coarse groups of adjacent levels."""
    return np.arange(K) * K_coarse // K


def coarse_world(p, m, group):
    """Exact p and m of the merged levels (what the estimator converges to with unlimited data)."""
    p_c = np.bincount(group, weights=p)
    return p_c, np.bincount(group, weights=p * m) / p_c


# ---------------------------------------------------------------------------
# 3. Part A: true K varies;  Part B: true K = 20, estimator merges levels
# ---------------------------------------------------------------------------
def part_a(K):
    p, m = make_world(K)
    true_curve = expected_utility(p, m)
    by_size = {n: run_setting(p, m, true_curve, n, K, np.arange(K)) for n in SAMPLE_SIZES}
    return {"true_curve": true_curve, "true_N": int(best_N(true_curve[None])[0]), "by_size": by_size}


def part_b(K_coarse):
    p, m = make_world(K_TRUE_B)
    true_curve = expected_utility(p, m)
    group = merge_levels(K_TRUE_B, K_coarse)
    by_size = {n: run_setting(p, m, true_curve, n, K_coarse, group) for n in SAMPLE_SIZES}

    # floor: regret of the best N of the merged world (unlimited calibration data)
    p_c, m_c = coarse_world(p, m, group)
    N_floor = int(best_N(expected_utility(p_c, m_c)[None])[0])
    return {"by_size": by_size, "floor": true_curve.max() - true_curve[N_floor - 1], "N_floor": N_floor}


results_a = {K: part_a(K) for K in K_VALUES}
results_b = {Kc: part_b(Kc) for Kc in K_COARSE}


# ---------------------------------------------------------------------------
# 4. Print tables
# ---------------------------------------------------------------------------
def regret_line(by_size):
    return " | ".join(f"{by_size[n]['regret'].mean():>7.4f}" for n in SAMPLE_SIZES)


header = " | ".join(f"{'n=' + str(n):>7}" for n in SAMPLE_SIZES)
print(f"Setup: top level p = {P_TOP}, m rises {M_LOW} -> {M_PEAK} then drops to {M_TOP}; {N_TRIALS} trials per cell.")

print("\nPart A: true K levels, estimator knows them. Mean regret:")
print(f"  {'K':>3} | {'N*':>3} | {'V(N*)':>6} | {'V(1)':>6} | {header}")
for K, r in results_a.items():
    c = r["true_curve"]
    print(f"  {K:>3} | {r['true_N']:>3} | {c.max():>6.3f} | {c[0]:>6.3f} | {regret_line(r['by_size'])}")

print("\nPart A: median N_hat (true N* in the second column):")
for K, r in results_a.items():
    print(f"  K={K:>2} | {r['true_N']:>3} | " + " | ".join(f"{np.median(r['by_size'][n]['N_hat']):>7.0f}" for n in SAMPLE_SIZES))

print(f"\nPart B: true K = {K_TRUE_B}, estimator merges levels into K_coarse groups. Mean regret:")
print(f"  {'K_coarse':>8} | {'floor':>6} | {'N_floor':>7} | {header}")
for Kc, r in results_b.items():
    print(f"  {Kc:>8} | {r['floor']:>6.4f} | {r['N_floor']:>7} | {regret_line(r['by_size'])}")


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

for K, r in results_a.items():                              # (a) true curves
    ax1.plot(N_VALUES, r["true_curve"], lw=2, label=f"K={K}  (N*={r['true_N']})")
ax1.set_xscale("log")
ax1.set_xlabel("N")
ax1.set_ylabel("true E[U of the kept response]")
ax1.set_title("True Best-of-N curves")
ax1.legend()

for K, r in results_a.items():                              # (b) Part A regret vs n
    mean = [r["by_size"][n]["regret"].mean() for n in SAMPLE_SIZES]
    err = [r["by_size"][n]["regret"].std() / np.sqrt(N_TRIALS) for n in SAMPLE_SIZES]
    ax2.errorbar(SAMPLE_SIZES, mean, yerr=err, marker="o", capsize=3, label=f"K={K}")
ax2.set_xscale("log")
ax2.set_xlabel("calibration samples n")
ax2.set_ylabel("mean regret")
ax2.set_title("A: regret vs n, estimator knows the K levels")
ax2.legend()

for Kc, r in results_b.items():                             # (c) Part B regret vs n, with floors
    mean = [r["by_size"][n]["regret"].mean() for n in SAMPLE_SIZES]
    err = [r["by_size"][n]["regret"].std() / np.sqrt(N_TRIALS) for n in SAMPLE_SIZES]
    line = ax3.errorbar(SAMPLE_SIZES, mean, yerr=err, marker="o", capsize=3, label=f"merged into {Kc}")
    ax3.axhline(r["floor"], ls=":", color=line[0].get_color(), alpha=0.7)
ax3.set_xscale("log")
ax3.set_xlabel("calibration samples n")
ax3.set_ylabel("mean regret")
ax3.set_title(f"B: true K={K_TRUE_B}, estimator merges levels (dotted: floor)")
ax3.legend()

plt.tight_layout()
figure_path = Path(__file__).resolve().parents[2] / "docs" / "k_level_simulation.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
print(f"\nsaved {figure_path}")
