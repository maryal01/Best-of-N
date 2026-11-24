"""
Finite-data synthetic simulation for the discrete 3-level Best-of-N model.

Model (from the proposal):
  - verifier score S takes 3 levels s1 < s2 < s3,   p_j = Pr(S = s_j)
  - true utility U is bounded,                      m_j = E[U | S = s_j]
  - Best-of-N keeps the candidate with the highest S.

For each parameter regime (how rare the top level s3 is) we:
1. Compute the TRUE expected utility V(N) = E[U_{I_N}] and the true best N*.
2. Draw n labeled calibration samples (S, U) for n = 50, 100, 500, 1000,
   estimate p_j, m_j, plug them into the same formula, get N*_hat.
3. Regret = V(N*) - V(N*_hat), where V is always the TRUE curve.

Run:  python experiments/synthetic/simulation.py        (needs numpy, matplotlib)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Index 0,1,2 <-> levels s1,s2,s3.  p_j = Pr(S = s_j),  m_j = E[U | S = s_j].
# All regimes are one-tail-crossover: m1 < m2 and m3 < m2 (the top level is over-optimized).
# They differ only in how rare the top level s3 is.
M = np.array([0.20, 0.70, 0.30])
REGIMES = {
    "top level p3 = 0.10": np.array([0.50, 0.40, 0.10]),
    "top level p3 = 0.03": np.array([0.50, 0.47, 0.03]),
    "top level p3 = 0.01": np.array([0.50, 0.49, 0.01]),
}

N_MAX = 300                        # we look for the best N in 1..N_MAX
N_VALUES = np.arange(1, N_MAX + 1)
SAMPLE_SIZES = [50, 100, 500, 1000]
N_TRIALS = 1000                    # random calibration sets per sample size


# ---------------------------------------------------------------------------
# 1. Exact expected utility of Best-of-N
# ---------------------------------------------------------------------------
def expected_utility(p, m):
    """
    V(N) = E[U_{I_N}] for every N in N_VALUES.

    Let a = Pr(S <= s2) = p1 + p2   and   b = Pr(S <= s1) = p1.
    The best of N scores is <= s2 with probability a^N, and <= s1 with probability b^N, so
        Pr(winner level = 3) = 1 - a^N
        Pr(winner level = 2) = a^N - b^N
        Pr(winner level = 1) = b^N
    Weighting m_j by these probabilities gives the closed form
        V(N) = m3 + (m2 - m3) a^N + (m1 - m2) b^N

    p and m have shape (..., 3), so one call can handle a single parameter set
    or a whole batch of estimated ones. Output shape: (..., N_MAX).
    """
    a = (p[..., 0] + p[..., 1])[..., None]   # shape (..., 1) so it broadcasts against N
    b = p[..., 0][..., None]
    m1, m2, m3 = (m[..., j][..., None] for j in range(3))

    return m3 + (m2 - m3) * a ** N_VALUES + (m1 - m2) * b ** N_VALUES


# ---------------------------------------------------------------------------
# 2. Draw calibration data and estimate p_j, m_j
# ---------------------------------------------------------------------------
def draw_calibration_sets(P, n):
    """
    N_TRIALS independent calibration sets, each with n labeled responses (S, U).
    Returns two arrays of shape (N_TRIALS, n):
      level[t, i]   which verifier level (0, 1 or 2) response i got
      utility[t, i] its true utility, 0 or 1 (Bernoulli(m_j), so U is bounded)
    """
    level = rng.choice(3, size=(N_TRIALS, n), p=P)
    utility = (rng.random((N_TRIALS, n)) < M[level]).astype(float)
    return level, utility


def estimate_parameters(level, utility):
    """
    p_hat_j = fraction of responses at level j
    m_hat_j = average utility of the responses at level j
    Returns p_hat, m_hat (shape (N_TRIALS, 3)) and count per level.
    """
    n = level.shape[1]

    # one-hot: is_level[t, i, j] = True if response i of trial t is at level j
    is_level = level[:, :, None] == np.arange(3)

    count = is_level.sum(axis=1)                         # (trials, 3) responses per level
    utility_sum = (is_level * utility[:, :, None]).sum(axis=1)

    p_hat = count / n
    # If a level never appears (can happen for the rare top level) its mean is undefined.
    # We use 0 there: this is harmless, because a level with p_hat = 0 never gets
    # selected, so m_hat for it drops out of the formula.
    m_hat = utility_sum / np.maximum(count, 1)
    return p_hat, m_hat, count


# ---------------------------------------------------------------------------
# 3. One regime, one sample size  ->  estimated N and regret
# ---------------------------------------------------------------------------
def run_one_sample_size(P, true_curve, n):
    """Estimate N* on many random calibration sets of size n and measure the loss."""
    level, utility = draw_calibration_sets(P, n)
    p_hat, m_hat, count = estimate_parameters(level, utility)

    # plug the estimates into the same formula, take the best N for each trial
    estimated_curves = expected_utility(p_hat, m_hat)            # (trials, N_MAX)
    best_N_hat = np.argmax(estimated_curves, axis=1) + 1         # (trials,)

    # Fix for an unseen top level. If s3 never appears, p_hat3 = 0, so the estimated curve is
    #   V(N) = m2 + (m1 - m2) * b^N,   which is INCREASING in N whenever m_hat2 > m_hat1.
    # Its best N inside the search range is N_MAX. But b^N underflows to exactly 0 for large N,
    # the curve becomes flat, and argmax returns the FIRST point of the flat part (N ~ 55)
    # instead of N_MAX. So we set N_MAX by hand in that case.
    no_top = count[:, 2] == 0
    increasing = m_hat[:, 1] > m_hat[:, 0]
    best_N_hat[no_top & increasing] = N_MAX

    # Regret: TRUE value at N* minus TRUE value at N_hat
    regret = true_curve.max() - true_curve[best_N_hat - 1]       # (trials,)

    top_level_missing = (count[:, 2] == 0).mean()                # how often we never saw s3
    p3_error = np.abs(p_hat[:, 2] - P[2]).mean()
    return {"N_hat": best_N_hat, "regret": regret, "missing": top_level_missing, "p3_error": p3_error}


def run_regime(P):
    """Run all sample sizes for one parameter regime."""
    true_curve = expected_utility(P, M)
    by_size = {n: run_one_sample_size(P, true_curve, n) for n in SAMPLE_SIZES}
    return {"true_curve": true_curve, "true_N": int(np.argmax(true_curve)) + 1, "by_size": by_size}


results = {name: run_regime(P) for name, P in REGIMES.items()}


# ---------------------------------------------------------------------------
# 4. Print tables
# ---------------------------------------------------------------------------
for name, res in results.items():
    curve = res["true_curve"]
    print(f"=== {name}:  p = {REGIMES[name]},  true N* = {res['true_N']}"
          f"  (V = {curve.max():.4f};  N=1: {curve[0]:.4f},  N={N_MAX}: {curve[-1]:.4f})")
    print(f"{'n':>6} | {'mean N_hat':>10} | {'median N_hat':>12} | {'mean regret':>11} | {'std err':>8}"
          f" | {'|p3_hat-p3|':>11} | {'s3 unseen':>9}")
    for n, r in res["by_size"].items():
        std_err = r["regret"].std() / np.sqrt(N_TRIALS)
        print(f"{n:>6} | {r['N_hat'].mean():>10.1f} | {np.median(r['N_hat']):>12.0f}"
              f" | {r['regret'].mean():>11.4f} | {std_err:>8.4f}"
              f" | {r['p3_error']:>11.4f} | {r['missing']:>9.1%}")
    print()


# ---------------------------------------------------------------------------
# 5. Plots: one row per regime, 3 panels each
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(REGIMES), 3, figsize=(16, 4 * len(REGIMES)))

for row, (name, res) in zip(axes, results.items()):
    ax1, ax2, ax3 = row
    by_size = res["by_size"]

    # (a) the true curve
    ax1.plot(N_VALUES, res["true_curve"], lw=2)
    ax1.axvline(res["true_N"], color="red", ls="--", label=f"N* = {res['true_N']}")
    ax1.set_xscale("log")
    ax1.set_xlabel("N")
    ax1.set_ylabel("true E[U of the kept response]")
    ax1.set_title(f"True Best-of-N curve ({name})")
    ax1.legend()

    # (b) distribution of the estimated N for each sample size
    ax2.boxplot([by_size[n]["N_hat"] for n in SAMPLE_SIZES], tick_labels=SAMPLE_SIZES)
    ax2.axhline(res["true_N"], color="red", ls="--", label="true N*")
    ax2.set_yscale("log")
    ax2.set_xlabel("calibration samples n")
    ax2.set_ylabel("estimated N_hat")
    ax2.set_title("Estimated N vs. data size")
    ax2.legend()

    # (c) regret vs. sample size (mean +- standard error)
    mean_regret = [by_size[n]["regret"].mean() for n in SAMPLE_SIZES]
    std_err = [by_size[n]["regret"].std() / np.sqrt(N_TRIALS) for n in SAMPLE_SIZES]
    ax3.errorbar(SAMPLE_SIZES, mean_regret, yerr=std_err, marker="o", capsize=4)
    ax3.set_xscale("log")
    ax3.set_xlabel("calibration samples n")
    ax3.set_ylabel("mean regret (true utility lost)")
    ax3.set_title("Regret vs. data size")

plt.tight_layout()
figure_path = Path(__file__).resolve().parents[2] / "docs" / "simulation.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")
print(f"saved {figure_path}")
