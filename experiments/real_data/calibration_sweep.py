"""
Step 2: finite-data question on REAL LLM data. How does the regret of the estimated best N
shrink as we use more calibration questions?

For each random split of the questions:
  - EVAL = 80% of the questions (fixed size): the "truth". Real best N and its curve come from here.
  - Calibration pool = the other 20%; we use the first `size` questions of it, size = 10, 25, ... up to the pool.
  - Levels are rank bins WITHIN each question, so p_j is known by design; only m_j is estimated:
    m_hat_j = average correctness of the answers in level j, over the calibration questions.
  - Plug m_hat into V(N) = sum_j m_j (F_j^N - F_{j-1}^N), take N_hat = argmax.
  - Regret = real_acc_EVAL(best N) - real_acc_EVAL(N_hat).
Baseline: the DIRECT EMPIRICAL method uses the same calibration questions to compute the real Best-of-N accuracy
curve (no levels, no model) and picks its argmax. Both methods are compared on the same eval questions.
Also reported: "discretization regret" = regret of the model's best N when m_j is estimated on the
whole EVAL set (i.e. with unlimited calibration data). It is the floor that more data cannot remove.

Run from the repo root:  python experiments/real_data/calibration_sweep.py results/math500_reward-model-deberta-v3-large-v2.csv [more tables ...]
Writes docs/calibration_sweep.png and prints one table per input.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discretization_check import load_questions, sort_by_score, real_weights, level_weights

rng = np.random.default_rng(0)

LAYOUTS = {                                   # level masses p_j, lowest score first
    "K=3 equal": [1 / 3, 1 / 3, 1 / 3],
    "K=5 equal": [0.2] * 5,
    "K=3 top 5%": [0.50, 0.45, 0.05],
    # Dyadic: level masses halve towards the top (bottom 50%, then 25%, ...), so every power of two N = 1, 2, 4, ... has its own level.
    # Best-of-N with N answers looks at the top ~1/N of the scores, which is where over-optimization shows.
    "K=6 dyadic": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125],
}
EMPIRICAL = "Empirical (direct)"            # name of the baseline in the tables and the figure
N_SPLITS = 1000                               # random calibration/eval splits per setting
EVAL_FRACTION = 0.8
CAL_SIZES = [10, 25, 50, 100, 200, 400, 600]   # calibration questions (cut to what the 20% pool has)


def model_coefficients(p, n_max):
    """C[N-1, j] = F_j^N - F_{j-1}^N: probability that the winner of N answers is from level j."""
    F = np.cumsum(p)
    F_prev = np.concatenate([[0.0], F[:-1]])
    N = np.arange(1, n_max + 1)[:, None]
    return F[None, :] ** N - F_prev[None, :] ** N            # (n_max, K)


def run_layout(L, p, cal_mean, eval_mean, real_eval, sizes, n_max):
    """
    L: answers per question.  cal_mean[size] (splits, L): mean sorted correctness of the first `size` calibration
    questions of each split.  eval_mean (splits, L): the same for the eval questions.
    real_eval: (splits, n_max) real accuracy curve on each split's eval set.
    Returns {size: dict(N_hat, regret)} plus the discretization regret per split.
    """
    weights = level_weights(L, np.asarray(p))                # (L, K)
    coeff = model_coefficients(np.asarray(p), n_max)         # (n_max, K)

    best_real = real_eval.max(axis=1)                        # (splits,)
    regret_of = lambda N_hat: best_real - real_eval[np.arange(len(N_hat)), N_hat - 1]

    def one_size(size):
        m_hat = cal_mean[size] @ weights                                      # (splits, K)
        N_hat = np.argmax(m_hat @ coeff.T, axis=1) + 1
        return {"N_hat": N_hat, "regret": regret_of(N_hat)}

    # unlimited calibration data: m_j estimated on the eval questions themselves
    m_full = eval_mean @ weights
    N_disc = np.argmax(m_full @ coeff.T, axis=1) + 1

    return {size: one_size(size) for size in sizes}, regret_of(N_disc)


def run_empirical(L, cal_mean, real_eval, sizes, n_max):
    """
    Direct empirical baseline: the real Best-of-N accuracy curve of the first `size` calibration questions
    (exact, N answers drawn without replacement), and N_hat = its argmax. Same output format as run_layout.
    """
    W = real_weights(L, n_max)                                             # (n_max, L)
    best_real = real_eval.max(axis=1)

    def one_size(size):
        curve = cal_mean[size] @ W.T                                       # (splits, n_max)
        N_hat = np.argmax(curve, axis=1) + 1
        return {"N_hat": N_hat, "regret": best_real - real_eval[np.arange(len(N_hat)), N_hat - 1]}

    return {size: one_size(size) for size in sizes}


def paired_difference_row(by_size, empirical, sizes):
    """'mean +- se' of (regret of this layout - regret of the empirical baseline) on the same splits, per size."""
    cells = []
    for size in sizes:
        diff = by_size[size]["regret"] - empirical[size]["regret"]
        cells.append(f"{diff.mean():+.4f}+-{diff.std() / np.sqrt(N_SPLITS):.4f}")
    return " | ".join(f"{c:>16}" for c in cells)


def sweep_table(path):
    questions = load_questions(path)
    n_max = min(len(s) for s, _ in questions)
    sorted_correct = np.array([sort_by_score(s, c) for s, c in questions])   # (Q, L)
    Q = len(questions)

    # random splits: a permutation per split; eval = last 80%, calibration pool = first 20%
    perm = rng.permuted(np.tile(np.arange(Q), (N_SPLITS, 1)), axis=1)
    n_eval = int(round(EVAL_FRACTION * Q))
    pool = Q - n_eval
    cal_idx = perm[:, :pool]
    sizes = [s for s in CAL_SIZES if s <= pool]
    L = sorted_correct.shape[1]

    # Mean sorted correctness of the first `size` calibration questions, for every size at once (cumulative sum),
    # and of the eval questions (everything not in the pool: total minus pool). Computed once, reused by all methods.
    cal_cumsum = np.cumsum(sorted_correct[cal_idx], axis=1)                       # (splits, pool, L)
    cal_mean = {size: cal_cumsum[:, size - 1] / size for size in sizes}
    eval_mean = (sorted_correct.sum(axis=0) - cal_cumsum[:, -1]) / n_eval         # (splits, L)
    del cal_cumsum

    # real accuracy curve on each split's eval questions (linear in the mean correctness)
    real_eval = eval_mean @ real_weights(L, n_max).T
    real_best_N = np.argmax(real_eval, axis=1) + 1

    print(f"\n=== {path}: {Q} questions x {n_max} answers; eval {n_eval} questions, calibration pool {pool}"
          f"\nreal best N on the eval sets: median {np.median(real_best_N):.0f} "
          f"(10-90%: {np.percentile(real_best_N, 10):.0f}-{np.percentile(real_best_N, 90):.0f}); "
          f"mean accuracy lost by N=1 / N={n_max}: "
          f"{(real_eval.max(1) - real_eval[:, 0]).mean():.4f} / {(real_eval.max(1) - real_eval[:, -1]).mean():.4f}")

    results = {name: run_layout(L, p, cal_mean, eval_mean, real_eval, sizes, n_max)
               for name, p in LAYOUTS.items()}
    results[EMPIRICAL] = (run_empirical(L, cal_mean, real_eval, sizes, n_max), np.zeros(N_SPLITS))

    for name, (by_size, disc_regret) in results.items():
        print(f"\n  {name}   (discretization regret with unlimited data: {disc_regret.mean():.4f})")
        print(f"  {'cal questions':>13} | {'mean regret':>11} | {'std err':>8} | {'mean N_hat':>10} | {'median N_hat':>12}")
        for size, r in by_size.items():
            print(f"  {size:>13} | {r['regret'].mean():>11.4f} | {r['regret'].std() / np.sqrt(N_SPLITS):>8.4f}"
                  f" | {r['N_hat'].mean():>10.1f} | {np.median(r['N_hat']):>12.0f}")

    print("\n  Paired difference in regret vs the empirical baseline (same splits). Negative = K-level model better.")
    print(f"  {'layout':<12} | " + " | ".join(f"{str(s) + ' questions':>16}" for s in sizes))
    for name, (by_size, _) in results.items():
        if name != EMPIRICAL:
            print(f"  {name:<12} | {paired_difference_row(by_size, results[EMPIRICAL][0], sizes)}")
    return {"sizes": sizes, "results": results, "n_max": n_max}


# ---------------------------------------------------------------------------
# Run all tables, then one figure: one panel per table, one line per layout
# ---------------------------------------------------------------------------
paths = sys.argv[1:]
summaries = [sweep_table(path) for path in paths]

fig, axes = plt.subplots(1, len(paths), figsize=(5.5 * len(paths), 4.2), squeeze=False)
for ax, path, summary in zip(axes[0], paths, summaries):
    for name, (by_size, disc_regret) in summary["results"].items():
        sizes = summary["sizes"]
        mean = [by_size[s]["regret"].mean() for s in sizes]
        err = [by_size[s]["regret"].std() / np.sqrt(N_SPLITS) for s in sizes]
        style = {"color": "black", "linestyle": "--"} if name == EMPIRICAL else {}
        line = ax.errorbar(sizes, mean, yerr=err, marker="o", capsize=3, label=name, **style)
        if name != EMPIRICAL:
            ax.axhline(disc_regret.mean(), ls=":", color=line[0].get_color(), alpha=0.7)   # unlimited-data floor
    ax.set_xscale("log")
    ax.set_xlabel("calibration questions")
    ax.set_ylabel("mean regret (accuracy lost)")
    ax.set_title(Path(path).stem, fontsize=9)
    ax.legend(fontsize=8)
plt.suptitle("Regret of the estimated best N vs. calibration size (dotted: floor with unlimited data)")
plt.tight_layout()
figure_name = os.environ.get("SWEEP_FIGURE", "calibration_sweep.png")          # e.g. SWEEP_FIGURE=ppe_sweep.png for another table set
plt.savefig(Path(__file__).resolve().parents[2] / "docs" / figure_name, dpi=150, bbox_inches="tight")
print(f"\nsaved docs/{figure_name}")
