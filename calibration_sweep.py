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
Also reported: "discretization regret" = regret of the model's best N when m_j is estimated on the
whole EVAL set (i.e. with unlimited calibration data). It is the floor that more data cannot remove.

Run:  python calibration_sweep.py results/math500_reward-model-deberta-v3-large-v2.csv [more tables ...]
Writes calibration_sweep.png and prints one table per input.
"""
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
}
N_SPLITS = 1000                               # random calibration/eval splits per setting
EVAL_FRACTION = 0.8
CAL_SIZES = [10, 25, 50, 100, 200]            # calibration questions (cut to what the 20% pool has)


def model_coefficients(p, n_max):
    """C[N-1, j] = F_j^N - F_{j-1}^N: probability that the winner of N answers is from level j."""
    F = np.cumsum(p)
    F_prev = np.concatenate([[0.0], F[:-1]])
    N = np.arange(1, n_max + 1)[:, None]
    return F[None, :] ** N - F_prev[None, :] ** N            # (n_max, K)


def run_layout(sorted_correct, p, cal_idx, eval_idx, real_eval, sizes, n_max):
    """
    sorted_correct: (questions, L).  cal_idx (splits, pool) and eval_idx (splits, n_eval): the splits.
    real_eval: (splits, n_max) real accuracy curve on each split's eval set.
    Returns {size: dict(N_hat, regret)} plus the discretization regret per split.
    """
    L = sorted_correct.shape[1]
    weights = level_weights(L, np.asarray(p))                # (L, K)
    coeff = model_coefficients(np.asarray(p), n_max)         # (n_max, K)

    best_real = real_eval.max(axis=1)                        # (splits,)
    regret_of = lambda N_hat: best_real - real_eval[np.arange(len(N_hat)), N_hat - 1]

    def one_size(size):
        m_hat = sorted_correct[cal_idx[:, :size]].mean(axis=1) @ weights      # (splits, K)
        N_hat = np.argmax(m_hat @ coeff.T, axis=1) + 1
        return {"N_hat": N_hat, "regret": regret_of(N_hat)}

    # unlimited calibration data: m_j estimated on the eval questions themselves
    m_full = sorted_correct[eval_idx].mean(axis=1) @ weights
    N_disc = np.argmax(m_full @ coeff.T, axis=1) + 1

    return {size: one_size(size) for size in sizes}, regret_of(N_disc)


def sweep_table(path):
    questions = load_questions(path)
    n_max = min(len(s) for s, _ in questions)
    sorted_correct = np.array([sort_by_score(s, c) for s, c in questions])   # (Q, L)
    Q = len(questions)

    # random splits: a permutation per split; eval = last 80%, calibration pool = first 20%
    perm = rng.permuted(np.tile(np.arange(Q), (N_SPLITS, 1)), axis=1)
    n_eval = int(round(EVAL_FRACTION * Q))
    pool = Q - n_eval
    eval_idx, cal_idx = perm[:, pool:], perm[:, :pool]
    sizes = [s for s in CAL_SIZES if s <= pool]

    # real accuracy curve on each split's eval questions (linear in the mean correctness)
    real_eval = sorted_correct[eval_idx].mean(axis=1) @ real_weights(sorted_correct.shape[1], n_max).T
    real_best_N = np.argmax(real_eval, axis=1) + 1

    print(f"\n=== {path}: {Q} questions x {n_max} answers; eval {n_eval} questions, calibration pool {pool}"
          f"\nreal best N on the eval sets: median {np.median(real_best_N):.0f} "
          f"(10-90%: {np.percentile(real_best_N, 10):.0f}-{np.percentile(real_best_N, 90):.0f}); "
          f"mean accuracy lost by N=1 / N={n_max}: "
          f"{(real_eval.max(1) - real_eval[:, 0]).mean():.4f} / {(real_eval.max(1) - real_eval[:, -1]).mean():.4f}")

    results = {name: run_layout(sorted_correct, p, cal_idx, eval_idx, real_eval, sizes, n_max)
               for name, p in LAYOUTS.items()}

    for name, (by_size, disc_regret) in results.items():
        print(f"\n  {name}   (discretization regret with unlimited data: {disc_regret.mean():.4f})")
        print(f"  {'cal questions':>13} | {'mean regret':>11} | {'std err':>8} | {'mean N_hat':>10} | {'median N_hat':>12}")
        for size, r in by_size.items():
            print(f"  {size:>13} | {r['regret'].mean():>11.4f} | {r['regret'].std() / np.sqrt(N_SPLITS):>8.4f}"
                  f" | {r['N_hat'].mean():>10.1f} | {np.median(r['N_hat']):>12.0f}")
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
        line = ax.errorbar(sizes, mean, yerr=err, marker="o", capsize=3, label=name)
        ax.axhline(disc_regret.mean(), ls=":", color=line[0].get_color(), alpha=0.7)   # unlimited-data floor
    ax.set_xscale("log")
    ax.set_xlabel("calibration questions")
    ax.set_ylabel("mean regret (accuracy lost)")
    ax.set_title(Path(path).stem, fontsize=9)
    ax.legend(fontsize=8)
plt.suptitle("Regret of the estimated best N vs. calibration size (dotted: floor with unlimited data)")
plt.tight_layout()
plt.savefig("calibration_sweep.png", dpi=150, bbox_inches="tight")
print("\nsaved calibration_sweep.png")
