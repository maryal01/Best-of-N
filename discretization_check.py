"""
Step 1: does the discrete K-level model capture the real Best-of-N curve? (NO calibration noise)

We use ALL questions of a table to estimate the level means m_j, so any gap between the model's
best N and the real best N comes from discretization alone, not from finite calibration data.

Levels are defined by RANK WITHIN EACH QUESTION (reward-model scores are not comparable across
questions): the lowest-scored p_1 fraction of a question's answers is level 1, the next p_2 fraction
is level 2, and so on. Every question then has the same p_j, so pooling m_j over questions makes the
model's V(N) exactly the average of the per-question curves (under i.i.d. sampling with replacement).

We compare three curves (accuracy of the kept answer vs N):
  real     : real Best-of-N (argmax of the continuous score), N answers drawn WITHOUT replacement
             from the question's answers.  This is what run_experiment.py reports.
  iid      : same, but drawn WITH replacement. This is the i.i.d. setting the model assumes.
  model    : discrete K-level model, V(N) = sum_j m_j (F_j^N - F_{j-1}^N)

Run:  python discretization_check.py results/math500_reward-model-deberta-v3-large-v2.csv ...
"""
import csv
import sys

import numpy as np
from scipy.special import gammaln

# Level layouts to try: the masses p_j of the levels, lowest score first.
LAYOUTS = {
    "K=3  top 10%":  [0.50, 0.40, 0.10],
    "K=3  top 5%":   [0.50, 0.45, 0.05],
    "K=3  top 2%":   [0.50, 0.48, 0.02],
    "K=3  top 1%":   [0.50, 0.49, 0.01],
    "K=3  equal":    [1 / 3, 1 / 3, 1 / 3],
    "K=5  equal":    [0.2] * 5,
    "K=10 equal":    [0.1] * 10,
}


def load_questions(path):
    """Returns a list of (scores, correct) per question, both numpy arrays."""
    groups = {}
    for row in csv.DictReader(open(path)):
        groups.setdefault(row["question_id"], []).append((float(row["score"]), int(row["correct"])))
    return [(np.array([s for s, _ in g]), np.array([c for _, c in g], dtype=float)) for g in groups.values()]


def sort_by_score(scores, correct):
    """Sort answers by score; answers with the same score share their mean correctness (ties
    are equally likely to be kept)."""
    order = np.argsort(scores)
    s, c = scores[order], correct[order]
    _, tie, counts = np.unique(s, return_inverse=True, return_counts=True)
    return np.bincount(tie, weights=c)[tie] / counts[tie]


def real_weights(L, n_max):
    """
    W[N-1, j] = P(the answer at sorted position j is the top-scored of N answers drawn WITHOUT
    replacement from a question's L answers) = C(j, N-1) / C(L, N).   Shape (n_max, L).
    The real accuracy curve of a question is W @ (its sorted correctness).
    """
    N = np.arange(1, n_max + 1)[:, None]
    j = np.arange(L)[None, :]
    log_c = lambda n, k: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    with np.errstate(invalid="ignore"):
        return np.where(j >= N - 1, np.exp(log_c(j, N - 1) - log_c(L, N)), 0.0)


def iid_weights(L, n_max):
    """Same, but drawing WITH replacement (the i.i.d. setting of the model): ((j+1)/L)^N - (j/L)^N."""
    N = np.arange(1, n_max + 1)[:, None]
    j = np.arange(L)[None, :]
    return ((j + 1) / L) ** N - (j / L) ** N


def real_and_iid_curves(sorted_correct, n_max):
    """Empirical accuracy curves (N = 1..n_max), averaged over the questions in `sorted_correct` (questions, L)."""
    L = sorted_correct.shape[1]
    mean_correct = sorted_correct.mean(axis=0)              # the curves are linear in the correctness
    return real_weights(L, n_max) @ mean_correct, iid_weights(L, n_max) @ mean_correct


def level_weights(L, p):
    """
    Weights (L, K) that turn per-position correctness into level means: m_j = sum_i weight[i, j] * c_i.
    An answer at sorted position i covers the quantile interval [i/L, (i+1)/L); its weight in level j
    is the overlap of that interval with the level's quantile interval [F_{j-1}, F_j).
    """
    F = np.concatenate([[0.0], np.cumsum(p)])                # level edges, shape (K+1,)
    lo = np.arange(L)[:, None] / L
    hi = (np.arange(L)[:, None] + 1) / L
    overlap = np.clip(np.minimum(hi, F[None, 1:]) - np.maximum(lo, F[None, :-1]), 0, None)   # (L, K)
    return overlap / overlap.sum(axis=0)                     # each level's weights sum to 1


def level_means(sorted_correct, p):
    """m_j = average correctness of the answers in rank-level j, pooled over questions."""
    return sorted_correct.mean(axis=0) @ level_weights(sorted_correct.shape[1], np.asarray(p))


def model_curve(p, m, n_max):
    """V(N) = sum_j m_j (F_j^N - F_{j-1}^N) for N = 1..n_max."""
    F = np.cumsum(p)
    F_prev = np.concatenate([[0.0], F[:-1]])
    N = np.arange(1, n_max + 1)[:, None]
    return (F[None, :] ** N - F_prev[None, :] ** N) @ m


def check_table(path):
    questions = load_questions(path)
    n_max = min(len(s) for s, _ in questions)
    sorted_correct = np.array([sort_by_score(s, c) for s, c in questions])   # (questions, L)

    real, iid = real_and_iid_curves(sorted_correct, n_max)
    best_real = int(np.argmax(real)) + 1
    print(f"\n=== {path}: {len(questions)} questions x {n_max} answers")
    print(f"real curve:  best N = {best_real} (acc {real.max():.4f});  N=1: {real[0]:.4f};  N={n_max}: {real[-1]:.4f}")
    print(f"iid  curve:  best N = {int(np.argmax(iid)) + 1} (acc {iid.max():.4f})   [with replacement]")
    print(f"{'layout':<14} | {'model best N':>12} | {'model acc':>9} | {'max |model-iid|':>15} | {'disc. regret':>12}   (levels m_j)")

    def one_layout(name, p):
        p = np.array(p)
        m = level_means(sorted_correct, p)
        curve = model_curve(p, m, n_max)
        best_model = int(np.argmax(curve)) + 1
        regret = real.max() - real[best_model - 1]          # loss on the REAL curve from using the model's N
        print(f"{name:<14} | {best_model:>12} | {curve.max():>9.4f} | {np.abs(curve - iid).max():>15.4f} |"
              f" {regret:>12.4f}   {np.round(m, 3)}")

    [one_layout(name, p) for name, p in LAYOUTS.items()]


if __name__ == "__main__":
    for path in sys.argv[1:]:
        check_table(path)
