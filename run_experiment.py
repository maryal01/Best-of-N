import csv
import sys
from math import comb

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

# python run_experiment.py                     -> table.csv, plot in accuracy_curve.png
# python run_experiment.py results/x.csv       -> that table, plot in results/x_accuracy_curve.png
# python run_experiment.py results/x.csv 64    -> also test the decline from N=64 (chosen in
#                                                 advance, e.g. on another dataset) to the largest N
TABLE = sys.argv[1] if len(sys.argv) > 1 else "table.csv"
PLOT = TABLE.replace(".csv", "_accuracy_curve.png") if len(sys.argv) > 1 else "accuracy_curve.png"
FIXED_N = int(sys.argv[2]) if len(sys.argv) > 2 else None

groups = {}
for row in csv.DictReader(open(TABLE)):
    groups.setdefault(row["question_id"], []).append((float(row["score"]), int(row["correct"])))

questions = [(np.array([s for s, _ in g]), np.array([c for _, c in g])) for g in groups.values()]

M = min(len(s) for s, _ in questions)  # answers per question = largest N we can test
print(f"{len(questions)} questions, {M} answers each")

def weights(L):
    """w[N-1, j]: chance that answer j (sorted by score, lowest first) is the top-scored of
    N answers drawn at random from the question's L answers, = C(j, N-1) / C(L, N)."""
    return np.array([[comb(j, N - 1) / comb(L, N) for j in range(L)] for N in range(1, M + 1)])

def accuracy_curve(scores, correct, W={}):
    """For one question: the exact chance that the kept answer is correct, for N = 1..M.
    Same as averaging infinitely many random draws of N answers, but exact and fast."""
    order = np.argsort(scores)
    s, c = scores[order], correct[order].astype(float)
    _, tie, counts = np.unique(s, return_inverse=True, return_counts=True)
    c = np.bincount(tie, weights=c)[tie] / counts[tie]  # tied answers are equally likely to be kept
    if len(s) not in W:
        W[len(s)] = weights(len(s))
    return W[len(s)] @ c

per_question = np.array([accuracy_curve(s, c) for s, c in questions]).T  # [N-1, question]
curve = per_question.mean(axis=1)
true_N = int(np.argmax(curve)) + 1
print(f"Best N = {true_N}  (accuracy {curve.max():.3f}; at N=1: {curve[0]:.3f}, at N={M}: {curve[-1]:.3f})")

# Is the drop from the best N to the largest N real, or just which questions we drew?
# Bootstrap over questions; the decline is significant if the interval excludes zero.
# The best N is picked on this same data, which makes that test optimistic; a fixed N chosen
# beforehand (FIXED_N) gives an honest test.
for test_N, label in [(true_N, "the best N"), (FIXED_N, f"N={FIXED_N}, chosen in advance")]:
    if test_N is None or test_N >= M:
        continue
    drop = per_question[test_N - 1] - per_question[-1]
    boot = [drop[rng.integers(0, len(drop), len(drop))].mean() for _ in range(20000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = "significant" if lo > 0 else "not significant"
    print(f"Decline after {label} = {drop.mean():.4f}, 95% CI [{lo:.3f}, {hi:.3f}] -> {verdict}")

plt.plot(range(1, M + 1), curve, lw=2)
plt.xlabel("N (answers sampled)")
plt.ylabel("accuracy of the kept answer")
plt.savefig(PLOT, dpi=150, bbox_inches="tight")