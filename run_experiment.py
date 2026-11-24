import csv

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

groups = {}
for row in csv.DictReader(open("table.csv")):
    groups.setdefault(row["question_id"], []).append((float(row["score"]), int(row["correct"])))

questions = [(np.array([s for s, _ in g]), np.array([c for _, c in g])) for g in groups.values()]

M = min(len(s) for s, _ in questions)  # answers per question = largest N we can test
print(f"{len(questions)} questions, {M} answers each")

def accuracy(N, trials=500):
    """For each question: pick N of its answers at random, keep the top-scored one,
    check it. Average over trials; returns one accuracy per question."""
    accs = []
    for scores, correct in questions:
        picks = rng.random((trials, len(scores))).argsort(axis=1)[:, :N]
        winner = picks[np.arange(trials), scores[picks].argmax(axis=1)]
        accs.append(correct[winner].mean())
    return np.array(accs)

per_question = np.array([accuracy(N) for N in range(1, M + 1)])  # [N-1, question]
curve = per_question.mean(axis=1)
true_N = int(np.argmax(curve)) + 1
print(f"Best N = {true_N}  (accuracy {curve.max():.3f}; at N=1: {curve[0]:.3f}, at N={M}: {curve[-1]:.3f})")

# Is the drop from the best N to the largest N real, or just which questions we drew?
# Bootstrap over questions; the decline is significant if the interval excludes zero.
if true_N < M:
    drop = per_question[true_N - 1] - per_question[-1]
    boot = [drop[rng.integers(0, len(drop), len(drop))].mean() for _ in range(20000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    verdict = "significant" if lo > 0 else "not significant"
    print(f"Decline after the best N = {drop.mean():.4f}, 95% CI [{lo:.3f}, {hi:.3f}] -> {verdict}")

plt.plot(range(1, M + 1), curve, lw=2)
plt.xlabel("N (answers sampled)")
plt.ylabel("accuracy of the kept answer")
plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")