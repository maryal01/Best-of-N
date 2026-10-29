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
    check it. Average over trials and questions."""
    accs = []
    for scores, correct in questions:
        picks = rng.random((trials, len(scores))).argsort(axis=1)[:, :N]
        winner = picks[np.arange(trials), scores[picks].argmax(axis=1)]
        accs.append(correct[winner].mean())
    return np.mean(accs)

curve = np.array([accuracy(N) for N in range(1, M + 1)])
true_N = int(np.argmax(curve)) + 1
print(f"Best N = {true_N}  (accuracy {curve.max():.3f}; at N=1: {curve[0]:.3f}, at N={M}: {curve[-1]:.3f})")

plt.plot(range(1, M + 1), curve, lw=2)
plt.xlabel("N (answers sampled)")
plt.ylabel("accuracy of the kept answer")
plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")