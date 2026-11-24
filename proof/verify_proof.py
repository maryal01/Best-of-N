"""Numerical checks for proof/bon_monotonicity.tex.

Part A checks the theorems against brute force on random parameters.
Part B recomputes the numbers quoted in the "Connection to the LLM experiment" section.

    python proof/verify_proof.py                      # everything
    python proof/verify_proof.py path/to/table.csv    # only analyse the given table(s)
"""
import csv
import sys
from math import lgamma
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)


def V(N, p, m):
    """Exact E[U_{I_N}] for a K-level model (Lemma 2)."""
    F = np.concatenate([[0.0], np.cumsum(p)])
    F[-1] = 1.0
    return float(np.dot(m, F[1:] ** N - F[:-1] ** N))


# ---------------- Part A: theorems ----------------
def check_theorems():
    print("A1. closed form vs Monte Carlo (uniform tie-breaking)")
    p, m = np.array([0.5, 0.35, 0.15]), np.array([0.3, 0.9, 0.5])
    for N in [1, 3, 8]:
        S = rng.choice(3, size=(200_000, N), p=p)
        U = (rng.random(S.shape) < m[S]).astype(float)
        pick = (S + 0.5 * rng.random(S.shape)).argmax(1)
        print(f"   N={N}: MC {U[np.arange(len(U)), pick].mean():.4f}   formula {V(N, p, m):.4f}")

    print("A2. three levels: two-point test (Thm 2) and N_opt formula (Thm 3)")
    Ns = np.arange(1, 20001)
    fail_test = fail_opt = n_test = n_opt = 0
    for _ in range(20_000):
        p, m = rng.dirichlet(np.ones(3)), rng.random(3)
        a, b = p[0] + p[1], p[0]
        al, be = m[1] - m[0], m[2] - m[1]
        D1 = al * (1 - b) * b + be * (1 - a) * a
        # sign of Delta(N) from the factored form Delta(N) = a^N g(N): no cancellation
        g = al * (1 - b) * (b / a) ** Ns + be * (1 - a)
        if abs(al) * (1 - b) * (b / a) ** Ns[-1] < abs(be) * (1 - a) * 1e-3:  # tail settled within range
            n_test += 1
            fail_test += (np.all(g >= 0) != (be >= 0 and D1 >= 0)) or (np.all(g <= 0) != (be <= 0 and D1 <= 0))
        if al > 0 > be:
            Nstar = np.log(al * (1 - b) / (-be * (1 - a))) / np.log(a / b)
            Nopt = max(1, int(np.floor(Nstar)) + 1)
            if Nopt < 390:
                v = np.array([V(N, p, m) for N in range(1, 401)])
                n_opt += 1
                fail_opt += abs(v[Nopt - 1] - v.max()) > 1e-12
    print(f"   two-point test failures: {fail_test}/{n_test}   N_opt failures: {fail_opt}/{n_opt}")

    print("A3. K levels: turning points of V <= turning points of m (Thm 4)")
    viol = 0
    for _ in range(20_000):
        K = rng.integers(3, 9)
        p, m = rng.dirichlet(np.ones(K)), rng.random(K)
        inc = np.sign(np.diff(m))
        d = np.diff([V(N, p, m) for N in range(1, 600)])
        d = np.sign(d[np.abs(d) > 1e-14])
        viol += np.sum(d[1:] != d[:-1]) > np.sum(inc[1:] != inc[:-1])
    print(f"   violations: {viol}/20000")


TABLES = sys.argv[1:]
if not TABLES:
    check_theorems()


# ---------------- Part B: LLM data ----------------
def rank_table(path):
    """Q x M 0/1 matrix; column j = correctness of the (j+1)-th lowest-scored answer."""
    g = {}
    for r in csv.DictReader(open(path)):
        g.setdefault(r["question_id"], []).append((float(r["score"]), int(r["correct"])))
    return np.array([[c for _, c in sorted(v)] for v in g.values()], float)


def weights_without_replacement(N, M):
    """P(top-scored of a uniform N-subset has ascending rank j) = C(j-1,N-1)/C(M,N)."""
    lc = lambda n, k: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    j = np.arange(1, M + 1)
    w = np.zeros(M)
    w[j >= N] = np.exp([lc(jj - 1, N - 1) - lc(M, N) for jj in j[j >= N]])
    return w


print("\nB. exact Best-of-N curves from the result tables (Prop. 1(ii))")
EDGES = [0, .25, .5, .75, .9, .95, .98, 1.0]
for name in TABLES or ["results/default64_table.csv", "results/weak_scorer64_table.csv",
                       "results/weak_both64_table.csv", "table.csv"]:
    path = Path(name) if TABLES else ROOT / name
    if not path.exists():
        print(f"   {name}: missing, skipped")
        continue
    C = rank_table(path)
    Q, M = C.shape
    mbar = C.mean(0)
    Vq = np.array([C @ weights_without_replacement(N, M) for N in range(1, M + 1)]).T  # Q x M
    v = Vq.mean(0)
    idx = [int(round(e * M)) for e in EDGES]
    bins = " ".join(f"{mbar[idx[i]:idx[i + 1]].mean():.3f}" for i in range(len(EDGES) - 1))
    print(f"   {name}: Q={Q} M={M}  argmax N={v.argmax() + 1}  max={v.max():.4f}  V(M)={v[-1]:.4f}")
    print(f"      rank-calibration by score-quantile bin (low->high): {bins};  top rank alone: {mbar[-1]:.2f}")
    if v.argmax() + 1 < M:  # apparent interior optimum: is the decline significant?
        d = Vq[:, v.argmax()] - Vq[:, -1]
        bs = np.array([d[rng.integers(0, Q, Q)].mean() for _ in range(20_000)])
        lo, hi = np.quantile(bs, [.025, .975])
        print(f"      decline V(peak)-V(M) = {d.mean():.4f}, bootstrap-over-questions 95% CI [{lo:.3f}, {hi:.3f}]")
        print("      -> " + ("significant: the interval excludes zero" if lo > 0 else "not significant: the interval includes zero"))
