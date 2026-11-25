"""Numerical checks for proof/bon_monotonicity.tex.

Part A checks the theorems against brute force on random parameters.
Part B recomputes the numbers quoted in the "Connection to the LLM experiment" section
(tied scores are broken uniformly at random; every three-level binning tried is printed).

    python proof/verify_proof.py                      # everything
    python proof/verify_proof.py path/to/table.csv    # only analyse the given table(s)
"""
import csv
import subprocess
import sys
from math import comb
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
MATH500_COMMIT = "7a412e2"  # the MATH-500 tables were later removed from results/


def open_table(name):
    """Lines of a result table, from disk or (MATH-500) from git history; None if neither."""
    for path in (Path(name), ROOT / name):
        if path.exists():
            return open(path)
    got = subprocess.run(["git", "-C", str(ROOT), "show", f"{MATH500_COMMIT}:{name}"],
                         capture_output=True, text=True)
    return got.stdout.splitlines() if got.returncode == 0 else None


def rank_table(lines, keep=lambda qid: True):
    """Q x M matrix; column j = correctness of the (j+1)-th lowest-scored answer.
    Tied answers share their mean correctness, i.e. ties are broken uniformly at random."""
    g = {}
    for r in csv.DictReader(lines):
        if keep(int(r["question_id"])):
            g.setdefault(r["question_id"], []).append((float(r["score"]), int(r["correct"])))
    rows = []
    for v in g.values():
        s, c = np.array(sorted(v)).T
        _, tie, n = np.unique(s, return_inverse=True, return_counts=True)
        rows.append(np.bincount(tie, weights=c)[tie] / n[tie])
    return np.array(rows)


def weights_without_replacement(M):
    """W[N-1, j-1] = P(top-scored of a uniform N-subset has ascending rank j)."""
    return np.array([[comb(j - 1, N - 1) / comb(M, N) for j in range(1, M + 1)] for N in range(1, M + 1)])


def boot_ci(d, B=20_000):
    g = np.random.default_rng(1)
    return np.quantile([d[g.integers(0, len(d), len(d))].mean() for _ in range(B)], [.025, .975])


def turns(x):
    s = np.sign(np.diff(x))
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def three_level_fits(mbar):
    """Every binning tried: tertiles, then bottom half..15/16 x top 1, 2 or 4 answers."""
    M = len(mbar)
    cuts = [(M // 3, M - M // 3)] + [(M - M // k, M - t) for k in (2, 4, 8, 16) for t in (1, 2, 4)]
    for lo, hi in cuts:
        if not lo < hi:
            continue
        p = np.array([lo, hi - lo, M - hi]) / M
        m = np.array([mbar[:lo].mean(), mbar[lo:hi].mean(), mbar[hi:].mean()])
        a, b = p[0] + p[1], p[0]
        D1 = (m[1] - m[0]) * (1 - b) * b - (m[1] - m[2]) * (1 - a) * a
        beta = m[2] - m[1]
        if beta < 0 < D1:
            Ns = np.log((m[1] - m[0]) * (1 - b) / ((m[1] - m[2]) * (1 - a))) / np.log(a / b)
            shape = f"peak N_opt={max(1, int(np.floor(Ns)) + 1)}"
        elif beta >= 0 and D1 >= 0:
            shape = "increasing"
        elif beta <= 0 and D1 <= 0:
            shape = "decreasing"
        else:
            shape = "dip"
        yield (lo, hi - lo, M - hi), m, shape, V(M, p, m)


RUNS = [(n, n, lambda q: True) for n in TABLES] or [
    ("GSM8K default scorer", "results/default64_table.csv", lambda q: True),
    ("GSM8K weak scorer (100q, M=64)", "results/weak_scorer64_table.csv", lambda q: True),
    ("GSM8K weak scorer + weak generator", "results/weak_both64_table.csv", lambda q: True),
    ("GSM8K weak scorer, first 100 questions", "results/table256.csv", lambda q: True),
    ("GSM8K weak scorer, full", "table.csv", lambda q: True),
    ("GSM8K weak scorer, questions 100-1318 only", "table.csv", lambda q: q >= 100),
    ("MATH-500 weak scorer", "results/math500_Skywork-Reward-V2-Qwen3-0.6B.csv", lambda q: True),
    ("MATH held-out weak scorer", "results/math_skywork.csv", lambda q: True),
    ("MATH-500 DeBERTa", "results/math500_reward-model-deberta-v3-large-v2.csv", lambda q: True),
]
print("\nB. exact Best-of-N curves from the result tables (Prop. 1(ii))")
for label, name, keep in RUNS:
    lines = open_table(name)
    if lines is None:
        print(f"\n   {label}: {name} missing, skipped")
        continue
    C = rank_table(lines, keep)
    Q, M = C.shape
    mbar = C.mean(0)
    Vq = C @ weights_without_replacement(M).T  # Q x M
    v = Vq.mean(0)
    print(f"\n   {label} ({name}): Q={Q} M={M}")
    print("      V(N): " + "  ".join(f"{n}:{v[n - 1]:.4f}" for n in (1, 2, 4, 8, 16, 32, 64, 94, 128, 256) if n <= M)
          + f"   argmax N={v.argmax() + 1} ({v.max():.4f})")
    idx = [int(round(e * M)) for e in (0, .25, .5, .75, .9, .95, .98)] + [M - 1, M]
    bins = np.array([mbar[i:j].mean() for i, j in zip(idx, idx[1:]) if i < j])
    print("      rank calibration, low->high (last = top answer): " + " ".join(f"{x:.3f}" for x in bins)
          + f"   turning points: binned {turns(bins)}, raw {turns(mbar)}")
    for i in (1, 16, 64, 94):
        if i < M:
            d = Vq[:, i - 1] - Vq[:, -1]
            lo, hi = boot_ci(d)
            print(f"      V({i})-V({M}) = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if v.argmax() not in (0, M - 1):
        d = Vq[:, v.argmax()] - Vq[:, 0]
        lo, hi = boot_ci(d)
        print(f"      V(argmax)-V(1) = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   (peak chosen on same data)")
    for split, m, shape, vM in three_level_fits(mbar):
        print(f"      3-level fit {split}: m = {np.round(m, 3)}  {shape:14s}  V({M}) fit {vM:.3f} vs data {v[-1]:.3f}")
