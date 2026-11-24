# K-level synthetic simulation (K = 3, 5, 10, 20)

**Question:** how do the number of verifier levels K and the amount of calibration data affect the regret of the estimated best N? This extends [`SIMULATION_RESULTS.md`](SIMULATION_RESULTS.md) (K = 3 only).

Code: [`k_level_simulation.py`](../experiments/synthetic/k_level_simulation.py) · Figure: [`k_level_simulation.png`](k_level_simulation.png)
`python experiments/synthetic/k_level_simulation.py` (12 seconds)

## Setup

World with K levels (index 0..K-1, lowest score first), `U | level j ~ Bernoulli(m_j)`:

- `p_j`: the top level is rare, `p_K = 0.03`; the other K-1 levels share the remaining mass equally.
- `m_j`: rises linearly from 0.2 to 0.7 over levels 1..K-1, then **drops at the top level** to 0.3 (the misleading level).
  For K = 3 this is `m = (0.2, 0.7, 0.3)`, the same pattern as the K=3 simulation.
- Estimator: plug-in `p_hat_j` (level fractions) and `m_hat_j` (level mean utility), `V_hat(N) = sum_j m_hat_j (F_hat_j^N - F_hat_{j-1}^N)`, `N_hat = argmax` over N = 1..300 (ties go to the largest N, which handles the flat end of an increasing curve when the top level is unseen).
- Regret = `V(N*) - V(N_hat)` on the TRUE curve. 1000 random calibration sets per cell, calibration sizes n = 50 to 2000.

Why there is no empirical baseline here: with discrete levels, resampling the n samples and keeping the top-level one has exactly the same expectation as the plug-in formula (checked by Monte Carlo), so the two methods coincide. The comparison is meaningful only for continuous scores and is done on real data in [`CALIBRATION_SWEEP_RESULTS.md`](CALIBRATION_SWEEP_RESULTS.md).

## Part A: the estimator knows the K levels

| K | True N* | V(N*) | V(N=1) | n=50 | n=100 | n=200 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | 0.630 | 0.446 | 0.1230 | 0.0501 | 0.0152 | 0.0025 | 0.0010 | 0.0004 |
| 5 | 6 | 0.606 | 0.446 | 0.1119 | 0.0551 | 0.0200 | 0.0037 | 0.0016 | 0.0007 |
| 10 | 6 | 0.589 | 0.445 | 0.0934 | 0.0514 | 0.0237 | 0.0054 | 0.0024 | 0.0011 |
| 20 | 6 | 0.581 | 0.445 | 0.0983 | 0.0514 | 0.0259 | 0.0066 | 0.0029 | 0.0012 |

(Mean regret.) The median `N_hat` equals the true N* (or is off by one) from n = 50 on, for every K.

- Regret falls with n for every K, as in the K=3 case.
- **More levels cost more data.** For n >= 200 the regret increases with K (n=500: 0.0025 for K=3 vs 0.0066 for K=20, about 2.6x); K=20 needs roughly twice the data of K=3 to reach the same regret (0.0029 at n=1000 vs 0.0025 at n=500). Each extra level is one more `m_j` to estimate, with fewer samples per level.
- At n = 50 the K values are close (about 0.09 to 0.12): there, the dominant error is not seeing the rare top level at all, which does not depend on K.
- The true curve itself flattens as K grows (peak 0.630 for K=3, 0.581 for K=20), because the fine levels reduce how much the top-score rule gains before the misleading level appears.

## Part B: true K = 20, the estimator only sees merged (coarser) levels

Adjacent levels are merged into K_coarse groups. "Floor" = regret of the best N of the merged world, i.e. with unlimited data.

| K_coarse | Floor (N of merged world) | n=50 | n=100 | n=200 | n=500 | n=1000 | n=2000 |
|---|---|---|---|---|---|---|---|
| 3 | 0.2813 (300) | 0.2158 | 0.2357 | 0.2648 | 0.2796 | 0.2813 | 0.2813 |
| 5 | 0.2813 (300) | 0.1689 | 0.1754 | 0.1854 | 0.1965 | 0.2119 | 0.2447 |
| 10 | 0.0037 (8) | 0.1168 | 0.0971 | 0.0801 | 0.0467 | 0.0310 | 0.0128 |
| 20 (no merging) | 0.0000 (6) | 0.0956 | 0.0518 | 0.0259 | 0.0067 | 0.0030 | 0.0013 |

- **Merging into 3 or 5 levels hides the misleading top level** (it is merged with a good level, so the merged curve looks increasing). The floor is 0.28, the regret rises toward it **as n grows**: with more data the coarse model becomes more confidently wrong.
- **Merging into 10 levels** has a tiny floor (0.0037) but converges much more slowly than the fine model (0.0310 vs 0.0030 at n=1000): the drop at the top level, from 0.66 to 0.30 when fine, is diluted to about 0.62 to 0.53 once merged, so more data is needed to see it.
- **In this world, finer is better at every n.** The expected benefit of coarse levels (fewer parameters, less variance) does not appear, because the information needed is concentrated in the rare top level, which merging blurs.

## Caveats

- One family of worlds (rare, sharply misleading top level, linear rise below). Other shapes (a gradual decline over several top levels) would favour coarser levels more.
- The levels here are the true discrete scores. On real data the levels are rank bins of a continuous score, which adds a discretization floor but can reduce variance (see the real-data sweep, where a coarse K=3 model beats the empirical method for 10 to 25 calibration questions). The two results are not contradictory: they study different mechanisms.
- Different K are different worlds (different true curves), so regret across K compares the difficulty of estimating that world, not one fixed target.
