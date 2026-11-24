# Calibration sweep on real LLM data

**Question:** on real data, how does the regret of the estimated best N shrink as we use more calibration questions? (Does the synthetic result of [`SIMULATION_RESULTS.md`](../synthetic/SIMULATION_RESULTS.md) hold on real data?)

Code: [`calibration_sweep.py`](calibration_sweep.py) · Figure: [`calibration_sweep.png`](calibration_sweep.png)
`python experiments/real_data/calibration_sweep.py results/math500_reward-model-deberta-v3-large-v2.csv results/math500_Skywork-Reward-V2-Qwen3-0.6B.csv table.csv`

## Procedure (per random split, 1000 splits)

1. Randomly split the questions: **80% eval** (the truth) and a **20% calibration pool** (MATH-500: 400 / 100; GSM8K: 1055 / 264).
2. Real best N and its accuracy curve are computed on the eval questions (exact formula, as in `run_experiment.py`).
3. Take the first `size` questions of the calibration pool. Levels are rank bins within each question (see [`DISCRETIZATION_RESULTS.md`](DISCRETIZATION_RESULTS.md)), so only `m_j` is estimated: `m_hat_j` = average correctness of the answers in level j.
4. Plug `m_hat` into `V(N) = sum_j m_j (F_j^N - F_{j-1}^N)`, take `N_hat = argmax`.
5. **Regret** = real eval accuracy at the real best N minus real eval accuracy at `N_hat`.
6. **Floor** (discretization regret with unlimited data): the same, with `m_j` estimated on the eval questions themselves.

The calibration unit is the question (each has 256 answers, but answers of one question are correlated, so effective sample size is far below 256 x questions).

## Results (mean regret; standard error <= 0.0007 everywhere)

Calibration questions: 10 / 25 / 50 / 100 (GSM8K also 200).

### MATH-500 + DeBERTa (real best N: median 2; N=1 loses 0.0020, N=256 loses 0.0492)

| Layout | 10 | 25 | 50 | 100 | Floor | Mean / median N_hat at 10 -> 100 |
|---|---|---|---|---|---|---|
| K=3 equal | 0.0105 | 0.0093 | 0.0079 | 0.0063 | 0.0004 | 29.2 -> 13.7 / 2 |
| K=5 equal | 0.0107 | 0.0082 | 0.0062 | 0.0042 | 0.0000 | 40.2 -> 11.9 / 2 |
| K=3 top 5% | 0.0153 | 0.0104 | 0.0071 | 0.0045 | 0.0005 | 69.6 -> 10.8 / 3 -> 2 |

### MATH-500 + Skywork (real best N: median 70, 10-90% range 53-88; N=1 loses 0.0609, N=256 loses 0.0120)

| Layout | 10 | 25 | 50 | 100 | Floor | Mean / median N_hat at 10 -> 100 |
|---|---|---|---|---|---|---|
| K=3 equal | 0.0067 | 0.0022 | 0.0006 | 0.0003 | 0.0003 | 69.3 -> 85.6 / 86 |
| K=5 equal | 0.0110 | 0.0069 | 0.0049 | 0.0040 | 0.0035 | 105.1 -> 140.8 / 150 |
| K=3 top 5% | 0.0153 | 0.0129 | 0.0123 | 0.0121 | 0.0120 | 185.3 -> 239.9 / 256 |

### GSM8K + Skywork (real best N: median 256; N=1 loses 0.0564, N=256 loses 0)

| Layout | 10 | 25 | 50 | 100 | 200 | Floor |
|---|---|---|---|---|---|---|
| K=3 equal | 0.0097 | 0.0070 | 0.0055 | 0.0049 | 0.0048 | 0.0047 |
| K=5 equal | 0.0096 | 0.0065 | 0.0046 | 0.0034 | 0.0026 | 0.0023 |
| K=3 top 5% | 0.0071 | 0.0043 | 0.0026 | 0.0014 | 0.0003 | 0.0000 |

## Findings

- **Regret decreases with calibration size in every setting**, qualitatively matching the synthetic simulation.
- **It levels off at the discretization floor** (Skywork K=3 top 5%: 0.0153 -> 0.0121 with floor 0.0120; GSM8K K=3 equal: 0.0048 vs floor 0.0047). Beyond that point more data does not help; a better layout does.
- **Absolute regrets are small** (<= 0.015, vs 0.25 in the synthetic rare-top-level case): on real data the curve near the optimum is flat, and a wrong N is not very costly except when it overshoots towards N=256 on DeBERTa.
- **DeBERTa:** the median `N_hat` is already 2 (the truth) with 10 questions; the mean `N_hat` is inflated by a few large overshoots, which matter because N=256 loses 0.049.
- **The synthetic failure mode ("top level never observed, so N_hat jumps to N_max") does not occur here**, because rank levels are defined within each question, so every question contributes answers to every level. The finite-data difficulty shows up as variance of `m_hat` instead.
- Because the layout sets the floor, the layout must be chosen without looking at the eval truth (here: fixed in advance: K=3 equal, K=5 equal, K=3 top 5%).

## Caveats

- One generator, 500 MATH-500 questions; the eval set (400 questions) is itself noisy and its questions overlap across splits, so the 1000 splits are not independent.
- Only the DeBERTa setting has a real peak (N=2, at the edge of the search range).
- Planned confirmation on the 4,500-question MATH held-out set (Skywork scorer) is not generated yet.
