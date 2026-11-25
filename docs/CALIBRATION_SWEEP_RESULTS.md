# Calibration sweep on real LLM data

**Question:** on real data, how does the regret of the estimated best N shrink as we use more calibration questions? (Does the synthetic result of [`SIMULATION_RESULTS.md`](SIMULATION_RESULTS.md) hold on real data?)

Code: [`calibration_sweep.py`](../experiments/real_data/calibration_sweep.py) · Figure: [`calibration_sweep.png`](calibration_sweep.png)
`python experiments/real_data/calibration_sweep.py results/math500_reward-model-deberta-v3-large-v2.csv results/math500_Skywork-Reward-V2-Qwen3-0.6B.csv table.csv`

## Procedure (per random split, 1000 splits)

1. Randomly split the questions: **80% eval** (the truth) and a **20% calibration pool** (MATH-500: 400 / 100; GSM8K: 1055 / 264).
2. Real best N and its accuracy curve are computed on the eval questions (exact formula, as in `run_experiment.py`).
3. Take the first `size` questions of the calibration pool. Levels are rank bins within each question (see [`DISCRETIZATION_RESULTS.md`](DISCRETIZATION_RESULTS.md)), so only `m_j` is estimated: `m_hat_j` = average correctness of the answers in level j.
4. Plug `m_hat` into `V(N) = sum_j m_j (F_j^N - F_{j-1}^N)`, take `N_hat = argmax`.
5. **Regret** = real eval accuracy at the real best N minus real eval accuracy at `N_hat`.
6. **Floor** (discretization regret with unlimited data): the same, with `m_j` estimated on the eval questions themselves.
7. **Direct empirical baseline** (no levels, no model): on the same calibration questions compute the real Best-of-N accuracy curve (exact, without replacement) and take its argmax as `N_hat`. Regret is measured on the same eval questions, and the same splits are used, so the two methods can be compared pair by pair.

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

### MATH held-out + Skywork (3,000 questions: eval 2,400, calibration pool 600; real best N: median 29, 10-90% range 24-35; N=1 loses 0.0558, N=256 loses 0.0035)

| Layout | 10 | 25 | 50 | 100 | 200 | 400 | 600 | Floor |
|---|---|---|---|---|---|---|---|---|
| K=3 equal | 0.0060 | 0.0027 | 0.0019 | 0.0018 | 0.0018 | 0.0018 | 0.0018 | 0.0018 |
| K=5 equal | 0.0083 | 0.0048 | 0.0038 | 0.0034 | 0.0034 | 0.0034 | 0.0034 | 0.0034 |
| K=3 top 5% | 0.0071 | 0.0046 | 0.0038 | 0.0036 | 0.0035 | 0.0035 | 0.0035 | 0.0035 |
| **Empirical (direct)** | 0.0065 | 0.0043 | 0.0033 | 0.0028 | 0.0025 | 0.0023 | 0.0021 | |

All regrets are at most 0.008: the real curve is nearly flat after N=30 (going to N=256 loses only 0.0035), so a wrong N costs little by construction (see [`REAL_CURVES_RESULTS.md`](REAL_CURVES_RESULTS.md)). Standard errors are at most 0.0004.

## Comparison with the direct empirical baseline

Mean regret of the baseline, and the **paired difference** (K-level model minus empirical, same splits; negative = our method is better, mean +- standard error).

### Empirical baseline regret

| Table | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| MATH-500 DeBERTa | 0.0170 | 0.0112 | 0.0070 | 0.0033 | |
| MATH-500 Skywork | 0.0122 | 0.0088 | 0.0067 | 0.0054 | |
| GSM8K Skywork | 0.0072 | 0.0058 | 0.0045 | 0.0039 | 0.0030 |

### Paired difference in regret (layout minus empirical)

| Table | Layout | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| MATH-500 DeBERTa | K=3 equal | **-0.0065** | **-0.0019** | +0.0009 | +0.0030 | |
| | K=5 equal | **-0.0063** | **-0.0030** | -0.0007 | +0.0009 | |
| | K=3 top 5% | **-0.0017** | **-0.0008** | +0.0001 | +0.0012 | |
| MATH-500 Skywork | K=3 equal | **-0.0055** | **-0.0065** | **-0.0061** | **-0.0051** | |
| | K=5 equal | **-0.0012** | **-0.0018** | **-0.0018** | **-0.0014** | |
| | K=3 top 5% | +0.0031 | +0.0042 | +0.0056 | +0.0067 | |
| GSM8K Skywork | K=3 equal | +0.0025 | +0.0012 | +0.0010 | +0.0010 | +0.0018 |
| | K=5 equal | +0.0024 | +0.0007 | +0.0001 | **-0.0004** | **-0.0004** |
| | K=3 top 5% | -0.0001 | **-0.0015** | **-0.0020** | **-0.0025** | **-0.0027** |

(Standard errors are 0.0001 to 0.0007; bold = more than 2 standard errors below zero.)

Held-out MATH (paired difference, layout minus empirical; 10 / 25 / 50 / 100 / 200 / 400 / 600 calibration questions):

| Layout | 10 | 25 | 50 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|---|---|
| K=3 equal | **-0.0005** | **-0.0016** | **-0.0013** | **-0.0010** | **-0.0007** | **-0.0004** | **-0.0002** |
| K=5 equal | +0.0017 | +0.0005 | +0.0005 | +0.0006 | +0.0009 | +0.0011 | +0.0013 |
| K=3 top 5% | +0.0006 | +0.0003 | +0.0006 | +0.0008 | +0.0010 | +0.0012 | +0.0014 |


**Reading:**
- **Very few calibration questions (10-25): the K-level model beats the empirical curve on MATH-500**, by up to 0.0065 (a 40-50% lower regret on DeBERTa). The empirical argmax is noisy when the curve is nearly flat, and the model smooths it by pooling answers into 3 to 5 levels.
- **With more data the empirical method catches up and overtakes** on DeBERTa (crossover around 50 questions for K=3, around 100 for K=5), because the model's discretization floor (0.0004 for K=3, 0.0000 for K=5) is no longer negligible against the shrinking estimation error.
- **The outcome depends on the layout and the table**, not uniformly in our favour: on Skywork-MATH-500 K=3 and K=5 equal win at every size while K=3 top 5% loses (its floor is 0.012); on GSM8K (monotone curve) the model with K=3 equal loses at every size, K=5 equal is roughly tied, and K=3 top 5% wins.
- Honest summary: the model is a **small-data advantage**, which matches the motivation of the project, but it is not a free win; a poorly chosen layout (floor too high) loses to the plain empirical method.

## Findings

- **Regret decreases with calibration size in every setting**, qualitatively matching the synthetic simulation.
- **It levels off at the discretization floor** (Skywork K=3 top 5%: 0.0153 -> 0.0121 with floor 0.0120; GSM8K K=3 equal: 0.0048 vs floor 0.0047). Beyond that point more data does not help; a better layout does.
- **Absolute regrets are small** (<= 0.015, vs 0.25 in the synthetic rare-top-level case): on real data the curve near the optimum is flat, and a wrong N is not very costly except when it overshoots towards N=256 on DeBERTa.
- **DeBERTa:** the median `N_hat` is already 2 (the truth) with 10 questions; the mean `N_hat` is inflated by a few large overshoots, which matter because N=256 loses 0.049.
- **The synthetic failure mode ("top level never observed, so N_hat jumps to N_max") does not occur here**, because rank levels are defined within each question, so every question contributes answers to every level. The finite-data difficulty shows up as variance of `m_hat` instead.
- Because the layout sets the floor, the layout must be chosen without looking at the eval truth (here: fixed in advance: K=3 equal, K=5 equal, K=3 top 5%).

## Held-out MATH: what it adds

- **The sweep conclusions hold on 6x more questions** (2,400 eval questions, calibration up to 600): regret falls with calibration size and then stops at the discretization floor of the layout (0.0018 for K=3 equal, 0.0034-0.0035 for K=5 equal and K=3 top 5%).
- **K=3 equal beats the direct empirical method at every calibration size** (by 0.0002 to 0.0016), but the gap shrinks: the empirical regret keeps falling (0.0065 to 0.0021) while K=3 equal is stuck at its floor 0.0018; with a few more calibration questions the empirical method would catch up, as it did on MATH-500 + DeBERTa after about 50 questions. K=5 equal and K=3 top 5% (higher floors) lose to the empirical method at all sizes.
- **Absolute differences are tiny** because the curve is flat after N=30; this is a setting where choosing N matters little, so it is a weak test of the method. The informative settings are those with a sharp peak (MATH-500 + DeBERTa).
- The empirical method's median `N_hat` keeps moving down with more data (127 to 33) towards the true N* of 29, while the K-level models settle at the N implied by their coarse levels (86, 150, 256): the floor in action.

## Caveats

- One generator, 500 MATH-500 questions; the eval set (400 questions) is itself noisy and its questions overlap across splits, so the 1000 splits are not independent.
- Only the DeBERTa setting has a real peak (N=2, at the edge of the search range).
- Planned confirmation on the 4,500-question MATH held-out set (Skywork scorer) is not generated yet.
