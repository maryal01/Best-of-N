# Discretization check

**Question:** if we replace the continuous reward score by K discrete levels, does the K-level model still predict the real best N, **even with unlimited calibration data**? (This isolates discretization error from finite-data error.)

Code: [`discretization_check.py`](discretization_check.py) (`python experiments/real_data/discretization_check.py <table.csv> ...`)

## Setup

- Levels are **rank bins within each question** (reward scores are not comparable across questions). The lowest-scored `p_1` fraction of a question's answers is level 1, and so on. Answers tied in score share their mean correctness.
- Because every question has the same `p_j`, `p_j` is known by design and the pooled `m_j` (average correctness per level, over all questions) makes the model curve exactly the i.i.d. curve averaged over questions.
- Model: `V(N) = sum_j m_j (F_j^N - F_{j-1}^N)`.
- Three curves per table: **real** (continuous argmax, N answers without replacement), **iid** (same, with replacement), **model** (K levels).
- **Disc. regret** = real accuracy at the real best N minus real accuracy at the model's best N.
- `m_j` here uses ALL questions of the table, so there is no calibration noise.

## Results

Real best N: DeBERTa 2 (acc 0.7513), MATH-500 Skywork 70 (0.8100), GSM8K 256 (0.9090).

| Layout (masses p_j) | MATH-500 DeBERTa: model N / disc. regret | MATH-500 Skywork: model N / disc. regret | GSM8K: model N / disc. regret |
|---|---|---|---|
| K=3, top 10% (0.5, 0.4, 0.1) | 2 / 0.0000 | 256 / 0.0120 | 256 / 0.0000 |
| K=3, top 5% | 2 / 0.0000 | 256 / 0.0120 | 256 / 0.0000 |
| K=3, top 2% | 3 / 0.0015 | 256 / 0.0120 | 256 / 0.0000 |
| K=3, top 1% | 4 / 0.0031 | 256 / 0.0120 | 256 / 0.0000 |
| K=3, equal | 2 / 0.0000 | 86 / 0.0002 | 86 / 0.0046 |
| K=5, equal | 2 / 0.0000 | 150 / 0.0034 | 150 / 0.0022 |
| K=10, equal | 2 / 0.0000 | 256 / 0.0120 | 256 / 0.0000 |

Largest `|model curve - iid curve|` (absolute accuracy) over N: 0.005 to 0.033 depending on layout and table.

## Findings

- On the setting that matters (MATH-500 + DeBERTa) every layout recovers the real best N or comes within 2 of it (regret <= 0.003). Very thin top levels (1-2%) overshoot slightly.
- On Skywork-MATH-500 (flat peak at N=70) layouts with a wide, coarse top level say "keep increasing" (N=256), costing 0.012, because inside the top bin the model picks at random while the real rule picks the best score. Equal-mass bins (K=3 or K=5) track the peak much better (0.0002 and 0.0034).
- Discretization is therefore not a problem large enough to invalidate the calibration experiment, but it sets a floor on regret that more data cannot remove (the dotted lines in `calibration_sweep.png`).
- Suggested layouts for the calibration experiment: K=3 equal, K=5 equal, and K=3 top-5% as the "rare, misleading top level" stress case.
