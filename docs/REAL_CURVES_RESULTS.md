# Real Best-of-N curves

**Question:** on real LLM data, does accuracy of the kept answer keep rising with N, or does it peak and decline (reward over-optimization)?

Code: [`run_experiment.py`](../run_experiment.py) · Data: `table.csv`, `results/math500_*.csv`

## Setup

- Generator: `Qwen/Qwen2.5-Math-1.5B-Instruct`, 256 answers per question, temperature 0.8, top-p 0.95, max 1024 tokens (2048 for MATH).
- Grading: GSM8K by last number / `\boxed{}`; MATH by symbolic comparison (`math_verify`).
- Scorers: `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (math-aware) and `OpenAssistant/reward-model-deberta-v3-large-v2` (2023 general-chat reward model, not trained on math, reads at most 512 tokens; the weakest scorer).
- Curve: for each question, the exact probability that the top-scored of N answers drawn at random (without replacement) is correct, averaged over questions. The decline after the best N gets a bootstrap-over-questions 95% interval.

## Results

| Data / scorer | Questions | Best N | Acc N=1 | Acc best N | Acc N=256 | Decline best N -> 256 (95% CI) |
|---|---|---|---|---|---|---|
| GSM8K / Skywork-0.6B | 1319 | 256 | 0.853 | 0.909 | 0.909 | none (monotone) |
| MATH-500 / Skywork-0.6B | 500 | 70 | 0.749 | 0.810 | 0.798 | 0.012 [-0.001, 0.026], not significant |
| MATH-500 / DeBERTa | 500 | **2** | 0.749 | 0.751 | 0.702 | 0.049 [0.026, 0.073], **significant** |

## Findings

- With a good scorer (Skywork) over-optimization is weak: monotone on GSM8K, a flat and statistically insignificant peak on MATH-500.
- With the weakest scorer (DeBERTa) there is a real peak at N=2: the second answer helps a little, and sampling more hurts (-0.049 by N=256).
- Therefore MATH-500 + DeBERTa is the main setting for the finite-data experiments (see [`CALIBRATION_SWEEP_RESULTS.md`](CALIBRATION_SWEEP_RESULTS.md)); the other two are controls.

## Caveats

- Best N for DeBERTa is at the very edge (N=2); for Skywork on MATH-500 the best N (70) is poorly determined because the curve is flat there.
- The peak is chosen on the same data it is tested on, so the "best N" interval is optimistic. A fixed-N test on new data is planned with the MATH held-out set (4,500 questions, not yet generated).
- Earlier 100-question exploratory tables (`results/default64_table.csv`, `weak_*`, `table256.csv`) show the same shape but are too small to judge the decline.

## MATH held-out confirmation (3,000 new questions, Skywork-0.6B)

Goal: re-test the MATH-500 peak (best N=70, decline to N=256 of 0.012, not significant) on new data, with the test fixed before looking: **accuracy at N=64 versus N=256** (N=64 chosen from MATH-500), on 3,000 questions (30 chunks of 100, chosen in advance for about 94% power to detect a 1-point decline).

Data: `results/math_skywork_gilbreth.csv` (our run): a pseudo-random sample (fixed hash order) of 3,000 of the 4,500 MATH test questions that are not in MATH-500; 256 answers per question; same generator, prompt, grader and scorer as MATH-500. 768,000 rows, question ids 0 to 2999, exactly 256 rows each, no duplicates. Accuracy of a random answer: 0.758. (Generated in parallel on three machines; chunks 22-29 and 14-17 used vLLM 0.11 on porsche/ferrari, the others vLLM 0.34 on Gilbreth; sampling settings are identical.)

| Quantity | Value |
|---|---|
| Accuracy at N=1 | 0.758 |
| Best N (chosen on this data) | 29, accuracy 0.814 |
| Accuracy at N=256 | 0.810 |
| Decline after the best N | 0.0034, 95% CI [-0.003, 0.010], not significant |
| **Pre-registered: decline from N=64 to N=256** | **0.0024, 95% CI [-0.003, 0.008], not significant** |

- **An independent run agrees.** Manish Aryal ran the same 3,000 questions separately on Scholar (`results/math_skywork.csv`, same generator, prompt, grader and scorer; different random samples and hardware). Its numbers: best N = 24 (accuracy 0.812), N=1: 0.758, N=256: 0.808, decline after N=64: 0.0028, 95% CI [-0.003, 0.009], not significant. Ours: best N = 29 (0.814), N=256: 0.810, 0.0024, [-0.003, 0.008]. The two runs give the same conclusion and agree to within about 0.002 everywhere.
- **The MATH-500 peak does not replicate as a significant decline.** On the new data accuracy rises until N of about 30 and is then flat; the decline is at most a few tenths of a point, and the 95% interval excludes a decline larger than 0.8 points.
- So with the Skywork-0.6B scorer on MATH there is no measurable over-optimization up to N=256. The only setting with a significant peak remains MATH-500 + DeBERTa (N*=2).
- The MATH-500 decline of 0.012 was most likely noise (its interval [-0.001, 0.026] already contained zero).

## MATH held-out + InternLM-2-1.8B (our 3,000 questions, N up to 256)

A direct test of the paper's pair (MATH + InternLM-2-1.8B, [`PPE_RESULTS.md`](PPE_RESULTS.md)) on more questions and larger N: the saved answers of our held-out run (3,000 questions x 256 answers by Qwen2.5-Math-1.5B-Instruct) were re-scored with InternLM-2-1.8B (`experiments/rescore/score_saved_answers.py`; table `results/math_gilbreth_internlm.csv`, same rows and correctness flags as `math_skywork_gilbreth.csv`).
**Pre-registered before looking:** accuracy at N0 = 11 (the N chosen for this pair from the other two PPE pairs) versus N = 256; secondary: N = 32 versus N = 256.

| Scorer (same 3,000 questions, same answers) | Mean score correct vs wrong | N=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | Best N (acc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| InternLM-2-1.8B | +1.22 vs +0.30 | 0.758 | 0.773 | 0.780 | 0.783 | 0.783 | 0.783 | 0.781 | 0.779 | 0.776 | 15 (0.783) |
| Skywork-0.6B | +8.24 vs +4.25 | 0.758 | 0.786 | 0.801 | 0.810 | 0.813 | 0.814 | 0.813 | 0.811 | 0.810 | 29 (0.814) |

| Test (bootstrap over 3,000 questions) | Decline (95% interval) | |
|---|---|---|
| **Pre-registered:** N=11 to N=256, InternLM | +0.0077 [-0.0004, +0.0158] | not significant (borderline) |
| Secondary: N=32 to N=256, InternLM | +0.0072 [+0.0000, +0.0140] | significant (borderline) |
| N=11 to N=256, Skywork-0.6B (same answers) | +0.0014 [-0.0063, +0.0090] | not significant |
| Paired difference InternLM minus Skywork | +0.0063 [-0.0045, +0.0171] | not significant |

- **A rise and a very small fall, at the edge of detectability even with 3,000 questions.** Accuracy rises from 0.758 to 0.783 (N about 8 to 32), then slowly falls to 0.776 at N=256. The pre-registered decline (0.0077) just misses significance, the secondary one just reaches it: the evidence is weak.
- **The effect is smaller and comes later than on PPE.** There, MATH + InternLM lost 0.023 between the peak (N=7) and N=32; here accuracy is flat between N=8 and N=32 and loses only 0.008 by N=256. The answers come from a different, much stronger generator (accuracy 0.76 here against 0.34 on PPE), so there are fewer hopeless questions for the scorer to be fooled on.
- **The weaker scorer falls more than the stronger one on the same answers** (InternLM 0.0077 vs Skywork 0.0014 from N=11), in the direction expected, but the difference is not significant. The rank profile agrees: the top-ranked answer of InternLM is slightly less accurate than ranks 5 to 16 (0.776 vs 0.785).
- **The stake is small:** moving from N=1 to N=11 gains 0.026 (InternLM) and the best fixed N is worth at most 0.008 over N=256.

## Coding feasibility (MBPP+, pre-generated solutions)

Data: `lmarena-ai/PPE-MBPP-Plus-Best-of-K` (507 rows = 298 distinct MBPP+ tasks x several generator models; each row has 32 solutions from ONE generator with a unit-test result per solution). A row is one "question" with 32 candidates, so N <= 32. Scored with `experiments/coding/score_ppe_mbpp.py`:
CodeScaler-1.7B (code reward model, gets the last code block) and the DeBERTa reward model (full text). Tables: `results/mbpp_plus_codescaler.csv`, `results/mbpp_plus_deberta.csv`, `results/mbpp_plus_meta.csv` (row -> task_id, generator).

| Scorer | Mean score, passing vs failing | Acc N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | Best N |
|---|---|---|---|---|---|---|---|---|
| CodeScaler-1.7B | 6.07 vs -2.05 | 0.561 | 0.712 | 0.812 | 0.858 | 0.870 | 0.874 | 32 |
| DeBERTa | 0.90 vs 0.35 | 0.561 | 0.640 | 0.700 | 0.736 | 0.749 | 0.753 | 32 |

- **Both curves are monotone increasing up to N=32: no over-optimization is visible.** The decline test from N=8 for DeBERTa is -0.018 (an increase), CI [-0.042, 0.006], not significant.
- Unlike on MATH, where DeBERTa (not trained on math) peaks at N=2, DeBERTa is informative on code (passing solutions score clearly higher), so it is not "weak" enough here.
- **Selection caveat:** every one of the 507 rows contains both passing and failing solutions (pass@32 = 1.0, no row with all or none passing), i.e. PPE kept tasks with mixed outcomes. Accuracies (0.56 to 0.87) are therefore not MBPP+ accuracies and the gain from Best-of-N is inflated.
- Conclusion: with N <= 32 this dataset behaves like a monotone control (similar to GSM8K), not like MATH-500 + DeBERTa. It is not a setting where estimating the best N is interesting (best N = N_max, regret is trivial), so no calibration sweep was run on it. A peak, if any, would need larger N (own generation, e.g. Qwen2.5-Coder-1.5B-Instruct with 256 samples) or a scorer that is easier to fool.
- Any calibration/eval split on this data must group by `task_id` (several rows share a task).
- With our own generation of 256 solutions per problem (so N up to 256), see [`CODING_RESULTS.md`](CODING_RESULTS.md): a flat curve with at most a very weak, not significant decline for CodeScaler.

