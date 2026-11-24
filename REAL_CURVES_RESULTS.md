# Real Best-of-N curves

**Question:** on real LLM data, does accuracy of the kept answer keep rising with N, or does it peak and decline (reward over-optimization)?

Code: [`run_experiment.py`](run_experiment.py) · Data: `table.csv`, `results/math500_*.csv`

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
