# Coding: Best-of-N beyond N = 32 (own generation, MBPP+ and HumanEval+)

**Question:** the pre-generated PPE MBPP+ data (32 answers per question) showed monotone curves up to N = 32 ([`REAL_CURVES_RESULTS.md`](REAL_CURVES_RESULTS.md)). With many more solutions, does accuracy of the kept solution peak and then fall at larger N?

Code: [`generate_code.py`](../experiments/coding/generate_code.py), [`evaluate_code.sh`](../experiments/coding/evaluate_code.sh), [`score_generated.py`](../experiments/coding/score_generated.py) · Data: `results/gen_{mbpp,humaneval}_{codescaler,deberta}.csv`

## Method

1. **Generation.** Qwen2.5-Coder-1.5B-Instruct (vLLM), **256 solutions per problem** (temperature 0.8, top-p 0.95, at most 768 tokens), EvalPlus-style prompt. MBPP+ (378 problems, 96,768 solutions) and HumanEval+ (164 problems, 41,984 solutions).
2. **Tests.** EvalPlus (`sanitize` to extract the code, then `evaluate` with the base and the extra "plus" tests). A solution is correct if it passes both.
3. **Sandbox.** The programs are model-written, so the whole test run is inside a `bubblewrap` sandbox: read-only file system with `/home`, `/data`, `/scratch` emptied (only the python env, the EvalPlus cache and the samples folder are bound back), no network, own process namespace, killed with the parent, plus EvalPlus' own time and memory limits per solution. The sandbox was checked first with programs that delete a file, write outside the work folder, open a socket and loop forever: none got through (the file survived, nothing was written outside, the network call and the endless loop failed).
4. **Scoring.** CodeScaler-1.7B (gets the extracted code) and the DeBERTa reward model (gets the full reply).

| | MBPP+ | HumanEval+ |
|---|---|---|
| Problems x solutions | 378 x 256 | 164 x 256 |
| Mean correctness (pass@1) | 0.565 | 0.613 |
| pass@100 / pass@256 | 0.859 / 0.878 | 0.933 / 0.951 |
| Problems no solution solves / all solutions solve | 46 / 32 | 8 / 24 |

## Results: accuracy of the kept solution

Exact curves (N solutions drawn without replacement from the 256), averaged over problems; "rank profile" = accuracy of the solution with the highest score, of those ranked 2-4, 5-16 and 17-64 (a top-ranked answer that is worse than the next ones is the signature of over-optimization).

| Table | N=1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | Best N (acc) | Rank profile: top-1 / 2-4 / 5-16 / 17-64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MBPP+ + CodeScaler | 0.565 | 0.619 | 0.649 | 0.667 | 0.678 | 0.684 | 0.686 | 0.685 | 0.681 | 71 (0.686) | 0.681 / 0.689 / 0.681 / 0.655 |
| HumanEval+ + CodeScaler | 0.613 | 0.668 | 0.699 | 0.718 | 0.731 | 0.739 | 0.740 | 0.731 | 0.726 | 49 (0.740) | 0.726 / 0.744 / 0.735 / 0.704 |
| MBPP+ + DeBERTa | 0.565 | 0.567 | 0.564 | 0.559 | 0.555 | 0.552 | 0.551 | 0.549 | 0.548 | 2 (0.567) | 0.548 / 0.553 / 0.553 / 0.566 |
| HumanEval+ + DeBERTa | 0.613 | 0.615 | 0.613 | 0.613 | 0.614 | 0.615 | 0.612 | 0.606 | 0.604 | 2 (0.615) | 0.604 / 0.612 / 0.623 / 0.607 |

Decline tests (bootstrap over problems, 95% interval):

| Table | After the best N | After N=32 (chosen in advance) |
|---|---|---|
| MBPP+ + CodeScaler | 0.0052 [-0.012, 0.024] | 0.0027 [-0.020, 0.026] |
| HumanEval+ + CodeScaler | 0.0148 [-0.013, 0.043] | 0.0133 [-0.019, 0.047] |
| MBPP+ + DeBERTa | 0.0194 [-0.010, 0.050] | 0.0045 [-0.022, 0.031] |
| HumanEval+ + DeBERTa | 0.0115 [-0.040, 0.063] | n.s. |

## Findings

- **With CodeScaler the curve rises until N of about 30, flattens, and drifts down very slightly after N of 50 to 70** (MBPP+ peak 0.686 at N=71, 0.681 at N=256; HumanEval+ peak 0.740 at N=49, 0.726 at N=256). The declines (0.005 and 0.015) are **not statistically significant**; the direction agrees in both datasets.
- **The rank profile points the same way:** the top-ranked solution is slightly less accurate than ranks 2-4 in both datasets (MBPP+ 0.681 vs 0.689, HumanEval+ 0.726 vs 0.744), but again within noise.
- So with a strong code reward model there is, at most, a weak, flat peak: over-optimization on code is very mild up to N=256, in contrast to MATH with the weak DeBERTa scorer (N*=2, loss 0.049) and to the PPE pairs of Khalaf et al. (peak at N of 7 to 13, loss about 0.02 by N=32).
- **DeBERTa is uninformative on code** (HumanEval+ mean score: correct -2.61 vs wrong -2.46; MBPP+: -0.37 vs -0.62): the accuracy does not rise at all (about 0.565 and 0.613 at every N), and the top-scored solution is slightly worse than a random one (MBPP+ 0.548 vs 0.565). This is not the "weak but informative scorer" regime in which a peak appears.
- Consistent with the PPE MBPP+ feasibility run, which was monotone up to N=32 (its accuracy at N=32 is higher, 0.874, only because PPE kept problems with mixed outcomes).

## Other reward models on the same solutions: no peak either

The same 96,768 + 41,984 solutions were scored with general-purpose (not code-specific) reward models, to look for the "informative but imperfect scorer" regime in which MATH shows a peak: Skywork-Reward-V2-Qwen3 0.6B and 1.7B, InternLM-2-1.8B, Skywork-Reward-Llama-3.1-8B (see [`scorers.py`](../scorers.py)). Accuracy of the kept solution:

| Dataset | Scorer | Mean score correct vs wrong | N=1 | N=8 | N=32 | N=256 | Best N (acc) | Loss at N=256 |
|---|---|---|---|---|---|---|---|---|
| HumanEval+ | CodeScaler-1.7B (code-specific) | +8.56 vs +2.64 | 0.613 | 0.718 | 0.739 | 0.726 | 49 (0.740) | 0.015 |
| | Skywork-V2-1.7B | +7.98 vs +3.84 | 0.613 | 0.726 | 0.757 | 0.787 | 185 (0.787) | 0.000 |
| | Skywork-V2-0.6B | +5.60 vs +2.50 | 0.613 | 0.677 | 0.692 | 0.707 | 256 (0.707) | 0.000 |
| | InternLM-2-1.8B | +1.08 vs +0.35 | 0.613 | 0.650 | 0.653 | 0.677 | 256 (0.677) | 0.000 |
| | Skywork-Llama-3.1-8B | +25.85 vs +12.22 | 0.613 | 0.677 | 0.687 | 0.707 | 256 (0.707) | 0.000 |
| | DeBERTa | -2.61 vs -2.46 | 0.613 | 0.613 | 0.615 | 0.604 | 2 (0.615) | 0.011 |
| MBPP+ | CodeScaler-1.7B (code-specific) | +8.36 vs +1.40 | 0.565 | 0.667 | 0.684 | 0.681 | 71 (0.686) | 0.005 |
| | Skywork-V2-1.7B | +7.20 vs +2.34 | 0.565 | 0.664 | 0.683 | 0.679 | 104 (0.689) | 0.010 |
| | Skywork-V2-0.6B | +6.42 vs +1.98 | 0.565 | 0.634 | 0.644 | 0.646 | 145 (0.648) | 0.002 |
| | InternLM-2-1.8B | +0.78 vs +0.33 | 0.565 | 0.596 | 0.601 | 0.598 | 26 (0.601) | 0.003 |
| | Skywork-Llama-3.1-8B | +21.19 vs +5.02 | 0.565 | 0.634 | 0.636 | 0.646 | 256 (0.646) | 0.000 |
| | DeBERTa | -0.37 vs -0.62 | 0.565 | 0.559 | 0.552 | 0.548 | 2 (0.567) | 0.019 |

Pooled over both datasets (542 tasks), selection-free (N=32 fixed), decline = accuracy at N=32 minus accuracy at N=256:

| Scorer | HumanEval+ | MBPP+ | Pooled (95% interval) |
|---|---|---|---|
| CodeScaler-1.7B | +0.0133 | +0.0027 | +0.0059 [-0.013, +0.025] |
| Skywork-V2-1.7B | -0.0292 | +0.0041 | -0.0060 [-0.022, +0.011] |
| Skywork-V2-0.6B | -0.0152 | -0.0014 | -0.0056 [-0.025, +0.013] |
| InternLM-2-1.8B | -0.0243 | +0.0030 | -0.0052 [-0.025, +0.015] |
| Skywork-Llama-3.1-8B | -0.0204 | -0.0093 | -0.0127 [-0.032, +0.006] |
| DeBERTa | +0.0115 | +0.0045 | +0.0066 [-0.017, +0.030] |

- **No scorer shows a significant decline up to N=256** (pooled intervals all include 0). The general-purpose reward models are informative on code (correct solutions score clearly higher) and their curves keep rising: on HumanEval+ the best N is 185 to 256 and nothing is lost at N=256; on MBPP+ the curves flatten after N of about 30 to 100 with losses of at most 0.010.
- **The only scorer with a (still non-significant) decline on both datasets is the code-specific CodeScaler** (0.015 and 0.005); the general models rise further. So on code the "weak but informative scorer" regime does not produce the peak that it does on MATH or on the PPE reasoning sets: over-optimization of code reward models is, at most, very mild up to N=256 in these 542 tasks.

## Caveats
- One small generator (1.5B), one strong code scorer; 164 and 378 problems, so the declines cannot be told apart from zero.
- Test-based correctness uses EvalPlus' base and plus tests only; a few problems have no correct solution (46 and 8).
- Results of generation and scoring were produced on different machines (vLLM 0.11), sampling settings as stated above.
