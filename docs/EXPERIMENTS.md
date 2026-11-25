# Experiments: status and next steps

Each experiment has its code in `experiments/` (or the repo root for the data pipeline) and its results in this folder.

| # | Experiment | Status | Code | Results |
|---|---|---|---|---|
| 1 | Real Best-of-N curves (GSM8K, MATH-500; Skywork-0.6B and DeBERTa scorers) | done | `run_experiment.py` | [REAL_CURVES_RESULTS.md](REAL_CURVES_RESULTS.md) |
| 2 | Synthetic finite-data simulation (3 levels, rare top level) | done | `experiments/synthetic/simulation.py` | [SIMULATION_RESULTS.md](SIMULATION_RESULTS.md) |
| 3 | Discretization check: does the K-level rank model reproduce the real best N with unlimited data? | done | `experiments/real_data/discretization_check.py` | [DISCRETIZATION_RESULTS.md](DISCRETIZATION_RESULTS.md) |
| 4 | Calibration sweep on real data: regret vs number of calibration questions | done on MATH-500 and GSM8K | `experiments/real_data/calibration_sweep.py` | [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |
| 4b | Direct empirical baseline vs our K-level method (same calibration questions, same eval questions) | done on MATH-500 and GSM8K (part of the sweep) | `experiments/real_data/calibration_sweep.py` | [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |
| 5 | Confirmation on the MATH held-out set (3,000 of the 4,500 test questions not in MATH-500, Skywork scorer; run independently by us and by Manish, both agree) | done: pre-registered N=64 vs N=256 test not significant (the MATH-500 peak does not replicate); sweep on it done (K=3 equal beats the empirical method at every size, small absolute gaps) | `initialize_experiment.py` (`DATASET=math`), `jobs/` | [REAL_CURVES_RESULTS.md](REAL_CURVES_RESULTS.md), [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |
| 6 | Synthetic simulation for K = 3, 5, 10, 20 levels (effect of K and calibration size on regret; coarse-graining of a K=20 world) | done | `experiments/synthetic/k_level_simulation.py` | [K_LEVEL_SIMULATION_RESULTS.md](K_LEVEL_SIMULATION_RESULTS.md) |
| 7 | Coding feasibility: MBPP+ (PPE pre-generated, 32 solutions per row), CodeScaler-1.7B and DeBERTa scorers | done (feasibility only): monotone curves, no peak, so no sweep | `experiments/coding/score_ppe_mbpp.py` | [REAL_CURVES_RESULTS.md](REAL_CURVES_RESULTS.md) |
| 8 | Reward-model pairs of Khalaf et al. on PPE Best-of-K (MATH and MMLU-Pro + InternLM-2-1.8B, GPQA + Skywork-Llama-3.1-8B, plus controls) | done: all three pairs peak at N of 7 to 13 (decline about 0.02, not significant individually); sweep: coarse levels fail, a dyadic layout matches the empirical method | `experiments/ppe/score_ppe.py` | [PPE_RESULTS.md](PPE_RESULTS.md) |
| 9 | Coding with N up to 256: own generation (Qwen2.5-Coder-1.5B, 256 solutions per problem) on HumanEval+ and MBPP+, EvalPlus tests in a sandbox, CodeScaler and DeBERTa scorers | done: with CodeScaler the curve flattens at N of 30 to 70 and falls by 0.005 to 0.015 by N=256 (not significant); DeBERTa is uninformative on code | `experiments/coding/{generate_code,evaluate_code,score_generated}` | [CODING_RESULTS.md](CODING_RESULTS.md) |

## Next

1. Add the sweep figure and a short conclusion to the project write-up.
2. Decide the final level layouts for the write-up (current suggestion: K=3 equal, K=5 equal, K=3 top 5% as the "rare, misleading top level" case).
3. Optional: shrinkage or a prior on `m_hat` for very small calibration sets.

## Known pitfalls

- Reward scores are not comparable across questions, so levels are defined by rank within each question.
- With a flat or monotone real curve the regret is small by construction; only MATH-500 + DeBERTa has a significant peak (N*=2).
