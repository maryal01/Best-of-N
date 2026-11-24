# Experiments: status and next steps

Each experiment has its code in `experiments/` (or the repo root for the data pipeline) and its results in this folder.

| # | Experiment | Status | Code | Results |
|---|---|---|---|---|
| 1 | Real Best-of-N curves (GSM8K, MATH-500; Skywork-0.6B and DeBERTa scorers) | done | `run_experiment.py` | [REAL_CURVES_RESULTS.md](REAL_CURVES_RESULTS.md) |
| 2 | Synthetic finite-data simulation (3 levels, rare top level) | done | `experiments/synthetic/simulation.py` | [SIMULATION_RESULTS.md](SIMULATION_RESULTS.md) |
| 3 | Discretization check: does the K-level rank model reproduce the real best N with unlimited data? | done | `experiments/real_data/discretization_check.py` | [DISCRETIZATION_RESULTS.md](DISCRETIZATION_RESULTS.md) |
| 4 | Calibration sweep on real data: regret vs number of calibration questions | done on MATH-500 and GSM8K | `experiments/real_data/calibration_sweep.py` | [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |
| 4b | Direct empirical baseline vs our K-level method (same calibration questions, same eval questions) | done on MATH-500 and GSM8K (part of the sweep) | `experiments/real_data/calibration_sweep.py` | [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |
| 5 | Confirmation on the MATH held-out set (4,500 test questions not in MATH-500, Skywork scorer) | in progress: answers are being generated | `initialize_experiment.py` (`DATASET=math`), `jobs/` | not yet |
| 6 | Synthetic simulation for K = 3, 5, 10, 20 levels (effect of K and calibration size on regret; coarse-graining of a K=20 world) | done | `experiments/synthetic/k_level_simulation.py` | [K_LEVEL_SIMULATION_RESULTS.md](K_LEVEL_SIMULATION_RESULTS.md) |

## Next

1. **Experiment 5.** Generate the held-out data (30 chunks of 100 questions = 3,000 questions planned, chosen in advance for about 94% power to detect a 1-point decline), merge the parts, then test the decline at the pre-chosen N=64 with `run_experiment.py <table> 64` and repeat the calibration sweep (experiment 4) on the new data. See the header of `jobs/gilbreth.sub` for how to run and merge the chunks.
2. Add the sweep figure and a short conclusion to the project write-up.
3. Decide the final level layouts for the write-up (current suggestion: K=3 equal, K=5 equal, K=3 top 5% as the "rare, misleading top level" case).
4. Optional: shrinkage or a prior on `m_hat` for very small calibration sets.

## Known pitfalls

- Reward scores are not comparable across questions, so levels are defined by rank within each question.
- With a flat or monotone real curve the regret is small by construction; only MATH-500 + DeBERTa has a significant peak (N*=2).
