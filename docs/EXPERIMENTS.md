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
| 9 | Coding with N up to 256: own generation (Qwen2.5-Coder-1.5B, 256 solutions per problem) on HumanEval+ and MBPP+, EvalPlus tests in a sandbox, six reward models (CodeScaler, Skywork-V2 0.6B/1.7B, InternLM-2-1.8B, Skywork-Llama-8B, DeBERTa) | done: no significant decline for any scorer up to N=256; only CodeScaler shows a very weak, non-significant one; general-purpose scorers keep rising; DeBERTa uninformative | `experiments/coding/{generate_code,evaluate_code,score_generated}`, `scorers.py` | [CODING_RESULTS.md](CODING_RESULTS.md) |
| 10 | Synthetic test of the level layout: continuous score with an over-optimized tail of width 20%, 5% or 1% (N* = 3, 7, 16); equal-mass, top-5% and dyadic layouts vs the direct empirical method | done: only layouts that resolve the tail work; dyadic has floor <= 0.003 everywhere and beats the empirical method with few data; empirical wins at large n when the tail is narrower than the finest level | `experiments/synthetic/layout_simulation.py` | [LAYOUT_SIMULATION_RESULTS.md](LAYOUT_SIMULATION_RESULTS.md) |
| 11 | Held-out MATH (our 3,000 questions x 256 answers) re-scored with InternLM-2-1.8B, the paper's pair, with a pre-registered test (N=11 vs N=256) | done: rise to N of about 15, then a very small fall (0.008 by N=256); pre-registered test borderline (not significant), secondary N=32 vs 256 borderline (significant); sweep: dyadic and empirical tied | `experiments/rescore/score_saved_answers.py` | [REAL_CURVES_RESULTS.md](REAL_CURVES_RESULTS.md), [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md) |

## Summary of findings

1. **Over-optimization (accuracy rises, then falls as N grows) exists but is usually small.** Clear and significant only with a weak scorer: MATH-500 + DeBERTa (best N = 2, loss 0.049 by N=256). The three pairs of Khalaf et al. on PPE peak at N of 7 to 13 and lose about 0.02 by N=32, significant only pooled over the pairs ([PPE_RESULTS.md](PPE_RESULTS.md)); on our 3,000 MATH questions the same InternLM pair loses only 0.008 by N=256 (borderline). Not detected: Skywork scorers on MATH and GSM8K (two independent runs agree) and code with six reward models up to N=256 ([CODING_RESULTS.md](CODING_RESULTS.md)).
2. **Estimating the best N from little data is risky when the harmful top level is rare.** In the synthetic worlds a rare top level is often missed in small samples, the estimated curve then looks increasing and the regret is large (0.25 at n=50 for p3=0.01); more levels need more data ([SIMULATION_RESULTS.md](SIMULATION_RESULTS.md), [K_LEVEL_SIMULATION_RESULTS.md](K_LEVEL_SIMULATION_RESULTS.md)).
3. **The K-level model works only if its levels resolve the top tail where Best-of-N looks.** Equal-mass K=3 and K=5 fail on tables with a peak (floors up to 0.27 synthetic, 0.02 real); a dyadic layout (one level per power of two N) matches the direct empirical method on real tables and has a floor <= 0.003 in every synthetic tail width ([LAYOUT_SIMULATION_RESULTS.md](LAYOUT_SIMULATION_RESULTS.md), [CALIBRATION_SWEEP_RESULTS.md](CALIBRATION_SWEEP_RESULTS.md)).
4. **Versus the direct empirical method:** the K-level model is better with very few calibration questions (10 to 25), and the empirical method catches up or wins with more data when the layout's floor is not negligible.
5. **The stakes are small in most real settings** (a wrong N costs at most 0.02 to 0.05 accuracy), larger with weak scorers.

## Next

1. **Replicate the one clearly significant peak on new questions (running).** Score our 3,000 held-out MATH answers with the DeBERTa reward model (`experiments/rescore/score_saved_answers.py --scorer deberta`, same input format as for MATH-500: question and answer as a pair, 512 tokens, left truncation, fp32).
   **Pre-registered before any DeBERTa score on these questions existed:** the primary test is the accuracy at N=2 (the best N on MATH-500 + DeBERTa) minus the accuracy at N=256, with the 95% bootstrap interval over the 3,000 questions (run_experiment.py with N=2); the peak replicates if the lower bound of the interval is above 0. Expected from MATH-500: a decline of about 0.05. Secondary: best N and shape of the curve, gain from N=1 to N=2.
2. Write the final report from this page (figures: `docs/*.png`).
3. Optional: a finer dyadic ladder for very narrow tails (1/64, 1/128, ...); shrinkage or a prior on `m_hat` for very small calibration sets.

## Known pitfalls

- Reward scores are not comparable across questions, so levels are defined by rank within each question.
- With a flat or monotone real curve the regret is small by construction.
- Only MATH-500 + DeBERTa has a clearly significant peak (N*=2) and it has not been replicated on new questions yet; the other peaks are small (about 0.02 or less) and significant only when pooled or borderline.
- Level layouts matter more than K: always check that the top levels are thinner than the tail where accuracy falls.
