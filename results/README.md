# Data tables

All tables have the columns `question_id,score,correct`: one row per generated answer, `score` from a reward
model, `correct` = 1 if the answer is right. The analysis scripts group rows by `question_id`, so the row order
does not matter. Every math table has 256 answers per question.

| File | Content | Used by |
|---|---|---|
| `../table.csv` | GSM8K, 1,319 questions, Skywork-Reward-V2-Qwen3-0.6B | `run_experiment.py` (default), `docs/` GSM8K results |
| `math500_Skywork-Reward-V2-Qwen3-0.6B.csv` | MATH-500, 500 questions, Skywork scorer | `experiments/real_data/`, `docs/` |
| `math500_reward-model-deberta-v3-large-v2.csv` | MATH-500, same answers, DeBERTa (weakest) scorer: the only table with a significant peak (N*=2) | `experiments/real_data/`, `docs/` |
| `math_skywork.csv` | Held-out MATH, 3,000 questions, Skywork scorer; **Manish's run** (Scholar) | README results table, `proof/verify_proof.py` |
| `math_skywork_gilbreth.csv` | The **same 3,000 questions, an independent second run** (Gilbreth, porsche, ferrari) | `docs/REAL_CURVES_RESULTS.md`, `docs/CALIBRATION_SWEEP_RESULTS.md` |
| `mbpp_plus_*.csv`, `mbpp_plus_meta.csv` | Coding feasibility: PPE MBPP+ (32 answers per row), CodeScaler and DeBERTa scorers; meta = row to task and generator | `docs/REAL_CURVES_RESULTS.md` |
| `ppe_<dataset>_<scorer>.csv`, `ppe_<dataset>_meta.csv` | PPE Best-of-K MATH, MMLU-Pro, GPQA (512 questions x 32 answers) with InternLM-2-1.8B and Skywork-Reward-Llama-3.1-8B; meta = generator of each row | `docs/PPE_RESULTS.md` |
| `gen_{mbpp,humaneval}_{codescaler,deberta}.csv` | Our own generation: 256 solutions per MBPP+ / HumanEval+ problem (Qwen2.5-Coder-1.5B), correct = passes EvalPlus base and plus tests | `docs/CODING_RESULTS.md` |
| `default64_table.csv`, `weak_*64_table.csv`, `table256.csv` | Early 100-question explorations | `proof/verify_proof.py` only |
| `*_accuracy_curve.png` | Plots written by `run_experiment.py` | none (derived) |

The two held-out files are two valid samples of the same experiment (same questions, generator, prompt, grader and
scorer; different random generations and hardware). They agree (best N 24 vs 29, no significant decline after N=64
in either), neither is "the correct one"; see `docs/REAL_CURVES_RESULTS.md`.

Raw chunk outputs (`*_part<k>.csv`, 100 questions each) are intermediate results of the chunked jobs. They are not
tracked (`.gitignore`); the chunked jobs write them locally and the job header shows how to merge them
(`head -n 1` of one part, then `tail -n +2` of all parts). They can also be recovered from a merged table by
splitting on `question_id // 100`.
