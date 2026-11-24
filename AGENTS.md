# AGENTS.md: project state for the next agent

Course project (CS578 Stat ML): **how well can the optimal N in Best-of-N be estimated from finite
calibration data?** Team: Baris Sarper Tezcan (user of this file), Merve Rabia Barin, Manish Aryal.
Proposal: `proposal.pdf`. Proofs: `proof/proof.pdf`, `proof/verify_proof.py`.

User preferences: chat in Turkish; code simple, readable, commented, **no nested loops**; state assumptions
explicitly; **commit/push only when asked**. Work on `main` (keep the number of branches small). Push needs SSH:
`origin` is `git@github.com:maryal01/Best-of-N.git`. Results go in their own `*_RESULTS.md` files, not in this file.

## Model
Verifier score S has K ordered levels, `p_j = P(S=s_j)`, `m_j = E[U|S=s_j]`, U in {0,1} (correct).
Best-of-N expected utility (exact): `V(N) = sum_j m_j (F_j^N - F_{j-1}^N)`, `F_j = p_1+..+p_j`.
For K=3: `V(N) = m3 + (m2-m3) a^N + (m1-m2) b^N`, `a=p1+p2`, `b=p1`.
Estimation: plug-in `p_hat_j`, `m_hat_j` from calibration data, `N_hat = argmax V_hat`.
**Regret = V(N*) - V(N_hat)**, always evaluated on the TRUE curve.

## Where the results are
| Doc | Contents | Code |
|---|---|---|
| [`REAL_CURVES_RESULTS.md`](REAL_CURVES_RESULTS.md) | real Best-of-N curves, GSM8K and MATH-500, two scorers; only MATH-500 + DeBERTa has a significant peak (N*=2) | `run_experiment.py` |
| [`SIMULATION_RESULTS.md`](SIMULATION_RESULTS.md) | synthetic finite-data simulation, 3 regimes of a rare top level | `simulation.py` |
| [`DISCRETIZATION_RESULTS.md`](DISCRETIZATION_RESULTS.md) | does the K-level rank model reproduce the real best N with unlimited data? | `discretization_check.py` |
| [`CALIBRATION_SWEEP_RESULTS.md`](CALIBRATION_SWEEP_RESULTS.md) | regret vs number of calibration questions on real data | `calibration_sweep.py` |

Gotcha already fixed in `simulation.py`: if the top level is unseen (`p_hat3=0`) the estimated curve is flat after `b^N`
underflows to 0, so `argmax` stopped at N~55 instead of `N_MAX` and understated regret about 2x. The code sets `N_MAX` explicitly.

## Repo map
| File | What |
|---|---|
| `initialize_experiment.py` | generate 256 answers/question (Qwen2.5-Math-1.5B-Instruct, vLLM), grade, score. Env vars: `DATASET` (gsm8k/math500/math), `START/END`, `SCORERS`, `ANSWERS` (cache), `SCORE_TOKEN_BUDGET`. Prints per-stage timings (`stamp()`) |
| `scoring.py` | batched reward-model scoring (validated against one-at-a-time scoring by `jobs/check_batched_scoring.py`) |
| `run_experiment.py` | real accuracy curve + bootstrap test of the decline after the best N |
| `jobs/setup_env.sh`, `run_chunk.sh`, `gilbreth.sub`, `run_local.sh` | environment setup and chunk runners (see Compute). `jobs/scholar.sub` is the old Scholar version |
| `results/*.csv`, `table.csv` | tables `question_id,score,correct`, 256 rows per question (GSM8K 1319 q; MATH-500 500 q with Skywork-0.6B and DeBERTa) |
| `proof/` | monotonicity / optimal-N proofs and numerical checks |

## In progress / next
1. **MATH held-out** (4,500 MATH test questions not in MATH-500, Skywork scorer only; 30 chunks of 100 = 3,000 q planned): re-test the peak and the calibration sweep on new data. Not generated yet. Chunk 0 is a timing run on Gilbreth (`logs/bon-init-*.out`); then split the chunks across Gilbreth, ferrari and porsche.
2. Add the sweep figure and a short conclusion to the write-up; decide final layouts (suggested: K=3 equal, K=5 equal, K=3 top 5%).
3. Optional: shrinkage/prior on `m_hat` for tiny calibration sets.

## Compute (never write big files to `$HOME`: use `/scratch` on Gilbreth, `/data/tezcanb/bon` on porsche/ferrari)
- **Gilbreth** (`gilbreth-fe02` is a login node: no heavy work). Account `aliaga`, only the preemptible `standby` QoS is practical (4 h limit, up to 24 GPUs).
  Env `/scratch/gilbreth/tezcanb/bon-env` (vLLM 0.34, torch 2.14+cu130, built by `jobs/setup_env.sh`).
  From the repo root: `mkdir -p logs; sbatch --array=0 jobs/gilbreth.sub math`, then `--array=1-29`. Chunk k -> `results/math_part<k>.csv`; merge as in the
  header of `jobs/gilbreth.sub`. Generated answers are cached in `/scratch/gilbreth/tezcanb/bon_answers/` (a requeued chunk only redoes scoring).
  Scratch quota 2M files (about 25% used); home is 25 GB (about 6 GB used).
- **porsche** (4x RTX 6000 Ada 48 GB, driver 550, about 100 users) and **ferrari** (4x RTX PRO 6000 Blackwell 96 GB, driver 570, other users' memory on 3 GPUs):
  env `/data/tezcanb/bon/bon-env` (vLLM 0.11.0, torch 2.8+cu128; both import fine), repo copy `/data/tezcanb/bon/Best-of-N` (re-sync with rsync before use).
  One process per GPU: `BASE=/data/tezcanb/bon nohup bash jobs/run_local.sh math <gpu> <chunk>... &`. **Not yet test-run end to end.**
  Different vLLM version than Gilbreth (same sampling settings). Copy `results/math_part*.csv` back to Gilbreth.
- Batched scoring vs one-at-a-time (A100): DeBERTa identical (max diff 1e-5), Skywork max diff 0.027 on scores with std 2.8 (fp16 noise), 5.9x / 3.9x faster.
  Grading (`math_verify`, CPU, sequential) may be the next bottleneck: check the `stamp()` output.
