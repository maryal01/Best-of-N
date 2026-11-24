# AGENTS.md

Course project (CS578 Stat ML): **how well can the optimal N in Best-of-N be estimated from finite calibration data?**
Proposal: `proposal.pdf`. Proofs: `proof/proof.pdf`, `proof/verify_proof.py`.

## Conventions
- Code: simple, readable, commented; **no nested loops**; state assumptions explicitly.
- Commit and push only when asked. Work on `main` and keep the number of branches small.
- Results go in `docs/*_RESULTS.md`, plans and status in `docs/EXPERIMENTS.md`, not in this file.
- Never put personal paths or machine-specific setup in the repo.
- Run scripts from the repo root.

## Model
Verifier score S has K ordered levels, `p_j = P(S=s_j)`, `m_j = E[U|S=s_j]`, U in {0,1} (correct).
Best-of-N expected utility (exact): `V(N) = sum_j m_j (F_j^N - F_{j-1}^N)`, `F_j = p_1+..+p_j`.
For K=3: `V(N) = m3 + (m2-m3) a^N + (m1-m2) b^N`, `a=p1+p2`, `b=p1`.
Estimation: plug-in `p_hat_j`, `m_hat_j` from calibration data, `N_hat = argmax V_hat`.
**Regret = V(N*) - V(N_hat)**, always evaluated on the TRUE curve.

## Where to look
- `docs/EXPERIMENTS.md`: list of experiments, their status, and what is next. Results are in `docs/*_RESULTS.md`.

| Path | What |
|---|---|
| `initialize_experiment.py` | generate 256 answers/question (Qwen2.5-Math-1.5B-Instruct, vLLM), grade, score. Env vars: `DATASET` (gsm8k/math500/math), `START/END`, `SCORERS`, `ANSWERS` (cache), `SCORE_TOKEN_BUDGET` |
| `scoring.py` | batched reward-model scoring (checked against one-at-a-time scoring by `jobs/check_batched_scoring.py`) |
| `run_experiment.py` | real accuracy curve and bootstrap test of the decline after the best N |
| `experiments/synthetic/`, `experiments/real_data/` | code of the finite-data experiments |
| `jobs/` | environment setup and chunk runners (Slurm array job and a plain-bash runner); usage is in each script's header |
| `docs/` | experiment index and results write-ups with figures |
| `results/`, `table.csv` | tables `question_id,score,correct`, 256 rows per question (GSM8K 1319 q; MATH-500 500 q with Skywork-0.6B and DeBERTa) |
| `proof/` | monotonicity / optimal-N proofs and numerical checks |
