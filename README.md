# Best-of-N starter

How many answers N should Best-of-N sample when the scorer can be fooled?
And how much graded data do you need to choose N well?

## Run

```bash
pip install numpy matplotlib

# on a GPU (e.g. Google Colab), once:
pip install vllm transformers datasets
python initialize_experiment.py # writes table.csv; download it

# after collecting table.csv
python run_experiment.py
```

## Testing whether the decline at large N is real

`run_experiment.py` computes the Best-of-N curve exactly and, when the best N is below the
largest N, prints a bootstrap-over-questions 95% interval for the decline. The best N is
picked on the same data, so that test is optimistic. For an honest test, pass an N chosen
beforehand: `python run_experiment.py results/math_skywork.csv 64`.

`jobs/scholar.sub` runs the experiment on Scholar (GSM8K, MATH-500, or held-out MATH) as
chunked jobs; its header explains how to submit and merge.

## Results

| run (256 answers per question) | questions | V(1) | V(16) | V(256) | shape |
|---|---|---|---|---|---|
| GSM8K, Skywork-0.6B | 1319 | 0.853 | 0.899 | 0.909 | rises |
| held-out MATH, Skywork-0.6B | 3000 | 0.758 | 0.812 | 0.808 | rises, then flat |
| MATH-500, DeBERTa | 500 | 0.749 | 0.738 | 0.702 | falls |

No interior optimum was confirmed. Two apparent peaks (GSM8K on 100 questions, Skywork on
MATH-500) did not replicate on new questions; for the second, N=64 and 3,000 questions were
fixed in advance. The DeBERTa decline (V(1) - V(256) = 0.047, 95% CI [0.023, 0.072]) is
exploratory and has not been replicated.

The proofs and full results are in `proof/proof.pdf`; `proof/verify_proof.py` recomputes the
numbers (the MATH-500 tables are read from git commit 7a412e2).

## Project overview

See [`AGENTS.md`](AGENTS.md) for the project state, the experiment index and the results write-ups (`docs/`).
