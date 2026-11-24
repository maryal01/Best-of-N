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

With 100 questions the drop after the peak is within noise. About 1,260 questions
are needed for 80% power, so `initialize_experiment.py` now uses the whole GSM8K
test set (`N_QUESTIONS = 1319`; 1319 x 256 answers takes several hours on a GPU).
On Scholar this runs as 7 chunked jobs: see `jobs/scholar.sub` for how to submit and merge.

`run_experiment.py` plots the curve and, when the best N is below the largest N,
prints a bootstrap-over-questions 95% interval for the decline. The decline is
significant only if that interval excludes zero.

The proofs are in `proof/bon_monotonicity.pdf`