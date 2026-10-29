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