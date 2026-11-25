"""
Feasibility step for a coding experiment: score the pre-generated PPE-MBPP-Plus-Best-of-K solutions with reward models.

Dataset (lmarena-ai/PPE-MBPP-Plus-Best-of-K): 507 rows; each row = one MBPP+ task and 32 solutions written by ONE
generator model (`model_name`), with a unit-test result per solution (`scores`, True = passes all tests).
The same task can appear in several rows (different generators), so a later calibration/eval split must be done
by `task_id`, not by row. A row is treated as one "question" with 32 i.i.d. candidates (so N can go up to 32).

Output (same format as the math tables, so run_experiment.py and the sweep scripts work unchanged):
  results/mbpp_plus_<scorer>.csv   question_id (= row index), score, correct (1 = passes the tests)
  results/mbpp_plus_meta.csv       question_id, task_id, model_name

Scorers: CodeScaler-1.7B (code reward model, same base as Skywork-Reward-V2-Qwen3-1.7B) and the DeBERTa reward
model we already use as the weak scorer. CodeScaler gets the last markdown code block of a solution (its own
pipeline extracts code); DeBERTa gets the full text.

Run from the repo root on a GPU:  python experiments/coding/score_ppe_mbpp.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scoring import score_batched

SOLUTIONS_PER_ROW = 32
SCORERS = {                                   # name -> (model id, is a (question, answer) pair model, extract code first)
    "codescaler": ("LARK-Lab/CodeScaler-1.7B", False, True),
    "deberta": ("OpenAssistant/reward-model-deberta-v3-large-v2", True, False),
}
RESULTS = Path(__file__).resolve().parents[2] / "results"


def last_code_block(text):
    """The last ```...``` block of a solution (without the fences), or the whole text if there is none."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else text


# ---------- 1. Load the data ----------
ds = load_dataset("lmarena-ai/PPE-MBPP-Plus-Best-of-K", split="train")
n_rows = len(ds)
solutions = np.array([ds[f"response_{i}"] for i in range(1, SOLUTIONS_PER_ROW + 1)]).T      # (rows, 32) strings
passes = np.array(ds["scores"], dtype=int)                                                  # (rows, 32) 1 = tests pass
question_id = np.repeat(np.arange(n_rows), SOLUTIONS_PER_ROW)                               # flat, row-major
prompts_flat = np.repeat(np.array(ds["prompt"], dtype=object), SOLUTIONS_PER_ROW)
solutions_flat = solutions.ravel()
print(f"{n_rows} rows x {SOLUTIONS_PER_ROW} solutions; {len(set(ds['task_id']))} distinct tasks; "
      f"fraction passing: {passes.mean():.3f}", flush=True)

RESULTS.mkdir(exist_ok=True)
with open(RESULTS / "mbpp_plus_meta.csv", "w") as f:
    f.write("question_id,task_id,model_name\n")
    f.writelines(f"{i},{t},{m}\n" for i, (t, m) in enumerate(zip(ds["task_id"], ds["model_name"])))

# ---------- 2. Score with each reward model ----------
for name, (model_id, pair_input, extract) in SCORERS.items():
    texts = [last_code_block(s) for s in solutions_flat] if extract else list(solutions_flat)
    pairs = list(zip(prompts_flat, texts))

    tok = AutoTokenizer.from_pretrained(model_id)
    if pair_input:
        tok.truncation_side = "left"                 # keep the end of the text (the code), as for the math tables
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_id, dtype=torch.float32 if pair_input else torch.float16, device_map="cuda", num_labels=1)
    scores = np.array(score_batched(rm, tok, pair_input, pairs))

    out = RESULTS / f"mbpp_plus_{name}.csv"
    with open(out, "w") as f:
        f.write("question_id,score,correct\n")
        f.writelines(f"{q},{s},{c}\n" for q, s, c in zip(question_id, scores, passes.ravel()))

    # quick sanity: do passing solutions score higher?
    print(f"{name}: saved {out.name}; mean score passing {scores[passes.ravel() == 1].mean():.3f} "
          f"vs failing {scores[passes.ravel() == 0].mean():.3f}", flush=True)
    del rm
    torch.cuda.empty_cache()
