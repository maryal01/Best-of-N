"""
Turn generated code solutions + EvalPlus test results into Best-of-N tables (question_id, score, correct).

Inputs (all produced by generate_code.py and evaluate_code.sh):
  --problems     EvalPlus problem file (prompt text per task; the task order defines question_id)
  --raw          raw model replies, one JSON line per solution          (gen/<name>.jsonl)
  --sanitized    the same solutions with the code extracted by EvalPlus (gen/<name>-sanitized.jsonl)
  --eval         EvalPlus results                                       (gen/<name>-sanitized_eval_results.json)
A solution is correct if it passes the base AND the plus tests. Scorers: CodeScaler-1.7B gets the extracted code,
the DeBERTa reward model gets the full reply (like on the PPE MBPP+ data).

Output: <out-dir>/<tag>_<scorer>.csv with question_id (task index), score, correct; 256 solutions per task.
Run from the repo root on a GPU:
  python experiments/coding/score_generated.py --tag gen_mbpp --problems <MbppPlus jsonl> --raw gen/mbpp_all.jsonl \
      --sanitized gen/mbpp_all-sanitized.jsonl --eval gen/mbpp_all-sanitized_eval_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scoring import score_batched

SCORERS = {"codescaler": ("LARK-Lab/CodeScaler-1.7B", False), "deberta": ("OpenAssistant/reward-model-deberta-v3-large-v2", True)}

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)                       # output name prefix, e.g. gen_mbpp
parser.add_argument("--problems", required=True)
parser.add_argument("--raw", required=True)
parser.add_argument("--sanitized", required=True)
parser.add_argument("--eval", required=True)
parser.add_argument("--scorers", nargs="+", default=["codescaler", "deberta"], choices=list(SCORERS))
parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "results"))
args = parser.parse_args()

# ---------- data: one row per solution, grouped by task ----------
problems = [json.loads(line) for line in open(args.problems)]
task_index = {p["task_id"]: i for i, p in enumerate(problems)}
prompt_of = {p["task_id"]: p["prompt"] for p in problems}

raw = [json.loads(line) for line in open(args.raw)]
sanitized = [json.loads(line) for line in open(args.sanitized)]
assert len(raw) == len(sanitized), "raw and sanitized files must have the same solutions in the same order"
assert all(r["task_id"] == s["task_id"] for r, s in zip(raw, sanitized))

results = json.load(open(args.eval))["eval"]                      # task_id -> list of per-solution results, in file order
seen = {}                                                         # per task: how many of its solutions we have passed so far
correct = []
for r in raw:
    k = seen.get(r["task_id"], 0)
    seen[r["task_id"]] = k + 1
    res = results[r["task_id"]][k]
    correct.append(int(res["base_status"] == "pass" and res["plus_status"] == "pass"))
correct = np.array(correct)
question_id = np.array([task_index[r["task_id"]] for r in raw])
per_task = np.bincount(question_id)
assert per_task.min() == per_task.max(), f"tasks have different numbers of solutions: {per_task.min()} to {per_task.max()}"
print(f"{len(raw)} solutions, {len(per_task)} tasks x {per_task[0]}; fraction correct: {correct.mean():.3f}", flush=True)

prompts = [prompt_of[r["task_id"]] for r in raw]
inputs = {"codescaler": [s["solution"] for s in sanitized], "deberta": [r["solution"] for r in raw]}

# ---------- score and save ----------
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
for name in args.scorers:
    model_id, pair_input = SCORERS[name]
    tok = AutoTokenizer.from_pretrained(model_id)
    if pair_input:
        tok.truncation_side = "left"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_id, dtype=torch.float32 if pair_input else torch.float16, device_map="cuda", num_labels=1)
    scores = np.array(score_batched(rm, tok, pair_input, list(zip(prompts, inputs[name]))))

    out = out_dir / f"{args.tag}_{name}.csv"
    with open(out, "w") as f:
        f.write("question_id,score,correct\n")
        f.writelines(f"{q},{s},{c}\n" for q, s, c in zip(question_id, scores, correct))
    print(f"{name}: saved {out.name}; mean score correct {scores[correct == 1].mean():.3f} vs wrong {scores[correct == 0].mean():.3f}", flush=True)
    del rm
    torch.cuda.empty_cache()
