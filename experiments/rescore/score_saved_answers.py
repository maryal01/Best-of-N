"""
Score the SAVED answers of the held-out MATH run (3,000 questions x 256 answers) with another reward model.

The generated answers are kept as one file per chunk of 100 questions (<answers-dir>/math_part<k>.jsonl, written by
initialize_experiment.py). The correctness flags are taken from the table of the same run
(results/math_skywork_gilbreth.csv: rows are the chunks 0..29 one after the other, answers in the file order), the
question texts are rebuilt with the same deterministic ordering as in initialize_experiment.py (the 4,500 MATH test
questions that are not in MATH-500, sorted by the hash of the problem text).

Output per chunk: <out-dir>/math_gilbreth_<scorer>_part<k>.csv with question_id (global), score, correct. Merge the
parts into one table (head -n 1 of one part, tail -n +2 of all parts, in chunk order).

Run from the repo root on a GPU, e.g.:
  python experiments/rescore/score_saved_answers.py --scorer internlm --answers-dir <dir> --chunks 14 15 26
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scorers import MODELS, load_scorer

CHUNK = 100                                   # questions per chunk
ANSWERS = 256                                 # answers per question
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

parser = argparse.ArgumentParser()
parser.add_argument("--scorer", required=True, choices=list(MODELS))
parser.add_argument("--answers-dir", required=True)
parser.add_argument("--table", default="results/math_skywork_gilbreth.csv")   # source of the correctness flags
parser.add_argument("--chunks", type=int, nargs="+", required=True)
parser.add_argument("--n-questions", type=int, default=3000)
parser.add_argument("--limit", type=int, default=None)                         # score only the first LIMIT answers of a chunk (test)
parser.add_argument("--out-dir", default="results")
args = parser.parse_args()

# ---------- question texts, in the order used when the answers were generated ----------
norm = lambda s: " ".join(s.split())
in_math500 = {norm(p) for p in load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"]}
data = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in SUBJECTS])
data = data.filter(lambda row: norm(row["problem"]) not in in_math500)
order = sorted(range(len(data)), key=lambda i: hashlib.sha256(data[i]["problem"].encode()).hexdigest())
problems = [data[i]["problem"] for i in order[:args.n_questions]]
print(f"{len(problems)} questions rebuilt", flush=True)

# ---------- correctness flags and question ids of the run ----------
rows = list(csv.DictReader(open(args.table)))
table_question = np.array([int(r["question_id"]) for r in rows])
table_correct = np.array([int(r["correct"]) for r in rows])
per_chunk = CHUNK * ANSWERS

score = load_scorer(args.scorer)
Path(args.out_dir).mkdir(exist_ok=True)
for k in args.chunks:
    answers = [json.loads(line) for line in open(f"{args.answers_dir}/math_part{k}.jsonl")]
    assert len(answers) == per_chunk, f"chunk {k}: {len(answers)} answers"
    rows_k = slice(k * per_chunk, (k + 1) * per_chunk)
    question_id = k * CHUNK + np.array([a["question_id"] for a in answers])
    assert (question_id == table_question[rows_k]).all(), f"chunk {k}: the answer order does not match the table"

    n = args.limit or per_chunk
    pairs = [(problems[q], a["answer"]) for q, a in zip(question_id[:n], answers[:n])]
    scores = score(pairs)

    out = Path(args.out_dir) / f"math_gilbreth_{args.scorer}_part{k}.csv"
    with open(out, "w") as f:
        f.write("question_id,score,correct\n")
        f.writelines(f"{q},{s},{c}\n" for q, s, c in zip(question_id[:n], scores, table_correct[rows_k][:n]))
    print(f"chunk {k}: saved {out.name}; mean score correct {scores[table_correct[rows_k][:n] == 1].mean():.3f} "
          f"vs wrong {scores[table_correct[rows_k][:n] == 0].mean():.3f}", flush=True)
