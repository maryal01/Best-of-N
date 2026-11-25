"""
Generate many code solutions per problem for MBPP+ or HumanEval+ with a code LLM, to study Best-of-N beyond N = 32.

Writes one JSON line per solution: {"task_id", "sample_id", "solution"} where "solution" is the raw model reply
(the EvalPlus sanitizer extracts the code later). Only reads the problem file, so it runs in the vLLM environment.

Run from the repo root on a GPU (shards = slices of the task list, so several GPUs can work in parallel):
  python experiments/coding/generate_code.py --problems <MbppPlus-v0.2.0.jsonl> --out gen/mbpp_shard0.jsonl --start 0 --end 95
Problem files are downloaded by EvalPlus: <cache>/evalplus/MbppPlus-v0.2.0.jsonl and HumanEvalPlus-v0.1.10.jsonl
"""
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

parser = argparse.ArgumentParser()
parser.add_argument("--problems", required=True)        # EvalPlus problem file (jsonl, one problem per line)
parser.add_argument("--out", required=True)
parser.add_argument("--start", type=int, default=0)     # slice of the problem list
parser.add_argument("--end", type=int, default=10**9)
parser.add_argument("--n", type=int, default=256)       # solutions per problem
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--max_tokens", type=int, default=768)
args = parser.parse_args()

problems = [json.loads(line) for line in open(args.problems)][args.start:args.end]
print(f"{len(problems)} problems x {args.n} solutions with {MODEL}", flush=True)

llm = LLM(MODEL, dtype="half", gpu_memory_utilization=0.5)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template(
    [{"role": "user", "content": f"{INSTRUCTION}\n```\n{p['prompt'].strip()}\n```\n"}],
    tokenize=False, add_generation_prompt=True) for p in problems]

outputs = llm.generate(prompts, SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    f.writelines(
        json.dumps({"task_id": p["task_id"], "sample_id": i, "solution": o.text}) + "\n"
        for p, out in zip(problems, outputs) for i, o in enumerate(out.outputs))
print(f"saved {args.out}", flush=True)
