"""
1. Take math questions with an answer key: GSM8K (default), MATH-500 (DATASET=math500), or
   the 4,500 other MATH test questions (DATASET=math), new data for re-testing MATH-500 results.
2. The generator answers each question ANSWERS_PER_QUESTION times.
3. Each answer is checked against the answer key.
4. Each scorer (a reward model) gives every answer a score.
5. Saves one table per scorer (question_id, score, correct).

Settings come from environment variables (jobs/scholar.sub sets them); the defaults
reproduce the original GSM8K run with the weaker scorer.
"""
import csv
import hashlib
import json
import os
import re

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams

ANSWERS_PER_QUESTION = 256

GENERATOR = "Qwen/Qwen2.5-Math-1.5B-Instruct"
WEAKER_GENERATOR = "Qwen/Qwen2.5-0.5B-Instruct"
SCORER = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
WEAKER_SCORER = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
# 2023 general-chat reward model, not trained on math, reads at most 512 tokens
WEAKEST_SCORER = "OpenAssistant/reward-model-deberta-v3-large-v2"

DATASET = os.environ.get("DATASET", "gsm8k")                 # "gsm8k", "math500" or "math"
SCORERS = os.environ.get("SCORERS", WEAKER_SCORER).split(",")  # comma-separated
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
# Run only questions START..END-1 (set by jobs/scholar.sub so each job fits in 4h); default: all.
START = int(os.environ.get("START", 0))
END = int(os.environ.get("END", 10**9))
OUT = os.environ.get("OUT", "results/table_256.csv")
# Optional .jsonl path: generated answers are saved there, and reused if the file already
# exists, so a rerun (or a new scorer) does not have to generate them again.
ANSWERS = os.environ.get("ANSWERS")

# ---------- 1. Questions + answer key ----------
if DATASET == "math500":
    data = load_dataset("HuggingFaceH4/MATH-500", split="test")
    data = data.select(range(START, min(END, len(data))))
    questions = [row["problem"] for row in data]
    gold = [row["answer"] for row in data]
elif DATASET == "math":
    def last_boxed(text):
        """Contents of the last \\boxed{...} (nested braces allowed): MATH's answer key."""
        i = text.rfind("\\boxed{") + len("\\boxed{")
        j, depth = i, 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        return text[i:j - 1]

    norm = lambda s: " ".join(s.split())
    in_math500 = {norm(p) for p in load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"]}
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    data = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects])
    data = data.filter(lambda row: norm(row["problem"]) not in in_math500)
    # Fixed pseudo-random order (by hash of the problem text, so it never depends on library
    # versions): the first k chunks are then a random sample of the 4,500 questions.
    order = sorted(range(len(data)), key=lambda i: hashlib.sha256(data[i]["problem"].encode()).hexdigest())
    data = data.select(order).select(range(START, min(END, len(data))))
    questions = [row["problem"] for row in data]
    gold = [last_boxed(row["solution"]) for row in data]
else:
    data = load_dataset("openai/gsm8k", "main", split="test")
    data = data.select(range(START, min(END, len(data))))
    questions = [row["question"] for row in data]
    gold = [row["answer"].split("####")[-1].strip().replace(",", "") for row in data]

# ---------- 2. Generate answers (or reuse saved ones) ----------
if ANSWERS and os.path.exists(ANSWERS):
    answers = [json.loads(line) for line in open(ANSWERS)]
    print(f"loaded {len(answers)} saved answers from {ANSWERS}")
else:
    llm = LLM(GENERATOR, dtype=  "half", gpu_memory_utilization=0.5)
    tok = llm.get_tokenizer()
    prompts = [ tok.apply_chat_template(
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q}
        ],
        tokenize=False,
        add_generation_prompt=True)
        for q in questions
    ]

    params = SamplingParams(
        n=ANSWERS_PER_QUESTION,
        temperature=0.8,
        top_p=0.95,
        max_tokens=MAX_TOKENS)

    outputs = llm.generate(prompts, params)

    answers = [
        {"question_id": qid, "answer": o.text}
        for qid, out in enumerate(outputs)
        for o in out.outputs
    ]

    if ANSWERS:  # write to a temporary file first so a crash never leaves a half-written one
        with open(ANSWERS + ".tmp", "w") as f:
            for a in answers:
                f.write(json.dumps(a) + "\n")
        os.replace(ANSWERS + ".tmp", ANSWERS)

print(f"{len(answers)} answers to {len(questions)} questions")

# ---------- 3. Check each answer against the answer key ----------
def final_answer(text):
    """The last \\boxed{...} in the answer, or else the last number."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1]
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1] if numbers else ""

def same_number(a, b):
    clean = lambda s: s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return abs(float(clean(a)) - float(clean(b))) < 1e-6
    except ValueError:
        return False

if DATASET in ("math500", "math"):
    # MATH answers are LaTeX (fractions, roots, intervals...), so compare them symbolically.
    from math_verify import parse, verify
    gold_parsed = [parse("\\boxed{" + g + "}") for g in gold]

    def is_correct(text, qid):
        try:
            return verify(gold_parsed[qid], parse(text))
        except Exception:  # an unparsable answer counts as wrong rather than crashing the job
            return False
else:
    def is_correct(text, qid):
        return same_number(final_answer(text), gold[qid])

for a in answers:
    a["correct"] = int(is_correct(a["answer"], a["question_id"]))

acc = sum(a["correct"] for a in answers) / len(answers)
print(f"Fraction of answers correct: {acc:.3f}")

# ---------- 4. Score every answer with each reward model ----------
for scorer in SCORERS:
    # The OpenAssistant (DeBERTa) reward model takes a (question, answer) pair and reads at
    # most 512 tokens. Cut from the front, so the end of the question and the final
    # \boxed{} answer are always kept ("longest_first" also handles questions over 512).
    pair_input = "deberta" in scorer.lower()
    rm_tok = AutoTokenizer.from_pretrained(scorer)
    if pair_input:
        rm_tok.truncation_side = "left"
    rm = AutoModelForSequenceClassification.from_pretrained(
        scorer,
        dtype=torch.float32 if pair_input else torch.float16,
        device_map="cuda",
        num_labels=1)

    out = OUT if len(SCORERS) == 1 else OUT.replace(".csv", f"_{scorer.split('/')[-1]}.csv")
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "score", "correct"])

        for i, a in enumerate(answers):
            q = questions[a["question_id"]]
            if pair_input:
                inputs = rm_tok(
                    q, a["answer"],
                    truncation="longest_first",
                    max_length=512,
                    return_tensors="pt").to("cuda")
            else:
                chat = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a["answer"]}
                ]
                inputs = rm_tok.apply_chat_template(
                    chat,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt").to("cuda")

            with torch.no_grad():
                score = rm(**inputs).logits[0][0].item()
            w.writerow([START + a["question_id"], score, a["correct"]])

            if i % 500 == 0:
                print(f"{scorer}: scored {i}/{len(answers)}")

    print(f"saved {out}")
    del rm
    torch.cuda.empty_cache()
