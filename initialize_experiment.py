"""
1. Take GSM8K math questions (they come with an answer key).
2. The generator answers each question ANSWERS_PER_QUESTION times.
3. The scorer (a reward model) gives every answer a score.
4. Each answer is checked against the answer key.
5. Saves table.csv  (question_id, score, correct)
"""
import csv
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams

N_QUESTIONS = 100
ANSWERS_PER_QUESTION = 256

GENERATOR = "Qwen/Qwen2.5-Math-1.5B-Instruct"
WEAKER_GENERATOR = "Qwen/Qwen2.5-0.5B-Instruct"
SCORER = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
WEAKER_SCORER = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

from google.colab import drive
drive.mount("/content/drive")

# ---------- 1. Questions + answer key ----------
data = load_dataset("openai/gsm8k", "main", split="test").select(range(N_QUESTIONS))
questions = [row["question"] for row in data]
gold = [row["answer"].split("####")[-1].strip().replace(",", "") for row in data]

# ---------- 2. Generate answers ----------
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
    max_tokens=1024)

outputs = llm.generate(prompts, params)

answers = [
    {"question_id": qid, "answer": o.text} 
    for qid, out in enumerate(outputs) 
    for o in out.outputs    
]

print(f"{len(answers)} answers to {N_QUESTIONS} questions")

# ---------- 3. Score every answer with the reward model ----------
rm_tok = AutoTokenizer.from_pretrained(WEAKER_SCORER)
rm = AutoModelForSequenceClassification.from_pretrained(
    WEAKER_SCORER, 
    dtype=torch.float16, 
    device_map="cuda", 
    num_labels=1)

for i, a in enumerate(answers):
    chat = [
        {"role": "user", "content": questions[a["question_id"]]},
        {"role": "assistant", "content": a["answer"]}
    ]
    
    inputs = rm_tok.apply_chat_template(
        chat, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt").to("cuda")

    with torch.no_grad():
        a["score"] = rm(**inputs).logits[0][0].item()
    
    if i % 500 == 0:
        print(f"scored {i}/{len(answers)}")

# ---------- 4. Check each answer against the answer key ----------
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

for a in answers:
    a["correct"] = int(
        same_number(
            final_answer(a["answer"]), 
            gold[a["question_id"]])
        )

with open("/content/drive/MyDrive/table_256.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "score", "correct"])

    for a in answers:
        w.writerow([a["question_id"], a["score"], a["correct"]])

acc = sum(a["correct"] for a in answers) / len(answers)
print(f"saved table.csv. Fraction of answers correct: {acc:.3f}")