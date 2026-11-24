"""
Checks that batched scoring (scoring.py) gives the same scores as scoring one answer at a time
(the original code), for both reward models. Needs a GPU.  Run from the repo root:
    python jobs/check_batched_scoring.py
Texts: MATH-500 reference solutions, some concatenated to be very long (DeBERTa truncates at 512).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import encode, score_batched

SCORERS = ["Skywork/Skywork-Reward-V2-Qwen3-0.6B", "OpenAssistant/reward-model-deberta-v3-large-v2"]
data = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(300))
repeat = np.arange(len(data)) % 5 + 1                       # 1x ... 5x concatenated -> varied lengths
pairs = [(row["problem"], (" " + row["solution"]) * r) for row, r in zip(data, repeat)]

for scorer in SCORERS:
    pair_input = "deberta" in scorer.lower()
    rm_tok = AutoTokenizer.from_pretrained(scorer)
    if pair_input:
        rm_tok.truncation_side = "left"
    rm = AutoModelForSequenceClassification.from_pretrained(
        scorer, dtype=torch.float32 if pair_input else torch.float16, device_map="cuda", num_labels=1)

    # reference: one answer at a time, no padding (the original implementation)
    t = time.time()
    reference = []
    for q, a in pairs:
        inputs = {k: torch.tensor([v]).cuda() for k, v in encode(rm_tok, pair_input, q, a).items()}
        with torch.no_grad():
            reference.append(rm(**inputs).logits[0][0].item())
    t_single = time.time() - t

    t = time.time()
    batched = score_batched(rm, rm_tok, pair_input, pairs)
    t_batched = time.time() - t

    diff = np.abs(np.array(reference) - np.array(batched))
    ranks_same = (np.argsort(reference) == np.argsort(batched)).mean()
    print(f"\n{scorer}\n  max |single - batched| = {diff.max():.5f}  (score std = {np.std(reference):.3f})"
          f"\n  corr = {np.corrcoef(reference, batched)[0, 1]:.6f}   same rank position: {ranks_same:.1%}"
          f"\n  time: single {t_single:.1f}s, batched {t_batched:.1f}s  ({t_single / t_batched:.1f}x faster)")
    del rm
    torch.cuda.empty_cache()
