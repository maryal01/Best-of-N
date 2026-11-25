"""
Score the pre-generated PPE Best-of-K datasets with the reward models used in "Inference-Time Reward Hacking in
Large Language Models" (Khalaf et al.), to see whether Best-of-N over-optimizes in those settings.

Datasets (lmarena-ai/PPE-<name>-Best-of-K): MATH, MMLU-Pro, GPQA. Each has 512 rows = 512 different questions;
each row has 32 answers from ONE generator (gemma-2-9b-it, gpt-4o-mini, Llama-3-8B-Instruct or claude-3-haiku,
128 rows each) and a correctness flag per answer, so N can go up to 32.

Scorers:
  internlm      internlm/internlm2-1_8b-reward      custom model (trust_remote_code), scored with its get_scores API
  skywork_llama Skywork/Skywork-Reward-Llama-3.1-8B standard sequence classifier; the tokenizer must add BOS

Output, in the same format as the other tables (run_experiment.py and the sweep scripts read them unchanged):
  results/ppe_<dataset>_<scorer>.csv    question_id (= row index), score, correct
  results/ppe_<dataset>_meta.csv        question_id, model_name (the generator of that row)

Run from the repo root on a GPU:
  python experiments/ppe/score_ppe.py --datasets MATH MMLU-Pro GPQA --scorers internlm skywork_llama
For skywork_llama, a check against one-answer-at-a-time scoring is printed first (padding must not change scores).
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scoring import score_batched

ANSWERS_PER_ROW = 32
RESULTS = Path(__file__).resolve().parents[2] / "results"
MODELS = {"internlm": "internlm/internlm2-1_8b-reward", "skywork_llama": "Skywork/Skywork-Reward-Llama-3.1-8B"}

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["MATH", "MMLU-Pro", "GPQA"])
parser.add_argument("--scorers", nargs="+", default=["internlm", "skywork_llama"], choices=list(MODELS))
args = parser.parse_args()


def load_scorer(name):
    """Returns a function pairs -> scores (numpy array) for the chosen reward model."""
    model_id = MODELS[name]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if name == "internlm":
        model = AutoModel.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
        chats = lambda pairs: [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in pairs]
        batch = 4                                                          # chats per get_scores call

        MAX_CHARS = 20_000       # about 0.1% of the answers are longer (degenerate, repetitive outputs of up to 100k characters);
                                 # InternLM's eager attention needs memory quadratic in the length. Keep the END of such an answer.

        def score(pairs):
            pairs = [(q, a[-MAX_CHARS:]) for q, a in pairs]
            # shortest first, so a batch has little padding and the longest ones (which need the most memory) come last
            order = np.argsort([len(a) for _, a in pairs])
            sorted_scores = np.concatenate(
                [np.array(model.get_scores(tok, chats([pairs[j] for j in order[i:i + batch]]))) for i in range(0, len(pairs), batch)])
            scores = np.empty(len(pairs))
            scores[order] = sorted_scores                                  # back to the original order
            return scores
        return score

    # Skywork-Llama: Llama tokenizers have no pad token. Use one that never occurs in the text, so the model's
    # "last non-pad token" lookup cannot mistake a real token for padding.
    tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else "<|reserved_special_token_0|>"
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float16, device_map="cuda", num_labels=1)   # fp16: batched and single scores agree better than in bf16
    model.config.pad_token_id = tok.pad_token_id
    return lambda pairs: np.array(score_batched(model, tok, False, pairs, token_budget=16384, via_text=True))


def check_padding(score_fn, pairs):
    """Batched scores must equal one-at-a-time scores (up to low-precision noise)."""
    single = np.concatenate([score_fn([pair]) for pair in pairs])
    batched = score_fn(pairs)
    print(f"  padding check on {len(pairs)} answers: max |single - batched| = {np.abs(single - batched).max():.4f}, "
          f"correlation {np.corrcoef(single, batched)[0, 1]:.6f} (score std {single.std():.3f})", flush=True)


RESULTS.mkdir(exist_ok=True)
loaded = {}                                                                  # scorer name -> score function
for dataset_name, scorer_name in itertools.product(args.datasets, args.scorers):
    ds = load_dataset(f"lmarena-ai/PPE-{dataset_name}-Best-of-K", split="train")
    n_rows = len(ds)
    answers = np.array([ds[f"response_{i}"] for i in range(1, ANSWERS_PER_ROW + 1)]).T.ravel()   # row-major
    prompts = np.repeat(np.array(ds["prompt"], dtype=object), ANSWERS_PER_ROW)
    passes = np.array(ds["scores"], dtype=int).ravel()
    question_id = np.repeat(np.arange(n_rows), ANSWERS_PER_ROW)
    pairs = list(zip(prompts, answers))

    with open(RESULTS / f"ppe_{dataset_name}_meta.csv", "w") as f:
        f.write("question_id,model_name\n")
        f.writelines(f"{i},{m}\n" for i, m in enumerate(ds["model_name"]))

    if scorer_name not in loaded:
        loaded.clear()                                                       # keep one model in memory at a time
        torch.cuda.empty_cache()
        loaded[scorer_name] = load_scorer(scorer_name)
        if scorer_name == "skywork_llama":
            check_padding(loaded[scorer_name], pairs[:48])
    scores = loaded[scorer_name](pairs)

    out = RESULTS / f"ppe_{dataset_name}_{scorer_name}.csv"
    with open(out, "w") as f:
        f.write("question_id,score,correct\n")
        f.writelines(f"{q},{s},{c}\n" for q, s, c in zip(question_id, scores, passes))
    print(f"{dataset_name} + {scorer_name}: saved {out.name}; mean score correct {scores[passes == 1].mean():.3f} "
          f"vs wrong {scores[passes == 0].mean():.3f}", flush=True)
