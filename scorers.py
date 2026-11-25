"""
The reward models used in the experiments, behind one function: load_scorer(name) -> score(pairs).

`pairs` is a list of (question, answer) texts; `score` returns one reward per pair (numpy array, same order).
Needs a GPU. Shared by experiments/ppe/score_ppe.py and experiments/coding/score_generated.py.

  name              model                                         notes
  deberta           OpenAssistant/reward-model-deberta-v3-large-v2 (question, answer) pair model, reads 512 tokens
  skywork_v2_0.6b   Skywork/Skywork-Reward-V2-Qwen3-0.6B          chat-template classifier
  skywork_v2_1.7b   Skywork/Skywork-Reward-V2-Qwen3-1.7B          same family, larger
  codescaler        LARK-Lab/CodeScaler-1.7B                      code reward model (fine-tuned from skywork_v2_1.7b)
  internlm          internlm/internlm2-1_8b-reward                custom model (trust_remote_code), own get_scores API
  skywork_llama     Skywork/Skywork-Reward-Llama-3.1-8B           the tokenizer must add BOS
"""
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from scoring import score_batched

MODELS = {
    "deberta": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "skywork_v2_0.6b": "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "skywork_v2_1.7b": "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    "codescaler": "LARK-Lab/CodeScaler-1.7B",
    "internlm": "internlm/internlm2-1_8b-reward",
    "skywork_llama": "Skywork/Skywork-Reward-Llama-3.1-8B",
}


def load_scorer(name):
    model_id = MODELS[name]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if name == "internlm":
        return _internlm_scorer(model_id, tok)

    if name == "deberta":
        tok.truncation_side = "left"                 # keep the end of the text (the final answer)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32, device_map="cuda", num_labels=1)
        return lambda pairs: np.array(score_batched(model, tok, True, pairs))

    if name == "skywork_llama":
        # Llama tokenizers have no pad token. Use one that never occurs in the text, so the model's "last non-pad token"
        # lookup cannot mistake a real token for padding.
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else "<|reserved_special_token_0|>"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float16, device_map="cuda", num_labels=1)
        model.config.pad_token_id = tok.pad_token_id
        return lambda pairs: np.array(score_batched(model, tok, False, pairs, token_budget=16384, via_text=True))

    # Qwen3-based chat-template classifiers
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float16, device_map="cuda", num_labels=1)
    return lambda pairs: np.array(score_batched(model, tok, False, pairs))


def _internlm_scorer(model_id, tok):
    model = AutoModel.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
    chats = lambda pairs: [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in pairs]
    batch = 4                    # chats per get_scores call
    MAX_TOKENS = 8192            # eager attention needs memory quadratic in the length. About 0.1% of the PPE answers are degenerate
                                 # repetitions of 10k-100k characters (up to 1 token per character): keep their last 8192 tokens.
    SHORT = 2048                 # up to this length (tokens) 4 answers share a batch, longer ones are scored one at a time

    def score(pairs):
        ids = [tok(a, add_special_tokens=False).input_ids for _, a in pairs]
        pairs = [(q, tok.decode(i[-MAX_TOKENS:]) if len(i) > MAX_TOKENS else a) for (q, a), i in zip(pairs, ids)]
        lengths = np.minimum([len(i) for i in ids], MAX_TOKENS)
        order = np.argsort(lengths)                      # shortest first: little padding, the longest come last
        scores = np.empty(len(pairs))
        pos = 0
        while pos < len(order):
            size = batch if lengths[order[min(pos + batch - 1, len(order) - 1)]] <= SHORT else 1
            chunk = order[pos:pos + size]
            scores[chunk] = model.get_scores(tok, chats([pairs[j] for j in chunk]))
            pos += size
        return scores
    return score
