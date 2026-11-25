"""Batched reward-model scoring, shared by initialize_experiment.py and jobs/check_batched_scoring.py."""
import torch


def encode(rm_tok, pair_input, question, answer, via_text=False):
    """Token ids (and masks) of one (question, answer) pair, as a plain dict."""
    if pair_input:
        # The OpenAssistant (DeBERTa) reward model takes a (question, answer) pair and reads at
        # most 512 tokens. Truncation cuts from the front (the caller sets truncation_side="left"),
        # so the end of the question and the final \boxed{} answer are always kept
        # ("longest_first" also handles questions over 512 tokens).
        enc = rm_tok(question, answer, truncation="longest_first", max_length=512)
    else:
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        if via_text:
            # Render the chat to text, then tokenize the text: the tokenizer adds special tokens such as BOS, which
            # some chat templates leave out (e.g. Skywork-Reward-Llama-3.1-8B).
            enc = rm_tok(rm_tok.apply_chat_template(chat, tokenize=False))
        else:
            enc = rm_tok.apply_chat_template(chat, tokenize=True, return_dict=True)
    return dict(enc)


def score_batched(rm, rm_tok, pair_input, pairs, token_budget=32768, via_text=False):
    """
    One reward score for each (question, answer) in `pairs`, in the same order.

    Answers are sorted by length and grouped so that a batch holds at most `token_budget` tokens.
    Padding is on the right, so a (causal) reward model sees exactly the same tokens as when
    scoring one answer at a time.
    """
    rm_tok.padding_side = "right"
    if rm.config.pad_token_id is None:       # needed to score a padded batch
        rm.config.pad_token_id = rm_tok.pad_token_id

    encodings = [encode(rm_tok, pair_input, q, a, via_text) for q, a in pairs]
    lengths = [len(e["input_ids"]) for e in encodings]
    order = sorted(range(len(encodings)), key=lambda i: -lengths[i])     # longest first
    scores = [0.0] * len(encodings)

    pos = 0
    while pos < len(order):
        # the first answer of a batch is its longest, so it sets the padded length
        size = max(1, token_budget // lengths[order[pos]])
        batch = order[pos:pos + size]
        padded = rm_tok.pad([encodings[i] for i in batch], return_tensors="pt").to("cuda")
        with torch.no_grad():
            batch_scores = rm(**padded).logits[:, 0].float().cpu().tolist()
        for i, sc in zip(batch, batch_scores):
            scores[i] = sc
        pos += size
        if pos % 5000 < size:                 # about every 5000 answers
            print(f"  scored {pos}/{len(order)}", flush=True)
    return scores
