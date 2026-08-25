import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VOCAB_QA_FILE, BPE_MERGES_FILE, MODELS_DIR
from core.tokenizer import encode, decode, tokens_to_id
from core.qa_model import forward, predict_span
from core.retrieval import clean_passage

weights = np.load(MODELS_DIR / "qa_weights.npz")
params = {k: weights[k] for k in weights.files}

max_len = params["positional_emb"].shape[0]

vocab = json.load(open(VOCAB_QA_FILE))
id_to_tok = {v: k for k, v in vocab.items()}

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]


def ids_to_text(ids):
    toks = [id_to_tok[i] for i in ids]
    return "".join(decode(toks, merges))


def answer(question, passage):
    q_tokens = encode(list(question), merges)
    p_tokens = encode(list(passage), merges)
    combined = q_tokens + ["<SEP>"] + p_tokens

    if len(combined) > max_len:
        print(
            f"  skipped — {len(combined)} tokens exceeds checkpoint limit {max_len}"
        )
        return

    input_ids = tokens_to_id(combined, vocab)
    start_logits, end_logits, _ = forward(input_ids, params)
    sp, ep = predict_span(start_logits, end_logits, offset=len(q_tokens) + 1)

    predicted = ids_to_text(input_ids[sp : ep + 1])
    print(f"  span=({sp},{ep})  answer: {predicted!r}")

    if not any(c.isalpha() for c in predicted):
        return None

    return predicted