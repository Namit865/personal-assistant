import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_FILE, VOCAB_QA_FILE, BPE_MERGES_FILE, MODELS_DIR
from core.tokenizer import decode
from core.qa_model import forward, predict_span

weights = np.load(MODELS_DIR / "qa_weights.npz")
params = {k: weights[k] for k in weights.files}

vocab = json.load(open(VOCAB_QA_FILE))
id_to_tok = {v: k for k, v in vocab.items()}

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]


def ids_to_text(ids):
    toks = [id_to_tok[i] for i in ids]
    return "".join(decode(toks, merges))


data = json.load(open(DATASET_FILE))

for ex in data[:5]:
    ids = ex["input_ids"]
    start_logits, end_logits, _ = forward(ids, params)
    sp, ep = predict_span(start_logits, end_logits)

    predicted = ids_to_text(ids[sp : ep + 1])
    true = ids_to_text(ids[ex["start_label"] : ex["end_label"] + 1])

    print(f"pred=({sp},{ep})  true=({ex['start_label']},{ex['end_label']})")
    print(f"  predicted: {predicted!r}")
    print(f"  true:      {true!r}\n")
