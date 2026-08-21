import sys
import json
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_FILE, MAX_SEQ_LEN, MODELS_DIR, VOCAB_QA_FILE
from core.qa_model import (
    init_params,
    forward,
    cross_entropy_loss,
    backward,
    update_params,
)

data = json.load(open(DATASET_FILE))
data = [ex for ex in data if len(ex["input_ids"]) <= MAX_SEQ_LEN]
print("after length filter:", len(data), "examples")

random.seed(42)
random.shuffle(data)
val_data = data[:500]
data = data[500:]
print("train:", len(data), "| val:", len(val_data))

lr = 0.03
epochs = 5
vocab = json.load(open(VOCAB_QA_FILE))
params = init_params(len(vocab))

for epoch in range(epochs):
    random.shuffle(data)
    total_loss = 0

    for idx, example in enumerate(data):
        ids = example["input_ids"]
        start_label = example["start_label"]
        end_label = example["end_label"]

        start_logits, end_logits, cache = forward(ids, params)

        loss, probs_start, probs_end = cross_entropy_loss(
            start_logits, end_logits, start_label, end_label
        )

        grads = backward(
            cache, params, probs_start, probs_end, start_label, end_label
        )

        params = update_params(params, grads, lr)
        total_loss += loss

        if (idx + 1) % 1000 == 0:
            print(
                f"  {idx+1}/{len(data)} | running loss {total_loss/(idx+1):.4f}"
            )

    val_loss = 0
    for example in val_data:
        s, e, _ = forward(example["input_ids"], params)
        l, _, _ = cross_entropy_loss(
            s, e, example["start_label"], example["end_label"]
        )
        val_loss += l

    print(
        f"Epoch {epoch} | Train: {total_loss/len(data):.4f} "
        f"| Val: {val_loss/len(val_data):.4f}"
    )

    np.savez(MODELS_DIR / "qa_weights.npz", **params)

print("done, weights at", MODELS_DIR / "qa_weights.npz")
