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
    init_adam_state,
    adam_update,
)

GRAD_ACCUM_STEPS = (
    16  # effective batch size, via averaging — not padded tensors
)

data = json.load(open(DATASET_FILE))
data = [ex for ex in data if len(ex["input_ids"]) <= MAX_SEQ_LEN]
print("after length filter:", len(data), "examples")

random.seed(42)
random.shuffle(data)
val_data = data[:500]
data = data[500:]
print(
    "train:",
    len(data),
    "| val:",
    len(val_data),
    "| grad accum:",
    GRAD_ACCUM_STEPS,
)

lr = 0.001
epochs = 5
vocab = json.load(open(VOCAB_QA_FILE))
params = init_params(len(vocab))
opt_state = init_adam_state(params)

best_val = float("inf")

for epoch in range(epochs):
    random.shuffle(data)
    total_loss = 0
    n_updates = 0

    grad_accum = {k: np.zeros_like(v) for k, v in params.items()}
    accum_count = 0

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

        for k in grads:
            grad_accum[k] += grads[k]
        accum_count += 1
        total_loss += loss

        if accum_count == GRAD_ACCUM_STEPS:
            avg_grads = {k: v / accum_count for k, v in grad_accum.items()}
            params, opt_state = adam_update(
                params, avg_grads, opt_state, lr=lr
            )
            grad_accum = {k: np.zeros_like(v) for k, v in params.items()}
            accum_count = 0
            n_updates += 1

        if (idx + 1) % 1000 == 0:
            print(
                f"  {idx+1}/{len(data)} | running loss {total_loss/(idx+1):.4f} | updates {n_updates}"
            )

    if accum_count > 0:
        avg_grads = {k: v / accum_count for k, v in grad_accum.items()}
        params, opt_state = adam_update(params, avg_grads, opt_state, lr=lr)

    val_loss = 0
    for example in val_data:
        s, e, _ = forward(example["input_ids"], params)
        l, _, _ = cross_entropy_loss(
            s, e, example["start_label"], example["end_label"]
        )
        val_loss += l
    val_loss /= len(val_data)

    print(
        f"Epoch {epoch} | Train: {total_loss/len(data):.4f} | Val: {val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        np.savez(MODELS_DIR / "qa_weights.npz", **params)
        print(f"  -> new best (val={val_loss:.4f}), saved")

print("done. best val loss:", best_val)
