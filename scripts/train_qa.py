import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_FILE, MAX_SEQ_LEN
from core.qa_model import (
    init_params,
    forward,
    cross_entropy_loss,
    backward,
    update_params,
)

data = json.load(open(DATASET_FILE))

data = [ex for ex in data if len(ex["input_ids"]) <= MAX_SEQ_LEN]
print("training on", len(data), "examples after length filter")

lr = 0.001
params = init_params(888)

epochs = 3

for epoch in range(epochs):
    total_loss = 0

    for example in data:
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

    print(
        f"Epoch: {epoch} | Loss: {loss} | Avg Loss: {total_loss / len(data)}"
    )
