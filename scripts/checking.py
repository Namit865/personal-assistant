import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_FILE
from core.qa_model import init_params, forward, cross_entropy_loss, backward

data = json.load(open(DATASET_FILE))
ex = data[0]
ids = ex["input_ids"]
start_label = ex["start_label"]
end_label = ex["end_label"]

print("example length:", len(ids), "| labels:", start_label, end_label)

np.random.seed(0)
params = init_params(888)

s, e, cache = forward(ids, params)
loss, ps, pe = cross_entropy_loss(s, e, start_label, end_label)
grads = backward(cache, params, ps, pe, start_label, end_label)
print("loss:", round(loss, 4))

eps = 1e-6
N = 12
worst = 0.0
t0 = time.time()

for k in params:
    flat = params[k].reshape(-1)
    num = np.zeros(N)
    for j in range(N):
        old = flat[j]
        flat[j] = old + eps
        l1, _, _ = cross_entropy_loss(
            *forward(ids, params)[:2], start_label, end_label
        )
        flat[j] = old - eps
        l2, _, _ = cross_entropy_loss(
            *forward(ids, params)[:2], start_label, end_label
        )
        flat[j] = old
        num[j] = (l1 - l2) / (2 * eps)

    diff = np.abs(num - grads[k].reshape(-1)[:N]).max()
    worst = max(worst, diff)
    print(f"{k:16} {diff:.3e}")

print("WORST:", worst, "| took", round(time.time() - t0, 1), "s")
