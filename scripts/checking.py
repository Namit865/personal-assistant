import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_FILE
from core.qa_model import init_params, embed, forward, cross_entropy_loss

data = json.load(open(DATASET_FILE))
ex = data[0]

ids = ex["input_ids"]

params = init_params(888)

start_logits, end_logits = forward(ids, params)

loss, probs_start, probs_end = cross_entropy_loss(
    start_logits, end_logits, 270, 284
)
print(loss)
