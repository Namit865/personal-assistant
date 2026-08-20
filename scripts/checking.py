import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_FILE
from core.qa_model import init_params, embed, forward

data = json.load(open(DATASET_FILE))

params = init_params(888)

example = data[0]
start_logits, end_logits = forward(example["input_ids"], params)

print(start_logits.shape, end_logits.shape)
