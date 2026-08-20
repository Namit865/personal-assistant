import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BPE_MERGES_FILE, SQUAD_TRAIN_FILE
from core.tokenizer import (
    encode,
    build_token_ranges,
    find_token_for_char,
    find_answer_span,
    build_bpe_vocab,
    tokens_to_id,
    find_token_position,
    build_training_example,
)
from core.qa_model import init_params

params = init_params(888)

for k, v in params.items():
    print(k, v.shape)

print("Total Entries:", len(params))
