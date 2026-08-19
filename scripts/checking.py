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
)

train_data = json.load(open(SQUAD_TRAIN_FILE))

articles = train_data["data"][:50]

all_text = []

for article in articles:
    for paras in article["paragraphs"]:
        all_text.append(paras["context"])

        for qa in paras["qas"]:
            all_text.append(qa["question"])

joined_text = " ".join(all_text)

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]

context = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary."
tokens = encode(list(context), merges)
ranges = build_token_ranges(context, merges)
result = find_token_for_char(ranges, 3)
span = find_answer_span(ranges, 34, "catholic")
vocab = build_bpe_vocab(merges, joined_text)
tok = tokens_to_id(tokens, vocab)
print(len(tokens) == len(tok))
print(tok[:10])
