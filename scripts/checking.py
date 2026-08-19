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
    build_training_example
)

train_data = json.load(open(SQUAD_TRAIN_FILE))

paragraph = train_data["data"][0]["paragraphs"][0]
context = paragraph["context"]
qa = paragraph["qas"][0]
question = qa["question"]
answer_start = qa["answers"][0]["answer_start"]
answer_text = qa["answers"][0]["text"]

all_text = []

articles = train_data["data"][:50]


for article in articles:
    for paras in article["paragraphs"]:
        all_text.append(paras["context"])

        for qa in paras["qas"]:
            all_text.append(qa["question"])

joined_text = " ".join(all_text)

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]


vocab = build_bpe_vocab(merges, joined_text)
vocab["<SEP>"] = len(vocab)

input_ids, start_label, end_label = build_training_example(
    question, context, answer_start, answer_text, merges, vocab
)
print(start_label, end_label)