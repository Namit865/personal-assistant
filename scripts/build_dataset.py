import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SQUAD_TRAIN_FILE,
    BPE_MERGES_FILE,
    DATASET_FILE,
    VOCAB_QA_FILE,
)

from core.tokenizer import (
    encode,
    build_token_ranges,
    build_bpe_vocab,
    tokens_to_id,
    find_token_position,
)

train_data = json.load(open(SQUAD_TRAIN_FILE))
merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]

articles = train_data["data"][:50]

all_text = []

for article in articles:
    for paras in article["paragraphs"]:
        all_text.append(paras["context"])

        for qa in paras["qas"]:
            all_text.append(qa["question"])

joined_text = " ".join(all_text)

vocab = build_bpe_vocab(merges, joined_text)
vocab["<SEP>"] = len(vocab)

with open(VOCAB_QA_FILE, "w") as f:
    json.dump(vocab, f)


dataset = []
skipped = 0

for article in articles:
    for paras in article["paragraphs"]:
        context = paras["context"]
        context_ranges = build_token_ranges(context, merges)
        context_tokens = encode(list(context), merges)

        for qa in paras["qas"]:
            question = qa["question"]
            ans = qa["answers"][0]
            answer_start = ans["answer_start"]
            answer_text = ans["text"]
            answer_end = answer_start + len(answer_text) - 1

            local_start = find_token_position(context_ranges, answer_start)
            local_end = find_token_position(context_ranges, answer_end)

            if local_start is None or local_end is None:
                skipped += 1
                continue

            question_tokens = encode(list(question), merges)
            combined_tokens = question_tokens + ["<SEP>"] + context_tokens
            offset = len(question_tokens) + 1

            input_ids = tokens_to_id(combined_tokens, vocab)
            dataset.append(
                {
                    "input_ids": input_ids,
                    "start_label": offset + local_start,
                    "end_label": offset + local_end,
                }
            )

with open(DATASET_FILE, "w") as f:
    json.dump(dataset, f)


print(len(dataset), "examples,", skipped, "skipped")
