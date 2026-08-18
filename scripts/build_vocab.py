import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SQUAD_TRAIN_FILE, SQUAD_DEV_FILE, BPE_MERGES_FILE

from core.tokenizer import train_bpe

train_data = json.load(open(SQUAD_TRAIN_FILE))

articles = train_data["data"][:50]

all_text = []

for article in articles:
    for paras in article["paragraphs"]:
        all_text.append(paras["context"])

        for qa in paras["qas"]:
            all_text.append(qa["question"])

joined_text = " ".join(all_text)

start = time.time()
merged, merges = train_bpe(list(joined_text), 400)

elapsed = time.time() - start

print(len(merges))
print(len(merged))


print(elapsed, "seconds")

with open(BPE_MERGES_FILE, "w") as f:
    json.dump(merges, f, indent=2)
