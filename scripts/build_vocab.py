import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SQUAD_TRAIN_FILE, SQUAD_DEV_FILE, BPE_MERGES_FILE

train_data = json.load(open(SQUAD_TRAIN_FILE))

articles = train_data["data"][:50]

all_text = []

for article in articles:
    for paras in article["paragraphs"]:
        all_text.append(paras["context"])

        for qa in paras["qas"]:
            all_text.append(qa["question"])

joined_text = " ".join(all_text)
