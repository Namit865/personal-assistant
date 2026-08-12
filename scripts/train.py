import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.trainer import train
from core.vectorizer import build_dataset, build_vocab
from core.data_loader import load_examples, build_label_map
from config import SEED_FILE, MIN_FREQ, VOCAB_FILE, WEIGHTS_FILE
from core.classifier import predict

examples = load_examples(SEED_FILE)

vocab = build_vocab(examples, MIN_FREQ)

label_map = build_label_map(examples)

X, y = build_dataset(examples, vocab, label_map)

print("X Shape:", X.shape)
print("Y Shape:", y.shape)
print("Vocabulary Len:", len(vocab))
print("Labels Length:", len(label_map))

w1, b1, w2, b2 = train(X, y)


np.savez(WEIGHTS_FILE, w1=w1, b1=b1, w2=w2, b2=b2)

VOCAB_FILE.write_text(json.dumps(vocab))


data = np.load(WEIGHTS_FILE)

w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], data["b2"]

for cmd in [
    "open discord",
    "search for pytorch tutorials",
    "check pc temprature",
    "make an note about the metting",
    "goodbye",
    "akagidbskj",
]:
    label, conf = predict(cmd, vocab, label_map, w1, b1, w2, b2)
    print(f"{cmd:35} -> {label:15} -> {conf:.2f}")