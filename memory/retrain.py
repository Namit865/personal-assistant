import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_loader import load_examples, split_examples, build_label_map
from config import (
    SEED_FILE,
    CORRECTIONS_FILE,
    MIN_FREQ,
    WEIGHTS_FILE,
    VOCAB_FILE,
)
from core.vectorizer import build_vocab, build_dataset
from core.trainer import train
from core.classifier import forward

seed = load_examples(SEED_FILE)

corrections = load_examples(CORRECTIONS_FILE)

train_ex, test_ex = split_examples(seed, 0.2, 42)

candidate_train = train_ex + corrections

label_map = build_label_map(seed)

baseline_vocab = build_vocab(train_ex, MIN_FREQ)
xtr_b, ytr_b = build_dataset(train_ex, baseline_vocab, label_map)
w1_b, b1_b, w2_b, b2_b = train(xtr_b, ytr_b)

candidate_vocab = build_vocab(candidate_train, MIN_FREQ)
xtr_c, ytr_c = build_dataset(candidate_train, candidate_vocab, label_map)
w1_c, b1_c, w2_c, b2_c = train(xtr_c, ytr_c)

xte_b, yte_b = build_dataset(test_ex, baseline_vocab, label_map)
xte_c, yte_c = build_dataset(test_ex, candidate_vocab, label_map)

probs_b, _, _, _ = forward(xte_b, w1_b, b1_b, w2_b, b2_b)
baseline_acc = (probs_b.argmax(axis=1) == yte_b).mean()

probs_c, _, _, _ = forward(xte_c, w1_c, b1_c, w2_c, b2_c)
candidate_acc = (probs_c.argmax(axis=1) == yte_c).mean()

tolerance = 0.02

print("Baseline Accuracy:", baseline_acc)
print("Candidate Accuracy:", candidate_acc)


if candidate_acc >= baseline_acc - tolerance:
    print("Gate Passed - retraining on full data.")

    production_data = seed + corrections

    production_vocab = build_vocab(production_data, MIN_FREQ)

    xp,yp = build_dataset(production_data, production_vocab, label_map)

    w1_p, b1_p, w2_p, b2_p = train(xp, yp)

    np.savez(WEIGHTS_FILE, w1=w1_p, b1=b1_p, w2=w2_p, b2=b2_p)

    VOCAB_FILE.write_text(json.dumps(production_vocab))
else:
    print("Gate Failed - corrections not merged.")
