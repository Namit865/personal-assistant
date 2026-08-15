import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.trainer import train
from config import SEED_FILE, VOCAB_FILE, WEIGHTS_FILE, MIN_FREQ
from core.data_loader import load_examples, build_label_map, split_examples
from core.vectorizer import build_dataset, build_vocab
from core.classifier import predict, forward

examples = load_examples(SEED_FILE)

vocab = build_vocab(examples, MIN_FREQ)

labels = build_label_map(examples)

# =========================== Production: Full Data ===========================

X, Y = build_dataset(examples, vocab, labels)

print(
    "========================== Production Full Data Training =========================="
)

w1, b1, w2, b2 = train(X, Y)

np.savez(WEIGHTS_FILE, w1=w1, b1=b1, w2=w2, b2=b2)

VOCAB_FILE.write_text(json.dumps(vocab))

full_probs, _, _, _ = forward(X, w1, b1, w2, b2)
full_probs_acc = (full_probs.argmax(axis=1) == Y).mean()


print("\n\n")
print(
    "==========================Split Train/Test (genralization)=============================="
)

# =========================== Split Train/Test (genralization) ===========================

train_data, test_data = split_examples(examples, 0.2, 42)

train_vocab = build_vocab(train_data, MIN_FREQ)

Xtr, Ytr = build_dataset(train_data, train_vocab, labels)
Xte, Yte = build_dataset(test_data, train_vocab, labels)

print("\n\n ==================== Train Section ====================\n\n")

w1_diag_tr, b1_diag_tr, w2_diag_tr, b2_diag_tr = train(Xtr, Ytr)

train_probs, _, _, _ = forward(
    Xtr, w1_diag_tr, b1_diag_tr, w2_diag_tr, b2_diag_tr
)
train_accuracy = (train_probs.argmax(axis=1) == Ytr).mean()

test_probs, _, _, _ = forward(
    Xte, w1_diag_tr, b1_diag_tr, w2_diag_tr, b2_diag_tr
)
test_accuracy = (test_probs.argmax(axis=1) == Yte).mean()

print("\n\n============================================================")
print("\n Final Outputs:")
print(f"Production Accuracy: {full_probs_acc * 100:.2f}")
print(f"Train Accuracy: {train_accuracy * 100:.2f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}")
print("============================================================")

data = np.load(WEIGHTS_FILE)

w1_pred, b1_pred, w2_pred, b2_pred = (
    data["w1"],
    data["b1"],
    data["w2"],
    data["b2"],
)


for cmd in [
    "open discord",
    "search for pytoch documentations",
    "check pc temprature",
    "make an note about metting",
    "ciya",
    "asdiuasgfa",
]:
    label, conf = predict(
        cmd, vocab, labels, w1_pred, b1_pred, w2_pred, b2_pred
    )
    print(f"{cmd:35} -> {label:15} -> {conf:2f}")
