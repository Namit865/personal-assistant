import json
import numpy as np

from config import VOCAB_FILE, WEIGHTS_FILE, SEED_FILE, CORRECTIONS_FILE
from core.classifier import predict
from core.data_loader import load_examples, build_label_map
from actions.registry import REGISTRY
from memory.corrections import log_correction
from actions.app_finder import build_app_index

THRESHOLD = 0.6


def load_model():
    vocab = json.loads(VOCAB_FILE.read_text())
    data = np.load(WEIGHTS_FILE)
    w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], data["b2"]

    label_map = build_label_map(load_examples(SEED_FILE))

    app_index = build_app_index()

    context = {"app_index": app_index}

    return vocab, label_map, w1, b1, w2, b2, context


def main():
    vocab, label_map, w1, b1, w2, b2, context = load_model()
    print("Write a message...")

    last_text = None

    while True:
        text = input("> ").strip()

        if text.startswith("!fix "):
            label = text[5:].strip()
            if last_text is None:
                print("Nothing to correct")
                continue

            if label not in REGISTRY:
                print(f"unknown label. valid: {list(REGISTRY)}")
                continue

            log_correction(CORRECTIONS_FILE, last_text, label)
            print(f"logged: {last_text!r} -> {label}")
            continue

        if not text:
            continue

        last_text = text

        label, confidence = predict(text, vocab, label_map, w1, b1, w2, b2)

        if confidence < THRESHOLD:
            print("Uncertainity")
            continue

        handler = REGISTRY[label]
        handler(text, context)

        if label == "exit":
            break

        if text == "exit":
            break


if __name__ == "__main__":
    main()
