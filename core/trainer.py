import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SEED_FILE, MIN_FREQ, VOCAB_FILE, WEIGHTS_FILE
from core.data_loader import load_examples, build_label_map
from core.vectorizer import build_vocab, vectorize, build_dataset
from core.classifier import (
    relu,
    softmax,
    forward,
    backward,
    compute_loss,
    update_params,
)


def train(X, y, hidden_size=32, epochs=2000, learning_rate=0.1):
    input_size = X.shape[1]
    num_classes = len(set(y))
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, num_classes) * 0.01
    b2 = np.zeros(num_classes)

    for epoch in range(epochs):

        probs, z1, a1, z2 = forward(X, w1, b1, w2, b2)

        loss = compute_loss(probs, y)

        dw1, db1, dw2, db2 = backward(X, y, z1, a1, z2, probs, w2)

        w1, b1, w2, b2 = update_params(
            w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate
        )

        if epoch/epochs*100 % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss}")

    return w1, b1, w2, b2
