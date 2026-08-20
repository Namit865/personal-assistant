import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN


def init_params(vocab_size):

    params = {}

    params["token_emb"] = np.random.randn(vocab_size, D_MODEL) / np.sqrt(
        D_MODEL
    )

    params["positional_emb"] = np.random.randn(MAX_SEQ_LEN, D_MODEL) / np.sqrt(
        D_MODEL
    )

    for i in range(N_LAYERS):
        params[f"Wq_{i}"] = np.random.randn(D_MODEL, D_MODEL) / np.sqrt(
            D_MODEL
        )
        params[f"Wk_{i}"] = np.random.randn(D_MODEL, D_MODEL) / np.sqrt(
            D_MODEL
        )
        params[f"Wv_{i}"] = np.random.randn(D_MODEL, D_MODEL) / np.sqrt(
            D_MODEL
        )
        params[f"Wo_{i}"] = np.random.randn(D_MODEL, D_MODEL) / np.sqrt(
            D_MODEL
        )

        params[f"W1_{i}"] = np.random.randn(D_MODEL, D_FF) / np.sqrt(D_MODEL)

        params[f"b1_{i}"] = np.zeros(D_FF)

        params[f"W2_{i}"] = np.random.randn(D_FF, D_MODEL) / np.sqrt(D_FF)

        params[f"b2_{i}"] = np.zeros(D_MODEL)

    params["W_start"] = np.random.randn(D_MODEL, 1) / np.sqrt(D_MODEL)
    params["W_end"] = np.random.randn(D_MODEL, 1) / np.sqrt(D_MODEL)

    return params
