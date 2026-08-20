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


def embed(ids, params):
    ids = np.array(ids)

    token_vector = params["token_emb"][ids]

    slice_tokens = params["positional_emb"][: len(ids)]

    return slice_tokens + token_vector


def attention(x, Wq, Wk, Wv, Wo):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    head_dim = D_MODEL // N_HEADS
    T = x.shape[0]

    Q = Q.reshape(T, N_HEADS, head_dim).transpose(1, 0, 2)
    K = K.reshape(T, N_HEADS, head_dim).transpose(1, 0, 2)
    V = V.reshape(T, N_HEADS, head_dim).transpose(1, 0, 2)

    scores = Q @ K.transpose(0, 2, 1)
    scaled = scores / np.sqrt(head_dim)

    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / exp.sum(axis=-1, keepdims=True)

    out = weights @ V

    merged = out.transpose(1, 0, 2).reshape(T, D_MODEL)

    return merged @ Wo


def relu(x):
    return np.maximum(0, x)


def feed_forward(x, W1, b1, W2, b2):
    hidden = relu(x @ W1 + b1)
    return hidden @ W2 + b2


def layernorm(x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def transformer_block(x, params, i):
    attn_output = attention(
        x,
        params[f"Wq_{i}"],
        params[f"Wk_{i}"],
        params[f"Wv_{i}"],
        params[f"Wo_{i}"],
    )
    x = layernorm(x + attn_output)

    ff_out = feed_forward(
        x,
        params[f"W1_{i}"],
        params[f"b1_{i}"],
        params[f"W2_{i}"],
        params[f"b2_{i}"],
    )
    x = layernorm(x + ff_out)

    return x


def forward(ids, params):
    x = embed(ids, params)
    for i in range(N_LAYERS):
        x = transformer_block(x, params, i)

    start_logits = (x @ params["W_start"]).flatten()
    end_logits = (x @ params["W_end"]).flatten()

    return start_logits, end_logits


def cross_entropy_loss(start_logits, end_logits, start_label, end_label):
    shifted = start_logits - start_logits.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()

    assigned_probs = probs[start_label]

    start_loss = -np.log(assigned_probs)

    shifted2 = end_logits - end_logits.max()
    exp2 = np.exp(shifted2)
    probs2 = exp2 / exp2.sum()

    assigned_probs2 = probs2[end_label]

    end_loss = -np.log(assigned_probs2)

    return (start_loss + end_loss) / 2
