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

    cache = {
        "X": x,
        "Q": Q,
        "K": K,
        "V": V,
        "weights": weights,
        "merged": merged,
    }

    return merged @ Wo, cache


def relu(x):
    return np.maximum(0, x)


def feed_forward(x, W1, b1, W2, b2):
    hidden = relu(x @ W1 + b1)
    cache = {"x": x, "hidden": hidden}
    return (hidden @ W2 + b2), cache


def layernorm(x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def transformer_block(x, params, i):
    attn_output, attn_cache = attention(
        x,
        params[f"Wq_{i}"],
        params[f"Wk_{i}"],
        params[f"Wv_{i}"],
        params[f"Wo_{i}"],
    )

    norm1_input = x + attn_output
    x = layernorm(norm1_input)

    ff_out, ff_cache = feed_forward(
        x,
        params[f"W1_{i}"],
        params[f"b1_{i}"],
        params[f"W2_{i}"],
        params[f"b2_{i}"],
    )
    norm2_input = x + ff_out
    x = layernorm(norm2_input)

    cache = {
        "attn": attn_cache,
        "ff": ff_cache,
        "norm1_input": norm1_input,
        "norm2_input": norm2_input,
        "norm1_output": x,
    }

    return x, cache


def forward(ids, params):
    x = embed(ids, params)
    caches = []

    for i in range(N_LAYERS):
        x, block_cache = transformer_block(x, params, i)

        caches.append(block_cache)

    cache = {"ids": ids, "final_x": x, "blocks": caches}

    start_logits = (x @ params["W_start"]).flatten()
    end_logits = (x @ params["W_end"]).flatten()

    return start_logits, end_logits, cache


def cross_entropy_loss(start_logits, end_logits, start_label, end_label):
    shifted = start_logits - start_logits.max()
    exp = np.exp(shifted)
    probs_start = exp / exp.sum()

    assigned_probs = probs_start[start_label]

    start_loss = -np.log(assigned_probs)

    shifted2 = end_logits - end_logits.max()
    exp2 = np.exp(shifted2)
    probs_end = exp2 / exp2.sum()

    assigned_probs2 = probs_end[end_label]

    end_loss = -np.log(assigned_probs2)

    return (start_loss + end_loss) / 2, probs_start, probs_end


def layernorm_backward(d_out, x, eps=1e-8):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm = (x - mean) / std

    d_x = (
        d_out
        - d_out.mean(axis=-1, keepdims=True)
        - norm * (d_out * norm).mean(axis=-1, keepdims=True)
    ) / std

    return d_x


def backward(cache, params, probs_start, probs_end, start_label, end_label):

    grads = {}

    dloss = 1.0

    d_start_logits = probs_start.copy()
    d_start_logits[start_label] -= 1
    d_start_logits *= 0.5

    d_start_logits = d_start_logits * dloss

    d_end_logits = probs_end.copy()
    d_end_logits[end_label] -= 1
    d_end_logits *= 0.5

    d_end_logits = d_end_logits * dloss

    d_start_out = d_start_logits.reshape(-1, 1)

    d_end_out = d_end_logits.reshape(-1, 1)

    grads["W_start"] = cache["final_x"].T @ d_start_out
    d_final_x_start = d_start_out @ params["W_start"].T

    grads["W_end"] = cache["final_x"].T @ d_end_out
    d_final_x_end = d_end_out @ params["W_end"].T

    d_final_x = d_final_x_start + d_final_x_end

    d_x = d_final_x

    for i in reversed(range(N_LAYERS)):
        block = cache["blocks"][i]

        d_n2_in = layernorm_backward(d_x, block["norm2_input"])

        d_ff_out = d_n2_in
        d_x_skip = d_n2_in

        grads[f"b2_{i}"] = d_ff_out.sum(axis=0)

        grads[f"W2_{i}"] = block["ff"]["hidden"].T @ d_ff_out

        d_hidden = d_ff_out @ params[f"W2_{i}"].T

        d_pre = d_hidden * (block["ff"]["hidden"] > 0)

        grads[f"b1_{i}"] = d_pre.sum(axis=0)

        grads[f"W1_{i}"] = block["ff"]["x"].T @ d_pre

        d_x_from_ff = d_pre @ params[f"W1_{i}"].T

        d_x = d_x_from_ff + d_x_skip

        d_n1_in = layernorm_backward(d_x, block["norm1_input"])

        d_attn_out = d_n1_in

        d_x_skip2 = d_n1_in

        grads[f"Wo_{i}"] = block["attn"]["merged"].T @ d_attn_out

        d_merged = d_attn_out @ params[f"Wo_{i}"].T

        T = d_merged.shape[0]
        head_dim = D_MODEL // N_HEADS
        d_out = d_merged.reshape(T, N_HEADS, head_dim).transpose(1, 0, 2)

        d_V = block["attn"]["weights"].transpose(0, 2, 1) @ d_out
        d_weights = d_out @ block["attn"]["V"].transpose(0, 2, 1)

        w = block["attn"]["weights"]

        d_scaled = w * (
            d_weights - (d_weights * w).sum(axis=-1, keepdims=True)
        )

        d_scores = d_scaled / np.sqrt(head_dim)

        Qh = block["attn"]["Q"]
        Kh = block["attn"]["K"]

        d_Q = d_scores @ Kh

        d_K = d_scores.transpose(0, 2, 1) @ Qh

        d_Q_flat = d_Q.transpose(1, 0, 2).reshape(T, D_MODEL)
        d_K_flat = d_K.transpose(1, 0, 2).reshape(T, D_MODEL)
        d_V_flat = d_V.transpose(1, 0, 2).reshape(T, D_MODEL)

        ax = block["attn"]["X"]
        grads[f"Wq_{i}"] = ax.T @ d_Q_flat
        grads[f"Wk_{i}"] = ax.T @ d_K_flat
        grads[f"Wv_{i}"] = ax.T @ d_V_flat

        d_x_from_attn = (
            d_Q_flat @ params[f"Wq_{i}"].T
            + d_K_flat @ params[f"Wk_{i}"].T
            + d_V_flat @ params[f"Wv_{i}"].T
        )

        d_x = d_x_from_attn + d_x_skip2

    grads["positional_emb"] = np.zeros_like(params["positional_emb"])
    grads["positional_emb"][:T] = d_x

    grads["token_emb"] = np.zeros_like(params["token_emb"])
    np.add.at(grads["token_emb"], cache["ids"], d_x)

    return grads


def update_params(params, grads, lr):
    for key in grads:
        params[key] = params[key] - lr * grads[key]

    return params
