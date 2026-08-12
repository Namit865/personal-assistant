import numpy as np
from core.vectorizer import vectorize, build_dataset
from core.data_loader import build_label_map


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    exp_sum = np.sum(exp_x, axis=1, keepdims=True)
    probs = exp_x / exp_sum

    return probs


def forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    probs = softmax(z2)

    return probs, z1, a1, z2


def predict(text, vocab, label_map, w1, b1, w2, b2):
    vec = vectorize(text, vocab).reshape(1, -1)

    probs, _, _, _ = forward(vec, w1, b1, w2, b2)

    intent_id = probs.argmax()

    confidence = probs.max()

    inv_map = {idx: val for val, idx in label_map.items()}
    label_name = inv_map[intent_id]

    return label_name, confidence


def compute_loss(probs, y):
    correct_probs = probs[np.arange(len(y)), y]
    loss = -np.log(correct_probs).mean()

    return loss


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate=0.01):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    return w1, b1, w2, b2


def backward(x, y, z1, a1, z2, probs, w2):
    correct_probs = probs[np.arange(len(y)), y]
    dloss = 1.0
    dcorrect_probs_mean = np.full_like(correct_probs, 1 / len(y)) * dloss
    dcorrect_probs = -1 / correct_probs * dcorrect_probs_mean
    dprobs = np.zeros_like(probs)
    dprobs[np.arange(len(y)), y] = dcorrect_probs

    x_shifted = z2 - np.max(z2, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    exp_sum = np.sum(exp_x, axis=1, keepdims=True)
    probs1 = exp_x / exp_sum

    dexp_sum = (-exp_x / exp_sum**2 * dprobs).sum(axis=1, keepdims=True)
    dexp_x = (1 / exp_sum) * dprobs + dexp_sum
    dx_shifted = exp_x * dexp_x
    dz2 = dx_shifted
    db2 = (dz2).sum(axis=0)
    dw2 = a1.T @ dz2
    da1 = dz2 @ w2.T
    dz1 = da1 * (z1 > 0)
    db1 = dz1.sum(axis=0)
    dw1 = x.T @ dz1

    return dw1, db1, dw2, db2
