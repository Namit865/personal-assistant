from collections import Counter
from core.text_utils import tokenize
import numpy as np


def build_vocab(examples, min_freq=2):
    counts = Counter()
    for example in examples:
        words = tokenize(example["text"])

        counts.update(words)
    kept = {}
    for w, c in counts.items():
        if c >= min_freq:
            kept[w] = c

    sorted_words = sorted(kept)
    vocab = {"<UNK>": 0}

    for i, w in enumerate(sorted_words):
        vocab[w] = i + 1

    return vocab


def vectorize(text, vocab):
    words = tokenize(text)
    zeros_list = np.zeros(len(vocab))

    for word in words:
        if word in vocab:
            zeros_list[vocab[word]] = 1
        else:
            zeros_list[0] = 1

    return zeros_list


def build_dataset(examples, vocab, label_map):
    X = np.zeros((len(examples), len(vocab)))
    Y = np.zeros(len(examples), dtype=int)

    for i, example in enumerate(examples):
        X[i] = vectorize(example["text"], vocab)
        Y[i] = label_map[example["intent"]]

    return X, Y
