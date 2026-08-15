import json
import numpy as np


def load_examples(path):
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"expected a list, got {type(data)}")

    for i, item in enumerate(data):
        if "text" not in item or "intent" not in item:
            raise ValueError(f"item {i} missing 'text' or 'intent': {item}")

    return data


def build_label_map(examples):
    words = []
    sorted_dict = {}

    for ex in examples:
        words.append(ex["intent"])

    sorted_version = sorted(set(words))

    for i, val in enumerate(sorted_version):
        sorted_dict[val] = i

    return sorted_dict


def split_examples(examples, test_ratio, seed):
    rng = np.random.default_rng(seed)

    idx = rng.permutation(len(examples))

    n_test = int(len(examples) * test_ratio)

    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train_ex = [examples[i] for i in train_idx]

    test_ex = [examples[i] for i in test_idx]

    return train_ex, test_ex
