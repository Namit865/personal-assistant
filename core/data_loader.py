import json


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
