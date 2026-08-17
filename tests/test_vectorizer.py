import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vectorizer import build_vocab, vectorize, build_dataset
from core.text_utils import tokenize


def test_vocab_reserves_unk_at_zero():
    vocab = build_vocab(
        [{"text": "hello world"}],
        min_freq=1,
    )
    assert vocab["<UNK>"] == 0


def test_vocab_min_freq_filtering():
    examples = [{"text": "hello world"}, {"text": "hello there"}]
    vocab = build_vocab(examples, min_freq=2)
    assert vocab == {"<UNK>": 0, "hello": 1}


def test_vocab_sorted_order():
    examples = [{"text": "zebra apple"}, {"text": "zebra apple"}]
    vocab = build_vocab(examples, min_freq=2)

    assert vocab == {"<UNK>": 0, "apple": 1, "zebra": 2}


def test_vocab_custom_min_freq():
    examples = [{"text": "unique"}]
    vocab = build_vocab(examples, min_freq=1)

    assert vocab == {"<UNK>": 0, "unique": 1}


def test_vectorizer_shape():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    assert vectorize("hello", vocab).shape == (3,)


def test_vectorize_known_words():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    assert vectorize("hello world", vocab).tolist() == [0, 1, 1]


def test_vectorize_unknown_words():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    assert vectorize("banana", vocab).tolist() == [1, 0, 0]


def test_vectorize_pressure_not_count():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    assert vectorize("hello hello hello", vocab).tolist() == [0, 1, 0]


def test_build_dataset_shapes():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    label_map = {"greet": 0, "farewell": 1}
    examples = [
        {"text": "hello", "intent": "greet"},
        {"text": "world", "intent": "farewell"},
    ]

    x, y = build_dataset(examples, vocab, label_map)

    assert x.shape == (2, 3)
    assert y.shape == (2,)


def test_build_dataset_label_mapping():
    vocab = {"<UNK>": 0, "hello": 1, "world": 2}
    label_map = {"greet": 0, "farewell": 1}
    examples = [
        {"text": "hello", "intent": "greet"},
        {"text": "world", "intent": "farewell"},
    ]

    x, y = build_dataset(examples, vocab, label_map)

    assert y.tolist() == [0, 1]
