import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.text_utils import tokenize


def test_lowercase():
    assert tokenize("OPEN Chrome") == ["open", "chrome"]


def test_wifi_special_case():
    assert tokenize("connect to Wi-Fi") == ["connect", "to", "wifi"]


def test_hyphen_join():
    assert tokenize("a well-known fact") == ["a", "wellknown", "fact"]


def test_apostrophe_stripped():
    assert tokenize("don't stop") == ["dont", "stop"]


def test_curly_apostrophe_stripped():
    assert tokenize("don't stop") == ["dont", "stop"]


def test_general_punctuation():
    assert tokenize("Hello world!")


def test_empty_string():
    assert tokenize("") == []
