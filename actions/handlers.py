import webbrowser
from urllib.parse import quote_plus
from datetime import datetime
from config import ROOT
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.text_utils import tokenize

FILLER = {
    "open",
    "launch",
    "start",
    "run",
    "up",
    "fire",
    "the",
    "my",
    "a",
    "an",
    "please",
    "can",
    "you",
    "for",
    "me",
    "app",
    "application",
    "program",
}


def open_app(text):
    print(f"[open app] {text}")


def web_search(text):
    print(f"[web search] {text}")
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(text)}")


def create_note(text):
    print(f"[create_note] {text}")
    notes = ROOT / "notes.txt"
    with open(notes, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {text}\n")
    print(f"note saved to {notes}")


def system_status(text):
    print(f"[system_status] {text}")


def exit_assistant(text):
    print(f"[exit] {text}")


def unknown(text):
    print("I didn't understand that.")


def extract_app_name(text):
    words = tokenize(text)
    kept = []

    for word in words:
        if word not in FILLER:
            kept.append(word)

    result = " ".join(kept)

    return result


for t in ["open discord", "can you launch spotify for me", "fire up vs code"]:
    print(f"{t!r} -> {extract_app_name(t)!r}")
