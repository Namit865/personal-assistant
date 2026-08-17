import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TIER1_OBJECTS = {
    "app",
    "program",
    "window",
    "script",
    "process",
    "thread",
    "task",
    "terminal",
    "shell",
    "dialog",
    "interface",
    "pipeline",
    "module",
    "tool",
    "daemon",
    "framework",
}

TIER2_KEEP = {
    "exit",
    "quit",
    "goodbye",
    "bye",
    "shutdown",
    "shut",
    "down",
    "off",
    "logoff",
    "log",
    "deactivate",
    "leave",
    "disconnect",
    "assistant",
    "session",
    "yourself",
    "interaction",
    "everything",
    "sleep",
}

fresh = json.load(open("data/seed_examples.json"))
print(Counter(ex['intent'] for ex in fresh))