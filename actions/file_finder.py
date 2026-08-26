import re
from pathlib import Path
from config import FILE_ROOTS
import subprocess
import os

DOC_EXT = {".txt", ".md", ".pdf", ".doc", ".docx", ".xlsx", ".csv", ".json"}
SKIP_PARTS = ("phone link", "screenshot")


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def find_files(name, limit=10):
    name = name.lower().strip()
    if not name:
        return []

    hits = []

    for root in FILE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            full = str(path).lower()
            if any(part in full for part in SKIP_PARTS):
                continue

            filename = path.name.lower()
            stem = path.stem.lower()
            toks = _tokens(stem)

            if name == stem or name == filename or name == path.stem.lower():
                rank = 0
            elif toks == [name] or toks == name.split():
                rank = 0
            elif name in toks and len(toks) <= 3:
                rank = 1
            elif stem.startswith(name) or filename.startswith(name):
                rank = 2
            else:
                continue

            is_doc = 0 if path.suffix.lower() in DOC_EXT else 1
            hits.append((rank, is_doc, len(filename), path))

    hits.sort()
    return [p for _, _, _, p in hits[:limit]]

def open_file(path):
    path = Path(path).resolve()
    if not path.exists():
        return False
    
    subprocess.Popen(["explorer",f"/select,{path}"])
    return True