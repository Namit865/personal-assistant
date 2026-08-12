import json


def load_corrections(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return json.loads(text)


def log_correction(path, text, correct_label):
    corrections = load_corrections(path)

    corrections.append({"text": text, "intent": correct_label})

    path.write_text(json.dumps(corrections, indent=2), encoding="utf-8")
