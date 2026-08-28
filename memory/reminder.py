import json
from datetime import datetime
from config import REMINDERS_FILE, ROOT

NOTES_FILE = ROOT / "notes.txt"
REMINDER_TAG = "[reminder] "


def load_reminders():
    if REMINDERS_FILE.exists():
        return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
    return []


def save_reminders(items):
    REMINDERS_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def append_reminder_note(body, when: datetime):
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"{when:%Y-%m-%d %H:%M} | {REMINDER_TAG}{body}\n")


def remove_note_for_body(body: str):
    if not NOTES_FILE.exists():
        return

    target = body.lower().strip()
    lines = NOTES_FILE.read_text(encoding="utf-8").splitlines()
    kept = []

    for line in lines:
        if " | " not in line:
            kept.append(line)
            continue

        _, note_body = line.split(" | ", 1)
        nb = note_body.strip()
        if nb.startswith(REMINDER_TAG):
            nb = nb[len(REMINDER_TAG):]

        if nb.lower() == target or target in nb.lower():
            continue

        kept.append(line)

    text = "\n".join(kept)
    if kept:
        text += "\n"
    NOTES_FILE.write_text(text, encoding="utf-8")


def add_reminder(body, when: datetime):
    items = load_reminders()
    items.append({
        "body": body,
        "when": when.isoformat(),
    })
    save_reminders(items)
    append_reminder_note(body, when)


def due_reminders(now=None):
    if now is None:
        now = datetime.now()

    items = load_reminders()
    pending = []
    out = []

    for item in items:
        if item.get("fired"):
            continue

        when = datetime.fromisoformat(item["when"])
        if when <= now:
            out.append(item["body"])
            remove_note_for_body(item["body"])
        else:
            pending.append({
                "body": item["body"],
                "when": item["when"],
            })

    save_reminders(pending)
    return out
