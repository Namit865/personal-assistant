# Personal Assistant (from scratch)

A Windows command-line personal assistant with a hand-built intent classifier — no ML libraries. Every gradient in the network is derived by hand and verified numerically against PyTorch autograd (max abs diff `1e-17`–`1e-19`). Held-out test accuracy: **88.1%**.

This isn't a wrapper around an existing NLU library. The classifier is a two-layer neural net written in raw NumPy, trained on a small seed dataset, with forward pass, backward pass, and parameter updates all implemented from the underlying math — not `model.fit()`.

## Why it's built this way

Most "personal assistant" tutorials wire together an intent-classification API and a handful of `if` statements. The point of this project was the opposite: understand and implement every layer myself, so debugging is possible at any depth — from a shape mismatch in a gradient to a Start Menu shortcut that doesn't resolve.

## Architecture

The project is split into two halves that don't know much about each other:

- **Understanding** (`core/`) — text goes in, an intent label and confidence score come out. This is the NumPy classifier: tokenization → vectorization → forward pass → softmax → prediction. It has no idea what `open_app` *does*, only that this text belongs to that label.
- **Acting** (`actions/`) — given a label and the original text, actually do something: launch an app, open a browser, write a file, report system stats. Handlers have no idea how the label was chosen.

`main.py` is the only place these two halves meet: it classifies, checks a confidence threshold, and dispatches to the matching handler.

```
text → [classifier: predict()] → (label, confidence)
                                        │
                              confidence < 0.6? → "not sure" and re-prompt
                                        │
                              REGISTRY[label](text, context) → action happens
```

## The six intents

| Intent | What it does |
|---|---|
| `open_app` | Finds and launches an installed desktop app by short name (e.g. "open vlc") |
| `web_search` | Opens a Google search for the typed query in the default browser |
| `create_note` | Appends a timestamped line to `notes.txt` |
| `system_status` | Reports time, CPU, RAM, disk, battery, power mode, and top memory-consuming processes — filtered to just what you asked for |
| `exit` | Prints a goodbye message and ends the session |
| `unknown` | Fallback for anything the classifier can't confidently place elsewhere |

### `open_app` in more detail

Rather than a hardcoded `{"discord": "C:/..."}` map — which breaks on any other machine — this scans both Start Menu folders (`%APPDATA%` and `%PROGRAMDATA%`) for `.lnk` shortcuts at startup, recursively, and builds an index (~149 unique entries on a typical Windows install). Lookup is two-tier:

1. **Exact match** against the shortcut's filename
2. **Prefix-ranked substring match** — e.g. "vlc" correctly resolves to `VLC media player.lnk` over `VLC media player - reset preferences and cache files.lnk`, because it prefers keys that *start with* the query, then breaks ties by shortest key

If the app was installed after the index was built, a miss triggers exactly one rescan before giving up.

### `system_status` in more detail

Computes all six sections (time, CPU, RAM, disk, battery, power mode, top processes) on every call, but only prints the sections your text actually asked for — "battery status" prints just battery; "current power plan and battery usage" prints exactly those two; plain "system status" or "check status" prints the full report. Power mode is read via `powercfg /getactivescheme` through `subprocess`, since Windows power plans aren't exposed by `psutil`.

## Setup

```bash
git clone https://github.com/Namit865/personal-assistant
cd personal-assistant
pip install psutil numpy
python main.py
```

`weights.npz` and `vocab.json` are committed deliberately, so this runs immediately after cloning — no training step required. Type a command at the `>` prompt; type `exit` (or any phrasing the classifier recognizes as the exit intent) to quit.

### Correcting mistakes

If the assistant misclassifies something, follow it immediately with:

```
!fix <correct_label>
```

This logs the text and the correct label to `data/corrections.json` for a future batched retrain. It does **not** retrain on the spot — one-off retraining after a single correction risks catastrophic forgetting, so corrections are meant to accumulate before being used.

## Known limitations

- **`.lnk` scanning has a real boundary.** It can launch traditionally-installed desktop apps, but it cannot reach Windows Store/UWP apps (modern Calculator, Notepad in some installs) or protocol-addressed system panels (`ms-settings:`), because those were never expressed as `.lnk` files to begin with. This isn't a bug to fix — it's a different addressing mechanism entirely, out of scope for this approach.
- **`close_app` doesn't exist yet.** There's no way to close an app the assistant opened — this is a genuine gap in the intent set (not just missing training data), planned as the next major addition.
- **The classifier's weakest intent is `system_status`** (~70% on held-out data at last check), mostly from leading-word ambiguity in the training examples. Phrasings not well-covered by seed data may return "not sure" or misroute — that's what `!fix` is for.
- **CPU/GPU temperature is intentionally excluded.** `psutil` doesn't reliably expose it on Windows; reporting a number here would mean faking it.

## Project structure

```
personal-assistant/
├── main.py                 # entry point: load model, run input loop, dispatch
├── config.py                # path constants only — no logic
├── core/
│   ├── text_utils.py         # tokenization
│   ├── vectorizer.py         # text → fixed-width vector
│   ├── data_loader.py        # seed examples, label map
│   ├── classifier.py         # forward/backward pass, predict()
│   └── trainer.py             # training loop
├── scripts/
│   └── train.py               # entry point for (re)training
├── actions/
│   ├── handlers.py            # one function per intent
│   ├── registry.py            # label → handler function map
│   └── app_finder.py          # build_app_index(), find_app_path()
├── memory/
│   └── corrections.py         # !fix logging (read-modify-write JSON)
├── data/                     # seed examples + corrections.json (gitignored)
├── weights.npz                # trained parameters (committed)
├── vocab.json                  # trained vocabulary (committed)
└── notes.txt                    # created at runtime by create_note (gitignored)
```

## Roadmap

- [ ] `memory/retrain.py` — gated batched retrain on seed + accumulated corrections
- [ ] `close_app` intent — new label, ~100 training examples, retrain, handler, registry entry
- [ ] Permanent train/test split in-repo
- [ ] Fill in `test_gradients.py`, `test_vectorizer.py`, `test_text_utils.py`