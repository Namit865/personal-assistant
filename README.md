# Personal Assistant (from scratch)

A Windows command-line personal assistant with a hand-built intent classifier — no ML libraries. Every gradient in the network is derived by hand and verified numerically against PyTorch autograd (max abs diff `1e-17`–`1e-19`, reproduced mechanically in `tests/test_gradients.py`). Held-out test accuracy: **86.4%** across 7 intents, measured via a permanent diagnostic split that runs on every retrain (see `scripts/train.py`).

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

Every handler takes the same `(text, context)` signature, where `context` is a dict of shared state built once at startup (currently the app index). Five of the seven handlers ignore it entirely. This was chosen over per-handler signatures deliberately: adding new shared state later means adding a dict key, not editing every function signature and the dispatch line.

## The seven intents

| Intent | What it does |
|---|---|
| `open_app` | Finds and launches an installed app by short name (e.g. "open vlc") |
| `close_app` | Finds every running process matching a short app name and terminates all of them |
| `web_search` | Opens a Google search for the typed query in the default browser |
| `create_note` | Appends a timestamped line to `notes.txt` |
| `system_status` | Reports time, CPU, RAM, disk, battery, power mode, and top memory-consuming processes — filtered to just what you asked for |
| `exit` | Prints a goodbye message and ends the session |
| `unknown` | Fallback for anything the classifier can't confidently place elsewhere |

Training data is balanced at 100 examples per real intent, with `unknown` at 173 — 773 examples total.

### `open_app` in more detail

Rather than a hardcoded `{"discord": "C:/..."}` map — which breaks on any other machine — the app index is built at startup from two sources:

1. **Start Menu `.lnk` shortcuts**, scanned recursively from `%APPDATA%` and `%PROGRAMDATA%` (~149 entries on a typical install)
2. **`Get-StartApps`** via a PowerShell subprocess, which returns UWP/Store apps that never had `.lnk` files at all — Settings, the modern Calculator, Store-installed apps like Telegram. These are indexed as `shell:AppsFolder\{AUMID}` and launch through the same `os.startfile()` path.

Lookup against that index is two-tier:

1. **Exact match** against the shortcut's filename
2. **Prefix-ranked substring match** — "vlc" resolves to `VLC media player.lnk` over `VLC media player - reset preferences and cache files.lnk`, because it prefers keys that *start with* the query, then breaks ties by shortest key

If the app was installed after the index was built, a miss triggers exactly one rescan before giving up.

### `close_app` in more detail

Queries running processes fresh on every call — never cached, unlike `open_app`'s index. A process list a few seconds stale is actively wrong, while an app index a few minutes stale is merely incomplete.

Matches by substring against process names, then **terminates every matching process**, not just one. Multi-process apps like Chrome or Electron-based tools spawn one process per tab/window/helper, so killing a single instance leaves the app visibly still running. In practice "close chrome" terminates around 14 processes.

Two things worth knowing about how it fails:

- `psutil.Process.terminate()` is a **forceful kill** (`TerminateProcess` on Windows) — there's no "save changes?" prompt. Fine for disposable apps, real risk for anything with unsaved work.
- A process can legitimately exit in the gap between listing it and killing it (`psutil.NoSuchProcess`), or be protected by the OS (`psutil.AccessDenied`). Both are caught per-process and reported, not allowed to crash the whole command.

**`close_app` vs `exit` is a deliberately resolved ambiguity, not an accident.** Bare phrases like "close" or "stop" with no named object could plausibly mean either "quit this app" or "shut down the assistant." The 100 original `exit` examples were re-sorted by an explicit three-tier rule: phrases containing an object word (app, window, process, terminal…) always go to `close_app`; phrases with self-referential or farewell vocabulary (goodbye, session, yourself, shutdown…) go to `exit`; anything matching neither defaults to `close_app`. That default is the point — a wrong `close_app` guess fails harmlessly, while a wrong `exit` guess ends the whole session. Cheaper failure wins.

Both intents share one `extract_app_name()` with a combined filler set covering launching and closing verbs, since the two verb sets never collide.

### `system_status` in more detail

Computes all six sections (time, CPU, RAM, disk, battery, power mode, top processes) on every call, but only prints the sections your text actually asked for — "battery status" prints just battery; "current power plan and battery usage" prints exactly those two; plain "system status" prints the full report. Each section is gated by an independent `if`, not `elif`, so compound requests print multiple sections. Power mode is read via `powercfg /getactivescheme` through `subprocess`, since Windows power plans aren't exposed by `psutil`.

## Training

`scripts/train.py` runs **two passes every time**, and this is permanent rather than a debugging convenience:

1. **Diagnostic pass** — 80/20 split (seed 42), reports honest held-out accuracy, then discards the weights.
2. **Production pass** — trains on the full dataset; *these* are the weights saved to `models/`.

The vocabulary is rebuilt separately for each pass. The diagnostic pass builds its vocab from training examples only, so test-set words never leak into the word-frequency counts that decide which tokens survive `min_freq=2`. Collapsing these into one shared vocab would quietly inflate the reported accuracy — the distinction is load-bearing.

Epoch-0 loss is verified against `ln(n_classes)` on every retrain (`ln(7) ≈ 1.9459`) as a cheap sanity check that the softmax and label mapping are wired correctly.

### Correcting mistakes

If the assistant misclassifies something, follow it immediately with:

```
!fix <correct_label>
```

This logs the text and correct label to `data/corrections.json`. It does **not** retrain on the spot — one-off retraining after a single correction risks catastrophic forgetting.

`memory/retrain.py` consumes those corrections in a **gated batch retrain**: it trains a baseline model on seed data alone and a candidate model on seed + corrections, each with its own vocabulary, and evaluates both on the same held-out split. New weights are saved only if the candidate stays within 2 percentage points of the baseline. Otherwise nothing is written and both numbers are reported. The tolerance is roughly twice the run-to-run drift already observed from unseeded weight initialization.

## Tests

```bash
pytest
```

21 tests across three files:

- `test_text_utils.py` (7) — tokenization and cleaning edge cases
- `test_vectorizer.py` (10) — vocab construction, `<UNK>` handling, dataset shapes
- `test_gradients.py` (4) — hand-derived `backward()` against PyTorch autograd at `float64`, `atol=1e-9`, seeded. Both sides must be `float64`; at `float32` the comparison is meaningless.

## Setup

```bash
git clone https://github.com/Namit865/personal-assistant
cd personal-assistant
pip install -r requirements.txt
python main.py
```

`models/weights.npz` and `models/vocab.json` are committed deliberately, so this runs immediately after cloning — no training step required. Type a command at the `>` prompt.

Windows only. `open_app`, `close_app`, and `system_status` all depend on Windows-specific mechanisms (Start Menu layout, `Get-StartApps`, `powercfg`).

## Known limitations

- **`close_app` terminates forcefully, with no save prompt.** A real risk for apps with unsaved work, not just a rough edge.
- **`system_status` is the weakest intent** (~70% on held-out data, measured before the 7-intent retrain), mostly from leading-word ambiguity in the seed examples.
- **`close_app` and `exit` sit close together in confidence space** by design — "quit spotify" and "shut down the assistant" land around 0.67–0.70, just above the 0.6 threshold. Expected given deliberately overlapping vocabulary, but worth monitoring through `!fix` in real use.
- **Overlapping `system_status` keywords can double-print a section.** Low priority, known.
- **CPU/GPU temperature is intentionally excluded.** `psutil` doesn't reliably expose it on Windows; reporting a number here would mean faking it.

## Project structure

```
personal-assistant/
├── main.py                    # entry point: load model, input loop, dispatch
├── config.py                  # path constants only — no logic
├── core/
│   ├── text_utils.py          # clean_text(), tokenize()
│   ├── vectorizer.py          # build_vocab(), vectorize(), build_dataset()
│   ├── data_loader.py         # load_examples(), build_label_map(), split_examples()
│   ├── classifier.py          # forward(), backward(), predict()
│   ├── trainer.py             # training loop
│   └── tokenizer.py           # BPE: train_bpe(), encode(), decode()
├── scripts/
│   ├── train.py               # two-pass training entry point
│   ├── build_vocab.py         # trains BPE merges on SQuAD text
│   ├── checking.py            # one-off close_app/exit reclassification audit
│   └── audit_data.py          # dataset inspection
├── actions/
│   ├── handlers.py            # one function per intent
│   ├── registry.py            # label → handler map
│   └── app_finder.py          # app index, process listing, process matching
├── memory/
│   ├── corrections.py         # !fix logging
│   └── retrain.py             # gated batch retrain
├── tests/                     # 21 pytest tests
├── models/
│   ├── weights.npz            # trained parameters (committed)
│   └── vocab.json             # trained vocabulary (committed)
├── data/
│   ├── seed_examples.json     # 773 training examples
│   ├── corrections.json       # !fix log (gitignored)
│   ├── bpe_merges.json        # 400 trained BPE merges
│   └── squad/                 # SQuAD v1.1 source files (gitignored)
└── notes.txt                  # created at runtime (gitignored)
```

## Where this is going

The seven intents above are a **closed set**. Every one of them maps a sentence to a fixed bucket and runs a fixed function. That's the right architecture for commands, and it is structurally incapable of answering an open question — no amount of extra training data turns a bag-of-words classifier into something that can produce a free-form sentence. More data gives you more buckets, never generation.

So the next phase splits along that exact seam.

### `knowledge_query` — an eighth intent, with a real model behind it

Real informational questions ("what is the definition of consciousness") get their own intent and their own pipeline:

1. **Retrieval** — [Tavily](https://tavily.com) fetches candidate passages for the question.
2. **Stage 1: extraction** — a small transformer, non-generative. Question + one passage + a separator token go in; two linear heads emit a softmax over sequence positions for the answer's start and end index. This is the same softmax the intent classifier uses, applied per-position instead of per-sentence. Trained on **SQuAD v1.1**, which already ships in exactly the `(passage, question, answer-span)` shape — no hand-labeling. Target: `d_model=64`, ~128-token context, tens of thousands of parameters.
3. **Stage 2: fusion** — generative. Question + Stage 1's extracted spans (not full passages) go in; one fused answer sentence comes out, autoregressively. Vocab-sized output head, ~100k–300k parameters. Trained on **HotpotQA**, which has the multi-source fusion shape.

Combined target is roughly 200k–450k parameters — about 0.3% of GPT-2 small. That number is the point, not an apology for it: the scope is narrow question answering over retrieved text, not open-domain chat, and picking extraction-then-fusion over end-to-end generation is what keeps it there.

Both stages share one BPE vocabulary, trained by `scripts/build_vocab.py` on SQuAD contexts and questions and saved to `data/bpe_merges.json` (400 merges over a ~2.5M-character sample). Stage 2 consumes Stage 1's output, so a token id has to mean the same thing to both — separately-trained vocabularies would be a silent correctness bug.

Build order:

- [x] Shared BPE vocabulary wired into `core/tokenizer.py`, trained and saved
- [ ] Stage 1 alone, verified against real SQuAD examples — gradient checks plus held-out span accuracy
- [ ] Stage 2, including the autoregressive sampling loop (temperature, top-k, multinomial draw)
- [ ] `knowledge_query` handler: Tavily fetch → extract per source → fuse → print
- [ ] The intent itself: ~100 examples mined from the existing `unknown` bucket, retrain, verify no crossover

### `unknown` narrows to noise

Once `knowledge_query` exists, `unknown` stops being a catch-all. It keeps genuine gibberish and casual chit-chat ("how are you") and hands everything informational to the new intent. Chit-chat responses will come from a small pretrained local model through Ollama — borrowed weights, not hand-built, and a deliberate exception to this project's rule rather than an oversight. Conversational small talk isn't what the from-scratch effort is for.

### Also queued

- Four more command intents in one batch — media control, calculator, timers, reminders — with candidate examples mined out of the current `unknown` bucket first
- Voice input as a front end: push-to-talk before continuous wake-word detection. The classifier needs zero changes for this; voice only becomes a new way to produce the `text` variable

## Roadmap

- [x] Permanent train/test split in-repo (`scripts/train.py`)
- [x] `memory/retrain.py` — gated batched retrain on seed + accumulated corrections
- [x] `close_app` intent — new label, retrain, handler, registry entry
- [x] Full test suite — `test_gradients.py`, `test_vectorizer.py`, `test_text_utils.py`
- [x] UWP/Store app coverage via `Get-StartApps`
- [x] Shared BPE vocabulary trained on SQuAD
- [ ] Stage 1 extraction model
- [ ] Stage 2 fusion model
- [ ] `knowledge_query` intent, end to end