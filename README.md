# Personal Assistant (from scratch)

A Windows personal assistant with a hand-built intent classifier — no ML libraries for the command brain. Every gradient in that network is derived by hand and verified numerically against PyTorch autograd (max abs diff ~`1e-17`, in `tests/test_gradients.py`).

This is not a wrapper around an NLU API. The classifier is a two-layer NumPy net: tokenize → bag-of-words → forward → softmax → intent. Handlers then act on your PC. Typed text and voice both feed the **same** turn function.

## Architecture

Three layers that stay separate on purpose:

1. **Input** — keyboard at `>`, or type `v` for a mic session (`listen()`). Both produce a normal string.
2. **Understanding** (`core/classifier.py`) — string → `(label, confidence)`. Does not know how to open Chrome.
3. **Acting** (`actions/`) — label → handler. Handlers **return** a short completion string; they should not be the only place that prints.

`main.py` owns the turn:

```
text (type or listen)
  → process(): predict → REGISTRY[label](text, context)
  → (label, message)
  → print(message) + speak(message)
```

Low confidence returns `(None, "Uncertainity")` so a weak guess never triggers `exit`. Voice mode stays on until you say `keyboard` / `stop` / `stop listening`, or until `exit`.

Every handler uses `(text, context)`. `context` currently holds the Start Menu app index.

## The eight intents

| Intent | What it does |
|---|---|
| `open_app` | Launch an installed app by short name (`open vlc`) |
| `close_app` | Terminate matching running processes |
| `web_search` | Open Google (or YouTube if you say `youtube`) for an **extracted** query |
| `create_note` | Append a timestamped **body** (command words stripped) to `notes.txt` |
| `system_status` | Report time / CPU / RAM / disk / battery / power / top processes (sections you asked for) |
| `knowledge_query` | Ask Tavily with `include_answer=True` and return that short factual answer |
| `exit` | Goodbye and end the session |
| `unknown` | Fallback when nothing else fits |

Seed data: **874** examples (100 each for most command intents; `knowledge_query` 134; `unknown` 140). Retrain with `scripts/train.py` after seed changes.

### Slots (not new intents)

The classifier only picks the **job**. Handlers pull **what** / **where** from the sentence:

- `extract_app_name` — strips launch/close filler for `open_app` / `close_app`
- `extract_search_query` — strips search filler; if `youtube` is present, open YouTube results and drop `on` / `in` / `youtube` from the query; else Google
- `extract_note_body` — strips note filler so the file gets content, not “note down that…”

A note is a **file line**, not a Windows reminder popup. Reminders are a later stage.

### `knowledge_query` (product path)

Informational questions (“who invented the telephone”) go to Tavily. The handler prints/speaks `response["answer"]`. Requires `TAVILY_API_KEY` in the user environment (restart the terminal after setting it).

A from-scratch SQuAD **span extractor** (`core/qa_model.py`, `scripts/train_qa.py`, `core/predict_qa.py`) was built as a learning lab. It is **not** on the critical path for answers in `main.py` — a ~240k-param extractive model without pretraining was too weak for live web text. Keep those files; do not wire them back into the handler unless you are experimenting.

### Voice

- `SpeechRecognition` + mic backend → `listen()`
- `pyttsx3` (local Windows SAPI) → `speak()` after every turn message
- Type `v` once → stay on mic; say `keyboard` to return to typing
- After TTS in voice mode, `time.sleep(1)` so the mic does not record the speaker
- `listen()` uses ambient calibration; Google STT needs network for recognition

## Training

`scripts/train.py` still runs two passes:

1. **Diagnostic** — 80/20 split (seed 42), held-out accuracy, weights discarded; vocab from train split only.
2. **Production** — full data; saves `models/weights.npz` and `models/vocab.json`.

Epoch-0 loss should sit near `ln(n_classes)` (`ln(8) ≈ 2.08` with eight intents).

### Corrections

```
!fix <correct_label>
```

Logs to `data/corrections.json`. `memory/retrain.py` does a gated batch retrain (candidate must stay within ~2 points of baseline).

## Tests

```bash
pytest
```

21 tests: text utils, vectorizer, gradient checks vs PyTorch.

## Setup

```bash
git clone https://github.com/Namit865/personal-assistant
cd personal-assistant
pip install -r requirements.txt
# also used by main today (add to requirements when you next touch that file):
# pip install tavily-python SpeechRecognition pyaudio pyttsx3
setx TAVILY_API_KEY "your_key"   # then open a new terminal
python main.py
```

Windows only for app launch / close / power status. Classifier weights are committed so clone-and-run works for typed commands; knowledge answers need the API key; voice needs mic permission and the packages above.

## Known limitations

- **`close_app` is a forceful terminate** — no save prompt.
- **`knowledge_query` depends on Tavily** — quality and availability are the API’s, not a local generative model.
- **YouTube path is search results**, not autoplay of the first video.
- **Notes are not reminders** — no popup until “done.”
- **Voice STT uses Google** by default — needs internet; room noise / wrong mic device can hang or miss speech.
- **Span-QA lab is parked** — do not expect `predict_qa` quality for live assistant answers.

## Project structure

```
personal-assistant/
├── main.py                 # load model, process(), listen/speak, voice_mode loop
├── config.py               # paths + QA hyperparameters (constants only)
├── core/
│   ├── text_utils.py
│   ├── vectorizer.py
│   ├── data_loader.py
│   ├── classifier.py
│   ├── trainer.py
│   ├── tokenizer.py        # BPE (lab / Stage 1)
│   ├── qa_model.py         # Stage 1 extractive transformer (lab, unused by main)
│   ├── predict_qa.py       # span inference helper (lab)
│   └── retrieval.py        # Tavily fetch_answer()
├── scripts/
│   ├── train.py            # classifier two-pass train
│   ├── train_qa.py         # Stage 1 train (lab)
│   ├── build_dataset.py
│   ├── build_vocab.py
│   └── audit_data.py
├── actions/
│   ├── handlers.py         # intents + slot extractors
│   ├── registry.py
│   └── app_finder.py
├── memory/
│   ├── corrections.py
│   └── retrain.py
├── tests/
├── models/
│   ├── weights.npz         # classifier (committed)
│   ├── vocab.json
│   └── qa_weights.npz      # Stage 1 (gitignored)
├── data/
│   ├── seed_examples.json  # 8 intents
│   ├── corrections.json    # gitignored
│   ├── bpe_merges.json
│   ├── qa_vocab.json
│   └── squad/              # gitignored
└── notes.txt               # runtime notes (gitignored)
```

## Where this is going

**Done for the assistant product path**

- [x] Eight intents including `knowledge_query` via Tavily answer
- [x] Handlers return a completion string; `process()` is the single turn
- [x] Optional mic session + local TTS
- [x] Slots for search (Google / YouTube) and note body
- [x] Stage 1 transformer lab (trained, parked off `main`)

**Next (still command/slots, not vision)**

- [ ] Stronger multi-slot lines (`open X and search Y`) without hardcoding one media intent
- [ ] Optional: Windows reminders / notifications (separate from file notes)
- [ ] Optional: local STT so voice does not depend on Google
- [ ] Screen / UI vision only when command + slots are solid

**Parked (lab, not product blockers)**

- [ ] Improve Stage 1 span EM (optional experiment)
- [ ] Stage 2 generative fusion — deliberately off the critical path
