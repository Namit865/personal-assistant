from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED_FILE = DATA_DIR / "seed_examples.json"
CORRECTIONS_FILE = DATA_DIR / "corrections.json"
VOCAB_FILE = MODELS_DIR / "vocab.json"
WEIGHTS_FILE = MODELS_DIR / "weights.npz"


MIN_FREQ = 2
