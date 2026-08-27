from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED_FILE = DATA_DIR / "seed_examples.json"
CORRECTIONS_FILE = DATA_DIR / "corrections.json"
VOCAB_FILE = MODELS_DIR / "vocab.json"
WEIGHTS_FILE = MODELS_DIR / "weights.npz"

SQUAD_DIR = DATA_DIR / "squad"
SQUAD_TRAIN_FILE = SQUAD_DIR / "train-v1.1.json"
SQUAD_DEV_FILE = SQUAD_DIR / "dev-v1.1.json"
BPE_MERGES_FILE = DATA_DIR / "bpe_merges.json"
DATASET_FILE = DATA_DIR / "squad_train_examples.json"

VOCAB_QA_FILE = DATA_DIR / "qa_vocab.json"

NUM_MERGES = 400

MAX_SEQ_LEN = 512
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 256

MIN_FREQ = 2

FILE_ROOTS = [
    Path.home() / "Desktop",
    Path.home() / "OneDrive" / "Desktop",
    Path.home() / "Documents",
    Path.home() / "OneDrive" / "Documents",
    Path.home() / "Downloads",
    Path.home() / "OneDrive" / "Downloads",
    Path.home() / "Pictures",
    Path.home() / "OneDrive" / "Pictures",
    Path.home() / "Music",
    Path.home() / "Videos",
]

PROFILE_FILE = DATA_DIR / "profile.json"