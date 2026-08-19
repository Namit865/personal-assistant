import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BPE_MERGES_FILE
from core.tokenizer import encode, build_token_ranges, find_token_for_char

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]

context = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary."
tokens = encode(list(context), merges)
ranges = build_token_ranges(context, merges)
result = find_token_for_char(ranges, 3)
print(result)
