import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VOCAB_QA_FILE, BPE_MERGES_FILE, MODELS_DIR
from core.tokenizer import encode, decode, tokens_to_id
from core.qa_model import forward, predict_span
from core.retrieval import clean_passage

weights = np.load(MODELS_DIR / "qa_weights.npz")
params = {k: weights[k] for k in weights.files}

max_len = params["positional_emb"].shape[0]

vocab = json.load(open(VOCAB_QA_FILE))
id_to_tok = {v: k for k, v in vocab.items()}

merges = json.load(open(BPE_MERGES_FILE))
merges = [(tok, tuple(pair)) for tok, pair in merges]


def ids_to_text(ids):
    toks = [id_to_tok[i] for i in ids]
    return "".join(decode(toks, merges))


def answer(question, passage):
    q_tokens = encode(list(question), merges)
    p_tokens = encode(list(passage), merges)
    combined = q_tokens + ["<SEP>"] + p_tokens

    if len(combined) > max_len:
        print(
            f"  skipped — {len(combined)} tokens exceeds checkpoint limit {max_len}"
        )
        return

    input_ids = tokens_to_id(combined, vocab)
    start_logits, end_logits, _ = forward(input_ids, params)
    sp, ep = predict_span(start_logits, end_logits, offset=len(q_tokens) + 1)

    predicted = ids_to_text(input_ids[sp : ep + 1])
    print(f"  span=({sp},{ep})  answer: {predicted!r}")


question = "who invented the telephone"

passages = [
    "Both Alexander Graham Bell and Elisha Gray, another speech-at-a-distance inventor, filed patents for telephone technology at the US Patent Office on the same day, February 14, 1876. Having filed his application a few hours earlier, Bell was awarded the first US telephone patent (174465) on March 7, 1876. A few days later, Bell made his first telephone call to his assistant in the next room: “Mr. Watson, come here. I want you.” Bell would continue his tinkering with Watson, and their work would  Another inventor in this story was Antonio Meucci. An Italian immigrant to the US, Meucci had begun working on a talking telegraph (or telephone) in 1849. He had first filed his first patent caveat in 1871 but failed to renew it, leaving the field wide open for Bell. Largely forgotten, Meucci’s role in the invention of the telephone was finally acknowledged by a 2002 Congressional resolution.  Alexander Graham Bell is remembered today as the father of telephony; his invention launched the world’",
    """Alexander Graham Bell (/ˈɡreɪ.əm/ ⓘ-Naomi_Persephone_Amethyst_(NaomiAmethyst)-Graham.wav "File:LL-Q1860 (eng)-Naomi Persephone Amethyst (NaomiAmethyst)-Graham.wav"); born Alexander Bell; March 3, 1847 – August 2, 1922) was a Scottish-born Canadian-American inventor, scientist, and engineer who is credited with patenting the first practical telephone. He also co-founded the American Telephone and Telegraph Company (AT&T) in 1885.  Inventor of the telephone (1847–1922)""",
    "Alexander Graham Bell was an inventor, scientist, and teacher who is best remembered for inventing the telephone. He was born on March 3, 1847, in Scotland, and he and his family moved to England in 1865 and Canada in 1870. A year later Bell moved to the United States, where he taught speech to deaf students and where he also invented and improved a number of electrical technologies. He became a U.S. citizen in 1882 while remaining a British subject; he later moved to Canada and lived there  Although Alexander Graham Bell is best remembered as the inventor of the telephone, he invented other devices too. Bell developed several sonic technologies, including the photophone (1880) and the Graphophone (1886). He also developed medical technology. After the shooting of U.S. Pres. James A. Garfield in July 1881, Bell teamed up with professor Simon Newcomb of the U.S. Nautical Almanac Office to develop an electrical bullet probe. The pair demonstrated the probe in the autumn of 1881. Bell",
]

for i, p in enumerate(passages):
    print(f"--- passage {i} ---")
    answer(question, clean_passage(p))
