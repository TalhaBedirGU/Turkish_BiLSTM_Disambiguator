from __future__ import annotations
from conllu import parse_incr
from pathlib import Path
from typing import Dict, List, TypedDict, Iterable, Optional
import unicodedata as _ud

class Token(TypedDict): # 
    form: str
    lemma: Optional[str]            # Optional = If missing, make it None
    upos: Optional[str]             # Optional = If missing, make it None
    feats: Dict[str, str]           # UD features
    feats_str: str                  # canonical, sorted "A=B|C=D" or "_"

Sentence = List[Token]

def _canon_feats_str(feats: Optional[Dict[str, str]]) -> str: # This basically ensures Dict[str, str] or None. Feats can be missing. If so. then "none"
    """Canonical UD Feats string with sorted keys; '_' if empty."""
    if not feats:
        return "_"
    items = sorted((k, v) for k, v in feats.items())
    return "|".join(f"{k}={v}" for k, v in items)

def _nfc(s: Optional[str]) -> Optional[str]: # To prevent some weird bugs we encountered during parsing
    return None if s is None else _ud.normalize("NFC", s)

def read_conllu(filepath: Path, normalize_nfc: bool = True) -> List[Sentence]:
    """
    Preprocess UD-annotated sentences from a .conllu file → list of sentences.
    Each sentence is a list of tokens (forms/lemma/upos/feats + canonical feats_str).
    Skips multi-word tokens and empty nodes.
    """
    sentences: List[Sentence] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            sent: Sentence = []
            for tok in tokenlist:
                if isinstance(tok["id"], int):  # skip MWTs and empty nodes
                    form = tok["form"]
                    lemma = tok.get("lemma")
                    upos = tok.get("upostag")
                    feats = tok.get("feats") or {}
                    if normalize_nfc:
                        form = _nfc(form)  # type: ignore
                        lemma = _nfc(lemma)  # type: ignore
                    sent.append({
                        "form": form,
                        "lemma": lemma,
                        "upos": upos,
                        "feats": feats,
                        "feats_str": _canon_feats_str(feats),
                    })
            sentences.append(sent)
    return sentences

DATA_DIR = Path("ud_turkish_boun")
TRAIN = DATA_DIR / "tr_boun-ud-train.conllu"
TEST  = DATA_DIR / "tr_boun-ud-test.conllu"
DEV   = DATA_DIR / "tr_boun-ud-dev.conllu"

train_sentences = read_conllu(TRAIN)
test_sentences = read_conllu(TEST)
dev_sentences = read_conllu(DEV)

### SANITY CHECKS ###

if __name__ == "__main__":
    # Basic existence check
    for p in (TRAIN, DEV, TEST):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    print(f"An example entry (sentence 0, first 5 tokens):\n{train_sentences[0][:5]}")
    all_words = [tok["form"] for sent in train_sentences for tok in sent]

    print("\nSome data about the input\n")
    print(f"The total count of word tokens: {len(all_words)}")
    longest_word = max(all_words, key=len)
    print(f"The longest word: {longest_word!r} (len={len(longest_word)})")

    # Quick sanity: show feats canonicalization on first sentence
    print("\nFirst sentence forms + UPOS + feats_str:")
    for tok in train_sentences[0]:
        print(f"{tok['form']}\t{tok['upos']}\t{tok['feats_str']}")