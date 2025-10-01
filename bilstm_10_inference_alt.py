from pathlib import Path
from typing import Dict, List, Tuple
import re

import torch

from bilstm_4_char2word_encoder_alt import CharWordEncoder
from bilstm_5_sentence_encoder_alt import SentenceEncoder
from bilstm_6_head_probes_alt import UPOSFeatsHeadProbes

# --------- char ids helper ---------

def sentence_forms_to_char_ids(forms: List[str], char2id: Dict[str, int], max_word_len: int | None = None
                               ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """
    forms: raw tokens
    returns:
      char_ids:  LongTensor [1, T, L]
      word_mask: BoolTensor [1, T]  (all True for inference on a single sentence)
    """
    pad_id = char2id["<pad>"]
    unk_id = char2id["<unk>"]

    words = []
    for w in forms:
        ids = [char2id.get(ch, unk_id) for ch in w]
        words.append(ids)

    T = len(words)
    L = max((len(w) for w in words), default=0)
    if max_word_len is not None:
        L = min(L, max_word_len)

    padded = [ (w[:L] + [pad_id] * max(0, L - len(w))) for w in words ]
    char_ids = torch.tensor([padded], dtype=torch.long)     # [1, T, L]
    word_mask = torch.ones(1, T, dtype=torch.bool)          # all real tokens
    return char_ids, word_mask


# --------- pretty printer ---------

def pretty_print_predictions(forms: List[str], upos_labels: List[str], feats_labels: List[Dict[str, str]]) -> None:
    """
    feats_labels: per-token dict of {slot: value}, omit <None> entries
    """
    for w, u, f in zip(forms, upos_labels, feats_labels):
        feats_str = "|".join(f"{k}={v}" for k, v in sorted(f.items())) if f else "_"
        print(f"{w:<20} {u:<6} {feats_str}")


# --------- model loader from checkpoint ---------

def load_model_from_checkpoint(ckpt_path: str | Path, device: str = "cpu"):
    """
    Loads: vocabs, hparams, and state_dicts; returns (char_enc, sent_enc, heads, metadata)
    metadata includes: char2id, upos2id, feat2id, id2upos, id2feat, feature_slots, feature_categories, max_word_len
    """
    state = torch.load(Path(ckpt_path), map_location=device)

    # --- frozen vocabs / slots / labels ---
    char2id = state["char2id"]
    upos2id = state["upos2id"]
    feat2id = state["feat2id"]
    feature_categories = state["feature_categories"]     # {slot: ["<PAD>", "<None>", ...]}
    feature_slots = state["feature_slots"]               # fixed global order

    id2upos = {i: u for u, i in upos2id.items()}
    id2feat = {slot: {i: lab for lab, i in lab2id.items()} for slot, lab2id in feat2id.items()}

    # --- hparams (with sane defaults just in case) ---
    hp = state.get("hparams", {})
    D_CHAR   = hp.get("D_CHAR", 64)
    H_CHAR   = hp.get("H_CHAR", 128)
    H_WORD   = hp.get("H_WORD", 256)
    DROPOUT  = hp.get("DROPOUT", 0.2)
    MAX_WORD = hp.get("MAX_WORD_LEN", 40)

    # --- build modules with exact dims, then load weights ---
    pad_char_id = char2id["<pad>"]
    char_enc = CharWordEncoder(vocab_size=len(char2id), pad_id=pad_char_id,
                               d_char=D_CHAR, h_char=H_CHAR, dropout=DROPOUT).to(device)
    sent_enc = SentenceEncoder(d_word=2 * H_CHAR, h_word=H_WORD, dropout=DROPOUT).to(device)
    heads    = UPOSFeatsHeadProbes(d_model=2 * H_WORD,
                                   n_upos=len(upos2id),
                                   feature_slots=feature_slots,
                                   feature_categories=feature_categories,
                                   dropout=DROPOUT).to(device)

    char_enc.load_state_dict(state["char_enc"])
    sent_enc.load_state_dict(state["sent_enc"])
    heads.load_state_dict(state["heads"])

    char_enc.eval(); sent_enc.eval(); heads.eval()

    meta = {
        "char2id": char2id,
        "upos2id": upos2id,
        "feat2id": feat2id,
        "id2upos": id2upos,
        "id2feat": id2feat,
        "feature_slots": feature_slots,
        "feature_categories": feature_categories,
        "max_word_len": MAX_WORD,
    }
    return char_enc, sent_enc, heads, meta


# --------- core predict() ---------

@torch.no_grad()
def predict_sentence(forms: List[str],
                     char_enc: CharWordEncoder,
                     sent_enc: SentenceEncoder,
                     heads: UPOSFeatsHeadProbes,
                     *,
                     id2upos: Dict[int, str],
                     id2feat: Dict[str, Dict[int, str]],
                     char2id: Dict[str, int],
                     max_word_len: int | None = None,
                     device: str = "cpu",
                     drop_none: bool = True) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Returns:
      upos_labels:  list[str] length T
      feats_labels: list[dict] length T  (omit '<None>' if drop_none)
    """
    # Build inputs
    char_ids, word_mask = sentence_forms_to_char_ids(forms, char2id, max_word_len)
    char_ids = char_ids.to(device)
    word_mask = word_mask.to(device)

    # Forward
    X  = char_enc(char_ids)             # [1, T, d_word]
    H  = sent_enc(X, word_mask)         # [1, T, d_model]
    outs = heads(H)                     # dict → [1, T, C]

    # UPOS
    upos_idx = outs["upos"].argmax(dim=-1)[0]      # [T]
    upos_labels = [id2upos[int(i)] for i in upos_idx]

    # Features
    feats_labels: List[Dict[str, str]] = []
    T = len(forms)
    for t in range(T):
        d: Dict[str, str] = {}
        for slot, logits in outs.items():
            if slot == "upos":
                continue
            idx = int(logits[0, t].argmax().item())
            label = id2feat[slot][idx]
            if drop_none and label == "<None>":
                continue
            if label == "<PAD>":
                continue
            d[slot] = label
        feats_labels.append(d)

    return upos_labels, feats_labels


# ---------------- example usage ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("checkpoints/best_model.pt")

    char_enc, sent_enc, heads, meta = load_model_from_checkpoint(ckpt_path, device=device)

    input_sentence = input("\nPlease input a Turkish sentence for the morphology parser (Example sentence: 'Talha nereye gidiyor?'): ")

    forms = re.findall(r"[\w']+|[.,!?;\"\-]", input_sentence)
    
    upos_labels, feats_labels = predict_sentence(
        forms,
        char_enc, sent_enc, heads,
        id2upos=meta["id2upos"],
        id2feat=meta["id2feat"],
        char2id=meta["char2id"],
        max_word_len=meta["max_word_len"],
        device=device,
        drop_none=True,
    )
    pretty_print_predictions(forms, upos_labels, feats_labels)