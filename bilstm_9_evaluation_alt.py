from pathlib import Path
from typing import Dict, List, Tuple

import torch

from bilstm_1_preprocess_alt import read_conllu, DEV, TEST
from bilstm_3_collate_alt import collate_sentences
from bilstm_4_char2word_encoder_alt import CharWordEncoder
from bilstm_5_sentence_encoder_alt import SentenceEncoder
from bilstm_6_head_probes_alt import UPOSFeatsHeadProbes
from bilstm_7_loss_with_mask_alt import compute_total_loss

# ------------ helpers (counts, not just ratios) ---------------

@torch.no_grad()
def acc_counts_upos(logits: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    """(#correct, #total) for UPOS under mask."""
    preds = logits.argmax(dim=-1)
    m = mask.bool()
    correct = (preds[m] == gold[m]).sum().item()
    total   = m.sum().item()
    return correct, total

@torch.no_grad()
def acc_counts_feat_twoways(logits: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor, none_index: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Returns:
      with_none:   (#correct, #total) over all real tokens
      present_only:(#correct, #total) where gold != <None>
    """
    preds = logits.argmax(dim=-1)
    m = mask.bool()

    # with <None>
    corr_all = (preds[m] == gold[m]).sum().item()
    tot_all  = m.sum().item()

    # exclude <None>
    keep = m & (gold != none_index)
    corr_pres = (preds[keep] == gold[keep]).sum().item()
    tot_pres  = keep.sum().item()

    return (corr_all, tot_all), (corr_pres, tot_pres)

@torch.no_grad()
def bundle_accuracy_counts(logits_dict: Dict[str, torch.Tensor],
                           gold_upos: torch.Tensor,
                           gold_feats: Dict[str, torch.Tensor],
                           mask: torch.Tensor) -> Tuple[int, int]:
    """Bundle acc = UPOS AND all feature slots simultaneously correct (on real tokens)."""
    preds_upos = logits_dict["upos"].argmax(dim=-1)
    eq = (preds_upos == gold_upos)
    for slot, logits in logits_dict.items():
        if slot == "upos":
            continue
        preds = logits.argmax(dim=-1)
        eq = eq & (preds == gold_feats[slot])
    m = mask.bool()
    correct = eq[m].sum().item()
    total   = m.sum().item()
    return correct, total

# ------------------------ main evaluation -----------------------------

@torch.no_grad()
def evaluate_dataset(char_enc: CharWordEncoder,
                     sent_enc: SentenceEncoder,
                     heads: UPOSFeatsHeadProbes,
                     sentences: List,
                     *,
                     char2id: Dict[str,int],
                     upos2id: Dict[str,int],
                     feat2id: Dict[str,Dict[str,int]],
                     feature_slots: List[str],
                     batch_size: int = 32,
                     device: str = "cpu",
                     max_word_len: int | None = None) -> Dict:
    """
    Computes:
      - average loss (mask-weighted)
      - UPOS accuracy
      - per-feature accuracy (with <None> / present-only)
      - bundle accuracy (UPOS + all FEATS correct simultaneously)
    """
    char_enc.eval(); sent_enc.eval(); heads.eval()

    # PAD / NONE ids
    pad_upos_id = upos2id["<PAD>"]
    pad_feat_ids = {slot: feat2id[slot]["<PAD>"] for slot in feature_slots}
    none_indices = {slot: feat2id[slot]["<None>"] for slot in feature_slots}

    total_loss_num = 0.0
    total_loss_den = 0
    upos_corr = 0; upos_tot = 0
    bundle_corr = 0; bundle_tot = 0

    feat_corr_with = {slot: 0 for slot in feature_slots}
    feat_tot_with  = {slot: 0 for slot in feature_slots}
    feat_corr_pres = {slot: 0 for slot in feature_slots}
    feat_tot_pres  = {slot: 0 for slot in feature_slots}

    # manual batching
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i+batch_size]

        pad_char_id = char2id['<pad>']
        # keyword-only call; include feature_slots
        char_ids, word_mask, gold_upos, gold_feats = collate_sentences(
            batch_sents,
            char2id=char2id,
            upos2id=upos2id,
            feat2id=feat2id,
            feature_slots=feature_slots,
            pad_char_id=pad_char_id,
            max_word_len=max_word_len,
        )
        char_ids  = char_ids.to(device)
        word_mask = word_mask.to(device)
        gold_upos = gold_upos.to(device)
        gold_feats = {k: v.to(device) for k, v in gold_feats.items()}

        # forward
        X  = char_enc(char_ids)                 # [B,T,d_word]
        H  = sent_enc(X, word_mask)             # [B,T,d_model]
        logits = heads(H)                       # dict of [B,T,C]

        # loss (mask-weighted)
        loss = compute_total_loss(
            logits, gold_upos, gold_feats, word_mask,
            pad_upos_id=pad_upos_id, pad_feat_ids=pad_feat_ids
        )
        n_real = int(word_mask.sum().item())
        total_loss_num += loss.item() * n_real
        total_loss_den += n_real

        # UPOS acc
        corr, tot = acc_counts_upos(logits["upos"], gold_upos, word_mask)
        upos_corr += corr; upos_tot += tot

        # Feature accs
        for slot in feature_slots:
            (c_all, t_all), (c_pres, t_pres) = acc_counts_feat_twoways(
                logits[slot], gold_feats[slot], word_mask, none_index=none_indices[slot]
            )
            feat_corr_with[slot] += c_all;  feat_tot_with[slot]  += t_all
            feat_corr_pres[slot] += c_pres; feat_tot_pres[slot]  += t_pres

        # Bundle acc
        bc, bt = bundle_accuracy_counts(logits, gold_upos, gold_feats, word_mask)
        bundle_corr += bc; bundle_tot += bt

    # --- weighted average over all feature heads ---
    total_corr_with = sum(feat_corr_with[cat] for cat in feat2id)
    total_tokens_with = sum(feat_tot_with[cat] for cat in feat2id)
    weighted_avg_with_none = total_corr_with / max(total_tokens_with, 1)

    total_corr_pres = sum(feat_corr_pres[cat] for cat in feat2id)
    total_tokens_pres = sum(feat_tot_pres[cat] for cat in feat2id)
    weighted_avg_present_only = total_corr_pres / max(total_tokens_pres, 1)

    results = {
        "loss": total_loss_num / max(total_loss_den, 1),
        "upos_acc": upos_corr / max(upos_tot, 1),
        "feat_acc": {
            cat: {
                "with_none": feat_corr_with[cat] / max(feat_tot_with[cat], 1),
                "present_only": feat_corr_pres[cat] / max(feat_tot_pres[cat], 1),
            }
            for cat in feat2id
        },
        "bundle_acc": bundle_corr / max(bundle_tot, 1),
        "feat_weighted_avg": {               # ← add this new entry
            "with_none": weighted_avg_with_none,
            "present_only": weighted_avg_present_only,
        },
    }

    return results

# ---------------- example usage (dev/test) -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint (contains frozen vocabs + heads)
    ckpt_path = Path("checkpoints/best_model.pt")
    state = torch.load(ckpt_path, map_location=device)

    # --- vocabs & slots (frozen) ---
    char2id = state["char2id"]
    upos2id = state["upos2id"]
    feat2id = state["feat2id"]
    feature_categories = state["feature_categories"]
    feature_slots = state["feature_slots"]

    # --- hparams ---
    hp = state.get("hparams", {})
    D_CHAR   = hp.get("D_CHAR", 64)
    H_CHAR   = hp.get("H_CHAR", 128)
    H_WORD   = hp.get("H_WORD", 256)
    DROPOUT  = hp.get("DROPOUT", 0.2)
    MAX_WORD = hp.get("MAX_WORD_LEN", 40)

    # --- rebuild model with exact dims, then load weights ---
    pad_char_id = char2id["<pad>"]
    char_enc = CharWordEncoder(len(char2id), pad_id=pad_char_id, d_char=D_CHAR, h_char=H_CHAR, dropout=DROPOUT).to(device)
    sent_enc = SentenceEncoder(d_word=2*H_CHAR, h_word=H_WORD, dropout=DROPOUT).to(device)
    heads    = UPOSFeatsHeadProbes(d_model=2*H_WORD,
                                   n_upos=len(upos2id),
                                   feature_slots=feature_slots,
                                   feature_categories=feature_categories,
                                   dropout=DROPOUT).to(device)

    char_enc.load_state_dict(state["char_enc"])
    sent_enc.load_state_dict(state["sent_enc"])
    heads.load_state_dict(state["heads"])

    char_enc.eval(); sent_enc.eval(); heads.eval()

    # --- load datasets ---
    dev_sentences  = read_conllu(DEV)
    test_sentences = read_conllu(TEST)

    print("DEV results:")
    results = evaluate_dataset(char_enc, sent_enc, heads, dev_sentences,
                         char2id=char2id, upos2id=upos2id, feat2id=feat2id, feature_slots=feature_slots,
                         device=device, max_word_len=MAX_WORD)
    print(results)

    print("\nTEST results:")
    results = evaluate_dataset(char_enc, sent_enc, heads, test_sentences,
                         char2id=char2id, upos2id=upos2id, feat2id=feat2id, feature_slots=feature_slots,
                         device=device, max_word_len=MAX_WORD)
    print(results)

    print("\nWeighted average (with None):",
      results["feat_weighted_avg"]["with_none"])
    print("Weighted average (present only):",
      results["feat_weighted_avg"]["present_only"])