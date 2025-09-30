from pathlib import Path
import math
import random
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# ---- Your own stages (only things you've defined before) ----
from bilstm_1_preprocess_alt import read_conllu, TRAIN, DEV
from bilstm_2_vocabularies_alt import (
    build_char_vocab,
    build_upos_vocab,
    build_feat_vocab_and_process_features_categories,
    save_vocab,
    load_vocab,
)
from bilstm_3_collate_alt import collate_sentences
from bilstm_4_char2word_encoder_alt import CharWordEncoder
from bilstm_5_sentence_encoder_alt import SentenceEncoder
from bilstm_6_head_probes_alt import UPOSFeatsHeadProbes
from bilstm_7_loss_with_mask_alt import compute_total_loss

# ----------------------- hyperparams -----------------------
SEED            = 13
BATCH_SIZE      = 32
D_CHAR          = 64
H_CHAR          = 128      # char BiLSTM hidden per direction  → d_word = 2*H_CHAR
H_WORD          = 256      # sent BiLSTM hidden per direction  → d_model = 2*H_WORD
DROPOUT         = 0.2
LR              = 1e-3     # For Adam, they suggest this learning rate
EPOCHS          = 10
MAX_WORD_LEN    = 40
GRAD_CLIP       = 5.0
WEIGHT_DECAY    = 0.0

VOCAB_DIR       = Path("artifacts/vocabs")
CKPT_DIR        = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- utils -----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def batches(data: List, batch_size=BATCH_SIZE, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [data[j] for j in idx[i:i + batch_size]]

@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, gold: torch.Tensor, word_mask: torch.Tensor) -> float:
    """
    logits: [B,T,C]  gold: [B,T]  word_mask: [B,T] (bool)
    """
    preds = logits.argmax(dim=-1)
    mask  = word_mask.bool()
    correct = (preds[mask] == gold[mask]).sum().item()
    total   = mask.sum().item()
    return correct / max(total, 1)

@torch.no_grad()
def feature_accuracy_with_and_without_none(
    logits: torch.Tensor,
    gold: torch.Tensor,
    word_mask: torch.Tensor,
    none_index: int
):
    """
    Returns (acc_with_none, acc_present_only)
    """
    preds = logits.argmax(dim=-1)
    mask  = word_mask.bool()

    # With <None>
    correct_all = (preds[mask] == gold[mask]).sum().item()
    total_all   = mask.sum().item()
    acc_with_none = correct_all / max(total_all, 1)

    # Excluding <None>
    keep = mask & (gold != none_index)
    correct_present = (preds[keep] == gold[keep]).sum().item()
    total_present   = keep.sum().item()
    acc_present_only = correct_present / max(total_present, 1)

    return acc_with_none, acc_present_only

# -------------------- vocab load/build ---------------------
def load_or_build_vocabs():
    """
    Try to load frozen vocabs from artifacts; if missing, build from TRAIN and save.
    Returns:
      char2id, upos2id, feat2id, feature_categories, feature_slots
    """
    files = {
        "char2id": VOCAB_DIR / "char2id.json",
        "upos2id": VOCAB_DIR / "upos2id.json",
        "feat2id": VOCAB_DIR / "feat2id.json",
        "feature_categories": VOCAB_DIR / "feature_categories.json",
        "feature_slots": VOCAB_DIR / "feature_slots.json",
    }
    if all(p.exists() for p in files.values()):
        char2id = load_vocab(files["char2id"])
        upos2id = load_vocab(files["upos2id"])
        feat2id = load_vocab(files["feat2id"])
        feature_categories = load_vocab(files["feature_categories"])
        feature_slots = load_vocab(files["feature_slots"])
        return char2id, upos2id, feat2id, feature_categories, feature_slots

    # Build from TRAIN, then save
    train_sents = read_conllu(TRAIN)
    char2id, _ = build_char_vocab(train_sents)
    upos2id, _ = build_upos_vocab(train_sents, add_pad=True)  # Design B
    _, feature_categories, feat2id, _, feature_slots = build_feat_vocab_and_process_features_categories(train_sents)

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    save_vocab(char2id, files["char2id"])
    save_vocab(upos2id, files["upos2id"])
    save_vocab(feat2id, files["feat2id"])
    save_vocab(feature_categories, files["feature_categories"])
    save_vocab(feature_slots, files["feature_slots"])

    return char2id, upos2id, feat2id, feature_categories, feature_slots

# ----------------------- epoch runner ----------------------
def run_epoch(models, optimizer, sentences, *,
              char2id, upos2id, feat2id, feature_slots,
              train_mode=True, device="cpu"):
    """
    One pass over `sentences`. If train_mode=True: updates weights; else eval only.
    Returns dict with running loss and simple accuracies.
    """
    char_enc, sent_enc, heads = models
    if train_mode:
        char_enc.train(); sent_enc.train(); heads.train()
    else:
        char_enc.eval();  sent_enc.eval();  heads.eval()

    total_loss = 0.0
    total_tokens = 0
    total_upos_correct = 0
    total_upos_count = 0

    feat_stats: Dict[str, Dict[str, List[float]]] = {cat: {"with": [0,0], "present": [0,0]} for cat in feature_slots}

    # PAD ids for loss ignore (Design B)
    pad_upos_id = upos2id["<PAD>"]
    pad_feat_ids = {slot: feat2id[slot]["<PAD>"] for slot in feature_slots}
    # NONE indices (usually 1)
    none_indices = {slot: (0 if "<None>" not in feat2id[slot] else feat2id[slot]["<None>"]) for slot in feature_slots}

    for batch_sents in batches(sentences, shuffle=train_mode):
        # 1) collate → tensors on device  (keyword-only args!)
        pad_char_id = char2id['<pad>']
        char_ids, word_mask, gold_upos, gold_feats = collate_sentences(
            batch_sents,
            char2id=char2id,
            upos2id=upos2id,
            feat2id=feat2id,
            feature_slots=feature_slots,
            pad_char_id=pad_char_id,
            max_word_len=MAX_WORD_LEN,
        )
        char_ids  = char_ids.to(device)
        word_mask = word_mask.to(device)
        gold_upos = gold_upos.to(device)
        gold_feats = {k: v.to(device) for k, v in gold_feats.items()}

        # 2) forward
        X  = char_enc(char_ids)                 # [B,T,d_word]
        H  = sent_enc(X, word_mask)             # [B,T,d_model]
        logits = heads(H)                       # {'upos': [B,T,U], slot: [B,T,K]...}

        # 3) loss (ignore PAD labels)
        loss = compute_total_loss(
            logits, gold_upos, gold_feats, word_mask,
            pad_upos_id=pad_upos_id,
            pad_feat_ids=pad_feat_ids,
            feat_weights=None,
        )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(list(char_enc.parameters()) +
                            list(sent_enc.parameters()) +
                            list(heads.parameters()),
                            max_norm=GRAD_CLIP)
            optimizer.step()

        # 4) logging (weighted by real tokens)
        with torch.no_grad():
            n_real = int(word_mask.sum().item())
            total_loss += float(loss.item()) * n_real
            total_tokens += n_real

            upos_acc = masked_accuracy(logits["upos"], gold_upos, word_mask)
            total_upos_correct += int(upos_acc * n_real)
            total_upos_count   += n_real

            for slot, gold in gold_feats.items():
                acc_with, acc_present = feature_accuracy_with_and_without_none(
                    logits[slot], gold, word_mask, none_index=none_indices[slot]
                )
                # with-None weighted by real tokens
                feat_stats[slot]["with"][0] += acc_with * n_real
                feat_stats[slot]["with"][1] += n_real
                # present-only weighted by count of present labels
                present_mask = (word_mask & (gold != none_indices[slot]))
                n_present = int(present_mask.sum().item())
                feat_stats[slot]["present"][0] += acc_present * max(n_present, 1)
                feat_stats[slot]["present"][1] += max(n_present, 1)

    avg_loss = total_loss / max(total_tokens, 1)
    upos_acc_epoch = total_upos_correct / max(total_upos_count, 1)

    feat_acc_epoch = {}
    for slot, d in feat_stats.items():
        with_w, with_tot = d["with"]
        pres_w, pres_tot = d["present"]
        feat_acc_epoch[slot] = {
            "with_none": with_w / max(with_tot, 1),
            "present_only": pres_w / max(pres_tot, 1),
        }

    return {"loss": avg_loss, "upos_acc": upos_acc_epoch, "feat_acc": feat_acc_epoch}

# ----------------------- main train ------------------------
def main(device=None):
    set_seed(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 0) Load or build frozen vocabs (no train_* globals)
    char2id, upos2id, feat2id, feature_categories, feature_slots = load_or_build_vocabs()
    pad_char_id = char2id['<pad>']

    # 1) Load datasets
    train_sentences = read_conllu(TRAIN)
    dev_sentences   = read_conllu(DEV)

    # 2) Build models with exact dims from vocabs
    char_enc = CharWordEncoder(
        vocab_size=len(char2id), pad_id=pad_char_id,
        d_char=D_CHAR, h_char=H_CHAR, dropout=DROPOUT
    ).to(device)

    sent_enc = SentenceEncoder(
        d_word=2 * H_CHAR, h_word=H_WORD, dropout=DROPOUT
    ).to(device)

    heads = UPOSFeatsHeadProbes(
        d_model=2 * H_WORD,
        n_upos=len(upos2id),                  # includes <PAD>
        feature_slots=feature_slots,          # fixed order
        feature_categories=feature_categories,
        dropout=DROPOUT,
    ).to(device)

    # 3) Optimizer
    params = list(char_enc.parameters()) + list(sent_enc.parameters()) + list(heads.parameters())
    optimizer = Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

    best_dev = -math.inf
    best_path = CKPT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(
            (char_enc, sent_enc, heads), optimizer, train_sentences,
            char2id=char2id, upos2id=upos2id, feat2id=feat2id, feature_slots=feature_slots,
            train_mode=True, device=device
        )
        with torch.no_grad():
            dev_stats = run_epoch(
                (char_enc, sent_enc, heads), optimizer=None, sentences=dev_sentences,
                char2id=char2id, upos2id=upos2id, feat2id=feat2id, feature_slots=feature_slots,
                train_mode=False, device=device
            )

        # ---- Logs ----
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  train: loss={train_stats['loss']:.4f}")
        # UPOS displayed as a single number (mask-weighted)
        print(f"  train: upos_acc={train_stats['upos_acc']:.3f}")
        print(f"    feats (withNone | presentOnly):")
        for slot, accs in train_stats['feat_acc'].items():
            print(f"      {slot:>10}: {accs['with_none']:.3f} | {accs['present_only']:.3f}")

        print(f"  dev:   loss={dev_stats['loss']:.4f}")
        print(f"  dev:   upos_acc={dev_stats['upos_acc']:.3f}")
        print(f"    feats (withNone | presentOnly):")
        for slot, accs in dev_stats['feat_acc'].items():
            print(f"      {slot:>10}: {accs['with_none']:.3f} | {accs['present_only']:.3f}")

        # ---- Model selection (dev UPOS acc) ----
        score = dev_stats['upos_acc']
        if score > best_dev:
            best_dev = score
            torch.save({
                "char_enc": char_enc.state_dict(),
                "sent_enc": sent_enc.state_dict(),
                "heads": heads.state_dict(),
                # stash exact vocabs/slots for later eval/inference
                "char2id": char2id,
                "upos2id": upos2id,
                "feat2id": feat2id,
                "feature_categories": feature_categories,
                "feature_slots": feature_slots,
                "hparams": {
                    "D_CHAR": D_CHAR, "H_CHAR": H_CHAR, "H_WORD": H_WORD,
                    "DROPOUT": DROPOUT, "MAX_WORD_LEN": MAX_WORD_LEN,
                },
            }, best_path)
            print(f" -> Saved new best to {best_path} (dev upos_acc={best_dev:.3f})")

    print("\nTraining finished.")
    print(f"Best dev UPOS acc: {best_dev:.3f}  (ckpt: {best_path})")

if __name__ == "__main__":
    main()