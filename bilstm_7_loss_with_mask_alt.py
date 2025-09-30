import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_token_cross_entropy(logits: torch.Tensor,
                               gold: torch.Tensor,
                               mask: torch.Tensor,
                               ignore_index: int | None = None) -> torch.Tensor:
    """
    Cross-entropy over only real tokens (mask=True). Optionally ignore a label id.
    logits: [B, T, C]
    gold  : [B, T] (Long)
    mask  : [B, T] (Bool)
    """
    assert logits.dim() == 3 and gold.dim() == 2 and mask.dim() == 2
    B, T, C = logits.shape
    assert gold.shape == (B, T) and mask.shape == (B, T)
    assert mask.dtype == torch.bool, "word_mask must be bool"

    # Flatten
    logits_f = logits.reshape(B*T, C)
    gold_f   = gold.reshape(B*T)
    mask_f   = mask.reshape(B*T)

    # Optional ignore_index (rarely needed if your mask is correct,
    # but safe to have with Design B)
    if ignore_index is not None:
        keep = mask_f & (gold_f != ignore_index)
    else:
        keep = mask_f

    if keep.any():
        return F.cross_entropy(logits_f[keep], gold_f[keep])
    else:
        # No real tokens in this batch (shouldn’t happen, but be robust)
        return logits.sum() * 0.0


def compute_total_loss(head_logits: dict[str, torch.Tensor],
                       gold_upos: torch.Tensor,
                       gold_feats: dict[str, torch.Tensor],
                       word_mask: torch.Tensor,
                       *,
                       pad_upos_id: int | None = None,
                       pad_feat_ids: dict[str, int] | None = None,
                       feat_weights: dict[str, float] | None = None) -> torch.Tensor:
    """
    Sums CE for UPOS + each FEAT head over real tokens, then normalizes by total weight.
    head_logits: {"upos": [B,T,U], slot: [B,T,K_slot], ...}
    gold_upos  : [B,T]
    gold_feats : {slot: [B,T]}
    word_mask  : [B,T] (bool)
    """
    assert word_mask.dtype == torch.bool

    # --- UPOS ---
    upos_loss = masked_token_cross_entropy(
        head_logits["upos"], gold_upos, word_mask,
        ignore_index=pad_upos_id  # with Design B, this = upos2id["<PAD>"]
    )

    # --- Features ---
    if feat_weights is None:
        feat_weights = {slot: 1.0 for slot in gold_feats.keys()}

    total = upos_loss
    weight_sum = 1.0

    for slot, gold_tensor in gold_feats.items():
        w = float(feat_weights.get(slot, 1.0))
        ignore_id = pad_feat_ids.get(slot) if pad_feat_ids is not None else None

        feat_loss = masked_token_cross_entropy(
            head_logits[slot], gold_tensor, word_mask,
            ignore_index=ignore_id  # = feat2id[slot]["<PAD>"]
        )
        total += w * feat_loss
        weight_sum += w

    # Normalizing so adding more heads doesn’t change overall loss scale
    return total / weight_sum