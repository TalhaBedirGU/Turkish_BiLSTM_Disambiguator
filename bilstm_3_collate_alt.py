import torch

def sentences_to_feat_ids(batch_sentences, feat2id, feature_slots, pad_tok="<PAD>", none_tok="<None>"):
    """
    Returns dict[slot] -> LongTensor [B, T] with PAD index at padded word positions,
    and <None> for real words where the slot is absent.
    """
    if not batch_sentences:
        return {slot: torch.zeros(0, 0, dtype=torch.long) for slot in feature_slots}

    B = len(batch_sentences)
    T_max = max(len(s) for s in batch_sentences)

    # init grids with PAD for each slot
    feats_grids = {
        slot: [[feat2id[slot][pad_tok]] * T_max for _ in range(B)]
        for slot in feature_slots
    }

    for b, sent in enumerate(batch_sentences):
        for t, tok in enumerate(sent):
            feats = tok.get("feats") or {}
            for slot in feature_slots:
                f2i = feat2id[slot]
                if slot in feats and feats[slot] is not None:
                    val = feats[slot]
                    if isinstance(val, list) and len(val) > 0:
                        val = val[0]             # safe: your corpus is single-valued
                    idx = f2i.get(str(val), f2i[none_tok])
                else:
                    idx = f2i[none_tok]          # slot not applicable on this token
                feats_grids[slot][b][t] = idx

    return {slot: torch.tensor(grid, dtype=torch.long) for slot, grid in feats_grids.items()}

def sentences_to_upos_ids(batch_sentences, upos2id, pad_tag="<PAD>"):
    pad_id = upos2id[pad_tag]
    T_max = max(len(s) for s in batch_sentences) if batch_sentences else 0
    Y = []
    for sent in batch_sentences:
        row = [upos2id[tok['upos']] for tok in sent]
        row += [pad_id] * (T_max - len(row))
        Y.append(row)
    return torch.tensor(Y, dtype=torch.long)       # [B, T_max]

def collate_sentences(
    batch_sentences,
    *,
    char2id,
    upos2id,
    feat2id,
    feature_slots,
    pad_char_id=0,
    max_word_len=40,
):
    """
    Returns:
      char_ids:  LongTensor [B, T_max, L_max]
      word_mask: BoolTensor [B, T_max]
      gold_upos: LongTensor [B, T_max]  (padded with IGNORE_INDEX)
      gold_feats: dict[slot] -> LongTensor [B, T_max] (PAD at padded tokens)
    """
    # ----- chars per word -----
    batch_char_lists = []
    for sent in batch_sentences:
        word_char_ids = []
        for tok in sent:
            ids = [char2id.get(ch, char2id['<unk>']) for ch in tok['form']]
            # truncate/pad to max_word_len (we’ll still find batch L_max below)
            if len(ids) > max_word_len:
                ids = ids[:max_word_len]
            word_char_ids.append(ids)
        batch_char_lists.append(word_char_ids)

    B = len(batch_char_lists)
    if B == 0:
        return (torch.zeros(0, 0, 0, dtype=torch.long),
                torch.zeros(0, 0, dtype=torch.bool),
                torch.zeros(0, 0, dtype=torch.long),
                {slot: torch.zeros(0, 0, dtype=torch.long) for slot in feature_slots})

    T_max = max(len(s) for s in batch_char_lists)
    L_max = min(
        max((len(w) for s in batch_char_lists for w in s), default=0),
        max_word_len
    )

    char_ids = []
    word_mask = []
    for sent in batch_char_lists:
        padded_sent = []
        mask_row = []
        # real words
        for w in sent:
            w2 = w[:L_max] + [pad_char_id] * max(0, L_max - len(w))
            padded_sent.append(w2)
            mask_row.append(True)
        # pad words
        for _ in range(T_max - len(sent)):
            padded_sent.append([pad_char_id] * L_max)
            mask_row.append(False)
        char_ids.append(padded_sent)
        word_mask.append(mask_row)

    char_ids  = torch.tensor(char_ids, dtype=torch.long)     # [B, T, L]
    word_mask = torch.tensor(word_mask, dtype=torch.bool)     # [B, T]

    # gold labels
    gold_upos  = sentences_to_upos_ids(batch_sentences, upos2id)
    gold_feats = sentences_to_feat_ids(batch_sentences, feat2id, feature_slots)

    return char_ids, word_mask, gold_upos, gold_feats