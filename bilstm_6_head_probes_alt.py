import torch
import torch.nn as nn

class UPOSFeatsHeadProbes(nn.Module):
    """
    Takes H_ctx [B, T, d_model] and outputs:
      - 'upos': [B, T, n_upos]
      - one head per feature slot in feature_slots: [B, T, n_vals(slot)]
    """

    def __init__(
        self,
        d_model: int,
        n_upos: int,
        feature_slots: list[str],
        feature_categories: dict[str, list[str]],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # UPOS head
        self.upos_head = nn.Linear(d_model, n_upos)

        # Feature heads in the fixed slot order
        self.feature_slots = list(feature_slots)  # preserve order
        self._feat_labels = {slot: list(feature_categories[slot]) for slot in self.feature_slots}
        self.feat_heads = nn.ModuleDict(
            {slot: nn.Linear(d_model, len(self._feat_labels[slot])) for slot in self.feature_slots}
        )

    # ---- handy inspectors / metadata ----
    @property
    def d_model(self) -> int:
        return self.upos_head.in_features

    @property
    def n_upos(self) -> int:
        return self.upos_head.out_features

    @property
    def categories(self) -> list[str]:
        return list(self.feature_slots)

    @property
    def category_sizes(self) -> dict[str, int]:
        return {slot: self.feat_heads[slot].out_features for slot in self.feature_slots}

    @property
    def pad_indices(self) -> dict[str, int]:
        # Assumes "<PAD>" exists in each slot's labels
        return {slot: self._feat_labels[slot].index("<PAD>") for slot in self.feature_slots}

    @property
    def none_indices(self) -> dict[str, int | None]:
        return {
            slot: (self._feat_labels[slot].index("<None>") if "<None>" in self._feat_labels[slot] else None)
            for slot in self.feature_slots
        }

    @property
    def label_vocab(self) -> dict[str, list[str]]:
        return {slot: list(self._feat_labels[slot]) for slot in self.feature_slots}

    def expected_output_shapes(self, B: int, T: int) -> dict[str, tuple[int, int, int]]:
        shapes = {"upos": (B, T, self.n_upos)}
        for slot in self.feature_slots:
            shapes[slot] = (B, T, self.feat_heads[slot].out_features)
        return shapes

    def forward(self, H_ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        H_ctx: [B, T, d_model]
        returns: dict with logits per task, each [B, T, num_classes]
        """
        X = self.dropout(H_ctx)
        out = {"upos": self.upos_head(X)}
        for slot in self.feature_slots:
            out[slot] = self.feat_heads[slot](X)
        return out


# ---- quick sanity ----
if __name__ == "__main__":
    from bilstm_1_preprocess_alt import read_conllu, TRAIN
    from bilstm_2_vocabularies_alt import (
        build_char_vocab,
        build_upos_vocab,
        build_feat_vocab_and_process_features_categories,
    )
    from bilstm_3_collate_alt import collate_sentences
    from bilstm_4_char2word_encoder_alt import CharWordEncoder
    from bilstm_5_sentence_encoder_alt import SentenceEncoder

    sents = read_conllu(TRAIN)
    char2id, _ = build_char_vocab(sents)
    upos2id, _ = build_upos_vocab(sents, add_pad=True)
    _, feature_categories, feat2id, _, feature_slots = build_feat_vocab_and_process_features_categories(sents)

    batch = sents[:3]
    char_ids, word_mask, gold_upos, gold_feats = collate_sentences(
        batch,
        char2id=char2id,
        upos2id=upos2id,
        feat2id=feat2id,
        feature_slots=feature_slots,
        pad_char_id=char2id["<pad>"],
        max_word_len=40,
    )

    enc = CharWordEncoder(vocab_size=len(char2id), pad_id=char2id["<pad>"], d_char=64, h_char=128, dropout=0.1)
    word_vecs = enc(char_ids)

    sent_enc = SentenceEncoder(d_word=word_vecs.size(-1), h_word=256, dropout=0.1)
    H_ctx = sent_enc(word_vecs, word_mask)  # << pass word_mask

    heads = UPOSFeatsHeadProbes(
        d_model=H_ctx.size(-1),
        n_upos=len(upos2id),                    # includes <PAD>
        feature_slots=feature_slots,            # fixed order
        feature_categories=feature_categories,  # labels per slot (with <PAD>, <None>)
        dropout=0.1,
    )
    logits = heads(H_ctx)

    print("H_ctx:", tuple(H_ctx.shape))
    for k, v in logits.items():
        print(k, tuple(v.shape))