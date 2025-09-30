import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentenceEncoder(nn.Module):
    """
    Inputs:
      X: FloatTensor [B, T, d_word]        (token vectors)
      word_mask: BoolTensor [B, T]         (True for real tokens, False for pad)
    Returns:
      H_ctx: FloatTensor [B, T, d_model]   (d_model = 2 * h_word)
    """

    def __init__(self, d_word: int, h_word: int = 256, dropout: float = 0.2):
        super().__init__()
        self.word_lstm = nn.LSTM(
            input_size=d_word,
            hidden_size=h_word,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    # --- handy properties ---
    @property
    def input_d_word(self) -> int: return self.word_lstm.input_size
    @property
    def h_word(self) -> int: return self.word_lstm.hidden_size
    @property
    def d_model(self) -> int: return 2 * self.word_lstm.hidden_size
    @property
    def device(self): return next(self.parameters()).device
    @property
    def n_params(self) -> int: return sum(p.numel() for p in self.parameters())
    @property
    def n_trainable_params(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """
        X         : [B, T, d_word]
        word_mask : [B, T]  (bool)
        returns   : [B, T, d_model]
        """
        B, T, _ = X.shape
        lengths = word_mask.sum(dim=1)  # [B], number of real tokens per sentence

        # Pack to ignore padded steps
        packed = pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.word_lstm(packed)

        # Unpack back to [B, T, d_model]
        H_ctx, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        # Dropout on outputs
        H_ctx = self.dropout(H_ctx)
        return H_ctx

### SANITY CHECKS ###
if __name__ == "__main__":
    from bilstm_1_preprocess_alt import read_conllu, TRAIN
    from bilstm_2_vocabularies_alt import (
        build_char_vocab, build_upos_vocab, build_feat_vocab_and_process_features_categories
    )
    from bilstm_3_collate_alt import collate_sentences
    from bilstm_4_char2word_encoder_alt import CharWordEncoder

    sents = read_conllu(TRAIN)
    char2id, _ = build_char_vocab(sents)
    upos2id, _ = build_upos_vocab(sents, add_pad=True)
    _, feat_cats, feat2id, _, feature_slots = build_feat_vocab_and_process_features_categories(sents)

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

    enc = CharWordEncoder(
        vocab_size=len(char2id),
        pad_id=char2id["<pad>"],
        d_char=64,
        h_char=128,
        dropout=0.1,
    )
    word_vecs = enc(char_ids)                       # [B, T, d_word]

    sent_enc = SentenceEncoder(d_word=word_vecs.size(-1), h_word=256, dropout=0.2)
    H_ctx = sent_enc(word_vecs, word_mask)          # [B, T, d_model]
    print("H_ctx:", tuple(H_ctx.shape))