import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharWordEncoder(nn.Module):
    """
    Inputs:
      char_ids: LongTensor [B, T, L]  (char ids; padded with pad_id)
    Returns:
      word_vecs: FloatTensor [B, T, 2*h_char]  (one vector per token)
    """

    def __init__(self, vocab_size, pad_id=0, d_char=64, h_char=128, dropout=0.2):
        super().__init__()
        self.pad_id = pad_id

        # Char embedding: PAD row stays zeros
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_char,
            padding_idx=pad_id,
        )

        # Char-level BiLSTM (batch_first=True => [batch, seq, feat])
        self.char_lstm = nn.LSTM(
            input_size=d_char,
            hidden_size=h_char,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    # --- handy properties ---
    @property
    def vocab_size(self) -> int: return self.embed.num_embeddings
    @property
    def char_pad_id(self) -> int: return self.pad_id
    @property
    def char_embedding_dim(self) -> int: return self.embed.embedding_dim  # d_char
    @property
    def h_char(self) -> int: return self.char_lstm.hidden_size
    @property
    def d_word(self) -> int: return 2 * self.char_lstm.hidden_size
    @property
    def device(self): return next(self.parameters()).device
    @property
    def n_params(self) -> int: return sum(p.numel() for p in self.parameters())
    @property
    def n_trainable_params(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, char_ids: torch.LongTensor) -> torch.Tensor:
        """
        char_ids: [B, T, L]
        returns:  [B, T, 2*h_char]
        """
        B, T, L = char_ids.shape

        # Embedding + dropout: [B, T, L, d_char]
        E = self.dropout(self.embed(char_ids))

        # Flatten words: [B*T, L, d_char]
        BT = B * T
        E2 = E.reshape(BT, L, E.size(-1))

        # Compute true char lengths per word (0 for padded words)
        lengths = (char_ids != self.pad_id).sum(dim=-1).reshape(BT)  # [B*T]

        # We'll pack only words with length > 0
        nonzero_mask = lengths > 0
        nz_idx = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
        z_idx = (~nonzero_mask).nonzero(as_tuple=False).squeeze(-1)

        # Prepare outputs tensor
        out_flat = E2.new_zeros((BT, 2 * self.h_char))  # [B*T, 2*h_char]

        if nz_idx.numel() > 0:
            E2_nz = E2.index_select(0, nz_idx)
            len_nz = lengths.index_select(0, nz_idx)

            packed = pack_padded_sequence(E2_nz, len_nz.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.char_lstm(packed)  # h_n: [2, N_nz, h_char]

            # Concatenate forward/backward final states -> [N_nz, 2*h_char]
            word_vec_nz = torch.cat([h_n[0], h_n[1]], dim=-1)
            out_flat.index_copy_(0, nz_idx, word_vec_nz)

        # For zero-length words (pure PAD), we keep zeros (good: embedding pad is zero too)

        # Restore [B, T, 2*h_char] + dropout
        word_vecs = out_flat.reshape(B, T, -1)
        word_vecs = self.dropout(word_vecs)
        return word_vecs

### SANITY CHECKS ###
if __name__ == "__main__":
    from bilstm_1_preprocess_alt import read_conllu, TRAIN
    from bilstm_2_vocabularies_alt import (
        build_char_vocab, build_upos_vocab, build_feat_vocab_and_process_features_categories
    )
    from bilstm_3_collate_alt import collate_sentences

    train_sents = read_conllu(TRAIN)
    char2id, _ = build_char_vocab(train_sents)
    upos2id, _ = build_upos_vocab(train_sents, add_pad=True)
    _, feature_categories, feat2id, _, feature_slots = build_feat_vocab_and_process_features_categories(train_sents)

    batch = train_sents[:3]
    char_ids, word_mask, gold_upos, gold_feats = collate_sentences(
        batch,
        char2id=char2id,
        upos2id=upos2id,
        feat2id=feat2id,
        feature_slots=feature_slots,
        pad_char_id=char2id["<pad>"],
        max_word_len=40,
    )
    print("char_ids:", tuple(char_ids.shape))
    print("word_mask:", word_mask.shape, word_mask.dtype)
    print("gold_upos:", gold_upos.shape)

    enc = CharWordEncoder(vocab_size=len(char2id), pad_id=char2id["<pad>"], d_char=64, h_char=128, dropout=0.1)
    word_vecs = enc(char_ids)
    print("word_vecs:", tuple(word_vecs.shape))  # [B, T, 256] if h_char=128