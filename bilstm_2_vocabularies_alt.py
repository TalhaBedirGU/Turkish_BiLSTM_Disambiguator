from collections import defaultdict
from pathlib import Path
import json

# Import only the READER, not precomputed data
from bilstm_1_preprocess_alt import read_conllu, TRAIN, DEV, TEST

############################
# CHARACTER VOCABULARY
############################

def build_char_vocab(sentences):
    """
    Returns char2id, id2char with '<pad>' at 0 and '<unk>' at 1.
    """
    all_chars = []
    for sent in sentences:
        for tok in sent:
            all_chars.extend(list(tok['form']))
    unique = sorted(set(all_chars))
    vocab = ['<pad>', '<unk>'] + unique

    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    return char2id, id2char

def sentence_to_char_ids(sentence, char2id, max_len=None):
    """
    sentence: List[Dict], one UD sentence (tokens)
    return: List[List[int]] (one char-ID list per token)
    """
    pad_id = char2id['<pad>']
    unk_id = char2id['<unk>']
    out = []
    for tok in sentence:
        ids = [char2id.get(ch, unk_id) for ch in tok['form']]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [pad_id] * (max_len - len(ids))
        out.append(ids)
    return out

############################
# UPOS VOCABULARY
############################

def build_upos_vocab(sentences, add_pad=True):
    """
    Collect unique UPOS tags. Optionally add '<PAD>' at index 0.
    """
    tags = []
    for sent in sentences:
        for tok in sent:
            tags.append(tok['upos'])
    uniq = sorted(set(tags))

    if add_pad:
        vocab = ['<PAD>'] + uniq
    else:
        vocab = uniq

    upos2id = {t: i for i, t in enumerate(vocab)}
    id2upos = {i: t for i, t in enumerate(vocab)}
    return upos2id, id2upos

############################
# FEATURE SLOTS & FEAT VOCABS
############################

def build_feat_vocab_and_process_features_categories(sentences):
    """
    Returns:
      violations: {slot: count_of_multivalued_tokens}
      feature_categories: {slot: ["<PAD>", "<None>", v1, v2, ...]}  # stable order
      feat2id: {slot: {value: idx}}
      id2feat: {slot: {idx: value}}
      feature_slots: [slot1, slot2, ...]  # deterministic global order
    """
    violations = defaultdict(int)
    cats2vals = defaultdict(set)

    for sent in sentences:
        for tok in sent:
            feats = tok.get('feats') or {}
            for slot, val in feats.items():
                if isinstance(val, list):
                    if len(val) > 1:
                        violations[slot] += 1
                    for v in val:
                        cats2vals[slot].add(str(v))
                else:
                    cats2vals[slot].add(str(val))

    # Deterministic slot order (critical!)
    feature_slots = sorted(cats2vals.keys())

    feature_categories = {}
    for slot in feature_slots:
        vals = sorted(cats2vals[slot])
        # <PAD> for padding positions; <None> means slot absent on this token
        feature_categories[slot] = ["<PAD>", "<None>"] + vals

    feat2id = {
        slot: {v: i for i, v in enumerate(feature_categories[slot])}
        for slot in feature_slots
    }
    id2feat = {
        slot: {i: v for i, v in enumerate(feature_categories[slot])}
        for slot in feature_slots
    }
    return violations, feature_categories, feat2id, id2feat, feature_slots

############################
# PERSIST / LOAD HELPERS
############################

def save_vocab(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_vocab(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

### SANITY CHECKS ###

if __name__ == "__main__":
    train_sents = read_conllu(TRAIN)

    # Chars
    char2id, id2char = build_char_vocab(train_sents)
    # UPOS 
    upos2id, id2upos = build_upos_vocab(train_sents, add_pad=True)

    # Feats
    (violations,
     feature_categories,
     feat2id,
     id2feat,
     feature_slots) = build_feat_vocab_and_process_features_categories(train_sents)

    print("Violations (multi-valued features):", dict(violations))
    print("Feature slots (fixed order):", feature_slots)
    print("UPOS size:", len(upos2id))

    # Persist
    outdir = Path("artifacts/vocabs")
    save_vocab(char2id, outdir / "char2id.json")
    save_vocab(id2char, outdir / "id2char.json")
    save_vocab(upos2id, outdir / "upos2id.json")
    save_vocab(id2upos, outdir / "id2upos.json")
    save_vocab(feature_categories, outdir / "feature_categories.json")
    save_vocab(feat2id, outdir / "feat2id.json")
    save_vocab(id2feat, outdir / "id2feat.json")
    save_vocab(feature_slots, outdir / "feature_slots.json")