"""Vocabulary construction and tokenization for ActivityNet Captions.

Provides:
- Special token constants (<pad>, <bos>, <eos>, <unk>)
- Vocabulary class with encode/decode methods
- Automatic vocabulary extraction from ActivityNet Captions annotations
"""

import os
import re
import json
from collections import Counter
from pathlib import Path


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def clean_text(text: str) -> list:
    """Tokenizes and normalizes input text."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens


class Vocabulary:
    def __init__(self, word2idx: dict = None, idx2word: dict = None):
        if word2idx is not None and idx2word is not None:
            self.word2idx = word2idx
            self.idx2word = {int(k): v for k, v in idx2word.items()}
        else:
            self.word2idx = {
                PAD_TOKEN: PAD_IDX,
                BOS_TOKEN: BOS_IDX,
                EOS_TOKEN: EOS_IDX,
                UNK_TOKEN: UNK_IDX,
            }
            self.idx2word = {
                PAD_IDX: PAD_TOKEN,
                BOS_IDX: BOS_TOKEN,
                EOS_IDX: EOS_TOKEN,
                UNK_IDX: UNK_TOKEN,
            }

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    def encode(self, text: str, max_len: int = 25, add_special: bool = True) -> list:
        """Encodes text to a fixed-length list of token IDs."""
        tokens = clean_text(text)
        token_ids = []
        if add_special:
            token_ids.append(BOS_IDX)

        for tok in tokens:
            token_ids.append(self.word2idx.get(tok, UNK_IDX))

        if add_special:
            token_ids.append(EOS_IDX)

        # Truncate or Pad
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            if add_special and token_ids[-1] != EOS_IDX:
                token_ids[-1] = EOS_IDX
        else:
            token_ids = token_ids + [PAD_IDX] * (max_len - len(token_ids))

        return token_ids

    def decode(self, token_ids, strip_special: bool = True) -> str:
        """Decodes token IDs back to a readable string."""
        words = []
        for tid in token_ids:
            if hasattr(tid, "item"):
                tid = tid.item()
            tid = int(tid)
            if tid == EOS_IDX:
                if strip_special:
                    break
            if strip_special and tid in (PAD_IDX, BOS_IDX):
                continue
            words.append(self.idx2word.get(tid, UNK_TOKEN))
        return " ".join(words)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word
            }, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(word2idx=data["word2idx"], idx2word=data["idx2word"])


def build_activitynet_vocab(
    train_annotations_path: str,
    output_vocab_path: str,
    max_vocab_size: int = 5000,
    min_freq: int = 2
) -> Vocabulary:
    """Builds vocabulary from ActivityNet Captions training sentences."""
    if os.path.exists(output_vocab_path):
        print(f"Loading existing vocabulary from {output_vocab_path}...")
        return Vocabulary.load(output_vocab_path)

    print(f"Building vocabulary from {train_annotations_path} (max {max_vocab_size} words)...")
    with open(train_annotations_path, "r") as f:
        data = json.load(f)

    counter = Counter()
    for vid_id, item in data.items():
        sentences = item.get("sentences", [])
        for sent in sentences:
            tokens = clean_text(sent)
            counter.update(tokens)

    vocab = Vocabulary()
    for word, count in counter.most_common(max_vocab_size - 4):
        if count >= min_freq:
            vocab.add_word(word)

    vocab.save(output_vocab_path)
    print(f"Built vocabulary with {len(vocab)} words and saved to {output_vocab_path}.")
    return vocab
