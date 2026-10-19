"""Vocabulary construction and caption encoding."""

from __future__ import annotations

import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)

# Deliberately conservative: keeps punctuation that carries meaning in a
# spoken description, drops the rest.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:()\-]")


def tokenize(text: str) -> list[str]:
    """Lowercase and split a caption into tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Vocab:
    """Word <-> index mapping with the four special tokens fixed at the front."""

    min_freq: int = 5
    max_size: int = 20000
    word2idx: dict[str, int] = field(default_factory=dict)
    idx2word: list[str] = field(default_factory=list)
    freqs: Counter = field(default_factory=Counter)

    # ------------------------------------------------------------- properties
    @property
    def pad_idx(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.word2idx[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.word2idx[END_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.idx2word)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        min_freq: int = 5,
        max_size: int = 20000,
    ) -> "Vocab":
        """Build a vocabulary from a corpus of captions."""
        vocab = cls(min_freq=min_freq, max_size=max_size)
        vocab.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        vocab.idx2word = list(SPECIAL_TOKENS)

        for caption in captions:
            vocab.freqs.update(tokenize(caption))

        # Most frequent first; alphabetical tiebreak so runs are reproducible.
        candidates = [
            (word, count)
            for word, count in vocab.freqs.items()
            if count >= min_freq
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        for word, _ in candidates[: max_size - len(SPECIAL_TOKENS)]:
            vocab.word2idx[word] = len(vocab.idx2word)
            vocab.idx2word.append(word)

        return vocab

    # ----------------------------------------------------------------- encode
    def encode(self, caption: str) -> list[int]:
        """Caption -> [start, ..., end] indices."""
        tokens = tokenize(caption)
        if not tokens:
            return []
        body = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        return [self.start_idx, *body, self.end_idx]

    def encode_batch(self, captions: Sequence[str]) -> list[list[int]]:
        return [self.encode(c) for c in captions]

    # ----------------------------------------------------------------- decode
    def decode(self, indices: Sequence[int], strip_special: bool = True) -> str:
        """Indices -> caption text, optionally removing special tokens."""
        words: list[str] = []
        special = set(SPECIAL_TOKENS) if strip_special else set()
        for idx in indices:
            if idx < 0 or idx >= len(self.idx2word):
                continue
            word = self.idx2word[idx]
            if word in special:
                continue
            words.append(word)
        return detokenize(words)

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with Path(path).open("rb") as handle:
            vocab = pickle.load(handle)
        if not isinstance(vocab, cls):
            raise TypeError(f"{path} does not contain a Vocab")
        return vocab


def detokenize(tokens: Sequence[str]) -> str:
    """Join tokens back into readable text.

    Sentence-initial capitals and no space before punctuation, so the composer
    can speak the result without artefacts.
    """
    if not tokens:
        return ""

    out: list[str] = []
    for i, token in enumerate(tokens):
        if i == 0:
            out.append(token.capitalize() if token.isalpha() else token)
        elif token in ".,!?;:" or token == "'":
            out.append(token)
        elif out and out[-1].endswith(("(", "-")):
            out.append(token)
        else:
            out.append(" " + token)
    return "".join(out)
