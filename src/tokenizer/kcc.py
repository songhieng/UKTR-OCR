"""Khmer Character Cluster (KCC) normalization, segmentation and tokenizer.

A KCC is one orthographic Khmer cluster: a base (consonant or independent vowel)
followed by any number of subscripts (COENG + base) and combining marks
(dependent vowels, diacritics). Every non-Khmer character (Latin letter, digit,
punctuation, space) is its own token. Orphan marks (e.g. a diacritic quoted on
its own, "( \u17C9)") form a cluster of their own so nothing is silently dropped.

Encoding is reversible for any text whose characters are in the vocabulary:
a cluster missing from the vocabulary falls back to its individual code points,
and the full Khmer block plus printable ASCII are always in the vocabulary.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

COENG = "\u17D2"
_BASE = "\u1780-\u17B3"                      # consonants + independent vowels
_MARK = "\u17B6-\u17D1\u17D3\u17DD\u200C\u200D"  # dependent vowels, signs, ZWNJ/ZWJ
_CLUSTER_RE = re.compile(
    f"[{_BASE}](?:{COENG}[{_BASE}]|[{_MARK}])*"   # regular cluster
    f"|(?:{COENG}[{_BASE}]|[{_MARK}])+"           # orphan subscripts / marks
    r"|.",                                         # anything else, one char
    re.DOTALL,
)
_WS_RE = re.compile(r"\s+")

DEFAULT_NORMALIZATION = {
    "unicode_form": "NFC",
    "remove_chars": ["\u200B", "\uFEFF", "\u00AD", "\u17B4", "\u17B5"],  # ZWSP, BOM, SHY, inherent vowels
    "remove_zwnj_zwj": True,
    "compose_split_vowels": True,   # \u17C1+\u17B8 -> \u17BE , \u17C1+\u17B6 -> \u17C4
    "replace_deprecated": True,     # \u17A3 -> \u17A2 , \u17A4 -> \u17A2\u17B6
    "reorder_cluster_marks": True,  # canonical order inside a cluster
    "merge_coeng_da_ta": False,     # \u17D2\u178A -> \u17D2\u178F (spelling-changing, off by default)
    "collapse_whitespace": True,
}

_SPLIT_VOWELS = {"\u17C1\u17B8": "\u17BE", "\u17C1\u17B6": "\u17C4"}
_DEPRECATED = {"\u17A3": "\u17A2", "\u17A4": "\u17A2\u17B6"}
_RO = "\u179A"


def _mark_rank(unit: str) -> int:
    """Sort key for the units following a cluster base (stable sort)."""
    if unit[0] == COENG and len(unit) == 2:
        return 1 if unit[1] == _RO else 0     # COENG RO goes after other subscripts
    c = unit
    if c in "\u17C9\u17CA":                    # register shifters before vowels
        return 2
    if c == "\u17CC":                          # robat
        return 3
    if "\u17B6" <= c <= "\u17C5":              # dependent vowels
        return 4
    return 5                                   # other signs


def _reorder(cluster: str) -> str:
    if not ("\u1780" <= cluster[0] <= "\u17B3") or len(cluster) < 3:
        return cluster
    units, i = [], 1
    while i < len(cluster):
        if cluster[i] == COENG and i + 1 < len(cluster):
            units.append(cluster[i:i + 2]); i += 2
        else:
            units.append(cluster[i]); i += 1
    return cluster[0] + "".join(sorted(units, key=_mark_rank))


def normalize(text: str, cfg: dict | None = None) -> str:
    cfg = {**DEFAULT_NORMALIZATION, **(cfg or {})}
    if cfg["unicode_form"]:
        text = unicodedata.normalize(cfg["unicode_form"], text)
    for ch in cfg["remove_chars"]:
        text = text.replace(ch, "")
    if cfg["remove_zwnj_zwj"]:
        text = text.replace("\u200C", "").replace("\u200D", "")
    if cfg["replace_deprecated"]:
        for a, b in _DEPRECATED.items():
            text = text.replace(a, b)
    if cfg["merge_coeng_da_ta"]:
        text = text.replace(COENG + "\u178A", COENG + "\u178F")
    if cfg["reorder_cluster_marks"]:
        text = "".join(_reorder(c) for c in _CLUSTER_RE.findall(text))
    if cfg["compose_split_vowels"]:            # after reordering so the pair is adjacent
        for a, b in _SPLIT_VOWELS.items():
            text = text.replace(a, b)
    if cfg["collapse_whitespace"]:
        text = _WS_RE.sub(" ", text).strip()
    return text


def segment(text: str) -> list[str]:
    """Split (already normalized) text into KCC tokens."""
    return _CLUSTER_RE.findall(text)


def has_khmer(text: str) -> bool:
    return any("\u1780" <= c <= "\u17FF" or "\u19E0" <= c <= "\u19FF" for c in text)


def detect_language(text: str) -> str:
    km = has_khmer(text)
    latin = any(c.isascii() and c.isalpha() for c in text)
    return "mixed" if km and latin else "km" if km else "en" if latin else "other"


# Characters always in the vocabulary, so unseen clusters can fall back to code points.
BASE_CHARSET = (
    [chr(c) for c in range(0x1780, 0x1800) if unicodedata.category(chr(c)) != "Cn"]
    + [chr(c) for c in range(0x19E0, 0x1A00)]
    + [chr(c) for c in range(0x20, 0x7F)]
    + [chr(c) for c in range(0xA1, 0x180) if c != 0xAD]          # Latin-1 Supplement + Extended-A
    + [chr(c) for c in list(range(0x2010, 0x2028)) + list(range(0x2030, 0x203B)) + [0x20AC, 0x2116, 0x2122]]
)
SPECIALS = ["<blank>", "<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]


class KCCTokenizer:
    def __init__(self, vocab: list[str], normalization: dict | None = None):
        assert vocab[:len(SPECIALS)] == SPECIALS, "vocab must start with the special tokens"
        assert len(set(vocab)) == len(vocab), "duplicate vocab entries"
        self.vocab = vocab
        self.normalization = {**DEFAULT_NORMALIZATION, **(normalization or {})}
        self.stoi = {s: i for i, s in enumerate(vocab)}
        (self.blank_id, self.pad_id, self.bos_id, self.eos_id,
         self.unk_id, self.mask_id) = range(len(SPECIALS))

    def __len__(self) -> int:
        return len(self.vocab)

    # ---- building / persistence -------------------------------------------------
    @classmethod
    def build(cls, texts, normalization=None, min_freq=1, max_size=None):
        """Build from TRAINING transcriptions only. Returns (tokenizer, cluster Counter)."""
        counts = Counter()
        for t in texts:
            counts.update(segment(t))          # texts are expected to be normalized already
        clusters = [c for c, n in counts.most_common() if n >= min_freq and len(c) > 1]
        vocab = SPECIALS + BASE_CHARSET
        seen = set(vocab)
        vocab += sorted({c for c in counts if len(c) == 1} - seen)  # extra single chars seen in train
        seen = set(vocab)
        for c in clusters:
            if max_size and len(vocab) >= max_size:
                break
            if c not in seen:
                vocab.append(c); seen.add(c)
        return cls(vocab, normalization), counts

    def to_dict(self) -> dict:
        return {"type": "kcc", "version": 1, "vocab": self.vocab,
                "normalization": self.normalization, "specials": SPECIALS}

    @classmethod
    def from_dict(cls, d: dict) -> "KCCTokenizer":
        return cls(d["vocab"], d["normalization"])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "KCCTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- encode / decode --------------------------------------------------------
    def normalize(self, text: str) -> str:
        return normalize(text, self.normalization)

    def encode(self, text: str, normalized: bool = False) -> list[int]:
        if not normalized:
            text = self.normalize(text)
        ids = []
        for tok in segment(text):
            i = self.stoi.get(tok)
            if i is not None:
                ids.append(i)
            else:                              # unseen cluster -> code points
                ids.extend(self.stoi.get(ch, self.unk_id) for ch in tok)
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i >= len(SPECIALS):
                out.append(self.vocab[i])
            elif i == self.unk_id:
                out.append("\uFFFD")
        return "".join(out)

    def decode_ctc(self, ids) -> str:
        """Greedy CTC collapse: merge repeats, drop blanks/specials."""
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != prev and i >= len(SPECIALS):
                out.append(self.vocab[i])
            elif i != prev and i == self.unk_id:
                out.append("\uFFFD")
            prev = i
        return "".join(out)
