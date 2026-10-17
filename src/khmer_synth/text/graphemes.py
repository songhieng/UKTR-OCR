"""Khmer-aware grapheme (orthographic syllable) segmentation.

Why not UAX #29?
----------------
The default extended-grapheme-cluster algorithm does **not** keep a Khmer
subscript sequence together: U+17D2 KHMER SIGN COENG has
``Grapheme_Cluster_Break=Extend`` but the consonant that follows it is ``Other``,
so UAX #29 inserts a break *before the subscript consonant*.  The Unicode 15.1
``Indic_Conjunct_Break`` machinery only covers Devanagari-family viramas, not the
Khmer invisible stacker.  Segmenting Khmer with UAX #29 therefore splits
``ក្ន`` into ``ក`` + ``្ន``, which would break per-cluster rasterisation,
per-grapheme deformation and the text<->glyph mapping.

The segmenter below implements the orthographic-syllable grammar instead:

    syllable := base (coeng base)* shifter* robat? vowel? sign* joiner?
    base     := consonant | independent vowel

Anything that is not Khmer (Latin, digits, punctuation, whitespace) is emitted
as its own cluster, with trailing combining marks (Mn/Mc) attached, so mixed
Khmer/Latin/numeric text round-trips correctly.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from . import khmer_chars as K


@dataclass(frozen=True)
class GraphemeCluster:
    """One orthographic syllable / user-perceived character."""

    text: str
    #: [start, end) code-point indices into the string it was segmented from
    start: int
    end: int
    #: 'khmer' | 'latin' | 'digit' | 'punct' | 'space' | 'other'
    kind: str

    @property
    def codepoints(self) -> list[int]:
        return [ord(c) for c in self.text]

    def __len__(self) -> int:  # number of code points
        return self.end - self.start


def _kind_of(cp: int) -> str:
    if cp in (0x20, 0x09, 0x0A, 0x0D, K.ZWSP) or unicodedata.category(chr(cp)) == "Zs":
        return "space"
    if K.is_base(cp):
        return "khmer"
    if K.is_digit(cp):
        return "digit"
    if cp in K.PUNCT:
        return "punct"
    if unicodedata.category(chr(cp)).startswith("N"):
        return "digit"
    if unicodedata.category(chr(cp)).startswith("L"):
        return "latin"
    if unicodedata.category(chr(cp)).startswith("P"):
        return "punct"
    return "other"


def _is_combining(cp: int) -> bool:
    return unicodedata.category(chr(cp)) in ("Mn", "Mc", "Me")


def segment_graphemes(text: str) -> list[GraphemeCluster]:
    """Split ``text`` into Khmer-aware grapheme clusters."""
    out: list[GraphemeCluster] = []
    n = len(text)
    i = 0
    while i < n:
        cp = ord(text[i])
        start = i

        if K.is_base(cp):
            i += 1
            # (coeng + base)*  -- subscript consonant stack
            while i + 1 < n and ord(text[i]) == K.COENG and K.is_base(ord(text[i + 1])):
                i += 2
            # a dangling coeng at end of string still belongs to this cluster
            if i < n and ord(text[i]) == K.COENG and (i + 1 >= n or not K.is_base(ord(text[i + 1]))):
                i += 1
            # shifters / robat / dependent vowel / signs, in any order (the
            # normaliser puts them in canonical order beforehand)
            while i < n:
                c = ord(text[i])
                # a coeng+base pair that appears *after* a vowel is mis-ordered
                # input (very common when typing); it still belongs to this
                # syllable, so absorb it here and let the normaliser reorder it
                if c == K.COENG and i + 1 < n and K.is_base(ord(text[i + 1])):
                    i += 2
                elif K.is_dependent_vowel(c) or K.is_sign(c) or c == K.SIGN_ROBAT or c in (K.ZWNJ, K.ZWJ):
                    i += 1
                elif _is_combining(c) and K.is_khmer(c):
                    i += 1
                else:
                    break
            out.append(GraphemeCluster(text[start:i], start, i, "khmer"))
            continue

        # non-base: single char + any following combining marks
        i += 1
        while i < n and _is_combining(ord(text[i])) and not K.is_base(ord(text[i])):
            i += 1
        out.append(GraphemeCluster(text[start:i], start, i, _kind_of(cp)))
    return out


def cluster_index_map(clusters: list[GraphemeCluster], length: int) -> list[int]:
    """Map every code-point index of the source string to its cluster index."""
    m = [-1] * length
    for ci, c in enumerate(clusters):
        for i in range(c.start, min(c.end, length)):
            m[i] = ci
    # defensive: any unmapped index attaches to the nearest preceding cluster
    last = 0
    for i in range(length):
        if m[i] < 0:
            m[i] = last
        else:
            last = m[i]
    return m


def split_words(clusters: list[GraphemeCluster]) -> list[list[int]]:
    """Group cluster indices into "words".

    Khmer does not use spaces between words; U+200B ZWSP and U+0020 are the only
    reliable separators in plain text, so a "word" here is a run of clusters
    between separators.  Used only for word-level deformation.
    """
    words: list[list[int]] = []
    cur: list[int] = []
    for i, c in enumerate(clusters):
        if c.kind == "space":
            if cur:
                words.append(cur)
                cur = []
            continue
        cur.append(i)
    if cur:
        words.append(cur)
    return words
