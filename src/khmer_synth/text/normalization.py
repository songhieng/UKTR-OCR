"""Unicode + Khmer-specific text normalisation.

Real Khmer corpora (web text, OCR ground truth, user input) contain a lot of
sequences that are visually identical but differ in code points.  Feeding those
straight into a shaper produces inconsistent glyph sequences and therefore
inconsistent training targets, so we canonicalise first:

1. Unicode normalisation (NFC by default).
2. Deprecated code points are replaced (U+17A3/U+17A4 independent vowels,
   U+17B4/U+17B5 inherent vowels, U+17D3 bathamasat).
3. Marks inside one orthographic syllable are re-ordered into the canonical
   order  ``base (coeng cons)* shifter robat vowel sign``.  Typing order in the
   wild is frequently vowel-before-coeng, which shapes incorrectly.
4. Duplicate identical marks inside a syllable are dropped.
5. Whitespace / ZWSP policy.

Every step is individually switchable from :class:`~khmer_synth.config.TextConfig`
and the function returns the normalised string only - the mapping back to the
original text is intentionally *not* claimed to be index-preserving (the length
can change), so downstream metadata always stores both strings.
"""

from __future__ import annotations

import re
import unicodedata

from ..config import TextConfig
from . import khmer_chars as K
from .graphemes import segment_graphemes

# whitespace incl. NBSP, U+2000..U+200A and the ideographic space
_WS_RE = re.compile("[ \t\r\n\x0b\x0c\xa0\u2000-\u200a\u3000]+")
# C0/C1 control characters and the BOM (never carry ink)
_CTRL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufeff]")


def _apply_deprecated(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in K.DEPRECATED_MAP:
            out.append(K.DEPRECATED_MAP[cp])
        else:
            out.append(ch)
    return "".join(out)


def _reorder_cluster(cluster: str, dedup: bool) -> str:
    """Canonically order the marks of one Khmer syllable."""
    if not cluster:
        return cluster
    cps = [ord(c) for c in cluster]
    if not K.is_base(cps[0]):
        return cluster

    head: list[int] = [cps[0]]
    stacks: list[int] = []  # coeng + subscript consonant pairs, in input order
    marks: list[int] = []
    joiners: list[int] = []

    i = 1
    while i < len(cps):
        c = cps[i]
        if c == K.COENG and i + 1 < len(cps) and K.is_base(cps[i + 1]):
            # a subscript stack, wherever it appears in the cluster: typing
            # order is frequently "base vowel coeng consonant", which shapes
            # incorrectly unless the stack is moved back next to its base
            stacks.extend(cps[i : i + 2])
            i += 2
            continue
        if c in (K.ZWNJ, K.ZWJ):
            joiners.append(c)
        elif c == K.COENG:
            pass  # orphan coeng: cannot form a subscript, would render as a
            # dotted circle -> dropped (the original string is kept in metadata)
        else:
            marks.append(c)
        i += 1

    if dedup:
        seen = set()
        deduped = []
        for c in marks:
            # at most one dependent vowel and one of each sign per syllable
            key = (K.mark_class(c), c)
            if key in seen:
                continue
            if K.mark_class(c) == 3 and any(K.mark_class(x) == 3 for x in deduped):
                continue  # keep only the first dependent vowel
            seen.add(key)
            deduped.append(c)
        marks = deduped

    marks.sort(key=lambda c: (K.mark_class(c),))  # stable: preserves input order within a class
    return "".join(chr(c) for c in head + stacks + marks + joiners)


def normalize_khmer(text: str, cfg: TextConfig | None = None) -> str:
    """Normalise ``text`` for shaping.  Safe on mixed Khmer/Latin/numeric text."""
    cfg = cfg or TextConfig()
    if not text:
        return ""

    s = _CTRL_RE.sub("", text)
    if cfg.unicode_form:
        s = unicodedata.normalize(cfg.unicode_form, s)
    if cfg.fix_deprecated:
        s = _apply_deprecated(s)
    if cfg.strip_joiners:
        s = s.replace(chr(K.ZWNJ), "").replace(chr(K.ZWJ), "")

    if cfg.reorder_marks or cfg.dedup_marks:
        parts = []
        for cl in segment_graphemes(s):
            if cl.kind == "khmer" and (cfg.reorder_marks or cfg.dedup_marks):
                parts.append(_reorder_cluster(cl.text, cfg.dedup_marks))
            else:
                parts.append(cl.text)
        s = "".join(parts)

    if cfg.zwsp_policy == "space":
        s = s.replace(chr(K.ZWSP), " ")
    elif cfg.zwsp_policy == "remove":
        s = s.replace(chr(K.ZWSP), "")

    if cfg.collapse_whitespace:
        s = _WS_RE.sub(" ", s).strip()
    return s


def looks_corrupted(text: str) -> bool:
    """Cheap sanity check used by the dataset validator.

    Flags: unpaired surrogates, replacement chars, a Khmer mark with no base to
    attach to, or a coeng that is not followed by a base.
    """
    if not text:
        return True
    if "�" in text:
        return True
    for i, ch in enumerate(text):
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            return True
        if cp == K.COENG and (i + 1 >= len(text) or not K.is_base(ord(text[i + 1]))):
            return True
        if (K.is_dependent_vowel(cp) or K.is_sign(cp)) and i == 0:
            return True
    return False
