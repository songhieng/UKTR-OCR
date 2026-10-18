"""Khmer Unicode block (U+1780-U+17FF) character classes.

Reference: The Unicode Standard, ch. 16 "South and Central Asia-I", section
"Khmer", plus the Khmer script shaping rules used by HarfBuzz/USE.
"""

from __future__ import annotations

# --- consonants and independent vowels ------------------------------------- #
CONSONANT_START, CONSONANT_END = 0x1780, 0x17A2  # ក .. អ
INDEP_VOWEL_START, INDEP_VOWEL_END = 0x17A5, 0x17B3  # ឥ .. ឳ

# --- marks ------------------------------------------------------------------ #
COENG = 0x17D2  # ្  invisible stacker ("subscript" marker)
ZWSP = 0x200B
ZWNJ = 0x200C
ZWJ = 0x200D

# dependent vowels ា .. ៅ
DEP_VOWEL_START, DEP_VOWEL_END = 0x17B6, 0x17C5
# inherent vowels: deprecated, never rendered
INHERENT_AQ, INHERENT_AA = 0x17B4, 0x17B5

# signs that sit above/below and are *not* vowels
SIGN_NIKAHIT = 0x17C6  # ំ
SIGN_REAHMUK = 0x17C7  # ះ
SIGN_YUUKALEAPINTU = 0x17C8  # ៈ
SIGN_MUUSIKATOAN = 0x17C9  # ៉  register shifter
SIGN_TRIISAP = 0x17CA  # ៊  register shifter
SIGN_BANTOC = 0x17CB  # ់
SIGN_ROBAT = 0x17CC  # ៌
SIGN_TOANDAKHIAT = 0x17CD  # ៍
SIGN_KAKABAT = 0x17CE  # ៎
SIGN_AHSDA = 0x17CF  # ៏
SIGN_SAMYOK = 0x17D0  # ័
SIGN_VIRIAM = 0x17D1  # ៑
SIGN_BATHAMASAT = 0x17D3  # ៓ (deprecated lunar-date sign)
SIGN_ATTHACAN = 0x17DD  # ៝

REGISTER_SHIFTERS = frozenset({SIGN_MUUSIKATOAN, SIGN_TRIISAP})
ABOVE_SIGNS = frozenset(
    {
        SIGN_NIKAHIT,
        SIGN_BANTOC,
        SIGN_ROBAT,
        SIGN_TOANDAKHIAT,
        SIGN_KAKABAT,
        SIGN_AHSDA,
        SIGN_SAMYOK,
        SIGN_VIRIAM,
        SIGN_BATHAMASAT,
        SIGN_ATTHACAN,
    }
)
POST_SIGNS = frozenset({SIGN_REAHMUK, SIGN_YUUKALEAPINTU})

# --- where a combining mark is actually DRAWN relative to its base ---------- #
# WHY this table exists: ``writer/deformation.py::displace_marks`` used to decide
# "this is a floating above-mark" purely from geometry (is the component's bottom
# above the body's top?).  That test cannot tell a diacritic from a detached part
# of the letter itself, and Khmer has several of those - the upper horn of
# ក គ ត, the top serif of ដ, the top element of ភ.  MEASURED on 40 fonts x 5
# writers with `real_match_v2`: the geometric test fired on 59.3% of renders of
# bare (completely unmarked) ក គ ត ដ ភ ន ស - ត/ភ 0.825, ក/គ 0.775, ដ 0.725 - and
# on 60.4% of clusters carrying a below- or side-vowel (កុ 0.775, កែ 0.950,
# ក្ក 0.775), i.e. it floated ink that is not a mark at all.  Those letters are
# exactly the model's top confusions (ក->ត 74, គ->ត 74, ន->ដ 88, ស->ត 79), so the
# generator was manufacturing the confusion it is meant to teach against.
# Displacement is now gated on the cluster TEXT containing one of these.
#
# Membership was verified by rendering ក/ន/ស/ម + each mark over 30 fonts and
# measuring how often the mark contributes a component lying above the base top:
# ិ .908  ី .892  ឹ .850  ឺ .875  ើ .892  ំ 1.00  ៉ 1.00  ៊ .942  ់ 1.00  ៌ .958
# ៍ .958  ៎ 1.00  ៏ .942  ័ .992  ៑ 1.00  ៓ .933   <- above, listed here
# ុ .133  ូ .133  ួ .133  ្ .133  (== the ក-horn false-positive floor)  <- below
# ា .025  ៀ .025  ោ .025  ៅ .083  េ .125  ឿ .117                       <- spacing
# ែ 1.00  ៃ .933 are LEFT-SIDE spacing vowels drawn as េ plus a small tick; that
# tick reads as "above" geometrically, but floating it turns ែ into េ, so they
# are deliberately excluded.  ះ ៈ (POST_SIGNS) are right-side spacing signs whose
# upper dot reads the same way, and are excluded for the same reason.
ABOVE_VOWELS = frozenset(
    {
        0x17B7,  # ិ  I
        0x17B8,  # ី  II
        0x17B9,  # ឹ  Y
        0x17BA,  # ឺ  YY
        0x17BE,  # ើ  OE  (left-side េ plus a hook drawn above the base)
    }
)
#: marks drawn *below* the base (subscript zone), plus the coeng stacker itself
BELOW_MARKS = frozenset({0x17BB, 0x17BC, 0x17BD, COENG})  # ុ ូ ួ ្
#: vowels that occupy their own advance to the left/right of the base
SPACING_VOWELS = frozenset({0x17B6, 0x17BF, 0x17C0, 0x17C1, 0x17C2, 0x17C3, 0x17C4, 0x17C5})
#: every mark that renders above the base and may legitimately be floated
ABOVE_RENDERING_MARKS = ABOVE_VOWELS | ABOVE_SIGNS | REGISTER_SHIFTERS

# --- digits, punctuation, symbols ------------------------------------------- #
DIGIT_START, DIGIT_END = 0x17E0, 0x17E9  # ០ .. ៩
LEK_ATTAK_START, LEK_ATTAK_END = 0x17F0, 0x17F9  # ៰ .. ៹ divination numerals
PUNCT = frozenset(
    {
        0x17D4,  # ។ khan
        0x17D5,  # ៕ bariyoosan
        0x17D6,  # ៖ camnuc pii kuuh
        0x17D7,  # ៗ lek too (repetition)
        0x17D8,  # ៘ beyyal
        0x17D9,  # ៙ phnaek muan
        0x17DA,  # ៚ koomuut
        0x17DB,  # ៛ riel currency sign
        0x17DC,  # ៜ avakrahasanya
        0x19E0,  # ᧠ .. lunar date symbols
    }
)

# --- deprecated / discouraged code points ----------------------------------- #
DEPRECATED_MAP = {
    0x17A3: "អ",  # KHMER INDEPENDENT VOWEL QAQ -> QA
    0x17A4: "អា",  # KHMER INDEPENDENT VOWEL QAA -> QA + AA
    0x17B4: "",  # KHMER VOWEL INHERENT AQ  (should not appear in text)
    0x17B5: "",  # KHMER VOWEL INHERENT AA
    0x17D3: "",  # KHMER SIGN BATHAMASAT (deprecated)
}


def is_consonant(cp: int) -> bool:
    return CONSONANT_START <= cp <= CONSONANT_END


def is_independent_vowel(cp: int) -> bool:
    return INDEP_VOWEL_START <= cp <= INDEP_VOWEL_END


def is_base(cp: int) -> bool:
    """A code point that can start a Khmer orthographic syllable."""
    return is_consonant(cp) or is_independent_vowel(cp)


def is_dependent_vowel(cp: int) -> bool:
    return DEP_VOWEL_START <= cp <= DEP_VOWEL_END


def is_sign(cp: int) -> bool:
    return cp in ABOVE_SIGNS or cp in POST_SIGNS or cp in REGISTER_SHIFTERS


def is_above_rendering_mark(cp: int) -> bool:
    """True for a combining mark drawn *above* its base consonant.

    False for below-base marks (ុ ូ ួ, coeng), for spacing vowels that take their
    own advance (ា េ ែ ៃ ោ ៅ ៀ ឿ) and for the post-base signs (ះ ៈ).
    See ``ABOVE_RENDERING_MARKS`` for the measurement behind the table.
    """
    return cp in ABOVE_RENDERING_MARKS


def count_above_marks(text: str) -> int:
    """How many above-rendering marks a grapheme cluster's source text carries."""
    return sum(1 for ch in text if ord(ch) in ABOVE_RENDERING_MARKS)


def is_digit(cp: int) -> bool:
    return DIGIT_START <= cp <= DIGIT_END or LEK_ATTAK_START <= cp <= LEK_ATTAK_END


def is_khmer(cp: int) -> bool:
    return 0x1780 <= cp <= 0x17FF or 0x19E0 <= cp <= 0x19FF


def mark_class(cp: int) -> int:
    """Canonical ordering class inside a syllable (lower is written first).

    0  coeng + consonant  (handled separately, always first)
    1  register shifter   (U+17C9 / U+17CA)
    2  robat              (U+17CC - a superscript RO)
    3  dependent vowel
    4  above / post sign
    """
    if cp in REGISTER_SHIFTERS:
        return 1
    if cp == SIGN_ROBAT:
        return 2
    if is_dependent_vowel(cp):
        return 3
    return 4
