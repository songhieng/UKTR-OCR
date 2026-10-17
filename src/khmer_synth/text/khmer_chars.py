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
