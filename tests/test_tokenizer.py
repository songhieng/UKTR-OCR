import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer.kcc import KCCTokenizer, normalize, segment, detect_language  # noqa: E402

KA, COENG, TA, RO, SA = "ក", "្", "ត", "រ", "ស"
E, II, OE, AA, OO = "េ", "ី", "ើ", "ា", "ោ"


def test_segment_clusters():
    # "ស្ត្រី" (woman) is one cluster: SA + COENG TA + COENG RO + II
    w = SA + COENG + TA + COENG + RO + II
    assert segment(w) == [w]
    assert segment("ab 12") == ["a", "b", " ", "1", "2"]
    assert segment("( ៉)") == ["(", " ", "៉", ")"]       # orphan mark kept
    assert "".join(segment("ការ Khmer ៩")) == "ការ Khmer ៩"


def test_normalization():
    assert normalize(SA + COENG + RO + COENG + TA + II) == SA + COENG + TA + COENG + RO + II  # coeng RO last
    assert normalize(KA + E + II) == KA + OE
    assert normalize(KA + E + AA) == KA + OO
    assert normalize("a​b   c ­") == "ab c"
    assert normalize("បុ៊") == "ប៊ុ"      # register shifter before vowel
    assert normalize("ឣ") == "អ"


def test_language():
    assert detect_language("ក abc") == "mixed"
    assert detect_language("កា") == "km"
    assert detect_language("hello") == "en"
    assert detect_language("123") == "other"


def test_build_roundtrip_and_fallback(tmp_path):
    train = [normalize(t) for t in ["ការ ស្ត្រី", "the cat 42"]]
    tok, _ = KCCTokenizer.build(train, min_freq=1)
    for t in train + ["គ្រូ unseen cluster"]:      # unseen cluster -> code points
        assert tok.decode(tok.encode(t)) == normalize(t)
    assert "ស្ត្រី" in tok.stoi
    ids = tok.encode("កាកា")
    assert tok.decode_ctc([tok.blank_id, ids[0], ids[0], tok.blank_id, ids[1]]) == "កាកា"
    tok.save(tmp_path / "v.json")
    tok2 = KCCTokenizer.load(tmp_path / "v.json")
    assert tok2.vocab == tok.vocab and tok2.normalization == tok.normalization
