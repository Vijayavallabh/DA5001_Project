"""feat-004: copying metrics (LCS word/char, ACS, nv-recall) on hand-checked cases."""
from dap.stats import acs_word, lcs_char, lcs_word, nv_recall, copying_metrics

REF = " ".join(f"w{i}" for i in range(60))  # 60 distinct words


def test_exact_copy():
    assert lcs_word(REF, REF) == 60 and acs_word(REF, REF) == 60 and nv_recall(REF, REF) == 1.0
    assert lcs_char("Hello, World", "hello,  world") == len("hello, world")


def test_partial_and_near_verbatim():
    hyp = " ".join(f"w{i}" for i in range(0, 25)) + " X " + " ".join(f"w{i}" for i in range(26, 50))
    assert lcs_word(hyp, REF) == 25
    assert acs_word(hyp, REF) == 49  # two blocks (25 + 24) both >= 5 words
    assert abs(nv_recall(hyp, REF, min_len=20) - 50 / 60) < 1e-9  # one-word gap merges into a 50-word span
    assert nv_recall(hyp, REF, min_len=20, tau_gap=0) == 25 / 60 + 24 / 60  # no merge: 25 and 24 both >= 20


def test_short_matches_ignored():
    hyp = "w3 w4 w5 w6 then unrelated text w20 w21"
    assert lcs_word(hyp, REF) == 4 and acs_word(hyp, REF) == 0 and nv_recall(hyp, REF) == 0.0
    m = copying_metrics(hyp, REF)
    assert set(m) == {"rouge_l", "minhash_5gram", "lcs_word", "lcs_char", "acs_word", "nv_recall"}
    assert copying_metrics("anything", None)["nv_recall"] == 0.0
