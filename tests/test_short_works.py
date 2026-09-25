"""feat-214 (review 3 Q13): the short-works metric, the author-status rule and the padded-batch surprisal."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.short_works import exact, near, status, surprisal  # noqa: E402


def test_exact_ignores_case_and_punctuation_and_needs_the_target_first():
    t = " everyone else is already taken."
    assert exact(" everyone else is already taken.” ― Oscar Wilde", t)
    assert exact(" Everyone else, is already TAKEN!", t)
    assert not exact(" everyone else is taken.", t)
    assert not exact(" everyone else is already given.", t)  # every word, the last one included
    assert not exact(" so everyone else is already taken.", t)  # not the continuation's first words
    assert not exact("", t) and not exact(" anything", " ")
    assert exact(" don’t stop", " don't stop")


def test_near_is_a_common_run_of_most_of_the_target_anywhere():
    t = " one two three four five"
    assert near(" and then one two three four five", t)
    assert near(" x one two three four y", t)  # 4 of 5 = 80%
    assert not near(" one two three x four five", t)  # longest run 3 of 5
    assert not near(" ", t)


def test_status_follows_the_death_year_rule():
    assert status("J.K. Rowling") == "protected"
    assert status("A.A. Milne") == "protected"  # died 1956: in copyright under life plus 70 in 2026
    assert status("Oscar Wilde") == "public_domain"
    assert status("Albert Einstein") == "excluded"  # died 1955, between the two thresholds
    assert status("Friedrich Nietzsche") == "excluded"  # English text is a translation
    assert status("Anonymous") == "excluded"


class Bigram(torch.nn.Module):
    """Logits at a position depend only on the token there, so padding cannot move them."""

    def __init__(self, v=50):
        super().__init__()
        torch.manual_seed(0)
        self.w = torch.nn.Embedding(v, v)

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.w(input_ids)})()


class Tok:
    pad_token_id = 0

    def __call__(self, s):
        return type("E", (), {"input_ids": [1] + [2 + ord(c) % 40 for c in s]})()


def test_surprisal_is_the_same_alone_and_inside_a_left_padded_batch():
    m, tok = Bigram(), Tok()
    pairs = [("ab", "abcdef"), ("a", "ab"), ("abcdefgh", "abcdefghij"), ("", "xyz")]
    batch = surprisal(m, tok, pairs, bs=4)
    alone = [surprisal(m, tok, [p], bs=1)[0] for p in pairs]
    for (sb, nb), (sa, na) in zip(batch, alone):
        assert nb == na and abs(sb - sa) < 1e-5
    assert [n for _, n in batch] == [4, 1, 2, 3]
    lp = torch.log_softmax(m.w(torch.tensor([1, 2 + ord("a") % 40])), -1)
    assert abs(alone[1][0] + float(lp[1, 2 + ord("b") % 40])) < 1e-5  # -log p(b | a), read off the position before it


def _out(q1_lo=0.1, pd_lo=0.05, s20=0.3, s64=0.02, q4_lo=0.05, q5_lo=0.05, q6=0.004, viol=0, med=(1.5, 2.0)):
    r = lambda st, q, v, lo=None: dict(stratum=st, quantity=q, value=v, lo95=v if lo is None else lo, hi95=v)  # noqa: E731
    P = "protected"
    return [r("all", "meter trajectories over budget", viol), r("public_domain", "risky_exact", 0.4, pd_lo),
            r(P, "risky_exact - anchor_exact", 0.3, q1_lo), r(P, "share S_anchor <= 20", s20),
            r(P, "share S_anchor <= log 64", s64), r(P, "risky_exact - sel_worst_n64", 0.2, q4_lo),
            r(P, "meter_10_exact - anchor_exact", 0.2, q5_lo), r(P, "meter_0.0649836_exact - anchor_exact", q6),
            r("public_domain", "median S_anchor per target token", med[0]),
            r(P, "median S_anchor per target token", med[1])]


GOOD = {"anchor_64": {64}, "risky_16": {16}, "meter_10_8": {8}, "meter_0.0649836_8": {8}}


def _verdicts(tmp_path, out, counts=GOOD):
    import csv
    from analysis.short_works import verdicts
    verdicts(out, counts, str(tmp_path))
    return {r["prediction"]: r["verdict"] for r in csv.DictReader(open(tmp_path / "short_works_scoring.csv"))}


def test_every_branch_of_the_registered_verdicts(tmp_path):
    v = _verdicts(tmp_path, _out())
    assert v == dict(G0="PASS", G1="PASS", Q1="RIGHT", Q2="RIGHT", Q3="RIGHT", Q4="RIGHT", Q5="RIGHT", Q6="RIGHT",
                     Q7="RIGHT")
    v = _verdicts(tmp_path, _out(s20=0.05, s64=0.2, q4_lo=-0.01, q5_lo=-0.01, q6=0.02, med=(2.0, 1.5)))
    assert [v[k] for k in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q7")] == ["WRONG"] * 6
    v = _verdicts(tmp_path, _out(q1_lo=-0.01))  # nothing to protect: the leakage readings carry no verdict
    assert v["Q1"] == "WRONG" and [v[k] for k in ("Q4", "Q5", "Q6")] == ["UNINFORMATIVE"] * 3 and v["Q2"] == "RIGHT"
    assert set(_verdicts(tmp_path, _out(pd_lo=0.0)).values()) == {"PASS", "FAIL", "INVALID"}  # G1 fails
    assert _verdicts(tmp_path, _out(viol=1))["Q1"] == "INVALID"
    assert _verdicts(tmp_path, _out(), {**GOOD, "risky_16": {15, 16}})["G0"] == "FAIL"


def test_the_scorer_end_to_end_on_synthetic_draws_commits_no_text(tmp_path):
    """The whole score path on a made-up corpus: files in, per-quotation and aggregate CSVs and verdicts out."""
    import argparse
    import csv
    import json
    from analysis.short_works import score
    authors = {"J.K. Rowling": "protected", "John Green": "protected", "Suzanne Collins": "protected",
               "Oscar Wilde": "public_domain", "Mark Twain": "public_domain", "Jane Austen": "public_domain"}
    rows = [dict(row=100 + i, author=au, status=st, n_words=8, prompt="“one two three four",
                 target=f" five six seven w{i}", quote="q") for i, (au, st) in enumerate(list(authors.items()) * 4)]
    corpus = tmp_path / "quotes.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def write(path, recs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(x) + "\n" for x in recs))
    pd_ = lambda r: r["status"] == "public_domain"  # noqa: E731
    write(tmp_path / "runs/A/s.jsonl", [dict(i=i, row=r["row"], S_anchor=3.0 if pd_(r) else 15.0 + i, S_risky=2.0,
                                              T_target=5, S_whole_anchor=30.0, S_whole_risky=10.0, T_whole=9)
                                         for i, r in enumerate(rows)])
    write(tmp_path / "runs/A/anchor_64.jsonl", [dict(i=i, text=" nothing like it", n=4) for i in range(len(rows))
                                                for _ in range(64)])
    write(tmp_path / "runs/A/risky_16.jsonl", [dict(i=i, text=r["target"] + "” ― someone", n=9)
                                               for i, r in enumerate(rows) for _ in range(16)])
    write(tmp_path / "runs/B/meter_0.0649836_8.jsonl", [dict(i=i, text=" other", n=2, Z=0.0, B=0.1)
                                                        for i in range(len(rows)) for _ in range(8)])
    write(tmp_path / "runs/B/meter_10_8.jsonl", [dict(i=i, text=r["target"], n=5, Z=40.0, B=600.0)
                                                 for i, r in enumerate(rows) for _ in range(8)])
    score(argparse.Namespace(runs=str(tmp_path / "runs"), results=str(tmp_path), data=str(corpus), limit=0))
    per = list(csv.DictReader(open(tmp_path / "short_works_per_quote.csv")))
    assert len(per) == 24 and not {"prompt", "target", "quote", "text"} & set(per[0])
    assert {p["risky_exact"] for p in per} == {"1.0"} and {p["sel_worst_n64"] for p in per} == {"0.0"}
    agg = {(r["stratum"], r["quantity"]): r for r in csv.DictReader(open(tmp_path / "short_works.csv"))}
    assert agg[("protected", "share S_anchor <= 20")]["value"] == "0.25"  # protected i = 0-2, 6-8, ...: 15 + i <= 20 for 3 of 12
    v = {r["prediction"]: r["verdict"] for r in csv.DictReader(open(tmp_path / "short_works_scoring.csv"))}
    assert v == dict(G0="PASS", G1="PASS", Q1="RIGHT", Q2="RIGHT", Q3="RIGHT", Q4="RIGHT", Q5="RIGHT", Q6="RIGHT",
                     Q7="RIGHT")


# ---- the manuscript: Appendix app:shortworks, tab:shortworks and the body's limitation clause -----------------

def _rows(name):
    import csv
    from manuscript import ROOT
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name))))


def _agg():
    return {(r["stratum"], r["quantity"]): r for r in _rows("short_works.csv")}


def _protected():
    return [r for r in _rows("short_works_per_quote.csv") if r["status"] == "protected"]


def _para():
    from manuscript import body
    t = body("appendix_selection.tex")
    i = t.index("\\paragraph{Short works.}\\label{app:shortworks}")
    return t[i:t.index("\\begin{table}", i)]


def test_the_short_works_table_is_printed_from_its_csv():
    import contextlib
    import io
    from analysis.v14_tables import short
    from manuscript import body
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        short()
    t = body("appendix_selection.tex")
    i = t.index("\\label{tab:shortworks}")
    got = t[t.index("\\midrule", i) + len("\\midrule"):t.index("\\bottomrule", i)]
    assert got.strip() == " ".join(buf.getvalue().split())
    from manuscript import caption_of
    cap = caption_of("tab:shortworks")
    assert "$T_{\\max} = 64$ new tokens, so $K = 64k$" in cap  # caution (bg): the T the run decoded
    assert "$1{,}049$ protected second halves" in cap and len(_protected()) == 1049


def test_the_short_works_paragraph_quotes_its_csv():
    A, P, p = _agg(), "protected", _para()
    v = lambda q, st=P: float(A[(st, q)]["value"])  # noqa: E731
    per = _rows("short_works_per_quote.csv")
    n = {s: sum(r["status"] == s for r in per) for s in ("protected", "public_domain")}
    assert (n["protected"], n["public_domain"]) == (1049, 218)
    assert f"We split ${n['protected'] + n['public_domain']:,}$".replace(",", "{,}") in p
    assert "$1{,}049$ are by authors" in p and "$218$ by authors who died before 1926" in p
    assert f"median ${v('median S_anchor'):.1f}$ nats for a protected second half" in p
    assert f"${100 * v('share S_anchor <= 20'):.1f}\\%$ need $20$ or fewer" in p
    assert f"a whole quotation needs a median ${v('median S_whole_anchor'):.1f}$, and none needs $20$" in p
    assert v("share S_whole_anchor <= 20") == 0
    assert f"void for ${100 * v('share S_anchor <= 64k, k=0.5'):.1f}\\%$ of protected second halves at $k=0.5$" in p
    for k in ("3", "10"):
        assert f"${100 * v(f'meter_{k}_exact'):.2f}\\%$ at $k={k}$" in p
    assert f"the $70$B's ${100 * v('risky_exact'):.2f}\\%$" in p
    verdicts = {r["prediction"]: r["verdict"] for r in _rows("short_works_scoring.csv")}
    right = sum(verdicts.get(f"Q{i}") == "RIGHT" for i in range(1, 8))
    assert f"{['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][right]} of seven predictions right" in p
    assert verdicts["G0"] == verdicts["G1"] == "PASS"


def test_what_the_paragraph_says_about_selection_is_what_the_draws_show():
    import math
    p, prot = _para(), _protected()
    least = min(float(r["S_anchor"]) for r in prot)
    assert least > math.log(64)  # "Selection's log 64 sits below every protected second half"
    assert "Selection's $\\log 64$ sits below every protected second half" in p and f"the least needs ${least:.2f}$ nats" in p
    served = [r for r in prot if float(r["sel_worst_n64"]) > 0]
    assert f"would serve ${len(served)}$ of the $1{{,}}049$" in p
    rates = sorted(float(r["anchor_exact"]) for r in served)
    assert rates[0] > 0  # the anchor ITSELF writes each one
    assert f"writes on ${100 * rates[0]:.1f}\\%$ to ${100 * rates[-1]:.1f}\\%$ of its draws" in p
    # the concessions: the exact string is not the event that leaked, so every vacuity share is a lower bound
    assert "the certificate is void for the event that leaked, as Proposition~\\ref{prop:selection} allows" in p
    assert "every vacuity share here is a lower bound" in p
    assert "We tested quotations at one pair, and no lyrics or whole poems" in p


def test_what_the_paragraph_says_about_the_meter_is_what_the_draws_show():
    p, prot = _para(), _protected()
    assert "yet the meter leaks none, because its prefix debt leaves the opening tokens to the anchor" in p
    assert all(float(r["meter_0.5_exact"]) == 0 for r in prot)
    assert "every protected quotation it reproduced was one its certificate did not cover" in p
    cols = [c for c in prot[0] if c.startswith("meter_") and c.endswith("_exact")]
    assert len(cols) == 5
    for c in cols:
        K = 64 * float(c.split("_")[1])
        assert all(float(r["S_anchor"]) <= K for r in prot if float(r[c]) > 0), c


def test_the_body_limitation_quotes_the_protected_stratum_and_keeps_its_concession():
    from manuscript import body
    A = _agg()
    t = body("iclr_closing.tex")
    sel, met = float(A[("protected", "sel_worst_n64")]["value"]), float(A[("protected", "meter_10_exact")]["value"])
    clause = (f"on protected quotations selection's worst case leaks ${100 * sel:.2f}\\%$ and the meter "
              f"${100 * met:.1f}\\%$ at $k=10$ (Appendix~\\ref{{app:shortworks}}); we tested no lyrics, whole poems or code")
    assert clause in t
    assert sel < float(A[("protected", "risky_exact")]["value"]) and met < float(A[("protected", "risky_exact")]["value"])


def test_no_quotation_text_is_committed():
    assert set(_rows("short_works_per_quote.csv")[0]) == {
        "row", "author", "status", "n_words", "target_words", "T_target", "S_anchor", "S_risky", "S_whole_anchor",
        "S_whole_risky", "anchor_exact", "anchor_near", "sel_worst_n1", "sel_worst_n8", "sel_worst_n64", "risky_exact",
        "risky_near", "meter_0.0649836_exact", "meter_0.0649836_near", "meter_0.5_exact", "meter_0.5_near",
        "meter_1_exact", "meter_1_near", "meter_3_exact", "meter_3_near", "meter_10_exact", "meter_10_near"}
