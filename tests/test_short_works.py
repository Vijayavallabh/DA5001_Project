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
