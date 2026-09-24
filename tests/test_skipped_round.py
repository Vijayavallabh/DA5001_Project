"""Guards for the round that ran the review points skipped on 2026-09-24 (feat-187..193 and the notes).

Every number the new paragraphs print is re-derived here from the CSV or the arithmetic that produced
it, and each guard reads its own paragraph by label, not the file at large (caution (an)).
"""
import csv
import math
import os
import re

import pytest

from tests.manuscript import body

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def para(label, *files):
    """The whitespace-normalised paragraph that carries \\label{label}, up to the next heading."""
    t = body(*files)
    i = t.index(f"\\label{{{label}}}")
    j = min([k for k in (t.find("\\paragraph{", i), t.find("\\section", i), t.find("\\subsection", i))
             if k > 0] or [len(t)])
    return t[i:j]


def rows(name):
    return list(csv.DictReader(open(os.path.join(RES, name), encoding="utf-8")))


def test_the_per_work_certificate_arithmetic():
    p = para("app:perwork", "appendix_selection.tex")
    assert "$19{,}200$" in p and round(3 * 64 / 0.01) == 19200
    assert "$3 \\times 64/25{,}600 = 0.75\\%$" in p and abs(3 * 64 / 25600 - 0.0075) < 1e-12
    assert "$e^{100} \\approx 2.7 \\times 10^{43}$" in p and 2.65e43 < math.exp(100) < 2.75e43
    assert "$S_w = 159.8$" in p and "$10^{-60}$" in p and "$m=10^4$" in p
    assert math.log10(64 * 1e4) - 159.8 / math.log(10) < -60


def test_the_pools_the_per_work_bound_rests_on_have_no_event():
    r = [x for x in rows("rouge_threshold.csv")
         if x["file"] == "selfix_clean_n256_per_passage.csv" and x["source"] == "pool maximum"]
    assert r and all(int(x["count"]) == 0 and int(x["n"]) == 100 for x in r)
    assert {x["theta"] for x in r} >= {"0.3", "0.5", "0.7"}


def test_the_no_separation_remark_names_its_counterexamples():
    p = para("app:noseparation", "appendix_proofs.tex")
    for s in ("false as a statement about laws", "V_t = \\mathbb{E}_{p_s}[e^{\\lambda U} \\mid y_{\\le t}]",
              "information", "Proposition~\\ref{prop:imitation}"):
        assert s in p, s


def test_rouge_threshold_clean_anchors_zero_at_every_theta():
    r = [x for x in rows("rouge_threshold.csv") if x["kind"] == "clean" and x["source"] != "memoriser alone"]
    assert r and all(int(x["count"]) == 0 for x in r)
    mem = {x["theta"]: int(x["count"]) for x in rows("rouge_threshold.csv")
           if x["file"] == "selfix_clean_grid64_per_passage.csv" and x["source"] == "memoriser alone"}
    assert mem["0.5"] == 47 and mem["0.3"] >= mem["0.5"] >= mem["0.7"]


def test_the_covert_channel_never_beats_its_certificate():
    for r in rows("covert_channel.csv"):
        n = int(r["n"])
        assert float(r["bits_per_response"]) < math.log2(n)
        assert int(r["odometer_max_responses"]) == int(400 // math.log(n))
    r64 = next(r for r in rows("covert_channel.csv") if r["n"] == "64")
    assert abs(float(r64["efficiency"]) - float(r64["bits_per_response"]) / 6) < 1e-3


def test_adaptive_rules_are_certified_at_log_n_max():
    for r in rows("adaptive_n.csv"):
        assert abs(float(r["certificate_nats"]) - math.log(int(r["n_max"]))) < 1e-4
        assert 1 <= float(r["mean_draws"]) <= int(r["n_max"])
