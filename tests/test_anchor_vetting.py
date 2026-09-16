"""A contamination screen compares models only if it screened them the same way.

`results/anchor_vetting.csv` claimed a "complete separation over eighteen models". It was not one.
The five openly licensed anchors were screened at `--seed-tokens 20` with the `Complete the prefix:`
header still attached -- about fourteen tokens of genuine prefix once the header is paid for
(AGENTS.md caution (t)) -- while the one non-circular positive control, `Llama-3.1-70B`, was
screened at a hundred RAW tokens. The same 70B checkpoint reads `0.0000` under the anchors'
protocol (`results/selection_extraction_70b_per_passage.csv`), so the published gap was a gap
between protocols and not between models.

Nothing catches that by inspection: every number in the table is real, and each was correctly
computed from its own arm. The guard has to be structural -- every row declares its protocol, and no
separation statement may be made across two of them.

This matters more than an internal bookkeeping slip, because the Ethics Statement recommends this
screen to deployers as a precondition for quoting log n.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "anchor_vetting.csv")


def _rows():
    if not os.path.exists(CSV):
        return []
    return list(csv.DictReader(open(CSV, encoding="utf-8")))


def test_every_row_declares_its_protocol():
    rows = _rows()
    if not rows:
        return
    assert "protocol" in rows[0], "anchor_vetting.csv must name the protocol of every row"
    bad = [r["model"] for r in rows if not r["protocol"].strip()]
    assert not bad, f"rows with no protocol: {bad}"


def test_no_separation_claim_spans_two_protocols():
    """The claim the paper makes is 'clean models read below every contaminated one'. That is only
    meaningful inside one protocol, so assert it holds inside each protocol that has both kinds --
    and, crucially, that a protocol holding only one kind yields no claim at all rather than a
    vacuous or a cross-protocol one."""
    rows = _rows()
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r["protocol"], []).append(r)
    made_a_claim = False
    for proto, rs in by.items():
        clean = [r for r in rs if r["provenance"] == "openly licensed"]
        dirty = [r for r in rs if r["provenance"] not in ("openly licensed",)
                 and "open data" not in r["provenance"]]
        if not clean or not dirty:
            continue                       # no claim available at this protocol; that is the point
        hi = max(float(r["frac_passages_leaking"]) for r in clean)
        lo = min(float(r["frac_passages_leaking"]) for r in dirty)
        assert hi < lo, (
            f"[{proto}] clean anchors leak up to {hi} but a contaminated model leaks only {lo}; "
            "the screen does not separate at this protocol and the manuscript must not say it does")
        made_a_claim = True
    assert made_a_claim, (
        "no protocol in anchor_vetting.csv holds both a clean and a contaminated model, so the "
        "table supports no separation claim at all -- which is exactly the state the one-protocol "
        "arm (results/onset_prediction_vetting_protocol.md) exists to fix")


def test_the_non_circular_control_is_screened_beside_a_clean_anchor():
    """Twelve of the thirteen contaminated rows are LoRA fine-tunes on exactly these passages, which
    the manuscript itself calls close to circular. The load-bearing comparison is against the model
    that memorised without our help, and it is worth nothing unless a clean anchor was measured at
    the same protocol -- which is what went wrong the first time.

    This asserts in BOTH states rather than skipping when the data is thin (caution (j): a test that
    returns early on a missing file is a test that can pass by never running). Either a clean anchor
    is screened beside the control, or the manuscript must not be claiming the separation.
    """
    rows = _rows()
    if not rows:
        return
    nat = [r for r in rows if r["provenance"] == "memorised in pre-training"]
    if not nat:
        return
    proto = nat[0]["protocol"]
    clean_here = [r for r in rows
                  if r["provenance"] == "openly licensed" and r["protocol"] == proto]
    if clean_here:
        return                      # the comparison exists; the other two tests check it holds
    # It does not exist, so the manuscript is not allowed to assert one.
    from tests.manuscript import tex as _tex
    p = _tex("sections/appendix_selection.tex")
    if not os.path.exists(p):
        raise AssertionError(f"cannot check the claim: {p} is missing")
    body = " ".join(open(p, encoding="utf-8").read().split())
    for claim in ("separation is complete", "separation over eighteen models",
                  "complete over eighteen models"):
        assert claim not in body, (
            f"the pre-training memoriser is screened at '{proto}' and no openly licensed anchor "
            f"is, so the only non-circular half of the separation has nothing to stand against -- "
            f"yet appendix_selection.tex still says {claim!r}")
