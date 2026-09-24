"""results/deecho_rejudge_note.md: the survival rule is the interval's sign, nothing else."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.deecho_rejudge import PASSES, score, sign  # noqa: E402


def test_sign_rule():
    assert sign(0.001, 0.2) == "POSITIVE"
    assert sign(-0.2, -0.001) == "NEGATIVE"
    assert sign(0.0, 0.1) == "COVERS ZERO" and sign(-0.1, 0.0) == "COVERS ZERO"


def _write(d, sfx, d3):
    with open(os.path.join(d, f"order_averaged_h2h{sfx}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "arm", "value", "lo95", "hi95", "n", "single_order", "reading"])
        w.writerow(["D1 selection gain, order-averaged", "", 0.1, 0.05, 0.15, 500, "", ""])
        w.writerow(["D2 metered gain, order-averaged", "", 0.05, 0.01, 0.09, 500, "", ""])
        w.writerow(["D3 difference of gains, paired", "", *d3, 500, "", ""])


def test_survives_iff_sign_unchanged(tmp_path):
    p, j, old, new = PASSES[0]
    _write(tmp_path, old, (0.06, 0.01, 0.11))
    _write(tmp_path, new, (0.02, -0.01, 0.05))          # moved into zero: must NOT survive
    d3 = [r for r in score(str(tmp_path)) if r["quantity"].startswith("D3")]
    assert len(d3) == 1 and d3[0]["survives"] == "NO" and d3[0]["reading_deecho"] == "COVERS ZERO"
    _write(tmp_path, new, (0.03, 0.001, 0.06))          # smaller but still positive: survives
    d3 = [r for r in score(str(tmp_path)) if r["quantity"].startswith("D3")]
    assert d3[0]["survives"] == "YES"


def test_no_new_suffix_overwrites_a_committed_file():
    assert all(new.endswith("_deecho") and new != old for _, _, old, new in PASSES)
