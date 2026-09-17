"""The burstiness correlation is quoted in four places, and two of them had the wrong denominator.

rho = +0.036 is Spearman over the SEVEN pairs the pre-registration committed to
(results/onset_prediction_burstiness.md: "Over the seven pairs of the main onset table"). Over all
nine it is -0.050 -- a different number with a different sign. appendix_robustness and
appendix_limitations said so; appendix_proofs and appendix_onset both attributed +0.036 to "the nine
pairs". Found in the fourth read-through, 2026-09-17.

Nothing caught it because +0.036 IS in a CSV-derived quantity and audit_numbers.py only asks
whether a literal appears somewhere -- the defect was the pair count attached to it, which is the
caution (ai) shape again. This test recomputes both correlations from the CSV and checks that every
place quoting either one names the right count.
"""
import csv
import glob
import re

from tests.manuscript import tex

CSV = "results/onset_burstiness.csv"
MANIFEST = "results/onset_pairs.tsv"
COMMITTED_N = 7          # the pre-registration's own set, the first seven added


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def _rho(n=None):
    rows = {r["pair"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}
    order = [l.split("\t")[0] for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    sub = [rows[p] for p in (order[:n] if n else order) if p in rows]
    return _spearman([float(r["burstiness"]) for r in sub], [float(r["ratio"]) for r in sub])


def test_the_two_correlations_are_the_ones_the_csv_gives():
    assert round(_rho(COMMITTED_N), 3) == 0.036, _rho(COMMITTED_N)
    assert round(_rho(), 3) == -0.050, _rho()


def test_no_section_attaches_the_committed_rho_to_the_wrong_pair_count():
    """+0.036 may never be described as a nine-pair result, and -0.050 never as a seven-pair one."""
    seven = f"{_rho(COMMITTED_N):+.3f}".replace("+0.036", "+0.036")
    nine = f"{_rho():+.3f}"
    bad = []
    for f in sorted(glob.glob(tex("sections/*.tex"))):
        if re.search(r"_v\d", f):
            continue
        flat = " ".join(open(f, encoding="utf-8").read().split())
        for m in re.finditer(re.escape(f"$\\rho = {seven}$"), flat):
            window = flat[max(0, m.start() - 160):m.start()]
            if "nine" in window and "seven" not in window:
                bad.append((f, "+0.036 called a nine-pair result", window[-110:]))
        for m in re.finditer(re.escape(nine), flat):
            # the count for -0.050 follows it ("-0.050 over all nine"), so look FORWARD; looking
            # back only finds the correct "+0.036 over the seven ... and -0.050 over all nine"
            window = flat[m.end():m.end() + 90]
            if "seven" in window and "nine" not in window:
                bad.append((f, "-0.050 called a seven-pair result", window[:110]))
    assert not bad, bad
