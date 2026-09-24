"""The two reviewer-driven arms of 2026-09-16, pinned to their CSVs rather than read by eye.

Both tables were hand-built from a scoring log, which is exactly how six of seventy-two cells in
`order_law.csv` once went one out in the last digit (caution (j)): the log rounds, and a table
transcribed from a rounded log rounds a second time. Two of the four confidence intervals here were
wrong that way on their first draft. So every cell is generated from the CSV and asserted as a
substring of the section source.

The latency test also guards the claim the measurement overturned. The FLOP model says the reward
model is the price; the clock says it is 9.3% of it. Anywhere the manuscript still asserted the
FLOP version unqualified, a reader would take away the opposite of what was measured."""
import csv

from tests.manuscript import tex

APP = tex("sections/appendix_selection.tex")

H2H = [("seed $42$ (original) & B", "results/order_averaged_h2h.csv"),
       ("seed $42$            & C", "results/order_averaged_h2h_seed42_judgeC.csv"),
       ("seed $52$ (fresh)    & B", "results/order_averaged_h2h_seed52_judgeB.csv"),
       ("seed $52$ (fresh)    & C", "results/order_averaged_h2h_seed52_judgeC.csv")]


def _q(path):
    return {r["quantity"]: r for r in csv.DictReader(open(path))}


def test_every_row_of_the_repeat_table_comes_from_its_own_csv():
    body = open(APP, encoding="utf-8").read()
    for lead, path in H2H:
        q = _q(path)
        d1 = float(q["D1 selection gain, order-averaged"]["value"])
        d2 = float(q["D2 metered gain, order-averaged"]["value"])
        d3 = q["D3 difference of gains, paired"]
        cell = "${:+.4f}$ & ${:+.4f}$ &".format(d1, d2)
        ci = "$[{:+.4f},{:+.4f}]$".format(float(d3["lo95"]), float(d3["hi95"]))
        assert lead in body, f"the repeat table lost its {lead!r} row"
        row = next(l for l in body.splitlines() if l.startswith(lead))
        assert cell in row, (lead, cell, row)
        assert ci in row, (lead, ci, row)
        assert "{:+.4f}".format(float(d3["value"])) in row, (lead, d3["value"], row)


def test_all_three_repeats_confirm_or_the_headline_is_not_allowed_to_stand():
    """The committed rule (results/onset_prediction_h2h_independent.md) was that the headline stands
    as written only if all three new estimates are positive with intervals excluding zero. Assert
    the data still says that, so a later re-run cannot silently leave the claim behind."""
    body = " ".join(open(APP, encoding="utf-8").read().split())
    # v10 (2026-09-24) states the outcome in Section 4 rather than as an appendix sentence about the
    # registered condition: "The difference survives fresh draws, judges and controls". That is the
    # headline being allowed to stand, so it is what the data must license.
    main = " ".join(open(tex("sections/experiments.tex"), encoding="utf-8").read().split())
    ok = True
    for _, path in H2H[1:]:
        d3 = _q(path)["D3 difference of gains, paired"]
        ok &= float(d3["value"]) > 0 and float(d3["lo95"]) > 0
    if not ok:
        assert "stand as written" not in body and "All three new estimates are positive" not in body
        assert "survives fresh draws" not in main, "Section 4 claims a replication the data refuses"
    else:
        assert "The difference survives fresh draws" in main


def test_the_latency_table_rounds_from_serving_latency_csv():
    """Row by row, the numbers in the table are the CSV's own, rounded once. Parsed out of the row
    rather than matched as a fixed string, because the table is space-aligned."""
    import re
    lines = open(APP, encoding="utf-8").read().splitlines()
    q = _q("results/serving_latency.csv")
    for quantity, lead in [("selection n=64, anchor draws", "selection $n=64$, the $64$ anchor"),
                           ("selection n=64, reward pass", "selection $n=64$, the reward pass"),
                           ("selection n=64, total", "selection $n=64$, total"),
                           ("metered k=10", "metered $k=10$")]:
        row = next((l for l in lines if l.startswith(lead)), None)
        assert row, f"the latency table lost its {quantity!r} row"
        got = [float(x) for x in re.findall(r"\$([\d.]+)\$", row.split("&", 1)[1])]
        r = q[quantity]
        want = [float(x) for x in r["reps"].split()] + [
            round(float(r["seconds"]), 2), round(float(r["per_served_token"]), 4)]
        assert got == want, (quantity, got, want)


def test_the_measured_ratio_and_the_reward_share_are_the_csv_s():
    body = " ".join(open(APP, encoding="utf-8").read().split())
    q = _q("results/serving_latency.csv")
    for key, fmt in [("ratio, measured", "${:.1f}\\times$"),
                     ("ratio if the scorer were free", "${:.1f}\\times$")]:
        lit = fmt.format(float(q[key]["per_served_token"]))
        assert lit in body, (key, lit)
    share = float(q["reward share of selection wall-clock"]["per_served_token"])
    assert "${:.1f}\\%$".format(share * 100) in body, share
    assert "${:.1f}\\%$".format(100 - share * 100) in body, "the draws' share is not stated"
    assert "$8{,}179$" in body, "the shared denominator is not quoted"


def test_the_flop_cost_claim_is_never_asserted_without_naming_its_currency():
    """`61.3x` is a FLOP proxy, and the measurement showed it is 1.73x pessimistic as a price and
    backwards about where the price goes. Every surviving statement that the scorer is the price
    must therefore say it is about FLOPs, or a reader takes away the opposite of what was
    measured."""
    # v10 (2026-09-24) dropped both appendix claims this list used to hold ("price of putting a
    # reward model in the loop", "is set by the reward model"); the one surviving statement that the
    # scorer is the price is Section 4.5's, whose currency the paper now names "forward passes"
    # (Table tab:cost: "forward passes, parameters x tokens", the same FLOP proxy).
    files = {"most of it the reward model's prefill": "sections/experiments.tex"}
    for claim, f in files.items():
        body = " ".join(open(tex(f), encoding="utf-8").read().split())
        i = body.find(claim)
        assert i > 0, f"{claim!r} is gone; drop it from this test"
        # A tight window on purpose: the currency has to be named in the clause that makes the
        # claim. A generous one passes vacuously off the "forward-pass FLOPs" two sentences later,
        # which is what a reader skimming the claim itself would never see.
        near = body[max(0, i - 70): i + 70]
        assert "FLOP" in near or "forward pass" in near, \
            f"{claim!r} is asserted without naming FLOPs as its currency: {near!r}"
