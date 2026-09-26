"""Score feat-216 (results/onset_prediction_tempering.md): does tempering the anchor recover selection's gain?

Reads the two judge passes analysis/matched_h2h.py wrote (results/matched_h2h_tempering_{B,G}.csv), checks the
registered gates and writes results/tempering_scoring.csv. No GPU.
  G0  each new arm covers the 500 prompts at its registered temperature and penalty, every step forced to the anchor
  G2  judge B (local A100s, comparators carried from the committed pass): comparator levels equal the committed ones;
      judge G (host B H100s, every arm re-judged, addendum): the same comparison, descriptive
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ARMS = {"t07": (0.7, 1.0), "t05": (0.5, 1.0), "t07pen": (0.7, 1.1)}
COMPARATORS = ("sel_n64", "sel_n1", "anchor_k0")


def g0(arm, temp, pen):
    recs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "output", "feat216", arm, "trajectories_k0_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    pids = {r["metadata"]["prompt_id"] for r in recs}
    ok = (len(recs) == 500 and len(pids) == 500
          and all(r["metadata"]["temperature"] == temp and r["metadata"]["repetition_penalty"] == pen
                  and r["metadata"]["k"] == 0.0 and not r["metadata"]["chat_template"]
                  and r["aggregate"]["steps_active"] == 0 and r["aggregate"]["steps_risky_unchanged"] == 0
                  for r in recs))
    empty = sum(not r["aggregate"]["generation"].strip() for r in recs)
    return ok, len(recs), empty


def rows(tag):
    return list(csv.DictReader(open(os.path.join(RES, f"matched_h2h_{tag}.csv"), encoding="utf-8")))


def main():
    out = []
    for arm, (t, p) in ARMS.items():
        ok, n, empty = g0(arm, t, p)
        out.append(dict(band="G0", judge="", quantity=f"anchor_{arm}: 500 prompts at T={t}, penalty {p}, k=0",
                        value=float(ok), lo95="", hi95="", reading="PASS" if ok else "FAIL", n=n))
        out.append(dict(band="desc", judge="", quantity=f"anchor_{arm}: empty completions", value=empty, lo95="",
                        hi95="", reading="", n=n))
    for jd in ("B", "G"):
        new = {(r["quantity"], r["arm"]): r for r in rows(f"tempering_{jd}")}
        old = {(r["quantity"], r["arm"]): r for r in rows(f"matched_plain_{jd}")}
        for c in COMPARATORS:
            a, b = float(new[("level", c)]["value"]), float(old[("level", c)]["value"])
            if jd == "B":
                reading = "PASS" if a == b else "FAIL"
            else:
                reading = f"descriptive: {a - b:+.4f} against the committed A100 pass"
            out.append(dict(band="G2", judge=jd, quantity=f"level {c}: this pass against the committed pass",
                            value=a, lo95="", hi95="", reading=reading, n=new[("level", c)]["n"]))
        for arm in list(ARMS) + ["k0"]:
            name = f"anchor_{arm}"
            r = new[("gain", f"{name} - sel_n1")]
            out.append(dict(band="desc", judge=jd, quantity=f"gain {name} over sel_n1", value=r["value"],
                            lo95=r["lo95"], hi95=r["hi95"], reading=r["reading"], n=r["n"]))
        r = new[("gain", "sel_n64 - sel_n1")]
        out.append(dict(band="desc", judge=jd, quantity="gain sel_n64 over sel_n1", value=r["value"], lo95=r["lo95"],
                        hi95=r["hi95"], reading=r["reading"], n=r["n"]))
        for arm, lab in (("t07", "D07"), ("t05", "D05"), ("t07pen", "D07p")):
            r = new[("difference", f"sel_n64 - anchor_{arm}")]
            out.append(dict(band=lab, judge=jd, quantity=f"sel_n64 - anchor_{arm} (paired levels)", value=r["value"],
                            lo95=r["lo95"], hi95=r["hi95"], reading=r["reading"], n=r["n"]))
        for arm in ARMS:
            r = new[("level", f"anchor_{arm}")]
            out.append(dict(band="desc", judge=jd, quantity=f"anchor_{arm} consistency (same verdict both orders)",
                            value=r["consistency"], lo95="", hi95="", reading="", n=r["n"]))
    with open(os.path.join(RES, "tempering_scoring.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["band", "judge", "quantity", "value", "lo95", "hi95", "reading", "n"])
        w.writeheader()
        w.writerows(out)
    for r in out:
        iv = f" [{r['lo95']}, {r['hi95']}]" if r["lo95"] != "" else ""
        print(f"{r['band']:5s} {r['judge']:1s} {r['quantity']:62s} {r['value']}{iv} {r['reading']}")


if __name__ == "__main__":
    main()
