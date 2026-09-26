"""Score feat-217 (results/onset_prediction_promptblind.md): selection with a vetted, prompt-blind scorer. No GPU.
Reads results/matched_h2h_promptblind_{B,G}.csv and the committed passes the comparators were carried from; writes
results/promptblind_scoring.csv."""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(path):
    return {(r["quantity"], r["arm"]): r for r in csv.DictReader(open(path, encoding="utf-8"))}


def main():
    rw = list(csv.DictReader(open(os.path.join(RES, "selection_rewards64_promptblind.csv"), encoding="utf-8")))
    by = {}
    for r in rw:
        by.setdefault(r["prompt_id"], []).append(float(r["reward"]))
    all_empty = sum(1 for v in by.values() if max(v) <= -1e8)
    out = [dict(band="G0", judge="", quantity="cache: 500 prompts x 64 ranks; prompts whose 64 draws are all empty",
                value=all_empty, lo95="", hi95="",
                reading="PASS" if len(rw) == 32000 and len(by) == 500 else "FAIL", n=len(by))]
    refs = {"B": os.path.join(RES, "matched_h2h_matched_plain_B.csv"),
            "G": os.path.join(RES, "matched_h2h_tempering_G.csv")}
    for jd in ("B", "G"):
        P, Q = load(os.path.join(RES, f"matched_h2h_promptblind_{jd}.csv")), load(refs[jd])
        for c in ("sel_n64", "sel_n1"):
            a, b = float(P[("level", c)]["value"]), float(Q[("level", c)]["value"])
            out.append(dict(band="G2", judge=jd, quantity=f"level {c} against the pass it was carried from", value=a,
                            lo95="", hi95="", reading="PASS" if a == b else "FAIL", n=P[("level", c)]["n"]))
        lv = P[("level", "sel_pb")]
        out.append(dict(band="G0", judge=jd, quantity="sel_pb: empty texts served (must equal the all-empty prompts)",
                        value=lv["n_empty"], lo95="", hi95="",
                        reading="PASS" if int(lv["n_empty"]) == all_empty else "FAIL", n=lv["n"]))
        g = P[("gain", "sel_pb - sel_n1")]
        rd = {"CONFIRMED": "RETAINS", "UNRESOLVED": "NONE", "REFUTED": "HURTS"}[g["reading"]]
        out.append(dict(band="gain", judge=jd, quantity="gain sel_pb over sel_n1", value=g["value"], lo95=g["lo95"],
                        hi95=g["hi95"], reading=rd, n=g["n"]))
        g = P[("gain", "sel_n64 - sel_n1")]
        out.append(dict(band="desc", judge=jd, quantity="gain sel_n64 over sel_n1 (the unvetted Qwen scorer)",
                        value=g["value"], lo95=g["lo95"], hi95=g["hi95"], reading=g["reading"], n=g["n"]))
        d = P[("difference", "sel_n64 - sel_pb")]
        out.append(dict(band="DQ", judge=jd, quantity="sel_n64 - sel_pb", value=d["value"], lo95=d["lo95"],
                        hi95=d["hi95"], reading=d["reading"], n=d["n"]))
    with open(os.path.join(RES, "promptblind_scoring.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["band", "judge", "quantity", "value", "lo95", "hi95", "reading", "n"])
        w.writeheader()
        w.writerows(out)
    for r in out:
        iv = f" [{r['lo95']}, {r['hi95']}]" if r["lo95"] != "" else ""
        print(f"{r['band']:5s} {r['judge']:1s} {r['quantity']:70s} {r['value']}{iv} {r['reading']}")


if __name__ == "__main__":
    main()
