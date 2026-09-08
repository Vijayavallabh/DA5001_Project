"""feat-057: the paper's onset table must come from the evidence, including the rule that a
re-measured pair is tabulated at its higher n."""
import csv, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def test_table_matches_the_ci_file(tmp_path):
    ci = tmp_path / "onset_ci.csv"
    ci.write_text(
        "pair,mode,n_passages,onset_point,onset_lo95,onset_hi95,boot_no_crossing_pct,k_grid\n"
        "A + mem. A,single,100,2.00,1.8,2.4,0.0,1 2 3\n"
        "A + mem. A (n=458),single,458,2.40,2.2,2.8,0.1,1 2 3\n"
        "B + mem. B,single,100,1.00,0.9,1.2,0.0,1 2 3\n")
    th = tmp_path / "onset_theory.csv"
    th.write_text("pair,s_safe_median,s_risky_median\nA + mem. A,4.0,0.4\nB + mem. B,2.0,0.2\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(REPO, "analysis/onset_table.py"),
                    "--ci", str(ci), "--theory", str(th), "--out", str(out)],
                   check=True, capture_output=True)
    rows = list(csv.DictReader(open(out / "onset_table.csv")))
    body = {r["pair"]: r for r in rows if not r["pair"].startswith("ALL")}
    assert body["A + mem. A"]["n_passages"] == "458"          # the larger measurement wins
    assert body["A + mem. A"]["onset"] == "2.4"
    assert body["A + mem. A"]["ratio"] == "0.6"               # 2.4 / 4.0
    assert body["B + mem. B"]["ratio"] == "0.5"
    summary = [r for r in rows if r["pair"].startswith("ALL")][0]
    assert summary["ratio"] == "0.55"                        # mean of 0.6 and 0.5
    assert abs(float(summary["boot_no_crossing_pct"]) - 0.05) < 1e-9   # sd carried here
