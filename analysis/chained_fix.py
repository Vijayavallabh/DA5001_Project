"""feat-215 (results/onset_prediction_chained_fix.md): put the re-run chained arms back.

scripts/run_feat215.sh re-ran every committed chained arm of analysis/composition_attack.py with the left-pad
slicing fixed (caution (bc)) and --modes chained only, into output/chainfix/<run>. This script:
  merge    writes output/chainfix/merged/<run>: the old run's composition.csv, composition_summary.csv and
           queries.jsonl with every chained row replaced by the re-run's and every other line byte-identical (G0: the
           chained keys match exactly, and no re-run query is over budget);
  gate     R0: old against new window-0 chained texts, which never pad and so must reproduce on the same hardware
           -> results/chained_fix_reproduction.csv;
  compare  old against fixed chained recall per (run, k, L), paired over passages -> results/chained_fix.csv, and the
           registered predictions -> results/chained_fix_scoring.csv;
  apply    replaces the committed copies, splices the chained rows of bank_cap.csv and prefix_debt_ablation.csv,
           regenerates composition_70b.csv, odometer and separation with their own scripts, and refuses to write any
           file whose non-chained lines differ from the committed ones (G1).

Usage: .venv/bin/python analysis/chained_fix.py merge|gate|compare|apply
"""
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.order_averaged_h2h import paired_boot  # noqa: E402

O = "output/chainfix"
NM = ("hp1_greedy", "hp1_B", "hp1_A", "1984_greedy", "1984_B", "1984_A")
RUNS = {  # re-run name: (old run directory, committed verbatim copies {file in the run dir: committed path})
    # the phase-1 run wrote straight into results/, which `apply` replaces, so its pre-fix files are kept here
    # (`merge` copies them from results/ the first time it runs, before any apply)
    "phase1": ("output/chainfix/old/phase1", {}),
    "comp8b_kl": ("output/phase2/comp8b_kl", {"composition_summary.csv": "results/composition_8b_kl.csv",
                                             "composition.csv": "results/composition_8b_kl_per_passage.csv"}),
    "comp8b_pathwise": ("output/phase2/comp8b_pathwise",
                        {"composition_summary.csv": "results/composition_8b_pathwise.csv",
                         "composition.csv": "results/composition_8b_pathwise_per_passage.csv"}),
    "bank_cap_k10_10": ("output/phase2/bank_cap_k10_10", {}),
    "bank_cap_k10_50": ("output/phase2/bank_cap_k10_50", {}),
    "bank_cap_k20_20": ("output/phase2/bank_cap_k20_20", {}),
    "comp_comma7b": ("output/phase4/comp_comma7b", {"composition_summary.csv": "results/composition_comma7b_summary.csv",
                                                    "composition.csv": "results/composition_comma7b.csv"}),
    **{f"nm/{r}": (f"output/phase2/nm/{r}", {}) for r in NM},
}
KEYS = {"composition.csv": ("k", "mode", "L", "prompt_id"), "composition_summary.csv": ("k", "mode", "L")}


def lines(path):
    return open(path, encoding="utf-8", newline="").read().splitlines(keepends=True)


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8", newline="")))


def g1(old_path, new_path):
    """Every line of a non-chained row (and the header) is byte-identical; the row count is unchanged."""
    a, b = lines(old_path), lines(new_path)
    assert len(a) == len(b) and a[0] == b[0], f"G1 {new_path}: {len(a)} vs {len(b)} lines or a new header"
    for la, lb, r in zip(a[1:], b[1:], rows(old_path)):
        if r["mode"] != "chained":
            assert la == lb, f"G1 {new_path}: a non-chained row changed:\n{la}{lb}"


def merge_csv(old_path, new_path, keys, out_path):
    old_lines, new_lines = lines(old_path), lines(new_path)
    assert old_lines[0] == new_lines[0], f"{new_path}: header differs from {old_path}"
    new = {tuple(r[k] for k in keys): line for r, line in zip(rows(new_path), new_lines[1:])}
    assert all(k[1] == "chained" for k in new) and len(new) == len(new_lines) - 1, new_path
    old_keys = [tuple(r[k] for k in keys) for r in rows(old_path) if r["mode"] == "chained"]
    assert set(old_keys) == set(new) and len(old_keys) == len(new), f"G0 {new_path}: chained keys differ from {old_path}"
    out = [old_lines[0]] + [new[tuple(r[k] for k in keys)] if r["mode"] == "chained" else line
                            for r, line in zip(rows(old_path), old_lines[1:])]
    open(out_path, "w", encoding="utf-8", newline="").write("".join(out))
    g1(old_path, out_path)


def merge_queries(old_path, new_path, out_path):
    new = defaultdict(list)
    for line in open(new_path, encoding="utf-8"):
        q = json.loads(line)
        assert q["mode"] == "chained", new_path
        new[q["k"]].append(line)
    out, done = [], set()
    for line in open(old_path, encoding="utf-8"):
        q = json.loads(line)
        if q["mode"] != "chained":
            out.append(line)
        elif q["k"] not in done:  # the chained block of each k, at the place the old one stood
            out += new[q["k"]]
            done.add(q["k"])
    assert done == set(new), f"G0 {new_path}: chained budgets differ from {old_path}"
    open(out_path, "w", encoding="utf-8").write("".join(out))


EPS = 1e-3


def over_budget(queries, pathwise):
    """G0 from the query log: the invariant each decoder guarantees (KL spend Z for the KL decoder, the realised
    log-ratio R for the pathwise one, at k > 0)."""
    n = 0
    for q in map(json.loads, open(queries, encoding="utf-8")):
        B = max(0.0, float(q["B"]))
        n += (float(q["R"]) > B + EPS and q["k"] > 0) if pathwise else float(q["Z"]) > B + EPS
    return n


def as_committed(new, old, out, pathwise):
    """The re-run's rows in the committed file's own terms. Columns: the phase-1 files predate the constraint,
    R and window columns, so each row is projected onto the committed header (every old column exists in the new
    schema, and values pass through as the strings the run wrote). Violations: composition_8b_pathwise.csv's
    invariant_violations counts KL excursions (Z > max(0, B)), as its README says, where the current script counts
    the pathwise invariant R, so a pathwise re-run's chained cells are recounted the committed way from its query log."""
    per = defaultdict(int)
    if pathwise:
        for q in map(json.loads, open(f"{new}/queries.jsonl", encoding="utf-8")):
            per[(float(q["k"]), int(q["L"]), q["prompt_id"])] += float(q["Z"]) > max(0.0, float(q["B"])) + EPS
    for f in KEYS:
        head, rs = lines(f"{old}/{f}")[0], rows(f"{new}/{f}")
        cols = next(csv.reader([head]))
        assert set(cols) <= set(rs[0]), f"{new}/{f} lacks committed columns {set(cols) - set(rs[0])}"
        for r in rs if pathwise else []:
            key = (float(r["k"]), int(r["L"]))
            r["invariant_violations"] = str(per[key + (r["prompt_id"],)] if "prompt_id" in r
                                            else sum(v for kk, v in per.items() if kk[:2] == key))
        with open(f"{out}/{f}", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator=head[len(head.rstrip("\r\n")):])
            w.writeheader()
            w.writerows(rs)


def merge(_):
    keep = RUNS["phase1"][0]
    if not os.path.exists(f"{keep}/composition.csv"):
        os.makedirs(keep, exist_ok=True)
        for f in KEYS:
            shutil.copyfile(f"results/{f}", f"{keep}/{f}")
    for name, (old, _copies) in RUNS.items():
        new, out = f"{O}/{name}", f"{O}/merged/{name}"
        os.makedirs(out, exist_ok=True)
        pathwise = rows(f"{new}/composition.csv")[0].get("constraint", "kl") == "pathwise"
        over = over_budget(f"{new}/queries.jsonl", pathwise)
        assert over == 0, f"G0 {name}: {over} queries over budget"
        src = f"{new}/as_committed"
        os.makedirs(src, exist_ok=True)
        as_committed(new, old, src, pathwise)
        for f, keys in KEYS.items():
            merge_csv(f"{old}/{f}", f"{src}/{f}", keys, f"{out}/{f}")
        if os.path.exists(f"{old}/queries.jsonl"):
            merge_queries(f"{old}/queries.jsonl", f"{new}/queries.jsonl", f"{out}/queries.jsonl")
        print(f"[chainfix] merged {name}")
    for d in ("hp1_B_nodebt", "hp1_B_pathwise"):  # no chained rows; natural_memorisation.py reads every nm run
        dst = f"{O}/merged/nm/{d}"
        os.makedirs(dst, exist_ok=True)
        for f in ("composition_summary.csv", "queries.jsonl"):
            shutil.copyfile(f"output/phase2/nm/{d}/{f}", f"{dst}/{f}")


def gate(a):
    out = []
    for name, (old, _) in RUNS.items():
        qo, qn = f"{old}/queries.jsonl", f"{O}/{name}/queries.jsonl"
        if not os.path.exists(qo):
            continue
        w0 = lambda p: {(q["k"], q["L"], q["prompt_id"]): q["text"] for q in map(json.loads, open(p, encoding="utf-8"))  # noqa: E731
                        if q["mode"] == "chained" and q["window"] == 0}
        o, n = w0(qo), w0(qn)
        assert set(o) == set(n), name
        same = sum(o[k] == n[k] for k in o)
        by_k = defaultdict(lambda: [0, 0])
        for key in o:
            by_k[key[0]][0] += o[key] == n[key]
            by_k[key[0]][1] += 1
        out.append(dict(run=name, window0_queries=len(o), identical=same, share=round(same / len(o), 4),
                        by_k=";".join(f"{k:g}:{s}/{t}" for k, (s, t) in sorted(by_k.items()))))
    with open(f"{a.results}/chained_fix{a.tag}_reproduction.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


def compare(a):
    out = []
    for name, (old, _) in RUNS.items():
        per = defaultdict(dict)
        for tag, path in (("old", f"{old}/composition.csv"), ("new", f"{O}/{name}/composition.csv")):
            for r in rows(path):
                if r["mode"] == "chained":
                    per[(float(r["k"]), int(r["L"]))][(tag, r["prompt_id"])] = (float(r["nv_recall"]), float(r["lcs_word"]))
        for (k, L), d in sorted(per.items()):
            pids = sorted(p for t, p in d if t == "old")
            assert pids == sorted(p for t, p in d if t == "new"), (name, k, L)
            diffs = [d[("new", p)][0] - d[("old", p)][0] for p in pids]
            lo, hi = paired_boot(diffs, random.Random(215)) if len(pids) > 1 else (diffs[0], diffs[0])
            m = lambda tag, i: sum(d[(tag, p)][i] for p in pids) / len(pids)  # noqa: E731
            out.append(dict(run=name, k=k, L=L, n_passages=len(pids), nv_recall_old=round(m("old", 0), 4),
                            nv_recall_new=round(m("new", 0), 4), diff=round(sum(diffs) / len(diffs), 4),
                            lo95=round(lo, 4), hi95=round(hi, 4), lcs_word_old=round(m("old", 1), 2),
                            lcs_word_new=round(m("new", 1), 2)))
    with open(f"{a.results}/chained_fix{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    big = [r for r in out if r["n_passages"] >= 50]
    p1 = [r for r in big if r["nv_recall_new"] < r["nv_recall_old"] - 0.02]
    score = [dict(prediction="P1", scope=f"{len(big)} cells, runs with >= 50 passages",
                  reading=f"{len(p1)} cells fall by more than 0.02" + (": " + "; ".join(
                      f"{r['run']} k={r['k']:g} L={r['L']} {r['diff']:+.4f}" for r in p1) if p1 else ""),
                  verdict="RIGHT" if not p1 else "WRONG")]
    repro = f"{a.results}/chained_fix{a.tag}_reproduction.csv"  # descriptive: P1 on the arms whose R0 held (>= 95%)
    clean = {r["run"] for r in rows(repro) if float(r["share"]) >= 0.95} if os.path.exists(repro) else set()
    cells = [r for r in out if r["run"] in clean]
    if cells:
        worst = min(cells, key=lambda r: r["diff"])
        score.append(dict(prediction="P1 (descriptive)", scope=f"{len(cells)} cells of the runs whose R0 held: {', '.join(sorted(clean))}",
                          reading=f"largest fall {-min(0.0, worst['diff']):.4f} ({worst['run']} k={worst['k']:g} L={worst['L']})",
                          verdict="within 0.02" if worst["diff"] >= -0.02 else "beyond 0.02"))
    for name in RUNS:
        s = {L: sum(r["diff"] for r in big if r["run"] == name and r["L"] == L) for L in (20, 50)}
        if any(r["run"] == name and r["L"] == 20 for r in big):
            score.append(dict(prediction="P2", scope=name, reading=f"sum over k: L=20 {s[20]:+.4f}, L=50 {s[50]:+.4f}",
                              verdict="RIGHT" if s[20] > s[50] else "WRONG"))
    with open(f"{a.results}/chained_fix{a.tag}_scoring.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score[0]))
        w.writeheader()
        w.writerows(score)
    for r in out + score:
        print(r)


def splice(target, source_of):
    """Replace the shared columns of every chained row of `target` from its merged source summary row. Every other
    column and row is left alone, and before anything changes each spliced row must equal its OLD source row on those
    columns (or, on a rerun after apply, its new one), so a wrong mapping fails instead of writing."""
    header, old_rows = lines(target)[0], rows(target)
    body = lines(target)[1:]
    out = [header]
    for r, line in zip(old_rows, body):
        if r["mode"] != "chained":
            out.append(line)
            continue
        old_src, new_src = source_of(r)
        key = lambda s: (float(s["k"]), s["mode"], int(s["L"]))  # noqa: E731
        o = next(s for s in rows(old_src) if key(s) == key(r))
        n = next(s for s in rows(new_src) if key(s) == key(r))
        shared = [c for c in r if c in o and c not in ("k", "mode", "L", "run")]
        same = lambda src: all(float(r[c]) == float(src[c]) for c in shared if r[c] != "")  # noqa: E731
        assert same(o) or same(n), f"{target}: {r} is a copy of neither {old_src} nor {new_src}"  # n: already applied
        vals = dict(r, **{c: n[c] for c in shared})
        buf = __import__("io").StringIO()
        csv.DictWriter(buf, fieldnames=list(r), lineterminator=line[len(line.rstrip("\r\n")):]).writerow(vals)
        out.append(buf.getvalue())
    return "".join(out)


def apply(a):
    staged = f"{O}/staged"
    os.makedirs(staged, exist_ok=True)
    plan = []  # (staged file, committed path)
    for name, (old, copies) in RUNS.items():
        for f, committed in copies.items():
            plan.append((f"{O}/merged/{name}/{f}", committed))
    plan += [(f"{O}/merged/phase1/composition.csv", "results/composition.csv"),
             (f"{O}/merged/phase1/composition_summary.csv", "results/composition_summary.csv")]
    src = lambda run: (f"{RUNS[run][0]}/composition_summary.csv", f"{O}/merged/{run}/composition_summary.csv")  # noqa: E731
    open(f"{staged}/bank_cap.csv", "w", newline="").write(splice("results/bank_cap.csv", lambda r: src(r["run"])))
    open(f"{staged}/prefix_debt_ablation.csv", "w", newline="").write(splice(
        "results/prefix_debt_ablation.csv", lambda r: src({"comp8b_kl": "comp8b_kl", "hp1_B": "nm/hp1_B"}[r["run"]])))
    plan += [(f"{staged}/bank_cap.csv", "results/bank_cap.csv"),
             (f"{staged}/prefix_debt_ablation.csv", "results/prefix_debt_ablation.csv")]
    py = sys.executable
    subprocess.run([py, "analysis/natural_memorisation.py", "--runs", f"{O}/merged/nm", "--out", staged], check=True)
    subprocess.run([py, "analysis/odometer.py", "--queries", f"{O}/merged/comp8b_kl/queries.jsonl", "--out", staged],
                   check=True)
    shutil.copyfile("results/per_trajectory.csv", f"{staged}/per_trajectory.csv")
    subprocess.run([py, "analysis/separation.py", "--results", staged, "--out", staged, "--figures", ""], check=True)
    for f in ("composition_70b.csv", "natural_memorisation.csv", "odometer.csv", "odometer_per_passage.csv",
              "separation.csv", "separation_summary.csv"):
        plan.append((f"{staged}/{f}", f"results/{f}"))
    for new, committed in plan:  # G1 on everything before anything is written
        if "mode" in rows(committed)[0]:
            g1(committed, new)
        else:
            assert open(new, "rb").read() == open(committed, "rb").read(), f"G1 {committed}: a mode-free file changed"
    for new, committed in plan:
        shutil.copyfile(new, committed)
        print(f"[chainfix] wrote {committed}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("merge", "gate", "compare", "apply"))
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="", help="suffix of the gate and compare files (the A100 re-run pass: _a100)")
    a = ap.parse_args()
    {"merge": merge, "gate": gate, "compare": compare, "apply": apply}[a.cmd](a)


if __name__ == "__main__":
    main()
