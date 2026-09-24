"""The 500 judged prompts, described: composition, length, source and difficulty (AC report, E5).

The AC asked for "composition, length distribution, complexity, and source domains" of the
in-house workload the paper's headline is judged on. Composition and sources were stated in
Section 3.1; the rest was not. Everything here is read off the committed files, restricted to
EXACTLY the prompt ids the headline pass judged (results/order_averaged_h2h_per_prompt.csv) --
not to the prompt files, which hold 850 and of which 500 were judged.

Complexity is reported as two measured quantities rather than a label: the length of the answer
the unconstrained opponent gives (what the task demands), and the anchor's order-averaged win rate
against that answer (how hard the task is for the model selection draws from). Lengths are words
of the task text; two classes are served with the fixed header `Complete the prefix:`
(dap/shared.py), which is counted separately so the length describes the task and not the wrapper.

Overlap with the protected works (a referee question, 2026-09-23) is screened two ways, both from
the committed corpus rather than from a list anyone typed: a prompt counts if it shares any word
8-gram with any protected passage (analysis/blocklist.py's index), or if it names any of the
sixteen works by the title recorded in data/copybench_*.jsonl.

Usage:
  .venv/bin/python analysis/prompt_set_profile.py --out results
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.blocklist import build_index, ngrams, toks  # noqa: E402
from analysis.selection_decoding import load_baseline  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402

HEADER = "Complete the prefix:"
FORM = {"neutral": ("general-knowledge question", "written for this project"),
        "factual": ("biography request, `Tell me a bio of X'", "FActScore entities"),
        "creative": ("story premise", "r/WritingPrompts")}


def quartiles(xs):
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], statistics.median(xs), q[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--judged", default="results/order_averaged_h2h_per_prompt.csv")
    ap.add_argument("--opponent", default="output/sweep_plain")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    judged = {r["prompt_id"]: float(r["u_anchor_k0"])
              for r in csv.DictReader(open(a.judged, encoding="utf-8"))}
    corpus = {p.prompt_id: p for p in load_prompt_corpus(a.data, "factscore_prompt")}
    missing = sorted(set(judged) - set(corpus))
    assert not missing, f"{len(missing)} judged ids are not in the corpus, e.g. {missing[:3]}"
    opp = load_baseline(a.opponent)
    assert set(judged) <= set(opp), "the opponent directory does not cover the judged prompts"

    protected = sorted(glob.glob(os.path.join(a.data, "copybench_*.jsonl")))
    idx8 = build_index(protected, 8)
    titles = sorted({json.loads(line)["source_novel"].replace("_", " ").lower()
                     for p in protected for line in open(p, encoding="utf-8")})
    assert len(titles) == 16, titles

    rows = []
    for cls in ("neutral", "factual", "creative"):
        ids = [p for p in judged if corpus[p].split == cls]
        texts = [corpus[p].prompt_text for p in ids]
        headed = sum(1 for t in texts if t.startswith(HEADER))
        words = [len(t[len(HEADER):].split() if t.startswith(HEADER) else t.split())
                 for t in texts]
        resp = [len(opp[p].split()) for p in ids]
        q1, med, q3 = quartiles(words)
        r1, rmed, r3 = quartiles(resp)
        rows.append(dict(
            cls=cls, form=FORM[cls][0], source=FORM[cls][1], n=len(ids),
            served_with_header=headed,
            prompt_words_median=med, prompt_words_q1=q1, prompt_words_q3=q3,
            prompt_words_min=min(words), prompt_words_max=max(words),
            opponent_words_median=rmed, opponent_words_q1=r1, opponent_words_q3=r3,
            anchor_win=round(statistics.fmean(judged[p] for p in ids), 4),
            shares_8gram_with_protected=sum(1 for x in texts if ngrams(toks(x), 8) & idx8),
            names_a_protected_title=sum(1 for x in texts if any(ti in x.lower() for ti in titles))))
    assert sum(r["n"] for r in rows) == len(judged), "a judged prompt fell outside the classes"
    for r in rows:
        print(f"{r['cls']:<9} n={r['n']:<4} header {r['served_with_header']:>3}/{r['n']:<4}"
              f" prompt words {r['prompt_words_median']:.0f} [{r['prompt_words_q1']:.0f},"
              f" {r['prompt_words_q3']:.0f}] range {r['prompt_words_min']}-{r['prompt_words_max']}"
              f"   opponent {r['opponent_words_median']:.0f} [{r['opponent_words_q1']:.0f},"
              f" {r['opponent_words_q3']:.0f}]   anchor wins {r['anchor_win']:.4f}"
              f"   8-gram overlap {r['shares_8gram_with_protected']}"
              f"   title mentions {r['names_a_protected_title']}")
    out = os.path.join(a.out, "prompt_set_profile.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
