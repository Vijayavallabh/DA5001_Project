"""feat-202 (results/onset_prediction_factscore_oracle.md): the best biography selection could serve, and
a scorer that asks for facts.

  --dump        (CPU) the 64 de-echoed draws of the headline pool's 150 biography prompts, in
                analysis/he_metrics.py's row format (arm d<rank>) -> output/he_metrics/pool64_factual.jsonl;
                then analysis/he_metrics.py --factscore --dump <that> --tag _pool64 scores every draft
  --factuality  (GPU) the committed scorer model and path with the question asking for accurate facts
                -> results/factuality_rewards64.csv (prompt_id, rank, reward)
  --report      (CPU) gates G0-G1 and readings F1-F3 -> results/factscore_oracle.csv
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POOL = "output/phase5/sel_anchor64"
DUMP = "output/he_metrics/pool64_factual.jsonl"
FACT_TMPL = """You are rating a single candidate response to an instruction.

Instruction: {prompt}

Response: {completion}

Is every factual claim in this response accurate? Answer Yes or No.
Answer:"""


def pick(r):
    """argmax, ties to the lowest index: the committed replay's rule."""
    return max(range(len(r)), key=lambda i: r[i])


def dump(out):
    from analysis.he_metrics import fact_meta
    from analysis.order_averaged_h2h import true_prompts
    from analysis.selection_decoding import load_candidates
    cands, prompts, fm = load_candidates(POOL, deecho=True), true_prompts("data"), fact_meta()
    pids = sorted(p for p in cands if p in fm)
    assert len(pids) == 150 and all(len(cands[p]) == 64 for p in pids), len(pids)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for p in pids:
            ent, wiki = fm[p]
            for rank, (_, cls, _, text) in enumerate(cands[p]):
                fh.write(json.dumps(dict(arm=f"d{rank:02d}", spec="pool64", prompt_id=p, cls=cls,
                                         prompt=prompts[p], text=text, entity=ent, wiki=wiki)) + "\n")
    print(f"wrote {out} ({len(pids) * 64} rows)")


def factuality(dump_path, out, model_id, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from analysis.selection_scaling import score_rewards
    assert not os.path.exists(out), f"{out} exists; a reward cache is never overwritten"
    rows = [json.loads(line) for line in open(dump_path, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    sc = score_rewards(m, tok, [(r["prompt"], r["text"]) for r in rows], m.device, batch_size=batch_size,
                       tmpl=FACT_TMPL)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_id", "rank", "reward"])
        for r, s in zip(rows, sc):
            w.writerow([r["prompt_id"], int(r["arm"][1:]), round(s, 5)])
    print(f"wrote {out}")


def boot(v, rng, B=10000):
    n = len(v)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def report(a):
    from analysis.selection_scaling import load_rewards
    per = {}
    for r in csv.DictReader(open(os.path.join(a.results, "he_factscore_per_item_pool64.csv"))):
        prec = float(r["precision"]) if r["precision"] != "" else None
        per.setdefault(r["prompt_id"], {})[int(r["arm"][1:])] = (prec, int(r["n_facts"]))
    pids = sorted(per)
    rows, rng = [], random.Random(202)

    def add(band, quantity, value, lo="", hi="", n="", reading=""):
        rows.append(dict(band=band, quantity=quantity, value=value, lo95=lo, hi95=hi, n=n, reading=reading))

    ok0 = len(pids) == 150 and all(len(per[p]) == 64 for p in pids)
    fr = load_rewards(os.path.join(a.results, "factuality_rewards64.csv"))
    ok0 = ok0 and set(fr) == set(pids) and all(len(fr[p]) == 64 for p in pids)
    add("G0", "150 prompts x 64 drafts scored, 9,600 factuality rewards", float(ok0),
        reading="PASS" if ok0 else "FAIL")

    qw = load_rewards(os.path.join(a.results, "selection_rewards64.csv"))
    qpick = {p: pick(qw[p][:64]) for p in pids}
    fpick = {p: pick(fr[p][:64]) for p in pids}
    drafts = {}
    for line in open(a.dump_path, encoding="utf-8"):
        r = json.loads(line)
        drafts[(r["prompt_id"], int(r["arm"][1:]))] = r["text"]
    sel64 = {}
    for line in open(a.arms, encoding="utf-8"):
        r = json.loads(line)
        if r["arm"] == "sel64":
            sel64[r["prompt_id"]] = r["text"]
    same = sum(drafts[(p, qpick[p])] == sel64.get(p) for p in pids)
    add("G1", "committed pick's text equals the committed FActScore pass's sel64 text", same, n=len(pids),
        reading="PASS" if same == len(pids) else "FAIL")

    def oracle(n, min_facts=0):
        best = [max((v[0] for r, v in per[p].items() if r < n and v[0] is not None and v[1] >= min_facts),
                    default=None) for p in pids]
        best = [b for b in best if b is not None]
        return best

    for n in (1, 8, 64):
        for mf in (0, 5):
            v = oracle(n, mf)
            lo, hi = boot(v, rng)
            add("F1", f"oracle precision, first {n} drafts" + (f", drafts with >= {mf} facts" if mf else ""),
                round(sum(v) / len(v), 4), round(lo, 4), round(hi, 4), n=len(v))

    def prec(p, rank):
        return per[p][rank][0]

    committed = {}
    for r in csv.DictReader(open(os.path.join(a.results, "he_factscore_per_item.csv"))):
        if r["arm"] in ("sel64", "met_k10") and r["precision"] != "":
            committed[(r["arm"], r["prompt_id"])] = float(r["precision"])

    for name, pk in (("committed reward's pick", qpick), ("factuality scorer's pick", fpick)):
        v = [prec(p, pk[p]) for p in pids if prec(p, pk[p]) is not None]
        lo, hi = boot(v, rng)
        add("desc", f"{name}: precision (non-abstaining)", round(sum(v) / len(v), 4), round(lo, 4),
            round(hi, 4), n=len(v))
        add("desc", f"{name}: abstentions", sum(prec(p, pk[p]) is None for p in pids), n=len(pids))
        f = [per[p][pk[p]][1] for p in pids]
        add("desc", f"{name}: mean facts", round(sum(f) / len(f), 3), n=len(pids))

    both = [p for p in pids if prec(p, fpick[p]) is not None and prec(p, qpick[p]) is not None]
    d = [prec(p, fpick[p]) - prec(p, qpick[p]) for p in both]
    lo, hi = boot(d, rng)
    add("F2", "factuality pick minus committed pick, paired", round(sum(d) / len(d), 4), round(lo, 4),
        round(hi, 4), n=len(both), reading="RAISES" if lo > 0 else "LOWERS" if hi < 0 else "TIE")
    both = [p for p in pids if prec(p, fpick[p]) is not None and ("met_k10", p) in committed]
    d = [prec(p, fpick[p]) - committed[("met_k10", p)] for p in both]
    lo, hi = boot(d, rng)
    add("F3", "factuality pick minus the meter at k=10, paired", round(sum(d) / len(d), 4), round(lo, 4),
        round(hi, 4), n=len(both), reading="ABOVE" if lo > 0 else "BELOW" if hi < 0 else "TIE")

    avg = [sum(v[0] for v in per[p].values() if v[0] is not None) /
           max(1, sum(v[0] is not None for v in per[p].values())) for p in pids
           if any(v[0] is not None for v in per[p].values())]
    add("desc", "a single draw's precision, averaged over the 64 then over prompts", round(sum(avg) / len(avg), 4),
        n=len(avg))
    rep = [(prec(p, qpick[p]), committed[("sel64", p)]) for p in pids
           if prec(p, qpick[p]) is not None and ("sel64", p) in committed]
    add("desc", "committed pick recomputed, mean |difference| from the committed pass",
        round(sum(abs(x - y) for x, y in rep) / len(rep), 4), n=len(rep))

    path = os.path.join(a.out, "factscore_oracle.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--factuality", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dump-path", default=DUMP)
    ap.add_argument("--arms", default="output/he_metrics/arms.jsonl", help="the committed FActScore pass's dump")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.dump:
        dump(a.dump_path)
    if a.factuality:
        factuality(a.dump_path, os.path.join(a.out, "factuality_rewards64.csv"), a.model, a.batch_size)
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
