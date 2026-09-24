"""He et al.'s own utility metrics on this paper's served texts: Prometheus-2 fluency and FActScore.

A referee: He et al. (arXiv 2602.07120) evaluate utility with Prometheus-v2 fluency (a five-point
rubric) and FActScore claim precision on biographies, and this paper reports neither. This scores
every arm of Table 1 (and the AnchoredByte arms, when present) with both, on the same de-echoed
texts the judge reads (results/he_metrics_note.md).

Three stages, so the GPU work can run on whichever host has a free card:
  --dump        (CPU, where output/ lives) every arm's served text per prompt, via levels_pass.py's
                arm specs, plus each prompt and, for the 150 FActScore prompts, the entity and its
                Wikipedia text from data/factscore.jsonl -> output/he_metrics/arms.jsonl
  --prometheus  prometheus-eval/prometheus-7b-v2.0, absolute grading, the library's no-reference
                template and system prompt, a fluency/coherence rubric stated below, greedy
                -> results/he_prometheus_per_item.csv
  --factscore   factual prompts only: atomic facts from an instruct LLM (paragraph level), BM25 over
                the entity's Wikipedia text split into ~120-word passages, top 5, then the FActScore
                verification prompt ("... True or False?") read off the True/False logits
                -> results/he_factscore_per_item.csv
  --report      per-arm means with 95% bootstrap intervals over prompts, and paired differences
                -> results/he_metrics.csv
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DUMP = "output/he_metrics/arms.jsonl"
PROM = "prometheus-eval/prometheus-7b-v2.0"
ABS_SYSTEM = ("You are a fair judge assistant tasked with providing clear, objective feedback based on "
              "specific criteria, ensuring each assessment reflects the absolute standards set for "
              "performance.")
ABS_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """
FLUENCY = """[Is the response fluent and coherent: grammatical, readable, free of degenerate repetition, and a sensible continuation of the instruction?]
Score 1: The response is incoherent or ungrammatical throughout, or empty; it cannot be read as prose.
Score 2: The response is mostly disfluent, with frequent grammatical errors, repetition or abrupt breaks.
Score 3: The response is readable but uneven, with noticeable errors, repetition or lapses in coherence.
Score 4: The response is fluent and coherent, with only minor lapses.
Score 5: The response is fully fluent, grammatical and coherent throughout."""
ATOMIC = """Break the following passage about {topic} into a list of atomic facts. Each atomic fact is one short, self-contained sentence that states exactly one piece of information and names {topic} instead of using a pronoun. Include only claims the passage actually makes; do not add, correct or judge anything. Output one fact per line, each line starting with "- ". If the passage makes no factual claim, output the single line "- NONE".

Passage:
{text}"""
VERIFY = """Answer the question about {topic} based on the given context.

{context}

Input: {atom} True or False?
Answer with exactly one word, True or False."""
RESULT = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])")


def fact_meta():
    """prompt_id -> (entity, wikipedia_text) for data/factscore.jsonl, ids as dap.shared builds them."""
    out = {}
    for i, line in enumerate(open("data/factscore.jsonl", encoding="utf-8")):
        r = json.loads(line)
        out[f"fact_{i:05d}_{r['entity']}"] = (r["entity"], r["wikipedia_text"])
    return out


def dump(arms, path):
    from analysis.levels_pass import load_texts
    from analysis.order_averaged_h2h import true_prompts
    from dap.shared import load_prompt_corpus
    prompts = true_prompts("data")
    cls = {r.prompt_id: r.split for r in load_prompt_corpus("data", "text")}
    fm = fact_meta()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for name, spec in arms:
            texts = load_texts(spec)
            for p in sorted(texts):
                ent, wiki = fm.get(p, (None, None))
                fh.write(json.dumps(dict(arm=name, spec=spec, prompt_id=p, cls=cls.get(p),
                                         prompt=prompts[p], text=texts[p], entity=ent,
                                         wiki=wiki)) + "\n")
                n += 1
            print(f"[dump] {name:12s} {len(texts)} prompts", flush=True)
    print(f"wrote {path} ({n} rows)")


def load_dump(path, arms=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [r for r in rows if arms is None or r["arm"] in arms]


def load_model(mid, device_map):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=device_map).eval()
    return m, tok


def chat(tok, content):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True)


def generate(m, tok, texts, max_new, bs):
    import torch
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if (i // bs) % 20 == 0:
            print(f"[gen] {len(out)}/{len(texts)}", flush=True)
    return out


def prometheus(a):
    rows = load_dump(a.dump, a.arms.split(",") if a.arms else None)
    m, tok = load_model(a.model or PROM, a.device_map)
    texts = [chat(tok, ABS_SYSTEM + "\n\n" + ABS_WO_REF.format(
        instruction=r["prompt"], response=r["text"] if r["text"].strip() else "(empty response)",
        rubric=FLUENCY)) for r in rows]
    outs = generate(m, tok, texts, 384, a.batch_size)
    path = os.path.join(a.out, f"he_prometheus_per_item{a.tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "prompt_id", "cls", "empty", "score", "parsed"])
        for r, o in zip(rows, outs):
            mm = RESULT.search(o)
            w.writerow([r["arm"], r["prompt_id"], r["cls"], int(not r["text"].strip()),
                        int(mm.group(1)) if mm else "", int(bool(mm))])
    print(f"wrote {path}")


def passages(wiki, words=120):
    w = wiki.split()
    return [" ".join(w[i:i + words]) for i in range(0, max(len(w), 1), words)]


def bm25_top(query, docs, k=5, k1=1.5, b=0.75):
    tokd = [re.findall(r"\w+", d.lower()) for d in docs]
    q = re.findall(r"\w+", query.lower())
    N, avg = len(tokd), sum(map(len, tokd)) / max(len(tokd), 1)
    df = Counter(t for d in tokd for t in set(d))
    scores = []
    for i, d in enumerate(tokd):
        tf, s = Counter(d), 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append((s, i))
    return [docs[i] for _, i in sorted(scores, reverse=True)[:k]]


def parse_atoms(o):
    atoms = [ln.strip()[2:].strip() for ln in o.splitlines() if ln.strip().startswith("- ")]
    return [x for x in atoms if x and x.upper() != "NONE"]


def factscore(a):
    import torch
    rows = [r for r in load_dump(a.dump, a.arms.split(",") if a.arms else None) if r["entity"]]
    m, tok = load_model(a.model or "Qwen/Qwen2.5-14B-Instruct", a.device_map)
    todo = [i for i, r in enumerate(rows) if r["text"].strip()]
    outs = generate(m, tok, [chat(tok, ATOMIC.format(topic=rows[i]["entity"], text=rows[i]["text"]))
                             for i in todo], 512, a.batch_size)
    atoms = {i: parse_atoms(o) for i, o in zip(todo, outs)}
    ids = {w: [t for t in {tok.encode(v, add_special_tokens=False)[0] for v in vs}]
           for w, vs in (("t", ("True", " True", "true")), ("f", ("False", " False", "false")))}
    items = []
    for i, al in atoms.items():
        docs = passages(rows[i]["wiki"])
        for atom in al:
            ctx = "\n\n".join(f"Title: {rows[i]['entity']}\nText: {p}" for p in bm25_top(atom, docs))
            items.append((i, atom, chat(tok, VERIFY.format(topic=rows[i]["entity"], context=ctx, atom=atom))))
    sup = []
    for j in range(0, len(items), a.batch_size):
        enc = tok([x[2] for x in items[j:j + a.batch_size]], return_tensors="pt", padding=True,
                  truncation=True, max_length=4096).to(m.device)
        with torch.no_grad():
            lg = m(**enc, logits_to_keep=1).logits[:, -1, :].float().log_softmax(-1)
        t = torch.logsumexp(lg[:, ids["t"]], -1)
        f = torch.logsumexp(lg[:, ids["f"]], -1)
        sup += (t > f).tolist()
        if (j // a.batch_size) % 50 == 0:
            print(f"[verify] {len(sup)}/{len(items)}", flush=True)
    per = {}
    for (i, _, _), s in zip(items, sup):
        per.setdefault(i, []).append(s)
    path = os.path.join(a.out, f"he_factscore_per_item{a.tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "prompt_id", "n_facts", "n_supported", "precision", "abstained"])
        for i, r in enumerate(rows):
            v = per.get(i, [])
            w.writerow([r["arm"], r["prompt_id"], len(v), sum(v),
                        round(sum(v) / len(v), 6) if v else "", int(not v)])
    with open(os.path.join(a.out, f"he_factscore_atoms{a.tag}.jsonl"), "w", encoding="utf-8") as fh:
        for (i, atom, _), s in zip(items, sup):
            fh.write(json.dumps(dict(arm=rows[i]["arm"], prompt_id=rows[i]["prompt_id"], atom=atom,
                                     supported=bool(s))) + "\n")
    print(f"wrote {path}")


def boot(v, rng, B=10000):
    n = len(v)
    ms = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def report(a):
    rng = random.Random(1914)
    rows = []
    import glob
    for metric, pat, col in (("prometheus_fluency", "he_prometheus_per_item*.csv", "score"),
                             ("factscore_precision", "he_factscore_per_item*.csv", "precision"),
                             ("factscore_n_facts", "he_factscore_per_item*.csv", "n_facts")):
        per = {}
        for p in sorted(glob.glob(os.path.join(a.out, pat))):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if r[col] != "":
                    assert r["prompt_id"] not in per.get(r["arm"], {}), (p, r["arm"], "scored twice")
                    per.setdefault(r["arm"], {})[r["prompt_id"]] = float(r[col])
        for arm, d in sorted(per.items()):
            v = list(d.values())
            lo, hi = boot(v, rng, 4000)
            rows.append(dict(metric=metric, arm=arm, contrast="", value=round(sum(v) / len(v), 4),
                             lo95=round(lo, 4), hi95=round(hi, 4), n=len(v)))
        for c in (a.contrast or []):
            x, y = c.split("-", 1)
            if x in per and y in per:
                common = sorted(set(per[x]) & set(per[y]))
                v = [per[x][q] - per[y][q] for q in common]
                lo, hi = boot(v, rng, 4000)
                rows.append(dict(metric=metric, arm="", contrast=c, value=round(sum(v) / len(v), 4),
                                 lo95=round(lo, 4), hi95=round(hi, 4), n=len(v)))
    path = os.path.join(a.out, "he_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--do-dump", action="store_true")
    ap.add_argument("--arm", action="append", default=[], help="name=spec (levels_pass.py syntax)")
    ap.add_argument("--prometheus", action="store_true")
    ap.add_argument("--factscore", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--contrast", action="append", default=[], help="armA-armB, paired, repeatable")
    ap.add_argument("--arms", default="", help="comma list: restrict a GPU stage to these arms")
    ap.add_argument("--model", default="")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", default="", help="suffix of the per-item files, so a later arm adds rather than overwrites")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.do_dump:
        dump([s.split("=", 1) for s in a.arm], a.dump)
    if a.prometheus:
        prometheus(a)
    if a.factscore:
        factscore(a)
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
