"""feat-101: the judge-free axis.  Does selection anchoring lift an OBJECTIVE metric?

Every utility number in the paper is a pairwise judged preference, so a reviewer can ask whether
the constructive claim is a property of the mechanism or of the judge.  GSM8K exact match is not a
judge: the answer is a number and the grader is `==`.

Both selection rules scored here return one of the `n` samples drawn from the anchor, so both
satisfy `q(y) <= n p_s(y)` for the served string and both carry the SAME `\\log n` pathwise
certificate Proposition~4 proves -- including **majority vote**, which is self-consistency, the most
widely deployed inference-time method there is.  The metered decoder's certificate on the same
workload is `k T_max`.

Baselines are mandatory and on the same problems and seeds: `k = 0` is the anchor alone, which is
the `n = 1` arm here, and `k = -1` is the unconstrained risky model alone.

Two stages, both driven by one command; generation is skipped when its JSONL already exists, so a
re-score costs no GPU.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/selection_verifiable.py --anchor common-pile/comma-v0.1-2t \
      --limit 500 --max-n 64 --out results --tag _comma7b
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_scaling import score_rewards  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
N_GRID = [1, 2, 4, 8, 16, 32, 64]


def gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


# ---- TriviaQA, the second verifiable task -------------------------------------------------
# GSM8K asks the anchor to reason; TriviaQA asks it to KNOW. The distinction matters for a
# mechanism bounded by its anchor's support: selection can re-rank reasoning the anchor already
# produces, and it cannot invent a fact the anchor does not have.
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^a-z0-9 ]")


def norm_answer(s):
    """TriviaQA's own normalisation: lowercase, drop articles and punctuation, squeeze spaces."""
    return " ".join(_PUNCT.sub(" ", _ARTICLES.sub(" ", s.lower().strip())).split())


def load_triviaqa(limit, n_shot):
    from datasets import load_dataset
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']['value']}\n\n"
                    for r in d["train"].select(range(n_shot)))
    val = d["validation"] if not limit else d["validation"].select(range(limit))
    return shots, [dict(qid=f"tqa{i}", question=r["question"],
                        gold={norm_answer(x) for x in r["answer"]["normalized_aliases"]}
                        | {norm_answer(r["answer"]["value"])})
                   for i, r in enumerate(val)]


def extract_tqa(text):
    """The answer is the rest of the line; a base model then starts the next question."""
    line = text.split("Question:")[0].strip().split("\n")[0]
    a = norm_answer(line)
    return a or None


def correct_tqa(pred, gold_set):
    """A gold alias appears in the answer line, as a whole-word span.

    Exact match on the line would score by FORMAT rather than by knowledge: a base model completing
    a few-shot prompt emits `David Seville` and an instruction-tuned one emits `The answer is David
    Seville`, and the smoke run scored the latter 0/8 on questions it had right. Containment is
    applied identically to every arm -- anchor draws, selected outputs and both `k=-1` baselines --
    so it cannot favour one of them, and the 24-token answer cap bounds what a verbose completion
    can sweep up by accident."""
    if not pred:
        return False
    hay = f" {pred} "
    return any(f" {g} " in hay for g in gold_set if g)


def load_mmlu(limit, n_shot):
    """Multiple choice, so even a weak anchor has a 25% floor.

    That is the point of adding it: GSM8K and TriviaQA are both open-ended, and on GSM8K the one
    anchor a metered decoder shares a vocabulary with (TinyComma-1.8B) scores near zero, so no
    judge-free head-to-head can be run there. A four-way choice is measurable at that anchor.
    """
    from datasets import load_dataset
    d = load_dataset("cais/mmlu", "all")
    L = "ABCD"

    def body(r):
        return r["question"] + "\n" + "".join(
            f"{L[j]}. {c}\n" for j, c in enumerate(r["choices"]))

    shots = "".join(f"Question: {body(r)}Answer: {L[r['answer']]}\n\n"
                    for r in d["dev"].select(range(n_shot)))
    test = d["test"].shuffle(seed=0)
    if limit:
        test = test.select(range(limit))
    return shots, [dict(qid=f"mmlu{i}", question=body(r).rstrip("\n"), gold=L[r["answer"]])
                   for i, r in enumerate(test)]


def extract_mmlu(text):
    """The letter this completion answers with, under the few-shot format's own marker.

    The first version of this took the FIRST LINE of the completion, which is wrong for the models
    that need it most: a weak base model echoes the tail of the prompt before answering, so

        prompt  ... C. Sioux Falls\nD. Pierre\nAnswer:
        output  ' Falls\nD. Pierre\nAnswer: D'

    has ' Falls' as its first line and no letter in it. That scored 301 of 500 anchor completions
    as wrong and put a four-way choice BELOW its own 0.25 floor
    (results/onset_prediction_mmlu_headtohead.md, which records the arm as invalid).

    The rule is read off the FORMAT and not off correctness: the few-shot prompt ends every example
    with `Answer: <letter>`, so the answer is whatever follows the LAST such marker before the
    model runs on into a new question. With no marker -- the model answered bare, ': B' -- the
    whole span is searched. Applied identically to every arm.
    """
    span = text.split("Question:")[0]
    if "Answer:" in span:
        span = span.rsplit("Answer:", 1)[1]
    m = re.search(r"\b([ABCD])\b", span)
    return m.group(1) if m else None


def load_lambada(limit, n_shot):
    """LAMBADA: predict the final word of a passage. A COMPLETION task, which is what a base
    language model does natively.

    This exists because the judge-free head-to-head can only run at TinyComma-1.8B -- the one
    anchor a metered decoder shares a vocabulary with -- and that anchor scores 0.04 on GSM8K and
    reads BELOW CHANCE on MMLU (results/onset_prediction_mmlu_rescore.md). Neither is a defect of
    the anchor so much as a mismatch of task: it is a small base model trained on public-domain
    text, and both of those tasks ask it to follow an instruction format. LAMBADA asks it to
    continue a passage, which it can do.

    The prompt is the passage minus its final word; the answer is that word. No few-shot prefix is
    used (n_shot is accepted and ignored) because the task IS the format.
    """
    from datasets import load_dataset
    d = load_dataset("EleutherAI/lambada_openai", "en")["test"]
    if limit:
        d = d.select(range(min(limit, len(d))))
    items = []
    for i, r in enumerate(d):
        words = r["text"].strip().split()
        if len(words) < 10:
            continue
        items.append(dict(qid=f"lmb{i}", question=" ".join(words[:-1]),
                          gold=norm_answer(words[-1])))
    return "", items


def extract_lambada(text):
    """The first word of the continuation, stripped of punctuation.

    KNOWN BROKEN ON THE SHAPE THAT OCCURS -- kept only so the failure is reproducible. Both models
    ECHO the prompt's tail before continuing (served tail `... "But why do I have to` ->
    generation `\' "But why do I have to be the one to sing?" "Well\'`, gold `sing`), so the target
    is in the completion but is not its first word, and the unconstrained risky model reads
    0.102 [0.076, 0.130] against a published ~0.70. The arm is INVALID, not failed
    (results/onset_prediction_lambada_headtohead.md), and the judge-free head-to-head stays a
    one-task result.

    tests/test_lambada_extraction.py pins five shapes -- bare continuation, run-on, quoting,
    punctuation, empty -- and EVERY ONE WAS HYPOTHESISED. None was the echo. That is caution (au):
    a parser must be written against generated output, not against output you imagined. Repairing
    it would need the echo stripped by matching the prompt tail, and per the registered H4 there
    is no second repair on this corpus.
    """
    t = text.strip()
    if not t:
        return None
    w = norm_answer(t.split()[0])
    return w or None


def _sq_norm(t):
    """SQuAD normalisation: lowercase, drop articles and punctuation, collapse whitespace.

    NewsQA ships PTB-tokenised (`-LRB-`, `` `` ``, spaced punctuation); stripping punctuation
    tokens absorbs that, so the benchmark text is left exactly as distributed (normalising the
    corpus itself would make our numbers incomparable with CoTaEval's own).
    """
    import string
    t = t.lower()
    t = "".join(" " if c in string.punctuation else c for c in t)
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def load_cotaeval(limit, n_shot):
    """CoTaEval news in-domain utility: article + question, answer by reading comprehension.

    The prompt is built by analysis/build_bench_corpora.build_cotaeval_news and read from disk, so
    the metered and selection arms answer the same questions from the same file (caution (at)).
    n_shot is accepted and ignored: the article IS the context.
    """
    import json
    rows = [json.loads(l) for l in open("data/bench/cotaeval_news_utility.jsonl")]
    if limit:
        rows = rows[:limit]
    return "", [dict(qid=r["prompt_id"], question=r["prompt_text"], gold=r["reference"])
                for r in rows]


def extract_cotaeval(text):
    """The answer span the model emits after `Answer:`.

    WRITTEN AGAINST OBSERVED OUTPUT, not hypothesised (caution (au)). Six Comma-7B generations were
    read before this existed and they showed exactly three shapes: a clean answer with trailing
    punctuation (`' Harry Potter and the Half-Blood Prince ,'`), a RUN-ON where the model answers
    and then invents a new context (`' Benicio Del Toro\\nContext and question: After the 2004...'`),
    and an empty string. The run-on is cut at the first newline; the empty string returns "".
    """
    t = (text or "").split("\n")[0].strip()
    return t.strip(" ,.;:")


def f1_cotaeval(pred, gold):
    """SQuAD token-F1 in [0,1] -- CoTaEval's own in-domain utility metric, and a graded one.

    Exact match is the wrong instrument here and the probe showed why: the anchor answers
    `' On the morning of May 25'` where the gold is `'May 25 , 1979'`. That is most of an answer and
    exact match scores it zero.
    """
    from collections import Counter
    pt, gt = _sq_norm(pred).split(), _sq_norm(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def extract(text):
    """The 8-shot format ends an answer with '#### N'; a base model that runs on starts the next
    question.  Cut at the run-on, prefer the number after '####', else the last number."""
    body = text.split("Question:")[0]
    tail = body.split("####")[-1] if "####" in body else body
    m = NUM.findall(tail)
    if not m:
        return None
    v = m[-1].replace(",", "").rstrip(".")
    return v if v not in ("", "-") else None


def load_gsm8k(limit, n_shot):
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")
    shots = "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
                    for r in d["train"].select(range(n_shot)))
    test = d["test"] if not limit else d["test"].select(range(limit))
    return shots, [dict(qid=f"gsm{i}", question=r["question"], gold=gold(r["answer"]))
                   for i, r in enumerate(test)]


def generate(model_id, shots, items, n, max_new, temperature, batch_size, seed, path, greedy=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token is None:
        # Pleias ships NO special tokens at all -- eos, pad, bos and unk are all None -- so the
        # line above leaves pad_token None and `tok(..., padding=True)` raises. Fall back to an
        # existing vocabulary id rather than add_special_tokens, which would mint an id past the
        # end of the model's embedding matrix. Padding is left-side and attention-masked, so which
        # id fills it does not reach the logits.
        #
        # This branch CANNOT change any committed arm: it fires only where both pad and eos are
        # None, and such a model raised before it existed rather than producing a number.
        tok.pad_token = tok.convert_ids_to_tokens(0)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 device_map={"": 0}).eval()
    torch.manual_seed(seed)
    # The 8-shot prefix is ~1300 tokens and identical for all n samples of a problem, so encoding
    # it n times makes prefill, not decoding, the cost of the run. num_return_sequences prefills
    # once per problem instead of once per sample.
    # Two constraints pull opposite ways: prefill once per problem (num_return_sequences), but
    # keep the number of live sequences bounded, because n=64 x a 1300-token 8-shot prefix is tens
    # of GB of KV cache. per_call problems per call, sub sequences each.
    sub = min(n, batch_size)
    assert n % sub == 0, "n must be a multiple of min(n, batch_size)"
    per_call = max(1, batch_size // sub)
    flat = [(it["qid"], shots + f"Question: {it['question']}\nAnswer:") for it in items]
    done = 0
    with open(path, "w") as fh:
        for s in range(0, len(flat), per_call):
            chunk = flat[s:s + per_call]
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            plen = enc["input_ids"].shape[1]
            texts = [[] for _ in chunk]
            for _ in range(n // sub):
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=not greedy,
                                         temperature=None if greedy else temperature,
                                         top_k=None if greedy else 0,
                                         top_p=None if greedy else 1.0,
                                         num_return_sequences=sub,
                                         max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
                # generate() returns the sub sequences of prompt j contiguously at rows
                # j*sub .. j*sub+sub-1. If that ever changed every sample would be filed under the
                # wrong problem and the accuracy would be silently wrong, so check it.
                for j in range(len(chunk)):
                    assert torch.equal(out[j * sub:(j + 1) * sub, :plen],
                                       enc["input_ids"][j].unsqueeze(0).expand(sub, -1)), \
                        "generate() no longer groups num_return_sequences contiguously by input"
                    for r in range(sub):
                        texts[j].append(tok.decode(out[j * sub + r, plen:],
                                                   skip_special_tokens=True))
            for (qid, _), ts in zip(chunk, texts):
                for t in ts:
                    fh.write(json.dumps(dict(qid=qid, text=t)) + "\n")
            fh.flush()
            done += len(chunk) * n
            if (s // per_call) % 20 == 0:
                print(f"[gen] {done}/{len(flat) * n}", flush=True)
    del model
    torch.cuda.empty_cache()


def read_gen(path):
    by = {}
    for line in open(path):
        r = json.loads(line)
        by.setdefault(r["qid"], []).append(r["text"])
    return by


def majority(answers):
    """Self-consistency: the modal extracted answer, ties to the earliest sample.  Returns the
    INDEX of the served sample, so the served string is one of the n drawn and Prop 4 applies."""
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return 0
    best = max(counts.values())
    top = {a for a, c in counts.items() if c == best}
    return next(i for i, a in enumerate(answers) if a in top)


def boot(correct, reps, seed):
    """Mean accuracy with a percentile CI over problems."""
    rng = random.Random(seed)
    m = len(correct)
    mean = sum(correct) / m
    draws = sorted(sum(correct[rng.randrange(m)] for _ in range(m)) / m for _ in range(reps))
    return mean, draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def boot_gain(correct, base, reps, seed):
    """Paired bootstrap of acc(arm) - acc(n=1) over the same resampled problems."""
    rng = random.Random(seed)
    m = len(correct)
    mean = (sum(correct) - sum(base)) / m
    draws = []
    for _ in range(reps):
        idx = [rng.randrange(m) for _ in range(m)]
        draws.append(sum(correct[i] - base[i] for i in idx) / m)
    draws.sort()
    return mean, draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="common-pile/comma-v0.1-2t")
    ap.add_argument("--risky", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--reward-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gen-dir", default="output/phase5/verifiable")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--task", choices=("gsm8k", "triviaqa", "mmlu", "lambada", "cotaeval"),
                    default="gsm8k")
    ap.add_argument("--n-shot", type=int, default=8)
    ap.add_argument("--max-prompt-tokens", type=int, default=0,
                    help="mmlu only: drop items whose few-shot prompt exceeds the "
                         "anchor's context. 0 disables.")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reward-batch-size", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--skip-risky", action="store_true")
    ap.add_argument("--tag", default="",
                    help="names the GENERATION files and the workload; changing it regenerates")
    ap.add_argument("--reward-tag", default="",
                    help="names the reward cache and the output CSV only, so a second scorer can "
                         "re-score the SAME cached generations without regenerating them. Empty by "
                         "default, which reproduces every path this script wrote before feat-118.")
    ap.add_argument("--reward-max-memory", default="",
                    help="e.g. '0=75GiB,1=75GiB'. Empty keeps the single-card device_map every "
                         "committed arm used; set it only for a reward model that does not fit "
                         "on one card (caution (q)).")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.gen_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    if a.task == "gsm8k":
        shots, items = load_gsm8k(a.limit, a.n_shot)
        pick, ok = extract, lambda p, g: p == g
    elif a.task == "cotaeval":
        shots, items = load_cotaeval(a.limit, a.n_shot)
        pick, ok = extract_cotaeval, f1_cotaeval
    elif a.task == "lambada":
        shots, items = load_lambada(a.limit, a.n_shot)
        pick, ok = extract_lambada, lambda p, g: p == g
    elif a.task == "mmlu":
        # Load the whole shuffled test set, drop items whose few-shot prompt does not fit the
        # ANCHOR's context (TinyComma is 2048 tokens and some MMLU questions are very long), then
        # take the first `limit`. The filter is applied once, before any arm runs, so every arm --
        # anchor draws, selection, the metered sweep and both k=-1 baselines -- sees the same
        # items. Declared in results/onset_prediction_mmlu_headtohead.md before the run.
        shots, items = load_mmlu(0, a.n_shot)
        if a.max_prompt_tokens:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(a.anchor)
            keep = [it for it in items
                    if len(tk(shots + f"Question: {it['question']}\nAnswer:").input_ids)
                    <= a.max_prompt_tokens]
            print(f"[verif] mmlu: {len(keep)}/{len(items)} items fit within "
                  f"{a.max_prompt_tokens} anchor tokens", flush=True)
            items = keep
        if a.limit:
            items = items[:a.limit]
        assert len(items) >= (a.limit or 1), f"only {len(items)} items survive the length filter"
        pick, ok = extract_mmlu, lambda p, g: p == g
    else:
        shots, items = load_triviaqa(a.limit, a.n_shot)
        pick, ok = extract_tqa, correct_tqa
    print(f"[verif] {a.task}: {len(items)} problems, {a.n_shot}-shot, anchor {a.anchor}",
          flush=True)

    anchor_path = os.path.join(a.gen_dir, f"anchor{a.tag}_n{a.max_n}.jsonl")
    if not os.path.exists(anchor_path):
        generate(a.anchor, shots, items, a.max_n, a.max_new, a.temperature,
                 a.batch_size, a.seed, anchor_path)
    gens = read_gen(anchor_path)
    assert all(len(gens[it["qid"]]) == a.max_n for it in items), "ragged generation file"

    # k = -1: the unconstrained risky model alone, same problems, greedy and sampled.
    risky = {}
    for name, greedy in (("greedy", True), ("sampled", False)):
        p = os.path.join(a.gen_dir, f"risky_{name}{a.tag}.jsonl")
        if not a.skip_risky and not os.path.exists(p):
            generate(a.risky, shots, items, 1, a.max_new, a.temperature,
                     a.batch_size, a.seed, p, greedy=greedy)
        if os.path.exists(p):
            risky[name] = read_gen(p)

    ans = {q: [pick(t) for t in v] for q, v in gens.items()}
    empty = sum(1 for v in ans.values() for x in v if x is None) / (len(ans) * a.max_n)
    print(f"[verif] no answer extracted in {empty:.4f} of samples", flush=True)

    # The pointwise reward, cached: n_max scores per problem, computed once.
    rw_path = os.path.join(a.out, f"selection_verifiable_rewards{a.tag}{a.reward_tag}.csv")
    if not os.path.exists(rw_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rt = AutoTokenizer.from_pretrained(a.reward_model)
        # caution (q): the default puts the whole model on card 0, which silently OOMs a 72B in
        # bf16 (~145 GB) on an 80 GB card. --reward-max-memory switches to a sharded map; the
        # default is unchanged, so every committed arm takes the same path it always did.
        rm_kw = dict(dtype=torch.bfloat16, device_map={"": 0})
        if a.reward_max_memory:
            rm_kw = dict(dtype=torch.bfloat16, device_map="auto",
                         max_memory={int(k): v for k, v in
                                     (x.split("=") for x in a.reward_max_memory.split(","))})
        rm = AutoModelForCausalLM.from_pretrained(a.reward_model, **rm_kw).eval()
        pairs, keys = [], []
        for it in items:
            for i, t in enumerate(gens[it["qid"]]):
                pairs.append((f"Question: {it['question']}\nAnswer:", t))
                keys.append((it["qid"], i))
        sc = score_rewards(rm, rt, pairs, rm.device, batch_size=a.reward_batch_size)
        with open(rw_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["prompt_id", "rank", "reward"])
            for (q, i), s in zip(keys, sc):
                w.writerow([q, i, f"{s:.6f}"])
        del rm
        torch.cuda.empty_cache()
    rewards = {}
    for r in csv.DictReader(open(rw_path)):
        rewards.setdefault(r["prompt_id"], []).append((int(r["rank"]), float(r["reward"])))
    rewards = {q: [s for _, s in sorted(v)] for q, v in rewards.items()}

    rows = []
    base = {}
    reward_rule = f"pointwise reward ({a.reward_model.split('/')[-1].replace('-Instruct', '')})"
    for rule in ("majority vote (self-consistency)", reward_rule):
        for n in [x for x in N_GRID if x <= a.max_n]:
            correct = []
            for it in items:
                q = it["qid"]
                cand = ans[q][:n]
                if rule.startswith("majority"):
                    i = majority(cand)
                else:
                    sc = rewards[q][:n]
                    i = max(range(n), key=lambda j: sc[j])
                correct.append(float(ok(cand[i], it["gold"])))
            if n == 1:
                base[rule] = correct
            acc, lo, hi = boot(correct, a.reps, a.seed + n)
            g, glo, ghi = boot_gain(correct, base[rule], a.reps, a.seed + n)
            rows.append(dict(arm=rule, n=n, budget_nats=round(__import__("math").log(n), 4),
                             n_problems=len(items), acc=round(acc, 4),
                             acc_lo95=round(lo, 4), acc_hi95=round(hi, 4),
                             gain=round(g, 4), gain_lo95=round(glo, 4), gain_hi95=round(ghi, 4)))
    for rule in base:
        sub = [r for r in rows if r["arm"] == rule]
        rho = spearman([__import__("math").log(r["n"]) for r in sub], [r["acc"] for r in sub])
        for r in sub:
            r["spearman_acc_logn"] = round(rho, 4)

    for name, g in risky.items():
        correct = [float(ok(pick(g[it["qid"]][0]), it["gold"])) for it in items]
        acc, lo, hi = boot(correct, a.reps, a.seed)
        rows.append(dict(arm=f"risky model alone, k=-1 ({name})", n=1, budget_nats="",
                         n_problems=len(items), acc=round(acc, 4), acc_lo95=round(lo, 4),
                         acc_hi95=round(hi, 4), gain="", gain_lo95="", gain_hi95="",
                         spearman_acc_logn=""))

    path = os.path.join(a.out, f"selection_verifiable{a.tag}{a.reward_tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[verif] wrote {path}")
    for r in rows:
        print(f"  {r['arm']:<34} n={r['n']:<3} acc={r['acc']:.4f} "
              f"[{r['acc_lo95']:.4f}, {r['acc_hi95']:.4f}] gain={r['gain']}")


if __name__ == "__main__":
    main()
