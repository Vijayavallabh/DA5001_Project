"""What one REQUEST costs a server: selection with its n draws batched, against the metered decoder.

Every serving number on record times either the audit harness, which realises n as n seed groups and
forwards both models at every step (caution (ay)), or feat-164's cell C, a plain generate() of the
anchor at a fixed batch width W with ONE completion per prompt -- so its 21.8x still charges n=64 as
64 sequential decodes. A server batches a request's n candidates into one call
(`num_return_sequences=n`): one decode loop of width W*n over a shared prompt. This times that, per
request, against the metered decoder at the 8B and at the 70B pair (results/batched_latency_note.md).

Cells, every one serving exactly T new tokens per completion (min_new_tokens=T, as feat-164's C did),
temperature 1, the first W*reps neutral prompts of the headline corpus, rep r on prompts [rW, (r+1)W):
  SEL    anchor draws, one generate(num_return_sequences=n), then ONE reward pass over the W*n
         candidates with the committed scorer (selection_scaling.score_rewards, Qwen2.5-7B-Instruct)
  MET    a_patch AnchoredDecodingFactory.generate at k (both models every step, as the mechanism must)
  RISKY  the risky model alone: what an undefended server pays
Models load once, the load is timed apart, one untimed warm-up precedes the timed reps.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \\
    .venv/bin/python analysis/batched_latency.py --arm sel --widths 1,8 --ns 1,8,64
  .venv/bin/python analysis/batched_latency.py --report --out results      # no GPU
"""
import argparse
import csv
import os
import re
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ANCHOR = "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"
REWARD = "Qwen/Qwen2.5-7B-Instruct"
LINE = re.compile(r"\[blat\] arm=(\w+) risky=(\S+) W=(\d+) n=(\d+) rep=(\d+) gen_s=([\d.]+) "
                  r"reward_s=([\d.]+) served=(\d+)x(\d+)")


def prompts(n_needed):
    from dap.shared import load_prompt_corpus
    recs = [r.prompt_text for r in load_prompt_corpus("data", "text") if r.split == "neutral"]
    assert len(recs) >= n_needed, f"neutral holds {len(recs)} prompts, need {n_needed}"
    return recs[:n_needed]


def sync():
    import torch
    torch.cuda.synchronize()


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    widths = [int(x) for x in a.widths.split(",")]
    ns = [int(x) for x in a.ns.split(",")] if a.arm == "sel" else [1]
    texts = prompts(max(widths) * (a.reps + 1))
    t0 = time.time()
    if a.arm == "met":
        from a_patch.factory import AnchoredDecodingFactory
        mm = ({int(k): v for k, v in (x.split("=") for x in a.max_memory.split(","))}
              if a.max_memory else None)
        fac = AnchoredDecodingFactory.from_pretrained(
            safe_model_path=a.anchor, risky_model_path=a.risky, k_radius=a.k, use_prefix_debt=True,
            prefix_n=5, log_kl_stats=True, dtype=torch.bfloat16,
            risky_device_map=a.risky_device_map or None, max_memory=mm)
        cfg = GenerationConfig(max_new_tokens=a.T, min_new_tokens=a.T, do_sample=True, temperature=1.0)

        def gen(batch, n):
            out = fac.generate(text=batch, generation_config=cfg, k_radius=a.k, seed=a.seed,
                               parallelize=a.parallelize, show_progress=False)
            seq = out.sequences if hasattr(out, "sequences") else out
            return seq.shape[0], a.T
    else:
        path = a.anchor if a.arm == "sel" else a.risky
        tok = AutoTokenizer.from_pretrained(a.risky_tokenizer or path, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dm = a.risky_device_map if (a.arm == "risky" and a.risky_device_map) else {"": 0}
        mm = ({int(k): v for k, v in (x.split("=") for x in a.max_memory.split(","))}
              if (a.arm == "risky" and a.max_memory) else None)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dm,
                                                 max_memory=mm).eval()
        if a.arm == "sel":
            from analysis.selection_scaling import score_rewards
            rtok = AutoTokenizer.from_pretrained(a.reward, padding_side="left")
            rm = AutoModelForCausalLM.from_pretrained(a.reward, dtype=torch.bfloat16,
                                                      device_map={"": 0}).eval()

        def gen(batch, n):
            torch.manual_seed(a.seed)
            enc = tok(batch, return_tensors="pt", padding=True).to(m.device)
            with torch.no_grad():
                out = m.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=n,
                                 max_new_tokens=a.T, min_new_tokens=a.T, pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gen.last = [(p, g) for p, g in zip([b for b in batch for _ in range(n)], gens)]
            return out.shape[0], out.shape[1] - enc["input_ids"].shape[1]
    sync()
    print(f"[blat] LOAD arm={a.arm} seconds={time.time() - t0:.3f}", flush=True)

    risky_tag = "-" if a.arm == "sel" else a.risky
    gen(texts[:1], ns[0])          # warm-up, untimed
    sync()
    for W in widths:
        for n in ns:
            for rep in range(1, a.reps + 1):
                batch = texts[rep * W:(rep + 1) * W]
                sync()
                t1 = time.time()
                rows, T = gen(batch, n)
                sync()
                t2 = time.time()
                rs = 0.0
                if a.arm == "sel":
                    score_rewards(rm, rtok, gen.last, rm.device, batch_size=a.reward_batch,
                                  log_every=10 ** 9)
                    sync()
                    rs = time.time() - t2
                assert rows == W * n and T == a.T, (rows, W * n, T, a.T)
                print(f"[blat] arm={a.arm.upper()} risky={risky_tag} W={W} n={n} rep={rep} "
                      f"gen_s={t2 - t1:.3f} reward_s={rs:.3f} served={rows}x{T}", flush=True)


def report(logs, out):
    """Per-request seconds per cell, keyed by the log PART it was timed in: each part times its
    selection cell on the same card, back to back with the meter it is compared with, so a ratio is
    only ever formed within one part (the registration's same-card rule). Then T1-T4."""
    cells, part = {}, "?"
    for log in logs:
        for line in open(log, encoding="utf-8"):
            if line.startswith("[blat] START"):
                part = re.search(r"part=(\S+)", line).group(1)
            m = LINE.search(line)
            if m:
                arm, risky, W, n = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                cells.setdefault((part, arm, risky, W, n), []).append(
                    (float(m.group(6)), float(m.group(7))))
    rows = []
    for (part, arm, risky, W, n), v in sorted(cells.items()):
        g, r = st.mean(x[0] for x in v), st.mean(x[1] for x in v)
        rows.append(dict(part=part, arm=arm, risky=risky, W=W, n=n, reps=len(v), gen_s=round(g, 4),
                         reward_s=round(r, 4), per_request_s=round((g + r) / W, 4),
                         spread=round((max(x[0] + x[1] for x in v) - min(x[0] + x[1] for x in v))
                                      / (g + r), 4)))
    by = {(r["part"], r["arm"], r["risky"], r["W"], r["n"]): r for r in rows}
    for r in rows:
        if r["arm"] == "SEL":
            for (pt, a, risky, W, n), met in by.items():
                if pt == r["part"] and a == "MET" and W == r["W"]:
                    r[f"vs_met_{risky.split('/')[-1]}"] = round(r["per_request_s"] / met["per_request_s"], 4)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = os.path.join(out, "batched_latency.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {path}")
    # the registered bands (results/onset_prediction_batched_latency.md)
    R8, R70 = "meta-llama/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-70B"
    bands = []

    def ratio(tag, W, num, den, lo, hi, predict):
        if num not in by or den not in by:
            return
        a, b = by[num], by[den]
        q = a["per_request_s"] / b["per_request_s"]
        noise = a["spread"] + b["spread"]
        ok = (lo is None or q > lo) and (hi is None or q < hi)
        read = ("descriptive" if predict is None else
                ("CONFIRMED" if ok else "REFUTED") if min(abs(q - x) / x for x in (lo, hi) if x) > noise
                else "WITHIN NOISE")
        bands.append(dict(band=tag, W=W, numerator=f"{a['arm']} n={a['n']} ({a['part']})",
                          denominator=f"{b['arm']} {b['risky'].split('/')[-1]} n={b['n']} ({b['part']})",
                          ratio=round(q, 4), cell_spreads=round(noise, 4), predicted=predict or "",
                          reading=read))
    for W in (1, 8):
        d = None if W == 8 else True
        ratio("T1" if W == 1 else "T4", W, ("70b", "SEL", "-", W, 64), ("70b", "MET", R70, W, 1),
              None, 1.0, "below 1" if d else None)
        ratio("T2" if W == 1 else "T4", W, ("single", "SEL", "-", W, 64), ("single", "MET", R8, W, 1),
              0.5, 2.0, "between 0.5 and 2" if d else None)
        ratio("T3" if W == 1 else "T4", W, ("single", "SEL", "-", W, 64), ("single", "SEL", "-", W, 1),
              None, 8.0, "below 8" if d else None)
        ratio("ref", W, ("70b", "SEL", "-", W, 64), ("70b", "RISKY", R70, W, 1), None, None, None)
        ratio("ref", W, ("single", "SEL", "-", W, 64), ("single", "RISKY", R8, W, 1), None, None, None)
    bpath = os.path.join(out, "batched_latency_bands.csv")
    with open(bpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bands[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(bands)
    for b in bands:
        print(b)
    print(f"wrote {bpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("sel", "met", "risky"))
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--risky", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--risky-tokenizer", default="")
    ap.add_argument("--reward", default=REWARD)
    ap.add_argument("--reward-batch", type=int, default=64)
    ap.add_argument("--widths", default="1,8")
    ap.add_argument("--ns", default="1,8,64")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--parallelize", action="store_true")
    ap.add_argument("--risky-device-map", default="")
    ap.add_argument("--max-memory", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--logs", default="output/logs/batched_latency.log")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.report:
        report(a.logs.split(","), a.out)
    else:
        assert a.arm, "--arm is required unless --report"
        run(a)


if __name__ == "__main__":
    main()
