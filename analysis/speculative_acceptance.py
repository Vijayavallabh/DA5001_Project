"""Review 2 Q3 (post hoc, descriptive, no band): how far could the meter, run as speculative decoding, narrow
selection's batched serving time?

The meter serves p*_t = normalize(p_s^{bc_t} p_r^{bd_t}) at step t (a_patch/factory.py; the per-step log records
bc_t and bd_t). Run as exact speculative sampling with the anchor as the draft, a drafted token is accepted with
probability alpha_t = sum_x min(p_s(x), p*_t(x)) = 1 - TV(p_s, p*_t) and the served law is unchanged. This
teacher-forces both models on the committed meter's own served tokens, rebuilds p*_t from the logged weights
(checked against the logged p*(served token)), and records alpha_t.

A round with draft length g costs g anchor decode steps plus one risky forward that verifies g+1 positions,
charged as ONE risky decode step (optimistic for the meter: exact at one request, where decoding is
memory-bound). It serves E_g tokens, the mean of a Monte-Carlo run over each trajectory's own alpha_t. With
per-token costs from results/batched_latency.csv at one request (W=1) -- c_s the anchor alone (SEL, n=1), c_r
the risky model alone (RISKY), c_m the meter (MET, k=10) -- the speculative meter is F_g = c_m E_g / (g c_s + c_r)
times faster, and selection's time at n=64 against it is the committed ratio times F. Selection keeps its
batched advantage against the best g exactly when that product stays below 1; alpha* is the constant acceptance
at which it would reach 1.

Writes <out>/speculative_acceptance_<tag>.csv (one row per k, plus the per-g rows).

Usage (one card for the 8B pair; `--device-map auto` over two for the 70B):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \\
    .venv/bin/python analysis/speculative_acceptance.py --run-dir output/phase2/conc_all --k 10 3 0.5 --tag 8b
"""
import argparse
import csv
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASSES = ("neutral", "factual", "creative")
EOS = {128001, 128009}


def served(run_dir, k, limit):
    """Lowest-seed record per prompt: (prompt_id, prefix_text, prefix_len, [(token, bc, bd, p_star)])."""
    best = {}
    for cls in CLASSES:
        path = os.path.join(run_dir, f"trajectories_k{k}_{cls}.jsonl")
        for line in open(path):
            r = json.loads(line)
            m = r["metadata"]
            if m.get("constraint", "kl") != "kl":
                continue
            if m["prompt_id"] in best and best[m["prompt_id"]][0] <= m["seed"]:
                continue
            steps = []
            for s in r["per_step_log"]:
                steps.append((int(s["sampled_token_id"]), float(s["bc"]), float(s["bd"]), float(s["p_star_prob"])))
                if s["sampled_token_id"] in EOS:
                    break
            pa = r["prefix_analysis"]
            best[m["prompt_id"]] = (m["seed"], pa["prefix_text"], pa["prefix_length_tokens"], steps)
    pids = sorted(best)[:limit]
    return [(p, *best[p][1:]) for p in pids]


def mean_rounds(alphas, g, rng, reps=20):
    """Mean number of risky forwards (rounds) to serve the trajectory when drafting g tokens from the anchor:
    each round accepts drafted tokens while they pass (probability alpha_t at position t) and ends on the first
    rejection with one corrected token, or after g acceptances with one bonus token."""
    tot = 0
    T = len(alphas)
    for _ in range(reps):
        t = 0
        while t < T:
            i = 0
            while i < g and t + i < T and rng.random() < alphas[t + i]:
                i += 1
            t += i + 1
            tot += 1
    return tot / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", nargs="+", required=True, help="filename tokens of the meter arms")
    ap.add_argument("--safe-model", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--risky-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device-map", default="")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--latency", default="results/batched_latency.csv")
    ap.add_argument("--latency-part", default="single", help="single (the 8B pair) or 70b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--break-even-only", action="store_true",
                    help="no model: write only alpha*, the constant acceptance at which the best round would bring "
                         "selection's ratio to 1, for --latency-part (the 70B pair's alpha needs two cards)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    lat = {r["arm"] + (f":{r['n']}" if r["arm"] == "SEL" else ""): r for r in csv.DictReader(open(a.latency))
           if r["part"] == a.latency_part and r["W"] == "1"}
    c_s = float(lat["SEL:1"]["gen_s"]) / 200
    c_r = float(lat["RISKY"]["gen_s"]) / 200
    c_m = float(lat["MET"]["gen_s"]) / 200
    sel64 = float(lat["SEL:64"]["per_request_s"]) / float(lat["MET"]["per_request_s"])
    print(f"[spec] per-token seconds: anchor {c_s:.5f}, risky {c_r:.5f}, meter {c_m:.5f}; "
          f"selection n=64 / meter = {sel64:.4f}", flush=True)

    def ratio_at(al):
        """Selection's ratio against the best draft length g <= 8 at constant acceptance al."""
        return max(sel64 * c_m * (g + 1 if al >= 1 else (1 - al ** (g + 1)) / (1 - al)) / (g * c_s + c_r)
                   for g in range(1, 9))

    def alpha_star():
        """The constant acceptance at which the best draft length brings selection's ratio to 1; '' when even
        perfect acceptance does not (g <= 8)."""
        if ratio_at(1.0) < 1:
            return ""
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ratio_at(mid) < 1 else (lo, mid)
        return lo

    cols = ["k", "quantity", "g", "trajectories", "steps", "alpha_mean", "alpha_median", "alpha_p10",
            "pstar_median_abs_err", "best_g", "tokens_per_risky_forward", "speedup", "selection_ratio", "alpha_star",
            "selection_ratio_at_alpha1", "c_anchor", "c_risky", "c_meter", "selection_ratio_committed"]
    if a.break_even_only:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"speculative_acceptance_{a.tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            s = alpha_star()
            w.writerow(dict(k="", quantity="break-even", alpha_star=s if s == "" else round(s, 4),
                            selection_ratio_at_alpha1=round(ratio_at(1.0), 4), c_anchor=round(c_s, 6),
                            c_risky=round(c_r, 6), c_meter=round(c_m, 6), selection_ratio_committed=round(sel64, 4)))
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.safe_model)
    kw = dict(torch_dtype=torch.bfloat16)
    ms = AutoModelForCausalLM.from_pretrained(a.safe_model, **kw).cuda().eval()
    mr = (AutoModelForCausalLM.from_pretrained(a.risky_model, device_map=a.device_map, **kw) if a.device_map
          else AutoModelForCausalLM.from_pretrained(a.risky_model, **kw).cuda()).eval()

    rows = []
    rng = random.Random(3)
    for k in a.k:
        recs = served(a.run_dir, k, a.limit)
        alphas, check = [], []
        for i in range(0, len(recs), a.batch_size):
            chunk = recs[i:i + a.batch_size]
            seqs, spans = [], []
            for _, text, plen, steps in chunk:
                ids = tok(text).input_ids
                assert len(ids) == plen, (len(ids), plen)          # the prompt the meter was given
                seqs.append(ids + [s[0] for s in steps])
                spans.append((plen, len(steps)))
            w = max(map(len, seqs))
            ids = torch.full((len(seqs), w), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), w), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            with torch.no_grad():
                ls = ms(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
                lr = mr(input_ids=ids.to(mr.device), attention_mask=att.to(mr.device)).logits.to(ls.device)
            for j, (_, _, _, steps) in enumerate(chunk):
                p0, n = spans[j]
                lps = F.log_softmax(ls[j, p0 - 1:p0 - 1 + n].float(), -1)
                lpr = F.log_softmax(lr[j, p0 - 1:p0 - 1 + n].float(), -1)
                bc = torch.tensor([s[1] for s in steps], device=lps.device)[:, None]
                bd = torch.tensor([s[2] for s in steps], device=lps.device)[:, None]
                lpq = F.log_softmax(bc * lps + bd * lpr, -1)
                al = torch.minimum(lps.exp(), lpq.exp()).sum(-1).tolist()
                tgt = torch.tensor([s[0] for s in steps], device=lps.device)
                pq = lpq.gather(-1, tgt[:, None]).squeeze(-1).exp().tolist()
                check += [abs(x - s[3]) for x, s in zip(pq, steps)]
                alphas.append(al)
            print(f"[spec] k={k}: {min(i + a.batch_size, len(recs))}/{len(recs)}", flush=True)
        flat = [x for al in alphas for x in al]
        # the rebuilt p* must reproduce what the meter logged for the token it served (bf16 re-forward)
        med_err = st.median(check)
        assert med_err < 0.01, f"p* reconstruction off: median |dp| {med_err}"
        best = None
        for g in range(1, 9):
            E = len(flat) / sum(mean_rounds(al, g, rng) for al in alphas)   # served tokens per risky forward
            Fg = c_m * E / (g * c_s + c_r)
            rows.append(dict(k=k, quantity="round", g=g, tokens_per_risky_forward=round(E, 4),
                             speedup=round(Fg, 4), selection_ratio=round(sel64 * Fg, 4)))
            best = max(best or (0, 0, 0), (Fg, g, E))
        lo = alpha_star()
        lo = lo if lo == "" else round(lo, 4)
        rows.append(dict(k=k, quantity="acceptance", trajectories=len(alphas), steps=len(flat),
                         alpha_mean=round(st.mean(flat), 4), alpha_median=round(st.median(flat), 4),
                         alpha_p10=round(sorted(flat)[len(flat) // 10], 4), pstar_median_abs_err=round(med_err, 6),
                         best_g=best[1], tokens_per_risky_forward=round(best[2], 4), speedup=round(best[0], 4),
                         selection_ratio=round(sel64 * best[0], 4), alpha_star=lo,
                         selection_ratio_at_alpha1=round(ratio_at(1.0), 4),
                         c_anchor=round(c_s, 6), c_risky=round(c_r, 6), c_meter=round(c_m, 6),
                         selection_ratio_committed=round(sel64, 4)))
        print(rows[-1], flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"speculative_acceptance_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
