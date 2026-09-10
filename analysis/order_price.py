"""What each Renyi order buys and leaks at one published budget (plan v5 / feat-072).

Table 1 compares four decoders at a single published k and Appendix D concedes the limit of that
comparison: it is at matched *budget*, not matched utility, and Section 5 shows the judge cannot
supply the missing axis -- re-judging the same generations inverts the ranking of the four orders.

feat-070 supplies an instrument the judge cannot match. On the geodesic

    G(theta) = theta m - psi(theta) = -D_KL(p_r || p_theta) + const,   m = psi'(1)

is fidelity to the model the decoder is imitating, which is the only thing a metered decoder's
budget can buy. It is deterministic, needs no sampling and no judge, and is computed to machine
precision from two teacher-forced forward passes. So for each order alpha we can read off, at one
published k and on the same passages:

    price    fidelity bought on an ordinary generation   -- what the deployer gives up
    leakage  fidelity bought on a protected passage      -- what the rights-holder gives up

theta is solved under that order's own charge with a_patch/renyi.py, the solver the decoder itself
uses, so "the same published k" means exactly what it means to a deployer.

Writes <out>/order_price.csv. Pre-registered in results/onset_prediction_orders_matched.md.

  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/order_price.py --safe-model output/phase5/anchor_kl3m-002-520m \
      --risky-model output/phase5/mem_kl3m-002-520m --k 3.0 --limit 25 --out results
"""
from __future__ import annotations

import argparse, csv, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.marginal_price import fidelity, logits_along, solve_theta_greedy  # noqa: E402
from a_patch.renyi import solve_theta_renyi  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def theta_for(order, log_ps, l, k):
    """theta under the given order's charge. alpha = 1 is the audited KL decoder, for which the
    dedicated bisection on D_KL is used so the KL column is the decoder's own rule and not a
    limit of the Renyi formula."""
    if order == 1.0:
        return solve_theta_greedy(log_ps, l, k)
    log_pd = log_ps + l
    kt = torch.full((log_ps.size(0),), float(k), device=log_ps.device, dtype=log_ps.dtype)
    return solve_theta_renyi(log_ps, log_pd, kt, order).to(log_ps.dtype)


def run_side(safe, risky, tok, ps, orders, k, device, seed_tokens, max_tokens,
             sample_tokens, sample_target):
    """Returns {alpha: (fidelity_sum, ceiling_sum, theta_mean)} over the passages."""
    acc = {o: [0.0, 0.0, []] for o in orders}
    n = 0
    for p in ps:
        if sample_target:
            prompt_ids = tok(p.prompt_text).input_ids[:max_tokens]
            with torch.no_grad():
                out = risky.generate(torch.tensor([prompt_ids], device=device),
                                     max_new_tokens=sample_tokens, do_sample=True,
                                     top_k=0, top_p=1.0, temperature=1.0,
                                     pad_token_id=tok.eos_token_id or 0)
            ids, seed = out[0].tolist(), len(prompt_ids)
        else:
            ids, seed = tok(join(p.prompt_text, p.reference)).input_ids[:max_tokens], seed_tokens
        if len(ids) <= seed + 8:
            continue
        log_ps = torch.log_softmax(logits_along(safe, ids, device)[seed - 1:], dim=-1)
        log_pr = torch.log_softmax(logits_along(risky, ids, device)[seed - 1:], dim=-1)
        l = torch.nan_to_num(log_pr - log_ps, nan=0.0, posinf=0.0, neginf=0.0)
        m = (log_pr.exp() * l).sum(-1)
        ceiling = float(fidelity(log_ps, l, 1.0, m).sum())
        for o in orders:
            th = theta_for(o, log_ps, l, k)
            acc[o][0] += float(fidelity(log_ps, l, th, m).sum())
            acc[o][1] += ceiling
            acc[o][2].append(float(th.mean()))
        n += 1
        del log_ps, log_pr, l
        if device == "cuda":
            torch.cuda.empty_cache()
    return {o: (v[0], v[1], st.mean(v[2]) if v[2] else 0.0) for o, v in acc.items()}, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe-model", required=True)
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--orders", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--sample-tokens", type=int, default=160)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="order_price")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.safe_model)
    safe = AutoModelForCausalLM.from_pretrained(a.safe_model, torch_dtype=torch.float32).to(device).eval()
    risky = AutoModelForCausalLM.from_pretrained(a.risky_model, torch_dtype=torch.float32).to(device).eval()

    corpus = load_prompt_corpus("data", "factscore_prompt")
    prot = [p for p in corpus if p.split == "test" and p.reference][:a.limit]
    ordn = [p for p in corpus if p.split == "neutral"][:a.limit]

    leak, n_prot = run_side(safe, risky, tok, prot, a.orders, a.k, device,
                            a.seed_tokens, a.max_tokens, a.sample_tokens, False)
    print(f"[op] protected side done, {n_prot} passages", flush=True)
    price, n_ord = run_side(safe, risky, tok, ordn, a.orders, a.k, device,
                            a.seed_tokens, a.max_tokens, a.sample_tokens, True)
    print(f"[op] ordinary side done, {n_ord} generations", flush=True)

    base_l, base_p = leak[a.orders[0]][0], price[a.orders[0]][0]
    rows = []
    for o in a.orders:
        lf, lc, lt = leak[o]
        pf, pc, pt = price[o]
        rows.append(dict(
            alpha=o, k=a.k, n_protected=n_prot, n_ordinary=n_ord,
            leak_nats=round(lf, 3), leak_frac_ceiling=round(lf / lc, 4),
            price_nats=round(pf, 3), price_frac_ceiling=round(pf / pc, 4),
            R_leak_vs_alpha1=round(lf / base_l, 4), P_price_vs_alpha1=round(pf / base_p, 4),
            dominance=round((pf / base_p) - (lf / base_l), 4),
            theta_mean_protected=round(lt, 4), theta_mean_ordinary=round(pt, 4)))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.prefix}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"\nat one published k = {a.k}: what each order buys, and what it lets through\n")
    print(f"{'alpha':>7s}{'price (ordinary)':>19s}{'leakage (protected)':>21s}"
          f"{'P':>8s}{'R':>8s}{'P - R':>9s}")
    for r in rows:
        print(f"{r['alpha']:>7.0f}{r['price_nats']:>13.1f} nats"
              f"{r['leak_nats']:>16.1f} nats"
              f"{r['P_price_vs_alpha1']:>8.3f}{r['R_leak_vs_alpha1']:>8.3f}"
              f"{r['dominance']:>+9.3f}")
    print("\nP and R are relative to alpha = 1, the audited decoder. P - R > 0 means the order gives")
    print("up less utility than it gives up leakage, i.e. it dominates at the same published k.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
