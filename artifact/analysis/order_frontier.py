"""Does a higher Renyi order dominate at matched *utility*, not matched budget? (plan v5 / feat-073)

Table 1 compares four decoders at one published `k`, and Appendix D concedes what that comparison
cannot settle: an order that leaks less at the same budget may simply be buying less, and the
deployer could have had the same reduction by lowering `k` on the audited decoder. Section 5 shows
the judge cannot supply the missing axis -- re-judging the same generations inverts the ranking.

feat-070's geodesic gives the axis without a judge. At each step, with `l = log p_r - log p_s` and
`psi(u) = log sum_v p_s(v) e^{u l(v)}`,

    fidelity  G(theta) = theta psi'(1) - psi(theta) = -D_KL(p_r || p_theta) + const

is what the budget buys, and it is monotone in `k` at fixed order. So for every order there is a
unique budget at which it buys exactly what the audited KL decoder buys at the published one. This
sweeps a grid of `k` in a single pass over the passages -- no new forward passes, only more theta
solves -- and reports, at that matched budget, the quantity whose exponential is the reproduction
probability:

    L(alpha, k) = sum_t log p_theta(x_t | x_<t)   over the protected token sequence

Two sides, two different functionals, because extraction and utility are different kinds of
quantity: fidelity is a bounded average and belongs on the ordinary generations; `L` is a rare event
and belongs on the protected passage. Using an average for both was the error feat-072 recorded.

The protected split must be one the risky model was fine-tuned on. On a held-out split a
LoRA-memorised model is *worse* than its own base and the bracket inverts, which is the check the
run prints and refuses to interpret without.

Writes <out>/<prefix>.csv (long: one row per alpha x k x side) and <out>/<prefix>_matched.csv.
Pre-registered in results/onset_prediction_orders_matched.md.

  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_kl3m-002-520m \
      --risky-model output/phase5/mem_kl3m-002-520m --limit 25 --out results --prefix of_kl3m
"""
from __future__ import annotations

import argparse, csv, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.marginal_price import fidelity, logits_along  # noqa: E402
from analysis.order_price import logp_target, theta_for  # noqa: E402
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def interp(xs, ys, y):
    """The x at which the piecewise-linear (xs, ys) first reaches y. xs ascending, ys ascending.
    None if y is off the end of the grid, which is a result and not an error: it means the order
    cannot buy that much utility at any budget on the grid."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - y) * (y1 - y) <= 0 and y1 != y0:
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return None


def run_side(safe, risky, tok, ps, orders, ks, device, seed_tokens, max_tokens,
             sample_tokens, sample_target):
    acc = {(o, k): 0.0 for o in orders for k in ks}
    extra, n = {"ceiling": 0.0, "risky": 0.0, "safe": 0.0, "ntok": 0.0}, 0
    for i, p in enumerate(ps):
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
        tgt = torch.tensor(ids[seed:], device=device)[: log_ps.size(0)]
        extra["ceiling"] += float(fidelity(log_ps, l, 1.0, m).sum())
        if not sample_target:
            extra["ntok"] += float(tgt.numel())
            extra["risky"] += float(log_pr.gather(1, tgt.reshape(-1, 1)).sum())
            extra["safe"] += float(log_ps.gather(1, tgt.reshape(-1, 1)).sum())
        for o in orders:
            for k in ks:
                th = theta_for(o, log_ps, l, k)
                acc[(o, k)] += (float(fidelity(log_ps, l, th, m).sum()) if sample_target
                                else logp_target(log_ps, l, th, tgt))
        n += 1
        if (i + 1) % 5 == 0:
            print(f"[of] {'ordinary' if sample_target else 'protected'} {i+1}/{len(ps)}", flush=True)
        del log_ps, log_pr, l
        if device == "cuda":
            torch.cuda.empty_cache()
    return acc, extra, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe-model", required=True)
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--orders", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--k-grid", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 14.0, 20.0])
    ap.add_argument("--published-k", type=float, nargs="+", default=[1.0, 3.0],
                    help="the budgets the audited alpha = 1 decoder is compared at; each must be on "
                         "the grid, since the matched target is read off it")
    ap.add_argument("--split", default="attack_train",
                    help="MUST be a split the risky model was fine-tuned on")
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--sample-tokens", type=int, default=160)
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="order_frontier")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="Every checkpoint in this repository is stored in bfloat16, so float32 "
                         "only upcasts and buys accumulation precision, not weight precision. "
                         "bfloat16 halves the memory, which is what lets a 7B pair share a card; "
                         "the control run in results/onset_prediction_orders_matched.md measures "
                         "what it costs. logits_along() casts to float32 either way.")
    a = ap.parse_args()
    for pk in a.published_k:
        assert pk in a.k_grid, f"published k {pk} must be on the grid"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.safe_model)
    dt = getattr(torch, a.dtype)
    safe = AutoModelForCausalLM.from_pretrained(a.safe_model, dtype=dt).to(device).eval()
    risky = AutoModelForCausalLM.from_pretrained(a.risky_model, dtype=dt).to(device).eval()

    corpus = load_prompt_corpus("data", "factscore_prompt")
    prot = [p for p in corpus if p.split == a.split and p.reference][:a.limit]
    ordn = [p for p in corpus if p.split == "neutral"][:a.limit]
    ks = sorted(a.k_grid)

    leak, lx, n_prot = run_side(safe, risky, tok, prot, a.orders, ks, device,
                                a.seed_tokens, a.max_tokens, a.sample_tokens, False)
    price, px, n_ord = run_side(safe, risky, tok, ordn, a.orders, ks, device,
                                a.seed_tokens, a.max_tokens, a.sample_tokens, True)

    ntok = lx["ntok"] or 1.0
    os.makedirs(a.out, exist_ok=True)
    rows = [dict(alpha=o, k=k, split=a.split, n_protected=n_prot, n_ordinary=n_ord,
                 price_nats=round(price[(o, k)], 3),
                 price_frac_ceiling=round(price[(o, k)] / px["ceiling"], 4),
                 logp_target=round(leak[(o, k)], 3),
                 logp_per_token=round(leak[(o, k)] / ntok, 4),
                 n_tokens=int(ntok),
                 logp_target_risky=round(lx["risky"], 3),
                 logp_target_safe=round(lx["safe"], 3))
            for o in a.orders for k in ks]
    with open(os.path.join(a.out, f"{a.prefix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # The bracket gates the whole reading, so it is checked at every cell rather than once. Cells
    # at the top of the grid legitimately touch the upper bound: once the budget stops binding,
    # theta saturates at 1 and the served distribution IS the risky model, so equality there is
    # the instrument working. Only a cell strictly outside is a failure.
    tol = 1e-6 * max(1.0, abs(lx["safe"]))
    bad = [r for r in rows
           if r["logp_target"] < lx["safe"] - tol or r["logp_target"] > lx["risky"] + tol]
    sat = [r for r in rows if abs(r["logp_target"] - lx["risky"]) <= tol]
    print(f"\nbracket: risky {lx['risky']:.1f} nats on the protected tokens, anchor {lx['safe']:.1f}; "
          f"{len(bad)} of {len(rows)} cells outside it, {len(sat)} at the unbinding ceiling")
    if bad:
        print("  OUTSIDE:", [(r["alpha"], r["k"], r["logp_target"]) for r in bad[:8]])

    matched, win = [], 50.0
    for pk in a.published_k:
        base_f, base_l = price[(a.orders[0], pk)], leak[(a.orders[0], pk)]
        for o in a.orders:
            fs = [price[(o, k)] for k in ks]
            kp = pk if o == a.orders[0] else interp(ks, fs, base_f)
            lp = (base_l if o == a.orders[0]
                  else (None if kp is None else _at(ks, [leak[(o, k)] for k in ks], kp)))
            matched.append(dict(
                published_k=pk, alpha=o, k_matched=(round(kp, 4) if kp is not None else ""),
                price_nats=round(base_f, 3),
                logp_target=(round(lp, 3) if lp is not None else ""),
                nats_per_window=(round(win * (base_l - lp) / ntok, 3) if lp is not None else ""),
                off_grid=(kp is None)))
    with open(os.path.join(a.out, f"{a.prefix}_matched.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matched[0])); w.writeheader(); w.writerows(matched)

    print("\nat MATCHED fidelity: the budget each order needs to buy what alpha = 1 buys at the")
    print("published k, and what it lets through there\n")
    print(f"{'published k':>12s}{'alpha':>7s}{'k matched':>11s}"
          f"{'log p/token':>13s}{'nats/window':>13s}{'window x less likely':>22s}")
    for r in matched:
        if r["off_grid"]:
            print(f"{r['published_k']:>12.2f}{r['alpha']:>7.0f}{'off grid':>11s}")
            continue
        d = r["nats_per_window"]
        print(f"{r['published_k']:>12.2f}{r['alpha']:>7.0f}{r['k_matched']:>11.3f}"
              f"{r['logp_target']/ntok:>13.4f}{d:>13.2f}"
              f"{('1' if d <= 0 else f'{math.exp(min(d, 700)):.3g}'):>22s}")
    print("\nSame utility to the deployer by construction. If the window factor is still large the")
    print("order dominates on the axis Appendix D concedes it lacks; if it is near 1 the four arms")
    print("trace one frontier and Table 1's ranking is an artefact of comparing at matched budget.")
    print(f"wrote {os.path.join(a.out, a.prefix)}.csv and _matched.csv")


def _at(xs, ys, x):
    """Piecewise-linear ys at x."""
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0
    return ys[0] if x < xs[0] else ys[-1]


if __name__ == "__main__":
    main()
