"""How much does greedy metering leave on the table? (plan v5 / feat-070)

The audited decoder is myopic: at every step it serves the largest theta whose charge fits the
token bucket's current allowance, so it spends its whole allowance at every step regardless of
whether tilting toward the risky model buys anything there. Theorem 1 says the budget needed for a
given mean utility is at least the Cramer rate function; the measured gap is three to four orders
of magnitude, and the paper's stated open problem is whether that gap is an artifact of the causal,
greedy token bucket or something structural.

This answers the first half of that offline, with no change to the decoder. On the geometric path
with l = log p_r - log p_s and psi(u) = log E_{p_s}[e^{u l}],

    charge   C(theta) = D_KL(p_theta || p_s) = theta psi'(theta) - psi(theta),   C' = theta psi''
    fidelity G(theta) = -D_KL(p_r || p_theta) + const = theta m - psi(theta),    G' = m - psi'

with m = psi'(1) = E_{p_r}[l]. Note C'(0) = 0 and G'(0) = m - psi'(0) is the Jeffreys divergence
between the two models, which is the quantity the exact reward-KL frontier turns out to be governed
by (Monteiro Paes et al.), so the first nat bought at a step is worth its Jeffreys divergence and
the last is worth nothing.

Greedy sets G'/C' to whatever it happens to be at the step's allowance. The optimal allocation of a
FIXED total budget across steps equalises the marginal price: serve theta_t solving
G'(theta) = lambda C'(theta) with one lambda shared by every step. So for each passage we

  1. walk the anchor and the memoriser along the target, teacher-forced,
  2. take greedy's per-step charge at the steady-state allowance k and total it,
  3. find the lambda whose equal-price allocation spends the same total,
  4. compare the fidelity the two allocations buy.

The gap between them is what a non-greedy decoder could recover at the same published budget, and
it needs no decoding run to measure. If it is small, greedy is near-optimal within this family and
the overhead Theorem 1 exposes is structural; if it is large, the decoder -- not the accounting --
is what makes a metered budget expensive.

Writes <out>/marginal_price.csv (per passage) and <out>/marginal_price_summary.csv.

  CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
    HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/marginal_price.py \
      --safe-model output/phase5/anchor_kl3m-002-520m \
      --risky-model output/phase5/mem_kl3m-002-520m --k 3.0 --limit 50 --out results
"""
from __future__ import annotations

import argparse, csv, math, os, statistics as st, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dap.shared import load_prompt_corpus  # noqa: E402
from recipes.finetune_memorizing import join  # noqa: E402


def _psi(log_ps, l, u):
    """psi(u) = log sum_v p_s(v) e^{u l(v)}, and the mean/variance of l under p_u.

    Shapes: log_ps, l are [T, V]; u is [T] or a scalar. Returns three [T] tensors.
    """
    if not torch.is_tensor(u):
        u = torch.full((log_ps.size(0),), float(u), device=log_ps.device, dtype=log_ps.dtype)
    w = log_ps + u.reshape(-1, 1) * l
    psi = torch.logsumexp(w, dim=-1)
    p = (w - psi.reshape(-1, 1)).exp()
    m1 = (p * l).sum(-1)
    m2 = (p * l * l).sum(-1)
    return psi, m1, (m2 - m1 * m1).clamp(min=0.0)


def charge(log_ps, l, theta):
    psi, m1, _ = _psi(log_ps, l, theta)
    if not torch.is_tensor(theta):
        theta = torch.full_like(m1, float(theta))
    return theta * m1 - psi


def fidelity(log_ps, l, theta, m):
    """G(theta) = theta*m - psi(theta), the negative of D(p_r || p_theta) up to a per-step constant.

    Reported relative to G(0) = 0, so it is the fidelity *bought* at that step, in nats, and
    G(1) = m - psi(1) = D(p_r || p_s) is the most any budget can buy there."""
    psi, _, _ = _psi(log_ps, l, theta)
    if not torch.is_tensor(theta):
        theta = torch.full_like(psi, float(theta))
    return theta * m - psi


def solve_theta_greedy(log_ps, l, k_t, iters: int = 40):
    """Largest theta in [0, 1] with C(theta) <= k_t, per step. C is increasing from C(0) = 0."""
    T = log_ps.size(0)
    lo = torch.zeros(T, device=log_ps.device, dtype=log_ps.dtype)
    hi = torch.ones_like(lo)
    if not torch.is_tensor(k_t):
        k_t = torch.full_like(lo, float(k_t))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        feas = charge(log_ps, l, mid) <= k_t
        lo = torch.where(feas, mid, lo)
        hi = torch.where(feas, hi, mid)
    return torch.where(charge(log_ps, l, torch.ones_like(lo)) <= k_t, torch.ones_like(lo), lo)


def solve_theta_dual(log_ps, l, m, lam: float, iters: int = 40):
    """theta solving G'(theta) = lambda C'(theta), i.e. m - psi'(theta) = lambda theta psi''(theta).

    F(0) = m - psi'(0) is the Jeffreys divergence >= 0 and F(1) = -lambda Var_{p_r}[l] <= 0, so a
    sign change exists in [0, 1] and bisection finds it. F is not guaranteed monotone -- theta psi''
    need not be increasing -- so this returns *a* stationary point, which is why the caller checks
    the realised objective rather than trusting the root.
    """
    T = log_ps.size(0)
    lo = torch.zeros(T, device=log_ps.device, dtype=log_ps.dtype)
    hi = torch.ones_like(lo)

    def F(th):
        _, m1, v = _psi(log_ps, l, th)
        return m - m1 - lam * th * v

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = F(mid) > 0
        lo = torch.where(pos, mid, lo)
        hi = torch.where(pos, hi, mid)
    return 0.5 * (lo + hi)


@torch.no_grad()
def logits_along(model, ids, device):
    x = torch.tensor([ids], device=device)
    return model(x).logits[0, :-1].float()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe-model", required=True)
    ap.add_argument("--risky-model", required=True)
    ap.add_argument("--k", type=float, default=3.0, help="steady-state per-token allowance")
    ap.add_argument("--seed-tokens", type=int, default=20)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--split", default="attack_train",
                    help="MUST be a split the risky model was fine-tuned on; `test` is a "
                         "held-out novel and the memoriser is worse than its own base on it")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="marginal_price")
    ap.add_argument("--max-tokens", type=int, default=400, help="cap per passage, for speed")
    ap.add_argument("--corpus", default="factscore_prompt",
                    help="prompt set; factscore_prompt carries the protected references")
    ap.add_argument("--sample-target", action="store_true",
                    help="imitate a sample from the risky model instead of a protected reference. "
                         "This is the utility side: the decoder is trying to reproduce p_r's own "
                         "behaviour on an ordinary prompt, which is what Theorem 1 prices.")
    ap.add_argument("--sample-tokens", type=int, default=160)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.safe_model)
    safe = AutoModelForCausalLM.from_pretrained(a.safe_model, torch_dtype=torch.float32).to(device).eval()
    risky = AutoModelForCausalLM.from_pretrained(a.risky_model, torch_dtype=torch.float32).to(device).eval()

    ps = [p for p in load_prompt_corpus("data", a.corpus)
          if p.split == a.split and (a.sample_target or p.reference)][:a.limit]
    rows = []
    for i, p in enumerate(ps):
        if a.sample_target:
            # the target is p_r's own sample, so "fidelity" is fidelity to the behaviour the
            # decoder is actually trying to buy on an ordinary prompt
            prompt_ids = tok(p.prompt_text).input_ids[: a.max_tokens]
            with torch.no_grad():
                out = risky.generate(torch.tensor([prompt_ids], device=device),
                                     max_new_tokens=a.sample_tokens, do_sample=True,
                                     top_k=0, top_p=1.0, temperature=1.0,
                                     pad_token_id=tok.eos_token_id or 0)
            ids = out[0].tolist()
            seed = len(prompt_ids)
        else:
            ids = tok(join(p.prompt_text, p.reference)).input_ids[: a.max_tokens]
            seed = a.seed_tokens
        if len(ids) <= seed + 8:
            continue
        ls = logits_along(safe, ids, device)[seed - 1:]
        lr = logits_along(risky, ids, device)[seed - 1:]
        log_ps = torch.log_softmax(ls, dim=-1)
        log_pr = torch.log_softmax(lr, dim=-1)
        l = torch.nan_to_num(log_pr - log_ps, nan=0.0, posinf=0.0, neginf=0.0)
        m = (log_pr.exp() * l).sum(-1)                      # psi'(1) = E_{p_r}[l]
        _, m0, _ = _psi(log_ps, l, 0.0)
        jeffreys = (m - m0)                                 # G'(0), the first nat's worth

        th_g = solve_theta_greedy(log_ps, l, a.k)
        c_g = charge(log_ps, l, th_g)
        g_g = fidelity(log_ps, l, th_g, m)
        total_budget = float(c_g.sum())
        ceiling = float(fidelity(log_ps, l, torch.ones_like(th_g), m).sum())

        # equal-marginal-price allocation at the same total spend: bisect lambda on total charge
        lo, hi = 1e-6, 1e6
        for _ in range(60):
            lam = math.sqrt(lo * hi)
            th_d = solve_theta_dual(log_ps, l, m, lam)
            if float(charge(log_ps, l, th_d).sum()) > total_budget:
                lo = lam
            else:
                hi = lam
        lam = math.sqrt(lo * hi)
        th_d = solve_theta_dual(log_ps, l, m, lam)
        c_d = charge(log_ps, l, th_d)
        g_d = fidelity(log_ps, l, th_d, m)

        # How heterogeneous is the marginal price greedy happens to land on? Only the unsaturated
        # steps can answer: where theta = 1 the price is exactly 0 because G'(1) = m - psi'(1) = 0,
        # and including those makes the quartile ratio diverge rather than describe anything.
        eps = 1e-9
        _, m_g, v_g = _psi(log_ps, l, th_g)
        price_g = (m - m_g) / (th_g * v_g + eps)
        finite = torch.isfinite(price_g) & (th_g < 0.999) & (th_g > 1e-3)
        rows.append(dict(
            prompt_id=p.prompt_id, novel=getattr(p, "novel", ""), n_steps=int(th_g.numel()),
            k=a.k,
            spend_greedy=round(total_budget, 4),
            spend_dual=round(float(c_d.sum()), 4),
            fidelity_greedy=round(float(g_g.sum()), 4),
            fidelity_dual=round(float(g_d.sum()), 4),
            fidelity_ceiling=round(ceiling, 4),
            gain_ratio=round(float(g_d.sum()) / max(float(g_g.sum()), eps), 4),
            frac_of_ceiling_greedy=round(float(g_g.sum()) / max(ceiling, eps), 4),
            frac_of_ceiling_dual=round(float(g_d.sum()) / max(ceiling, eps), 4),
            theta_greedy_mean=round(float(th_g.mean()), 4),
            theta_dual_mean=round(float(th_d.mean()), 4),
            theta_greedy_at_one_pct=round(100 * float((th_g >= 0.999).float().mean()), 2),
            theta_dual_at_zero_pct=round(100 * float((th_d <= 1e-3).float().mean()), 2),
            jeffreys_median=round(float(jeffreys.median()), 4),
            price_iqr_ratio=round(float(torch.quantile(price_g[finite], 0.75)
                                        / (torch.quantile(price_g[finite], 0.25) + eps)), 3)
            if int(finite.sum()) > 4 else "",
            lam=round(lam, 6),
        ))
        # the per-step tensors are [T, V] and the lambda bisection allocates thousands of them;
        # without this the caching allocator grew to 80 GB on a 0.5B model and the run crawled.
        del ls, lr, log_ps, log_pr, l, th_g, th_d, c_g, c_d, g_g, g_d, price_g
        if device == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"[mp] {i+1}/{len(ps)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.prefix}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    f = lambda key: [float(r[key]) for r in rows]
    summary = dict(
        n_passages=len(rows), k=a.k,
        spend_greedy_mean=round(st.mean(f("spend_greedy")), 3),
        fidelity_greedy_mean=round(st.mean(f("fidelity_greedy")), 3),
        fidelity_dual_mean=round(st.mean(f("fidelity_dual")), 3),
        gain_ratio_mean=round(st.mean(f("gain_ratio")), 4),
        gain_ratio_median=round(st.median(f("gain_ratio")), 4),
        frac_of_ceiling_greedy=round(st.mean(f("frac_of_ceiling_greedy")), 4),
        frac_of_ceiling_dual=round(st.mean(f("frac_of_ceiling_dual")), 4),
        theta_greedy_at_one_pct=round(st.mean(f("theta_greedy_at_one_pct")), 2),
        theta_dual_at_zero_pct=round(st.mean(f("theta_dual_at_zero_pct")), 2),
    )
    with open(os.path.join(a.out, f"{a.prefix}_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary)); w.writeheader(); w.writerow(summary)

    print(f"\nat k = {a.k}, on {len(rows)} passages, both allocations spending the same total:")
    print(f"  greedy   fidelity {summary['fidelity_greedy_mean']:>9.3f} nats "
          f"({100*summary['frac_of_ceiling_greedy']:.1f}% of the theta=1 ceiling)")
    print(f"  equal-price  fidelity {summary['fidelity_dual_mean']:>9.3f} nats "
          f"({100*summary['frac_of_ceiling_dual']:.1f}% of the ceiling)")
    print(f"  ratio {summary['gain_ratio_mean']:.3f} mean, {summary['gain_ratio_median']:.3f} median")
    print(f"  greedy serves theta=1 at {summary['theta_greedy_at_one_pct']:.1f}% of steps; "
          f"equal-price serves theta=0 at {summary['theta_dual_at_zero_pct']:.1f}%")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
