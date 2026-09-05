"""Teacher-forced logit warping (feat-022): the repetition penalty and temperature that He et al. (App. B, D.1) apply to
both models' logits before the KL solve, so that surprisals and certificate caps can be computed under the warped
anchor p_s^(tau, rho). Semantics match HF's RepetitionPenaltyLogitsProcessor (score / rho if score > 0 else
score * rho, for tokens already in the context) followed by TemperatureLogitsWarper (score / tau)."""
import torch


def warp_logits(logits: torch.Tensor, ids: torch.Tensor, temperature: float = 1.0, repetition_penalty: float = 1.0, offset: int = 0) -> torch.Tensor:
    """logits [T, V]; row t is warped with the context ids[:t + offset] (offset=1 when row t predicts ids[t + 1] of a
    full sequence whose logits were sliced as logits[:-1], ids[:-1])."""
    out = logits
    if repetition_penalty != 1.0:
        T, V = logits.shape
        first = torch.zeros(T, V, dtype=torch.bool, device=logits.device)
        ctx = ids[: T - 1 + offset]
        rows = torch.arange(1 - offset, T, device=logits.device)[: ctx.numel()]
        rows = rows[rows >= 0]
        ctx = ctx[ctx.numel() - rows.numel():] if rows.numel() < ctx.numel() else ctx
        first[rows, ctx.to(logits.device)] = True
        seen = first.cumsum(0) > 0
        pen = torch.where(out > 0, out / repetition_penalty, out * repetition_penalty)
        out = torch.where(seen, pen, out)
    if temperature != 1.0:
        out = out / temperature
    return out
