"""feat-159: give the METERED decoder the same reward model, and price the parity.

The Program Chairs' report says selection anchoring's gain may be the external reward model rather
than the budget placement, since the metered decoder fuses per-token probabilities and has no
sequence-level scorer. This arm gives it one: draw n trajectories from the metered decoder, score
every one with the SAME pointwise reward every selection arm uses, serve the argmax.

Best-of-n over a decoder with sequence budget K gives q(y) <= n q_K(y) by Proposition 1's union
bound, so D_inf(q || p_s) <= K + log n -- the budgets ADD. This is not a mechanism we propose; it
is the measurement the parity objection asks for, and its certificate is worse than either
component alone.

Two stages, so each card does its own reward pass and the reading is computed once on CPU:

  --score-dir output/phase5/tqaP_k3 --tag k3      # GPU: writes results/meter_parity_rewards_k3.csv
  --report                                        # CPU: gates, bands, results/meter_parity.csv

Bands and gates: results/onset_prediction_meter_parity.md, committed before any generation.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.selection_verifiable import boot, boot_gain, correct_tqa, extract_tqa  # noqa: E402
from analysis.verifiable_metered import arms, gold_map  # noqa: E402

CORPUS = "data/bench/triviaqa_factual.jsonl"
N_GRID = (1, 2, 4, 8, 16)
REWARD_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# k -> the committed arm in results/verifiable_metered_tqa.csv that G3 takes its interval from.
COMMITTED = "results/verifiable_metered_tqa.csv"
SELECTION_DIR = "output/phase5/tqa_sel64"


def questions(path=CORPUS):
    """prompt_id -> the TARGET question, which is the last few-shot block of prompt_text.

    selection_verifiable.py builds its reward prompt as `Question: {q}\\nAnswer:`, so the metered
    arm must be handed the same string or the two mechanisms are scored under different templates.
    """
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["prompt_id"]] = r["prompt_text"].split("\n\n")[-1]
    return out


def pipeline_of(gen_dir):
    """(target_model, anchor_model) as the run itself recorded them -- caution (at)'s two lines."""
    import glob
    for p in sorted(glob.glob(os.path.join(gen_dir, "trajectories_k*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = json.loads(line).get("metadata", {})
            return (m.get("target_model"), m.get("anchor_model"))
    return (None, None)


def score_dir(a):
    """GPU stage: one reward per (prompt, draw) for every k in this directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from analysis.selection_scaling import score_rewards
    qs = questions()
    by_k = arms(a.score_dir)
    assert by_k, f"no trajectories under {a.score_dir}"
    pairs, keys = [], []
    for k, by in sorted(by_k.items()):
        for pid, draws in sorted(by.items()):
            for i, (gen, _spend) in enumerate(draws):
                pairs.append((qs[pid], gen))
                keys.append((k, pid, i))
    print(f"[parity] {len(pairs)} (prompt, draw) pairs over k={sorted(by_k)}", flush=True)
    tok = AutoTokenizer.from_pretrained(REWARD_MODEL)
    rm = AutoModelForCausalLM.from_pretrained(REWARD_MODEL, dtype=torch.bfloat16,
                                              device_map={"": 0}).eval()
    sc = score_rewards(rm, tok, pairs, rm.device, batch_size=a.reward_batch_size)
    out = os.path.join(a.out, f"meter_parity_rewards_{a.tag}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "prompt_id", "rank", "reward"])
        for (k, pid, i), s in zip(keys, sc):
            w.writerow([k, pid, i, f"{s:.6f}"])
    print(f"wrote {out}")


def load_rewards(out_dir):
    """tag-independent: every cache in results/, keyed (k, prompt_id) -> [reward by rank]."""
    import glob
    rw = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "meter_parity_rewards_*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rw.setdefault((r["k"], r["prompt_id"]), []).append((int(r["rank"]), float(r["reward"])))
    return {key: [s for _, s in sorted(v)] for key, v in rw.items()}


def committed_interval(k_token):
    """G3's tolerance, derived from the committed CSV. Never typed into the pre-registration."""
    want = f"k={k_token}"
    for r in csv.DictReader(open(COMMITTED, encoding="utf-8")):
        if r["mechanism"] == "metered decoder" and r["arm"] == want:
            return float(r["acc_lo95"]), float(r["acc_hi95"]), float(r["acc"])
    return None


def best_of(cands, rewards, gold, n):
    """Serve the argmax of the first n draws; score by the committed alias containment."""
    ok = []
    for pid, draws in sorted(cands.items()):
        sc = rewards.get(pid)
        if not sc or len(sc) < n or len(draws) < n:
            continue
        i = max(range(n), key=lambda j: sc[j])
        ok.append(float(correct_tqa(extract_tqa(draws[i][0]), gold[pid])))
    return ok


def report(a):
    gold = gold_map(CORPUS)
    rw = load_rewards(a.out)
    rows, verdicts = [], {}
    dirs = [d for d in a.dirs if os.path.isdir(d)]
    assert dirs, "no metered directories found; --report must not pass by never running"

    ref_pipe = pipeline_of("output/phase5/tqa_metered")
    for d in dirs:
        by_k = arms(d)
        pipe = pipeline_of(d)
        for k, cands in sorted(by_k.items()):
            tag = f"metered k={k}"
            print(f"--- {tag}   ({d})")
            # G2 first: a pipeline mismatch is STRUCTURAL and nothing else is computed.
            if pipe != ref_pipe:
                print(f"    G2 pipeline {pipe} != committed {ref_pipe} "
                      f"-> FAIL (STRUCTURAL, no comparison was made)")
                rows.append(dict(arm=tag, gate="FAIL-G2"))
                continue
            print(f"    G2 pipeline matches {ref_pipe} -> PASS")
            rewards = {pid: rw.get((k, pid), []) for pid in cands}
            base = best_of(cands, rewards, gold, 1)
            if not base:
                print("    no rewards cached for this k; band NOT computed")
                rows.append(dict(arm=tag, gate="NO-REWARDS"))
                continue
            acc1, _, _ = boot(base, a.reps, a.seed + 1)
            ci = committed_interval(k)
            if ci is None:
                print(f"    G3 no committed counterpart at k={k} "
                      f"-> FAIL (STRUCTURAL, no comparison was made)")
                rows.append(dict(arm=tag, gate="FAIL-G3"))
                continue
            lo, hi, was = ci
            g3 = lo <= acc1 <= hi
            print(f"    G3 n=1 acc {acc1:.4f} in the committed k={k} arm's own "
                  f"[{lo:.4f}, {hi:.4f}] -> {'PASS' if g3 else 'FAIL (arm INVALID)'}")
            if not g3:
                rows.append(dict(arm=tag, gate="FAIL-G3", acc_n1=round(acc1, 4)))
                continue
            for n in N_GRID:
                ok = best_of(cands, rewards, gold, n)
                if len(ok) != len(base):
                    continue
                acc, alo, ahi = boot(ok, a.reps, a.seed + n)
                g, glo, ghi = boot_gain(ok, base, a.reps, a.seed + n)
                rows.append(dict(arm=tag, k=k, n=n, gate="PASS", n_questions=len(ok),
                                 acc=round(acc, 4), acc_lo95=round(alo, 4), acc_hi95=round(ahi, 4),
                                 gain=round(g, 4), gain_lo95=round(glo, 4),
                                 gain_hi95=round(ghi, 4)))
                if n == max(N_GRID):
                    hw = (ghi - glo) / 2
                    hwr = abs(g) / hw if hw else float("inf")
                    v = "HELPS" if glo > 0 else "HURTS" if ghi < 0 else "NO EFFECT"
                    verdicts[tag] = (v, g, glo, ghi, hwr)
                    print(f"    B1 acc(1)={acc1:.4f} acc(16)={acc:.4f}  gain {g:+.4f} "
                          f"[{glo:+.4f}, {ghi:+.4f}]  {hwr:.2f} hw  {v}"
                          f"  {'MARGINAL' if hwr < 2.0 else 'not marginal'}")

    # The selection side, same scorer, same call, re-used generations.
    if os.path.isdir(SELECTION_DIR):
        by_k = arms(SELECTION_DIR)
        for k, cands in sorted(by_k.items()):
            rewards = {pid: rw.get((k, pid), []) for pid in cands}
            base = best_of(cands, rewards, gold, 1)
            if not base:
                continue
            print(f"--- selection over the anchor (reward-selected), {SELECTION_DIR} k={k}")
            for n in N_GRID:
                ok = best_of(cands, rewards, gold, n)
                if len(ok) != len(base):
                    continue
                acc, alo, ahi = boot(ok, a.reps, a.seed + n)
                rows.append(dict(arm="selection (reward)", k=k, n=n, gate="PASS",
                                 n_questions=len(ok), acc=round(acc, 4),
                                 acc_lo95=round(alo, 4), acc_hi95=round(ahi, 4)))
                print(f"    n={n:<3d} acc {acc:.4f} [{alo:.4f}, {ahi:.4f}]")

    p = os.path.join(a.out, "meter_parity.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {p}")
    met = [r for r in rows if r.get("arm", "").startswith("metered") and r.get("acc")]
    sel = [r for r in rows if r.get("arm") == "selection (reward)" and r.get("acc")]
    if met and sel:
        bm, bs = max(met, key=lambda r: r["acc"]), max(sel, key=lambda r: r["acc"])
        print(f"\nB2 best metered  {bm['arm']} n={bm['n']}  acc {bm['acc']}")
        print(f"B2 best selection n={bs['n']}  acc {bs['acc']}")
        print("B2 READING: " + ("PARITY MATTERS" if bm["acc"] > bs["acc"]
                                else "PARITY DOES NOT MATTER"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", help="GPU stage: reward-score every draw in this run directory")
    ap.add_argument("--tag", default="", help="names the reward cache for --score-dir")
    ap.add_argument("--report", action="store_true", help="CPU stage: gates, bands, CSV")
    ap.add_argument("--dirs", nargs="*", default=[], help="metered run directories for --report")
    ap.add_argument("--reward-batch-size", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.score_dir:
        assert a.tag, "--tag names the reward cache and must be given"
        return score_dir(a)
    assert a.report, "give --score-dir (GPU) or --report (CPU)"
    report(a)


if __name__ == "__main__":
    main()
