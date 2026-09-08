# Session handoff — 2026-09-08 (plan v5, day 1)

## Current objective
Convert the audit into a **predictive theory** for an ICLR 2027 oral. Abstract Sep 18, paper Sep 25.
Plan is `.claude-private/plans/radiant-stargazing-newell.md` (plan v5). Branch `iclr-2027`.
`master` holds the verified SaTML paper at `dd7e801` as the fallback and must not be deleted.

## State
**The paper is structurally complete and compiles**: `~/sub/satml/iclr_2027.tex`, main text **9 of 9
pages**, 0 overfull, 0 `??`, 16 pages total. Abstract, intro, four result sections, related work,
limitations, conclusion, and the ICLR-required Ethics / Reproducibility / LLM Usage statements.
93 tests pass. **Note: the manuscript is NOT in this git repo** (it lives in `~/sub/satml`, inside a
stray home repo that must never be committed to) — only `results/`, `analysis/`, `recipes/` and the
harness files are versioned here.

Done today:
- **Theorem 1, the no-free-lunch** (feat-046). Sequence-level Donsker-Varadhan: `K >= Lambda*_s(E[U])`.
  Paired with Prop 1 it says utility and extraction draw on one scalar. Proof in the appendix.
- **Prop 1 and k_crit repositioned** as the known Renyi change-of-measure bound and the Loynes /
  network-calculus workload maximum; our proof of the latter deleted.
- **alpha=4 priced** — the run simply had never been done. It does NOT interpolate (83% of steps
  touched vs 2.4% for alpha=2), which corrected the section's claim about "the middle of the family".
- **Errata**: the per-token robustness sentence had no CSV (now measured, and the effect is *smaller*
  per token than claimed); the curves cross at 0.024 not 0.022; "88.2%" matched no cell (90.1%).
- **Bibliography**: 8 entries added, 6 verified field-by-field against primary records.

## Running right now
- `output/phase5/ft_v2.log` — five self-paired memorisers at 40 epochs / rank 128 / lr 3e-4.
  A monitor is armed on this log for ADMISSIBLE/failure lines.
- `output/phase5/util_fine.log` — `h1.py` at k in {1.5, 2.0, 2.5}, 1500 ordinary trajectories each,
  to pin the utility crossover that currently jumps k=1 -> k=3 across s(x)=3.24.

## Recommended next step
1. When `ft_v2` finishes, keep only pairs whose **sampled** k=-1 recall >= 0.10 (the recipe now
   prints an ADMISSIBLE verdict). For each, run `budget_path.py` then `composition_attack.py` on a
   dense k grid, append to `results/onset_pairs.tsv` and `results/onset_theory_pairs.tsv`, and rerun
   `analysis/onset.py` and `analysis/onset_theory.py` (both are N-pair general now).
2. Judge the k in {1.5,2,2.5} arms plus alpha=4:
   `analysis/utility.py --extra-arm 'renyi a=4|renyi:4|3.0|output/phase4/util_renyi_4' ...`
3. The **anchor-temperature lever** (`--temperature` on `token_nats`/`onset_theory`) moves s(x) with
   the pair held fixed. Ready to use, not yet run.

## Open decision for the user
Plan v5 item F says to retitle around the no-free-lunch. The title is unchanged
("What a Divergence Budget Can and Cannot Certify About a Language Model") because AGENTS.md
requires asking before a title change.
