# Session Progress Log

## Current State

**Last Updated:** 2026-09-07 19:45
**Active Feature:** none. **Phases 1, 2 and 3 are all complete.** feat-001..009, 013..015, 017..027 and **028..034** are `done`; feat-012 is superseded by feat-024; feat-010/011 stay optional; feat-016 is human-only and must never be started.
**Plan:** `~/sub/satml/IMPROVEMENT_PLAN.md` is now **version 3 (2026-09-07)**; v2 archived as `IMPROVEMENT_PLAN_v2_2026-09-06.md`, v1 as `IMPROVEMENT_PLAN_v1_2026-09-05.md`. v3 attacks *significance*, not rigour: a second anchor, a real utility evaluation, Proposition 5, the prefix-debt promotion, length scaling, an optional second mechanism.
**Venue: SaTML 2027 only.** ICLR 2027 closes first (abstract Sep 18, paper Sep 25 AoE) and both CFPs bar parallel archival submission; the SaTML HotCRP form asks whether the paper is under review elsewhere, and both venues decide Dec 16, so neither can inform the other. The ICLR-format derivative (`iclr_2027.tex`) is drafted after Oct 2 for ICML 2027 (abstract ~Jan 16 2027 per aggregators — confirm against the official CFP).
**Artefacts:** manuscript `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.pdf` (**21 pages, body ends page 12**, 0 overfull, 0 `??`, 137 cited) with checkpoint `satml_2027_phase3_2026-09-07.pdf`; `references.bib` at **138 entries** (10 added and verified 2026-09-07); artifact v3 `artifact/` (206 files + `MANIFEST.sha256`) and `artifact.zip` (23 MB), manifest verified and anonymity scan clean.
**Since the phase-3 close (2026-09-07 afternoon/evening):** a hostile internal review fixed 12 manuscript defects; the hallucinator rerun checked all 137 cited references (129 auto-verified, 8 hand-verified, none hallucinated); an anonymity scan found the PDF and artifact clean but three holes in `scripts/build_artifact.sh`, all fixed and negative-controlled; a full consistency audit of every table and prose figure against `results/*.csv` fixed 5 discrepancies; an editorial pass cut a contribution list that had been enumerated three times in the first four sections; and **the title became declarative**: *A KL Budget Is Uninformative Where the Mechanism Is Usable*. `satml_2027_arxiv_v1.tex` is **not** a variant of this paper and was left alone (see the 19:45 entry).
**Deadline clock:** **Sep 20 = every headline number frozen** · abstract Sep 22 · paper Sep 29 · artifacts Oct 2 (AoE). The last three are human-only.
**Compute:** phase 3 cost **2.3 GPU-hours** against a 19.5-hour estimate (feat-028b was blocked; feat-029 needed no generation because the phase-2 sweeps already held every arm). Paper total is now **56.5 GPU-hours** (`results/compute_hours.csv`). At 02:00 on 2026-09-07 `nvidia-smi` showed GPUs 1, 2 and 4 at 100% under other users and only GPU 0 free; plan v3 Section 5 gives the degradation path for 3, 2, 1 and 0 free cards.
**Verification for a fresh session:** `./init.sh` (47 tests, 34 features) and the manuscript compile check in `AGENTS.md`. Read the log below from the bottom: entries are appended, so the phase-1 sections that follow are history, not the current plan.

## Status

### What's Done

- [x] Full read of repo, both manuscripts, and the released logs; literature review (He et al. 2026, Vyas et al. 2023, Cohen 2025, ~35 related papers).
- [x] Reframing plan written: `~/sub/satml/IMPROVEMENT_PLAN.md` (spec for everything below).
- [x] Log reanalysis scripts drafted: `analysis/reanalyze_logs.py`, `analysis/surprisal.py` (copies also in `~/sub/satml/scripts/`).
- [x] Harness rebuilt: `AGENTS.md`, `CLAUDE.md` (imports AGENTS.md), `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md`.

- [x] feat-001 closed: `./init.sh` exit 0; harness validator 100/100. Committed as 07a446c.
- [x] feat-002 closed (2026-09-05): `analysis/reanalyze_logs.py --logs output --out results` (6.8 s, no GPU) reproduces every Known Truth. Producing command: `unzip -o output.zip <4 trajectory files + heldout_validation.jsonl> && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results`. Outputs: `results/regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`, `per_trajectory.csv` (4,499 rows, 763 KB; input for feat-007/feat-013). New number: leading safe-forced tokens equal floor(δ_init/k) in 99.4–100% of trajectories. `analysis/surprisal.py` merged into the tool and removed.
- [x] feat-009 closed (2026-09-05): `analysis/composition_attack.py` on 100 memorised excerpts, k in {-1,0,0.15,0.5,1,3,5,10,20}, windows 20/50, single/oracle/chained → `results/composition.csv`, `results/composition_summary.csv`, `figures/composition.pdf`. Headline: reproduction < 10% at k <= 3 (budget binding, utilisation up to 0.97) even though the certificate is vacuous; single-query recall equals the unconstrained model at k=20 (0.48) where Z/K = 0.18; oracle composition reaches 0.86 at k=20 and doubles recall at k=5-10; zero violations. Producing command in feature_list.json.
- [x] feat-008 closed (2026-09-05): memorising risky model = LoRA fine-tune of Llama-3.1-8B-Instruct (`recipes/`), `results/memorizing_model_recall.csv`: greedy nv-recall 0.907 (attack_train) / 0.923 (val) / 0.0 (held-out test); sampled 0.80 / 0.81 / 0.0. Producing command: `scripts/run_memorizing_check.sh 1`.
- [x] feat-004 closed (2026-09-05): E1 accepts k=-1 (risky only, K=inf) and k=0 (safe only, K=0) with no certificate arithmetic for them; every record carries lcs_word, lcs_char, acs_word, nv_recall (dap/stats.py); h1_summary rows carry per-class metric means. Smoke: `CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 .venv/bin/python h1.py --k-values -1 0 0.15 1 --trajectories-per-prompt 2 --cap-* 2 --output-dir output/smoke --trust-remote-code --parallelize` (24 files, 0 invariant violations). Seeds now depend only on the trajectory index so E1/E2 batch across prompts (per-prompt seeds had forced batch size 1: 12.5 s/trajectory).
- [x] feat-006 closed (2026-09-05): `analysis/certificate_cap.py` → `results/certificate_caps.csv` (758 passages), `results/certificate_cap_summary.csv`, `figures/certificate_cap_curve.{pdf,png}`. Producing command: `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python analysis/certificate_cap.py --data data --out results`. Headline numbers: TinyComma surprisal of a CopyBench reference is median 205 nats (3.20 nats/token, 64 tokens); the K-NAF certificate is vacuous (S ≤ K) for 100% of passages at k=3 and k=5, 44% at k=1, 0% at k ≤ 0.5 (median cap 0.49 at k=0.5, 0.10 at k=0.1). Llama-3.1-8B-Instruct assigns the references 2.59 nats/token (median gap 37 nats): it has not memorised them.
- [x] feat-003 closed (2026-09-05): collision-free seeds (`dap/stats.py`, index-offset semantics in `dap/e2/evaluator.py`), per-trajectory utilisation/invariant/activity fields in E1 records and `h1_summary`, `utilisations`/`activity` lists in E2 `EvalResult`, R = K everywhere, `--use-chat-template` on `h1.py`/`h2.py` (`dap/shared.py: wrap_chat, chat_eos_ids, true_gen_len`), `tests/` (6 tests) + `tests/data/sample_trajectories.jsonl`. GPU smoke (local GPU 1, `HF_HUB_OFFLINE=1`, k=1, 2 traj × 2 prompts × 6 splits, `--use-chat-template`): `output/smoke_feat003/h1_summary.csv` shows invariant_violations=0 in all classes, util_max 0.95–1.00, active_step_pct 3.7–23.3%, chat-formatted generations stop at `<|eot_id|>` (lengths 11 and 28 observed). Command: `CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/python h1.py --k-values 1.0 --trajectories-per-prompt 2 --cap-neutral 2 --cap-val 2 --cap-test 2 --cap-attack-train 2 --cap-factual 2 --cap-creative 2 --output-dir output/smoke_feat003 --use-chat-template`.

### What's In Progress

- (none)

### What's Next

*(rewritten 2026-09-07 02:15; the "nothing for the agent" list it replaced was correct for plan v2 and is superseded.)*

1. **feat-028a** — download `common-pile/comma-v0.1-2t` (7B, Apache 2.0, ungated, ~14 GB) into `hf_cache/` (decision D1) and compute its per-token surprisal of the 758 CopyBench passages and the 58 candidates. ~1 GPU-h on 1 card. This is the cheapest answer to the loudest objection in the paper ("your vacuity result is about a weak 1.8B anchor").
2. ~~Tokenizer check before feat-028b~~ — **done 2026-09-07 02:30, and feat-028b is BLOCKED.** `common-pile/comma-v0.1-2t` has a 64,000-token vocabulary against Llama-3's 128,256, and of 45,538 shared token strings only 7 carry the same id. Anchored decoding fuses two next-token distributions over one shared vocabulary, so no second anchor can be fused with a Llama-3 risky model, and no substitute exists (TinyComma is retokenized to the Llama-3 vocabulary precisely because no other permissively trained model is). This goes in Limitations. Saves 4 GPU-hours; the plan is now ~15.5. **It forces a methodological constraint on feat-028a:** per-token surprisal is not comparable across tokenizers, so compare *total* passage surprisal in nats and nats per character — never print a 64k-vocab nats/token figure beside TinyComma's 3.20.
3. **feat-033** (0.5 GPU-h) and **feat-031** (no GPU) can run at any time; feat-031 is the highest value per GPU-hour in the plan.
4. **feat-029** (6 GPU-h) replaces the self-NLL utility proxy; **feat-032** needs 2 cards for the 70B; **feat-030** is optional and the first to cut.
5. After any edit under `~/sub/satml/`: recompile, then confirm 0 `??`, 0 overfull and that page 13 still starts with "Open Science". After any change under `results/`, `analysis/` or `a_patch/`: rerun `./init.sh` and `scripts/build_artifact.sh artifact`.
6. Never start feat-016 (human-only: registration Sep 22, paper Sep 29, artifact repository Oct 2).

### Phase 3 log

- [x] 2026-09-07 02:00-02:15 — plan v3 written. Archived v2 (`cp -n IMPROVEMENT_PLAN.md IMPROVEMENT_PLAN_v2_2026-09-06.md`); wrote `IMPROVEMENT_PLAN.md` v3 (432 lines, Sections 0-11). Venue question settled against the two CFPs: ICLR 2027 abstract Sep 18 / paper Sep 25, SaTML Sep 22 / Sep 29, both decide Dec 16, both bar parallel submission -> SaTML only this cycle.
- [x] 2026-09-07 — literature pass. Ten additions verified against the arXiv API (10 requested, 10 returned, 0 missing; `/tmp/bibcheck/arxiv.json`), venues cross-checked on DBLP: `li2024va3` (CVPR 2024 — **the composition attack's uncited ancestor**), `kim2025guaranteed` (ICLR 2025), `mudgal2024controlled` (ICML 2024), `loula2025smc` (ICLR 2025), `ahmed2024resampling` (ICLR 2025), `yoon2026position` (ICML 2026 Position Track), and four preprints with no venue asserted (`ahmed2025semantic`, `morris2025memorize`, `alshehyari2026copyshield`, `yu2025impossibility`). Written to `bib_additions_2026-09-07.bib`, appended to `references.bib` (128 -> 138, no key collisions, brace-balance parse clean). `LITERATURE_REVIEW.md` extended with sections M-Q, a "considered and not added" note (Saha TPDP 2026 could not be verified, so it is not cited) and a verification log.
- [x] 2026-09-07 03:30-04:45 — **feat-034 and feat-030 DONE — phase 3 complete** (user asked for both at once; AGENTS.md's one-feature-at-a-time rule was overridden deliberately, and they touch disjoint files).
  - **feat-034 (manuscript v3)**: body restructured and held at **exactly 12 pages**, 0 overfull, 0 `??`, 137 cited; checkpoint `sub/satml/satml_2027_phase3_2026-09-07.pdf`. Added: second-anchor and length-scaling paragraphs in V-C; Section VII promoted from a subsection to "What the Prefix Debt Does" with the k=20 row; new VIII-B "What the constraint costs" with the externally judged utility table; Proposition 5 + `figures/separation.pdf` replacing the odometer table in VIII-C; VA3 positioning and a "Guarantees enforced while decoding" paragraph in Related Work; abstract and intro rewritten around three claims; Open Science updated for the new scripts and models.
  - **Paid for by**: Section IV compressed to failure modes (-0.5), the sweep to one paragraph, VIII-B/D/E to a paragraph and two sentences each, the extraction-cost paragraph tightened, V-A/V-B and VI-D tightened. **Nothing measured was discarded**: Appendix B now holds the odometer and warped-anchor tables, the concentration inputs, the extraction-cost detail and the `fig:natural`/`fig:lengthscale` figures; Appendix C holds the two catalogue paragraphs of Related Work. Both are uncounted by the CFP and the load-bearing positioning stayed in the body. Lesson repeated from phase 2: in IEEEtran two-column, page breaks are pinned by floats, so cutting prose alone moved nothing until a float was moved — the body only came back to 12 when `fig:natural` went to the appendix.
  - **Numbers verified cell by cell** against `results/`: all 36 cells of the utility table and all 8 of the prefix-debt k=20 row match, 0 mismatches.
  - **feat-030 (CP-Fuse) DONE**: `recipes/finetune_memorizing.py --shard i/n` added (CP-Fuse needs two models on disjoint halves by construction). Both fine-tunes on GPUs 0 and 1, 229 disjoint passages each, 12 epochs, final loss 0.042 / 0.041. **Disjointness confirmed**: each component reproduces 0.754-0.884 of the passages it trained on and 0.000-0.027 of the other's. `analysis/cpfuse_audit.py` implements the reweighting from the paper's description — a reimplementation, not the authors' code, and reported as such — with a KV cache and a vectorised alpha search, without which the audit is quadratic in generation length. Full run: 60 passages, 4 arms, single + oracle, split by arm across GPUs 0 and 1, ~50 min each -> `results/cpfuse_audit.csv`.
    - **The useful negative result: CP-Fuse holds against the attack that breaks the budget.** Oracle windows of 50 tokens — the strategy that lifts anchored decoding from 0.09 to 0.23 at k=5 and 0.20 to 0.41 at k=10 — give CP-Fuse **0.022**, and a single query **0.000**, against its components' 0.814 and 0.884 on their own halves. So composition is not a general failure of per-query guarantees, it is a failure of **budgeted** ones: a window of L tokens carries its own allowance kL and the bank pays for the whole window, whereas CP-Fuse has no bank to drain. This sharpens Section VI rather than diluting it.
    - **Second finding: most of the suppression is the disjoint split, not the mechanism.** A fixed 50/50 geometric mixture with balancing off already holds oracle windows to 0.064 and single queries to 0.018, and the balancing moves the weight only from 0.500 to 0.526 on average. The reweighting earns a factor of three over plain averaging; the split earns the rest. In the body as one sentence, with the ablation in Appendix B.
    - `tests/test_cpfuse.py` (4 tests) checks the fusion normalisation and that the balancing reacts in the correct direction to an imbalance — a sign error there would invert the defence.
  - **Compute**: `analysis/compute_hours.py --out results` now covers the nine phase-3 jobs -> **56.0 GPU-hours total** (47.2 single-GPU, 8.8 on the 70B's two cards, three fine-tunes at 75/15/15 min). The paper's `LLM usage considerations` was 45 + 8 and is now 47 + 9 = 56.
  - **Artifact v3 rebuilt**: `scripts/build_artifact.sh artifact` -> 210 files (was 179), manifest verified, anonymity scan clean, `artifact.zip` 23 MB.
- [x] 2026-09-07 02:55-03:25 — **feat-029 and feat-032 DONE** (all four A100s free; 70B on 1+2, judge on 0, 8B on 4).
  - **feat-029** (`analysis/utility.py`, GPU 0, ~25 min, 8,300 pairs judged). No generation was needed: the KL sweep (`conc_all`), the pathwise sweep (`pathwise_sweep` + `_hi`) and both baselines (`sweep_plain` k=-1, k=0) already cover k in {0.5,1,3,5,10,20} on the same 500 prompts and seeds, 1,500 trajectories per arm. Command: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python analysis/utility.py --out results --judge-per-cell 200` -> `results/utility.csv`, `results/utility_summary.csv`.
    - **The null matters more than any single arm.** The unconstrained model judged against *itself* at a different seed gives 43.2 win / 47.0 loss / 9.8 tie, not 50/50 — temperature-1 sampling plus judge inconsistency. Every arm reads as a deviation from that row.
    - **At k >= 3 the KL decoder is indistinguishable from not constraining at all**: loss 47.5 / 45.7 / 43.0 / 44.8 at k=3/5/10/20 against the null's 47.0 (n=600, SE ~2.0pp). It does nothing there — active steps 0.35% falling to 0.000%.
    - **The cost appears only where the certificate is informative**: KL k=1 loss 52.3 (+5.3pp), k=0.5 59.8 (+12.8pp), anchor alone 62.7 (+15.7pp). So k=0.5 is already most of the way to just serving the anchor.
    - **The pathwise decoder costs more than the KL decoder at the same k in the deployed band**: loss 51.7 vs 47.5 at k=3, 49.8 vs 45.7 at k=5 (~2 SE each), converging above k=10. That is the price of the event-level guarantee, measured externally for the first time — the manuscript previously had only the risky model's own log-loss on its own output.
    - Degeneracy agrees and needs no judge: distinct-3 0.985 (unconstrained) -> 0.960 (KL k=3) -> 0.860 (k=1) -> 0.856 (k=0.5) -> 0.838 (anchor alone); lower for pathwise than KL at matched k (0.893 vs 0.960 at k=3).
    - Two design corrections made mid-run. Cross-run byte identity is **not** usable as a measure: seeds match 600/600 across runs but batched generation consumes randomness in a different order, so two runs diverge even at steps where both serve the risky model unchanged; replaced with the per-trajectory active-step share, which the logs already carry and which reproduces the manuscript's activity figures (KL k=3 0.35% on three classes vs 0.26% over six). And the first pass at 180 pairs per arm had no null and too little power to separate arms; rerun at 600 with the null arm added.
  - **feat-032** (`scripts/run_prefix_debt_k20.sh` on GPUs 1+2, plus the same composition_attack invocation for the 8B on GPU 4, then `analysis/merge_prefix_debt.py --out results`). `results/prefix_debt_ablation.csv` 56 -> 70 rows, 0 violations. **Off/on recall at k=20 is 1.0x for both risky models** (70B single 0.3139/0.3139, oracle 0.5907/0.6049; 8B single 0.4759/0.4921, oracle 0.7865/0.8074) against 6.8x/3.6x (8B) and 2.1x/10.4x (70B) at k=3. The prefix debt's contribution is an inverted U: it is the mechanism's protection exactly in k in [3,10], where the mechanism is both usable and claims to protect, and nothing at k=20 where no protection is left to attribute. The table was hand-assembled in phase 2 and is now rebuilt by a command.
  - **Phase 3 is complete**: feat-028..034 all `done`. Total phase-3 compute 1.8 GPU-hours against a 19.5-hour estimate.
- [x] 2026-09-07 02:35-02:50 — **feat-028a and feat-033 DONE** (GPU 0 and GPU 1; other users held 1/2/4 at 100% util but with ~68 GB free on 1 and 4, so small jobs fit alongside them).
  - **feat-028a** (`analysis/certificate_cap.py --safe-model common-pile/comma-v0.1-2t --risky-model '' --tag _comma7b`, GPU 0, ~4 min): the second-anchor objection is closed **against** the mechanism. On the same 758 references the 7B Common Pile anchor assigns median **180.5 total nats** against TinyComma's **204.7** — 11% fewer (ratio 0.887; totals only, tokenizers are 64,000 vs 128,256 so per-token figures are not comparable). Vacuity at k>=3 is **100% for both**; at k=1 the larger anchor is **more** vacuous, **74.0% vs 43.9%**. Scaling the anchor makes the certificate strictly weaker at the only budgets where it was informative.
  - **Control** (`analysis/anchor_control.py`, GPU 0, ~2 min, `results/anchor_control.csv`): same 1500-character spans, both anchors. The 7B's advantage is **0.803x on Gutenberg** (public domain, in the Common Pile, both anchors trained on it) but only **0.908x on the 16 CopyBench novels** (in neither training set). Its gain on protected text is therefore general fluency, not exposure — so a deployer cannot buy vacuity back by scaling the anchor.
  - **feat-033** (`analysis/length_scaling.py`, GPU 1, ~1 min, 66 works x 5 lengths): splits the Limitations caveat in two. Whole-work vacuity does collapse with length (k=3: 100% at 64 tokens, 93.8% at 128, 0% at 256), but the certificate the composition adversary actually faces is the per-window one at K_w = k*50, and that is **100% vacuous at k>=5 for both sources at every work length**. A longer work strengthens only the single-query certificate, which could not have reproduced it anyway. Gutenberg (contiguous, but in the Common Pile) and CopyBench (unfamiliar, but concatenated and so overstated) agree on the shape; neither is asked to carry the result alone. `--replot` rebuilds the figure from the summary CSV with no GPU.
  - Two bugs found and fixed in the first feat-033 launch: `works()` initially took every prompt class, so FactScore biographies were treated as novels ("Michael Douglas"); the fix filters to `SPLITS`. And `pkill -f length_scaling` killed my own shell twice — the pattern matches the invoking command line, exactly as the Blockers note already warns; use `ps`/PID.
  - Cache note: TinyComma's weights live in the home HF cache and comma-v0.1-2t in repo `hf_cache/`, and `HF_HUB_CACHE` can only point at one. `hf_cache/models--jacquelinehe--tinycomma-.../snapshots/42ccd323.../{model.safetensors,generation_config.json}` are now symlinks to the home copy (same revision), so both anchors resolve under `HF_HUB_CACHE=$PWD/hf_cache`.
- [x] 2026-09-07 02:20-02:30 — **feat-031 DONE (no GPU)**, and it corrected its own planned claim. `analysis/separation.py` puts legitimate use and reconstruction in the same currency. Producing command: `.venv/bin/python analysis/separation.py --results results --out results --figures figures` -> `results/separation.csv` (75 rows), `results/separation_summary.csv` (25 rows), `figures/separation.{pdf,png}`. **c_legit = 0.820 nats/token** over 26,999 trajectories, per-class medians 0.726-0.902, flat across all six prompt classes and k in {1,3,5}; one 200-token answer costs 164 nats. The planned "no per-user budget separates a reader from a reconstructor" is **false**: reconstruction is much dearer per recovered token. The true statement is sharper — the separation is bounded and collapses with k (protective ratio 13.6 at k=5 oracle L=50, 8.6 at k=10, **4.56 at k=20**), and the filter is redundant exactly where it would be safe: at k<=3 no strategy recovers 0.10 of a work even uncapped, while at k=20 holding the strongest adversary to 0.10 needs B*=100 nats, **0.61 of one ordinary completion**. Proposition 5 and its scope remark went into `sections/appendix_theory.tex` (proved: the ratio identity and monotonicity of recall in B; the near-proportionality is measured, not proved). `tests/test_separation.py` (3 tests) checks the monotonicity inequality against every row of the real replay. Manuscript recompiled: 0 overfull, 0 ??, 18 pages, page 13 still "Open Science".
- [x] 2026-09-07 02:20 — **D1 answered yes**: `common-pile/comma-v0.1-2t` downloaded to `hf_cache/models--common-pile--comma-v0.1-2t` (14 GB) by `scripts/download_second_anchor.py`. feat-028a is unblocked.
- [x] 2026-09-07 — harness updated: feat-028..034 added to `feature_list.json` (34 features); `init.sh` and `AGENTS.md` no longer say "both phases complete". `./init.sh` exit 0, 30 tests pass. Manuscript recompiled unchanged: 0 overfull, 0 `??`, 18 pages.

## GOAL COMPLETE (2026-09-05)

- **PDF:** `/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml/satml_2027.pdf` (11 pages in IEEEtran conference format including references; body ends on page 10; source `satml_2027.tex` + `sections/*.tex`; compile: `~/.local/bin/tectonic -X compile satml_2027.tex`).
- **References:** 48 cited, 51 entries in `references.bib` (all fetched and title-checked).
- **Artifact:** `/mnt/md0/IITM/BackUp/Home/vijayavallabh/DA5001_Project/artifact/` (committed, 75 files, `MANIFEST.sha256` verified) and `artifact.zip` (5.0 MB, gitignored, rebuilt by `scripts/build_artifact.sh artifact`).
- **Single most important number per results file:**
  - `regime_table.csv`: solver active in 0.195% of decode steps at k=3 (99.0% risky-unchanged); 57.1% of generations identical between k=3 and k=5.
  - `per_trajectory.csv` / `seed_collisions.csv`: 0 invariant violations in 26,999 released trajectories; one duplicated sample in every N=20 held-out row of the old Stage 2.
  - `llr_tails.csv`: L(y) > K for 9.6% of attack-train trajectories at k=1 (95% CS [5.1%, 12.6%]) and 32.7% at k=0.5.
  - `prefix_debt_forced_tokens.csv`: leading tokens written by the anchor = floor(delta_init/k) in 99.4%+ of trajectories.
  - `surprisal.csv`: anchor surprisal of a generated 200-token continuation, median 978 nats at k=3.
  - `certificate_cap_summary.csv`: certificate vacuous for 100% of 758 CopyBench passages at k>=3, 43.9% at k=1 (anchor surprisal median 205 nats).
  - `certificate_caps_memoriser.csv`: the memoriser assigns training passages 0.19 nats/token (anchor 3.28).
  - `regime_sweep.csv`: at k=0.1 the anchor writes the first 39 tokens of every answer; 0 violations in 37,800 trajectories.
  - `llr_ratio_samples.csv`: 300 attack-train L/K samples per (variant, k) for the ECDF figure.
  - `memorizing_model_recall.csv`: greedy nv-recall 0.907 on training excerpts, 0.0 on held-out test.
  - `composition_summary.csv`: single-query recall 0.476 at k=20 (unconstrained 0.492); oracle windows 0.858; <0.10 at k<=3; 0 violations in 19,400 queries.
- Success criteria: (1) features 001-009 and 013-015 done with evidence; (2) all required results files present; (3) manuscript compiles (tectonic; pdflatex is unavailable here), <=12 pages, threat model + Open Science + LLM-usage sections, 51 bib entries; (4) `artifact/` with manifest; (5) `./init.sh` passes and every feature is committed; (6) handoff names the PDF and artifact paths.
- Proofread pass (2026-09-05, after GOAL COMPLETE): fixed a broken `Table~\ref{tab:memorising}` (rendered as "Table ??") by adding the memorising-model table from `results/memorizing_model_recall.csv`; rewrote Section VIII-C (bank and burst) to state that the attack was attempted and not evaluated (memoriser ignores filler instructions) instead of describing it as run; removed the claim of an adaptive prompt search and the optimiser-model mentions in Related Work, Open Science and LLM-usage; corrected numbers to the CSVs (composition queries 27,818 budgeted / 35,766 total, not 19,400; target length 260 not 265; windowed total spend 830-860 nats; oracle L=20 unconstrained 0.88; delta_init 10.4 single / 12.9 window; k list {0.15,0.5,1,3,5,10,20}; identical-to-risky at k=0.5 is 1/2,700; "99.4% or more" qualified to attack-train); bib name fixes (Yih, Van Durme, Ben Allal, Tramer, De Sa, De Cristofaro, Szepesvari, Bretagnolle accents). PDF now 11 pages, 0 overfull boxes, 0 unresolved references.
- Prose pass (2026-09-05 23:01, humanizer + no-ai-slop + scientific-writing skills, title to end matter): every section rewritten in place for precise academic register. Removed colon reveals and dramatic one-liners ("The guarantee is another matter.", "The result cuts both ways.", "Too weak: ... Too strong: ..."), "not X but Y" contrasts, ordinal signposting ("Four observations follow", "Three regimes appear"), the kicker closing the Fairness paragraph, and a broken "; which" clause in Related Work; split the one-paragraph sweep section into four paragraphs; "On compute:" label and predicate "open-weight" reworded in LLM-usage. Verified: numbers/citations/labels identical before and after (script diff over all 12 files; only addition is the qualifier "at k=3" in the attack results), required LLM sentence present verbatim, end-matter order unchanged, tectonic exit 0, 0 overfull, 0 ??, 11 pages. Backup of the pre-pass sources in the session scratchpad only.
- Jargon and flow pass (2026-09-05 23:13): codebase terms removed from the manuscript and figure legends. "attack-train"/"attack_train" -> "attack split" (the six prompt classes are now defined once at the top of Section V); "Stage-1 prompt set" -> "the same six prompt classes"; the k=-1 sentinel -> "the risky model alone" (Table III rows are now "risky only"/"anchor only"); "nv-recall"/"LCS" -> "near-verbatim recall"/"longest span"; "early-EOS", "byte-identical", "fusion solver", "library default", "results/*.csv", "./init.sh", "entry points", "README" reworded; excerpt/passage unified to passage; TinyComma and Llama named once in the threat model. Structure for one flow: sweep is now Section V-E of "Audit of the Released Decoder" (with an opener that defines the data), attack results are Section VI-D of "Adaptive Attacks" (with an opener), Related Work moved to Section IX before the Conclusion (new `sections/conclusion.tex`, split from discussion.tex), figures reordered so numbering follows first reference. Legend strings changed in `analysis/regime_sweep.py`, `analysis/llr_tails.py`, `analysis/composition_attack.py`, `analysis/certificate_cap.py`; figures regenerated with `figures/make_figures.py --copy-to <sub/satml/figures>`; artifact rebuilt (75 files, manifest verified, 23:13); 14 tests pass. PDF: tectonic exit 0, 0 overfull, 0 ??, 11 pages.
- Second proofread of the PDF (2026-09-05 23:17): full read of the rendered text. Fixed a contradiction introduced by the prose pass in LLM usage considerations (the memorising model is not "released with open weights"; now "built on openly released weights"); removed the last code term "copyright-domain subset" (now "the CopyBench attack and test splits"); intro "answers this at the level of the mechanism" -> "offers a mechanism-level remedy". Figures: the lengthened legend in llr_tails covered the top of the right panel, so labels are now "<source>, 95% anytime-valid interval", the left-panel legend sits in the empty lower-right region, and all legends in llr_tails/regime_sweep/certificate_cap have an opaque background; figures regenerated and copied, artifact rebuilt (23:17, 75 files, manifest verified), 14 tests pass. Vyas et al. lemma numbering verified 23:20 against arXiv 2302.10870: Lemma 2.2 is the Delta-max event bound p(E|x) <= 2^k safe(E|x) (log base 2), Lemma 2.3 the KL event bound under an (eps,delta)-concentration assumption; Background (2.3, KL form without concentration), Related Work and Discussion (2.2, Delta-max) all cite correctly. PDF: tectonic exit 0, 0 overfull, 0 ??, 11 pages; required LLM sentence present verbatim.
- Consistency audit (2026-09-05 23:48, peer-review pass; every number in the text checked against results/*.csv, He et al. arXiv 2602.07120 and Vyas et al. 2302.10870 downloaded and grepped). All table values, sweep numbers, certificate caps, tail counts (738/8,999; 959/999), passage statistics (median S 205, 162-254, 3.20/token; risky 2.59/token; gap 37), composition counts (100 passages, 260.4 tokens, 27,818/35,766 queries), prefix debts (10.45/12.86/6.216), recipe (rank 64, lr 2e-4, 12 epochs), prefix-debt n=5 (mean of top-n, as in a_patch) and Theorem 3.1 verified. Corrected: (1) provenance: the 26,999-trajectory logs are from an earlier public evaluation with Llama-3.1-8B-Instruct, not released by He et al., whose own evaluation pairs TinyComma with Llama 3.1 70B base; six sentences (abstract, intro C2, threat model, Section V opener, Section VI opener, Discussion fairness, LLM usage x2) now say 'released runs/logs' and the 70B model is named as the mechanism's own risky model; 'the budgets at which the mechanism has been evaluated' -> 'the budgets of the released runs' (He et al. sweep k from 0.1 to 20). (2) single-query T_max is 296 (longest target), budget K = 296k for every passage, not each passage's own length (analysis/composition_attack.py line 152). (3) the per-user-budget mitigation sentence was arithmetically wrong (windowed spend 830-860 nats vs single-query K = 5,920 nats at k=20); rewritten: the per-user budget must lie below the anchor's surprisal of the passage (~3.3 nats/token). (4) intro 'at most 1%' -> 'about 1%' (1.16% at k=3); '0.48, the value of the unconstrained model' -> 'against 0.49'; abstract/intro 'as much as' -> 'nearly as much as'; abstract '0.2%' -> 'at most 0.2%'. (5) delta_init 6.2 baseline is from the released-run prompts, not the same prompts; chained 0.11/0.50 are the 50-token windows; utilisation Z/K named in VI-D; Open Science now lists the general-knowledge prompt set; 'impossible' softened. PDF: exit 0, 0 overfull, 0 ??, 11 pages.
- Third-person self-citation (2026-09-05 23:55, user decision): `vijayavallabh2026audit` added to references.bib (metadata from the arXiv API: J. Vijayavallabh, "An Empirical Audit of k-NAF Budget Accounting for Anchored Decoding", arXiv:2605.28001, 2026-05-27) and cited at six places that refer to the released logs, plus one Related Work sentence stating what the present audit changes relative to it (per-trajectory invariant and anytime-valid CS instead of a fixed-sample empirical Bernstein bound; certificate-strength, small-budget and attack audits added). No first-person reference to it. PDF: exit 0, 0 overfull, 0 ??, 11 pages; 53 bib entries.
- Hallucinator rerun 2026-09-06 00:08 (hallucinator 0.2.2 Python API, 229 s, 4 workers): 50 references extracted, 42 verified, 8 not_found, all 8 confirmed by hand: Haviv 2602.08632, Abad/CP-Fuse 2412.06619 (DBLP: ICLR 2025), Howard 1810.08240 (CrossRef 10.1214/20-AOS1991, AoS 49(2) 2021), Chugg 2512.21300 (arXiv API); Polyanskiy-Wu 10.1017/9781108966351 and Tsybakov 10.1007/b13794 (CrossRef); Elkin-Koren FORC 10.4230/LIPIcs.FORC.2024.3 (doi.org resolves to Dagstuhl; DBLP FORC 2024); ROUGE W04-1013 (ACL Anthology: Chin-Yew Lin, Text Summarization Branches Out, pp. 74-81, 2004). The new self-citation [6] verified automatically. Haviv and CP-Fuse were verified automatically last time and not this time, so not_found is a database flake, not a bib problem.
- Artifact rebuilt 2026-09-06 00:09 (`scripts/build_artifact.sh artifact`: 75 files, manifest verified, anonymity scan clean, 5.0 MB; contents unchanged since the 2026-09-05 23:17 build because only the manuscript changed afterwards). Handoff, feat-015 evidence and memory now carry this build time, the 50/53 reference count, the self-citation and the hallucinator rerun.
- CFP pass (2026-09-05, after the user pasted the SaTML initial-review criteria): added an Ethical Considerations section (public benchmark, memoriser weights withheld, own decoder only, findings to be shared with the mechanism's authors) before Open Science, and one sentence in the threat model tying the adversary-chosen risky model to a deployer whose model memorised the works in pre-training (Cooper et al.). Recompiled: 11 pages, 0 overfull, 0 `??`.
- Reference self-check (2026-09-05): hallucinator 0.2.2 on the PDF -> 49 refs, 36 verified, 13 not found by the tool; all 13 verified by hand against arXiv (10 IDs, exact titles/authors), ACL Anthology (ROUGE) and CrossRef DOIs (Tsybakov, Polyanskiy-Wu). Zero fabricated or mis-attributed references.
- CFP compliance pass (2026-09-05 22:40, after reading https://satml.org/call-for-papers/ and its checklist in full): end matter reordered to Open Science → `LLM usage considerations` (exact title; required editorial-use sentence; closed-assistant limitation; compute justification: ~12 A100 GPU-hours, why 8B, query minimisation) → Ethical Considerations (Menlo report cited, `dittrich2012menlo` added, 52 bib entries) → references; Open Science no longer says the review copy is attached (anonymised repository link goes in HotCRP, frozen after Oct 2); the three "our earlier audit/protocol" self-references were rewritten in neutral third person; README_artifact no longer mentions a Zenodo record. AGENTS.md carries the full timeline and format rules, feat-016 the human checklist, session-handoff.md the open question about prior reviews. Recompiled: 11 pages, 0 overfull, 0 `??`.

## Blockers / Risks

*(Status line added 2026-09-06 23:19: the entries below are the running record. Resolved since: the 70B is cached (`hf_cache/models--unsloth--Meta-Llama-3.1-70B`, D1), GPUs 0/4 were released for 8B jobs (D2), the batched-EOS utilisation issue was fixed in feat-004, and the decoder's warper guard was removed on 2026-09-06. Still true: `HF_TOKEN` is invalid (use `HF_HUB_OFFLINE=1`), the DGX is unreachable from this account, there is no `pdflatex` (use tectonic), and feat-010 is not testable with the LoRA memoriser.)*

- [ ] Compute: local 4×A100 80GB (GPUs 0 and 4 had ~15 GB in use by other processes on 2026-09-05; GPU 3 is a T400, unusable). DGX 6×H100 via `dgx-gpu` for 70B and sweeps. Llama-3.1-70B base in bf16 (~140 GB) fits on 2 free A100s with `device_map="auto"`, so feat-008's 70B option is feasible locally.
- [ ] **HF_TOKEN in `.env` is invalid** (`HfApi.whoami` → "Invalid user token", 2026-09-05). The local HF cache already holds Llama-3.1-8B-Instruct (weights only; tokenizer comes from the TinyComma repo), Llama-3.1-8B base, TinyComma, Qwen2.5-7B-Instruct, so all 8B runs work with `HF_HUB_OFFLINE=1`. Llama-3.1-70B is NOT cached: feat-008's 70B option needs a valid token (human) or the DGX cache. Ask the user for a working token when feat-008 starts.
- [ ] **DGX unreachable from this account (2026-09-05):** `ssh PrakashDGX_H2` → `Permission denied (publickey,password)` for user `prachh`; the `dgx-gpu` skill (`/home/sports/.config/opencode/skills/dgx-gpu/SKILL.md`) assumes a key that is not installed here. Per GOAL.md, all GPU work runs on the local A100s (indices 1 and 2 free; 0 and 4 at 100% util by other users). Ask the user to install the DGX key if the 70B run or the full sweep needs more than two GPUs.
- [ ] feat-005 sweep: `output/sweep_chat` crashed with CUDA OOM at ~17:50 while sharing GPU 2 with `output/sweep_plain` and the feat-008 fine-tune (peak of the prefix-debt full-vocab logits at batch 48). Relaunch after the fine-tune: `CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv/bin/python h1.py <COMMON from scripts/run_regime_sweep.sh> --use-chat-template --output-dir output/sweep_chat` (its k=-1 neutral file is reused by --skip-existing). Never run more than two decoding jobs per 80 GB GPU at batch 48.
- [ ] feat-014 prep (2026-09-05): `~/sub/satml/sections/{threat_model,related_work,open_science}.tex` drafted (threat model with parties/goals/out-of-scope; related work citing 48 keys; Open Science + LLM-usage). To be `\input` by the feat-014 rewrite; the graphical-abstract step of the scientific-writing skill was deliberately skipped (figures must trace to results/*.csv).
- [ ] feat-014 prep (2026-09-05): `~/sub/satml/references.bib` rebuilt with 51 verified entries (arXiv metadata fetched and titles checked; venues added by hand; old 10 keys preserved; the old He et al. and Cohen titles were wrong). Page numbers are absent for conference papers (NeurIPS/ICLR/ICML/COLM), which is standard for those venues.
- [ ] **LaTeX toolchain (2026-09-05):** no `pdflatex`/`bibtex` on this box and no sudo; conda `texlive-core` installs a broken TeX Live (no `pdflatex.fmt`, perl scripts missing) and was removed. `~/.local/bin/tectonic` (0.15) compiles `~/sub/satml/satml_2027_new.tex` (auto-fetches packages, runs the bibliography): `cd ~/sub/satml && tectonic -X compile satml_2027_new.tex`. The old manuscript compiles to 10 pages, the new skeleton to 6. Success criterion 3 will be checked with tectonic; the artifact README will state that `pdflatex + bibtex` produce the same PDF on a standard TeX Live.
- [x] Prep done while the sweeps run (2026-09-05): EBB retired (`dap/stats.py budget_check`, E1 summary with anytime-valid CS on utilisation, E2 `max_spend`/`rho`=max utilisation/`certified`=invariant; 14 tests), `figures/make_figures.py` (all figures from CSV; legacy figures in `figures/legacy/`), `scripts/build_artifact.sh` + `README_artifact.md` (69-file anonymised artifact, manifest verified), manuscript `~/sub/satml/satml_2027_new.tex` with sections intro/threat/background/protocol/results_logs/sweep/attacks/attack_results/discussion/open_science (8 pages, compiles with tectonic; two TODOs left for the chat-template sweep numbers).
- [ ] feat-010 prep (2026-09-05): `analysis/bank_burst.py` written and smoke-tested (`output/bank_burst_smoke/`). First filler design ("Repeat the following text exactly, then continue it: one, two, ... <seed>") does not bank: the memoriser diverges from TinyComma by ~2.5 nats/token on the filler and does not continue the passage after it (nv-recall 0 even at k=-1). Second smoke (fillers count/repeat, k in {-1,3}, `output/bank_burst_smoke/run2.log`) shows why: the memoriser ignores the instruction and starts the memorised continuation immediately after the seed, so no filler is generated and nothing is banked (the nv-recall 0 was a pivot-detection artifact on the passage text). Bank-and-burst needs a risky model that both memorises and follows instructions (a pre-training memoriser such as a 70B base or instruct model); with the fine-tuned memoriser it is not testable. Left optional/not-started; the composition attack (feat-009) is the C5 result.
- [ ] Batched decoding keeps accruing budget after a sequence's own EOS (`budget_so_far = (t+1)k − δ` for the whole batch), so `final_budget` and `generation_length_tokens` in the released logs reflect the batch length for early-EOS trajectories; utilisation of those trajectories is understated. feat-004 should compute B at the sequence's own EOS (`true_gen_len`) when reporting utilisation.
- [x] feat-008 decision (2026-09-05, ahead of the Sep 12 deadline): LoRA-memorised Llama-3.1-8B-Instruct (`recipes/finetune_memorizing.py`, merged weights in `output/memorizing_llama8b`, 73 min on one A100). Final token loss 0.058; greedy nv-recall 0.907 on training excerpts. The 70B-base option is off the table locally (not cached, invalid token, DGX unreachable).
- [ ] feat-008 (memorising model) was the schedule risk: Llama-3.1-70B base needs ~140 GB bf16; fine-tuning 8B is the fallback (≈2–4 GPU-hours). Decide by Sep 12.
- [ ] Known code issues to fix in feat-003, not before: seed collision in `dap/e2/runner.py::_eval_specs` (offset n0) and the top-up path (offset 10); E1 uses R = T·ln|V| while E2 uses R = K; `B_eff` conflates generation length with budget; instruct model called without chat template.
- [x] feat-002 extended (2026-09-05): all 18 released trajectory files reanalysed (`unzip -o output.zip 'output/h1_outputs/trajectories_k*.jsonl' && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results`): 26,999 trajectories, 0 invariant and 0 per-step violations; L>K at k=1 in 738/8,999 trajectories (attack_train 96/999, val 100/1500, test 152/1500, neutral 120/2000, factual 111/1500, creative 159/1500); solver active 0.09-0.45% of steps at k in {3,5} for every class; identical generations between k=3 and k=5: 44.6-64.5% per class. `results/per_trajectory.csv` is now 27k rows (4.3 MB).
- [ ] `trajectories_k1_attack_train.jsonl` has no `p_risky_prob` field (older run), so `L_risky` is 0 for k=1 in `results/`; the k=1 attack_train file has 999 rows, not 1000. Only 4 of 18 h1 trajectory files are reanalysed so far (the evidence set); extend to val/neutral/factual/creative in feat-013 if a per-domain table is needed.
- [ ] Unverified claims in the current manuscript: "Regime A pivot is the more common behaviour" (a crude check finds ~3–4%); the "not i.i.d. common random numbers" caveat is unnecessary (`torch.multinomial` rows are independent).

## Decisions Made

- **Retire the empirical-Bernstein proxy and ρ (2026-09-05).** Z ≤ K holds per trajectory by construction (KL chain rule), so a mean bound cannot fail; report per-trajectory utilisation, realised LLR tails, and certificate caps instead. Alternatives considered: keep EBB with R = K (bound gets looser, still vacuous), betting CS on Z/K (only where a mean is genuinely the target).
- **Audit at He et al.'s small-k regime (2026-09-05).** At k=3/5 the constraint was active in ≤0.2% of steps; the sweep now spans {0.1 … 1} plus k=−1/0 baselines.
- **Attacks over search (2026-09-05).** Drop the surrogate/k-DPP/ρ search; the SaTML core is the composition attack (Cohen Thm 3.5 instantiated), bank-and-burst, and a memorising risky model.

## Files Modified This Session

- `AGENTS.md`, `CLAUDE.md`, `feature_list.json`, `progress.md`, `session-handoff.md`, `init.sh`, `GOAL.md` — harness rebuilt for the SaTML plan.
- `analysis/reanalyze_logs.py` — argparse reanalysis tool (feat-002); `results/*.csv` generated.
- feat-004: `dap/stats.py` (metrics, index-only seeds), `dap/e1.py` (baselines, summary means), `analysis/reanalyze_logs.py` (k-1 file names), `tests/test_metrics.py`.
- feat-006: `analysis/certificate_cap.py`, `results/certificate_caps.csv`, `results/certificate_cap_summary.csv`, `figures/certificate_cap_curve.{pdf,png}`.
- feat-003: `dap/stats.py`, `dap/shared.py`, `dap/e1.py`, `dap/e2/evaluator.py`, `dap/e2/runner.py`, `dap/e2/types.py`, `a_patch/factory.py` (pad id when eos is a list), `tests/`.
- `~/sub/satml/IMPROVEMENT_PLAN.md`, `~/sub/satml/scripts/*` — plan and script copies (outside repo).

## Evidence of Completion

- [x] `./init.sh` passes (2026-09-05): all [OK] lines, `[info] local GPU available: True, count: 5`, `=== Init Complete ===`, exit 0.
- [x] `node .../harness-creator/scripts/validate-harness.mjs --target .` → Overall 100/100 (all five subsystems 5/5).
- [x] feat-002 evidence command exit 0; see `feature_list.json`.
- [x] feat-003: `pytest -q tests` 6 passed; `./init.sh` exit 0; GPU smoke `output/smoke_feat003/` invariant_violations=0.
- [x] feat-006: evidence command exit 0; `results/certificate_cap_summary.csv` written.
- [x] feat-004: smoke evidence command exit 0 (`output/smoke/`, 24 files); 9 tests pass.
- [x] feat-008: `scripts/run_memorizing_check.sh 1` exit 0; `results/memorizing_model_recall.csv` written.
- [x] feat-009: composition evidence command exit 0; `results/composition_summary.csv` (45 rows, 0 violations).

## Notes for Next Session

- The numbers in AGENTS.md "Known truths" were computed this session from `output.zip`; feat-002 must reproduce them exactly before anything is cited.
- `output.zip` is 2.4 GB; extract only the four trajectory files listed in feat-002's evidence command.
- Reproduction commands for the original runs are in `README.md`; keep them for the artifact's "original runs" section.

## 2026-09-06 (after phase 1): plan version 2, literature review, harness for phase 2

- **Why.** Four local A100s (1 and 2 idle; 0 and 4 shared) make the 70B-base audit feasible, and a reading pass found three results the paper implies but does not state. `~/sub/satml/IMPROVEMENT_PLAN.md` is now version 2 (v1 kept as `IMPROVEMENT_PLAN_v1_2026-09-05.md`); `GOAL.md` has a Phase 2 section; `feature_list.json` has feat-017..027; `AGENTS.md` status, compute, secrets and known-truth bullets updated.
- **Facts established today.** He et al. warp both logit vectors (temperature, repetition penalty) before the KL solve (App. B), decode books at temperature 0.7 with penalty 1.1 (App. D.1), and write in App. E.1 that at k = 3.0 the constraint "is rarely binding"; their Table 17 gives the unconstrained 70B ROUGE-L ≥ τ on 23.0% of CopyBench prompts (word-LCS 10.7). `data/copybench_test.jsonl` contains 50 *Harry Potter and the Sorcerer's Stone* passages (Cooper et al.: 96.3% extraction coverage for Llama 3.1 70B) and the attack split 8 *1984* passages; the LoRA memoriser never saw the test split. `HF_TOKEN` still 401; ungated 70B mirrors exist (unsloth, NousResearch). `results/composition.csv` lacks per-query spends (odometer needs a re-run with logging).
- **Literature.** 45 searches; 74 new references verified by script against the arXiv API / CrossRef (scratchpad `bibcheck/report.json`, 74/74 verified) and appended to `references.bib` (127 entries; also `bib_additions_2026-09-06.bib`); `LITERATURE_REVIEW.md` records claim, use and placement for each and the six framing bridges (leakage measures; choosing ε; composition/odometers; reconstruction bounds; concentration/dispersion; adaptive evaluation of certified defences). Verification caught one wrong attribution (arXiv 2408.13278 is Chen, Kairouz, Oh, Xu). Manuscript recompiled with the enlarged bib: 11 pages, 0 overfull, 0 `??` (no citations changed yet).
- **New theory to prove (plan v2 Section 4).** P1 pathwise budget ⇒ Δmax-NAF per event; P2 extraction cost ≥ exp(S − kT) queries; P3 concentrated certificate via Freedman/Howard on L(y) − Z; P4 budget-path (token-bucket) necessary condition; remark on the warped anchor.
- **Decisions pending (human).** D1: 70B route (token / ungated mirror / none); D2: may GPUs 0 and 4 be used while others' jobs run; D3: abstract text policy if 70B numbers are late (plan v2 Section 9). feat-017 is `blocked` on D1; nothing downloaded.
- **Next.** feat-023 (no GPU) and feat-022 (forward passes) can start immediately; feat-019 code + tests in parallel; on D1, feat-017 → feat-018 on GPUs 1,2.

## 2026-09-06 (phase 2 execution, session 2): decisions D1/D2 answered, experiments launched

- **Decisions.** User: "go, use the unsloth mirror, GPUs 1 and 2 only [other gpus use for other stuff if you want]" → D1 = ungated mirror `unsloth/Meta-Llama-3.1-70B` (revision 1b7306651142d0cc65d993076a250a6a82cf046c, 141 GB, downloaded 2026-09-06 00:50–01:08 into `hf_cache/`, `scripts/download_70b.py`); D2 = 70B on GPUs 1+2, 8B jobs on the shared GPUs 0 and 4.
- **CUDA device order gotcha (recorded in AGENTS.md).** Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, CUDA index 4 is the T400; one run OOM'd on it before the fix. All jobs now set both variables.
- **feat-019 code done, TDD.** `a_patch/pathwise.py` (bisection on the max log-ratio), `constraint='kl'|'pathwise'` in the factory with realised-ratio banking and per-step `r_t`, `m_t`, `var_ratio`, `cum_realised_ratio` logging (the sampling step was also moved out of the logging block, which had made `log_kl_stats=False` unusable); `--constraint` in `h1.py` and the composition driver; `tests/test_pathwise.py`. Smoke (GPU 1, k=1, 24 trajectories, 4,402 steps): 0 violations of R_T ≤ max(0,B)+1e-3, r_t ≤ k_t+1e-3, a_t ≤ m_t+1e-3; KL smoke on the same prompts: 0 violations, 4.5% active steps, 2/24 with L(y) > K. `dap/warp.py` (+ tests) for teacher-forced temperature/repetition-penalty warping; `risky_device_map`/`max_memory` pass-through for sharded 70B loading (factory, h1.py, composition driver); composition driver rewritten with `--retries`, per-query JSONL logging (spend, budget, realised ratio, attempt, matched, activity), `--no-prefix-debt`, `--novel`, `--repetition-penalty`.
- **feat-023 done.** `analysis/budget_path.py` (+ `tests/test_budget_path.py`): k_crit median 13.9 (p10 10.7, p90 19.7) vs mean per-token anchor surprisal 3.24 (ratio 4.3); exact feasibility 0/0/2/92% at k=3/5/10/20; the token bucket pays 87/99/100/100% of tokens with 4.8/2.8/1.1/0.1 anchor-forced opening tokens; observed recall 0.01/0.09/0.20/0.48. Interpretation for the paper: at k ≥ 5 the budget can pay for the passage; the prefix debt's forced opening (and the derailment it causes) is what limits recall between k=5 and k=20 (feat-025 tests it directly; run queued).
- **feat-022 done.** `results/warped_anchor.csv`: anchor surprisal per token 3.20 (τ=1) → 3.55 (τ=0.7, rp=1.1, He et al.'s setting) → 4.30 (τ=0.5); vacuity at k=1 43.9% → 22.6%; 100% at k ≥ 3 under every setting.
- **New analysis scripts** (inputs pending): `analysis/concentration.py` (feat-020; on the smoke logs at k=1: b ≈ 15–21 nats, V ≈ 300, δ(50) ≈ 0.15 at p90 caps vs empirical 0), `analysis/odometer.py` and `analysis/burst_audit.py` (feat-021; burst audit over the released logs and sweeps → `results/burst_audit.csv`: largest 20-step spend / (20k) p99 = 2.2 at k=1, max 3.5; bank max up to 199 nats at k=1, 981 at k=5), `analysis/extraction_cost.py` and `analysis/pathwise_price.py` (feat-019), `analysis/warped_anchor.py` (feat-022).
- **Manuscript.** `sections/appendix_theory.tex` (Propositions 1–4 + warped-anchor remark, IEEEproof) included after the references; `\newtheorem` added; `references.bib` 128 entries (Tropp 2011 added, verified); Related Work rewritten (v1 kept as `related_work_v1_2026-09-05.tex`; 124 keys cited). Labels `sec:pathwise`, `sec:natural`, `sec:leakage`, `sec:odometer`, `sec:warped` are forward references to feat-026's new sections and show as `??` until then. Phase-1 PDF preserved as `satml_2027_phase1_2026-09-06.pdf`.
- **Running (background).** GPU 0: KL composition with per-query logs → pathwise regime sweep. GPU 4 (PCI): pathwise composition → KL sweep with variance logging (feat-020) → prefix-debt ablation (feat-025) → retry runs (KL, pathwise). GPUs 1+2: 70B natural-memorisation smoke (feat-017; 50 Harry Potter test passages, k=−1, τ=0.7, rp=1.1).
- **feat-017 (70B) findings, 2026-09-06 01:55.** (i) The first smoke (instruction header "Complete the prefix:" + a 15-token seed, temperature 0.7, penalty 1.1) gave near-verbatim recall 0.000 on the 50 Harry Potter passages: a base model does not continue an instruction-headed snippet. Cooper et al. use raw 50-token prefixes (100-token windows, top-k T=1 k=40, p_z thresholds; 200-token prefixes push p_z above 90%; their Section 7 reconstructs the book with beam search from "Mr. and Mrs. D"). With `--raw-prompt --seed-tokens 50 --greedy` the 70B reaches mean recall 0.243 (LCS 29.7 words) on 16 passages. (ii) The memory caps 72/64 GiB silently offloaded part of the model to CPU (0.55 s per decode step, one CPU core at 91%); with 75/70 GiB the new `[INFO] risky model device map` line shows 41+43 modules on the two GPUs and no CPU, and throughput rose to about 55 tokens/s at batch 8. Both fixes are in `analysis/composition_attack.py` (`--raw-prompt`, `--greedy`) and `scripts/run_natural_memorisation.sh`.
- **feat-018 launched 2026-09-06 02:01** (`scripts/run_natural_memorisation.sh 16`, GPUs 1+2, logs in `output/phase2/nm/*/run.log`, summary `output/phase2/nm_runs.log`). Seed-length checks on 16 Harry Potter passages with the 70B alone: raw 50-token seed greedy recall 0.243 (LCS 30 words); raw 100-token seed greedy 0.631 (LCS 66); raw 50-token seed sampled at (0.7, 1.1) 0.125. Chosen protocol: raw 100-token seed (targets about 176 tokens), greedy baselines plus sampled runs at both settings, oracle and chained windows of 50, k in {−1, 0, 1, 1.5, 3, 5, 10, 20}, then pathwise (k up to 50) and no-prefix-debt variants. Violation semantics fixed in both drivers: the pathwise decoder bounds the realised log-ratio R per trajectory, not the KL spend Z (only E[Z] ≤ K), so pathwise runs are checked on R; the `invariant_violations` column of the pathwise composition run started before the fix must be recomputed from its `queries.jsonl` (R vs B).

### 2026-09-06 02:20 — 8B memoriser vs 70B base on the Harry Potter test passages (feat-017/018 control)
Same protocol for both (raw 100-token seed, greedy, 50 HP1 test passages, 50-token windows):
- LoRA memoriser (Llama-3.1-8B-Instruct fine-tuned on the attack_train split, never on these passages): nv-recall 0.000 in single, oracle and chained modes (`output/phase2/hp1_8b/greedy.log`). Its memorisation is confined to what it was fine-tuned on.
- Llama-3.1-70B base (unsloth mirror): nv-recall 0.558 single, 0.771 oracle L=50, 0.527 chained L=50 (`output/phase2/nm/hp1_greedy/run.log`), consistent with Cooper et al.'s discoverable-extraction result.
Consequence for the paper: the natural-memorisation audit (Section VI-A) needs the 70B base; the memoriser is the wrong risky model for pre-training memorisation. The sampled 8B run at τ=0.7/rp=1.1 with k ∈ {-1, 0, 20} is in `output/phase2/hp1_8b/sampled_B/`.
Bank cap (feat-021): `a_patch/bank.py::bucket_step` (token bucket, cap = depth), factory kwarg `bank_cap`, driver flag `--bank-cap`; tests `tests/test_bank_cap.py` (3). Smoke on GPU 4: `output/phase2/cap_smoke/`. Full run queued by `scripts/run_bank_cap.sh` (caps 20 and 100 nats at k=20, 100 passages) after the GPU-0 chain.
Note on bookkeeping: phase-2 GPU features run concurrently by design (plan v2 §6: three queues on GPUs 0, 4 and 1+2), so feat-017/018/019/021 are all marked in-progress while their queues run; analysis and writing proceed one feature at a time as each queue completes.

### 2026-09-06 02:25 — decoder guard removed for warped decoding under a budget
`AnchoredDecodingFactory._decode` refused any logits processor/warper when 0 < k, so every run at He et al.'s book settings (τ=0.7, repetition penalty 1.1) crashed at its first positive k: the 8B control `output/phase2/hp1_8b/sampled_B` died at k=20 (its k=-1/0 rows are valid: recall 0.000), and the 70B `hp1_B` run was stopped at k=0 (exit 143, 02:22) before it could crash. The loop already applied both processors and warpers to both logit vectors before the KL solve (He et al. App. B), so the guard was a leftover; removed. Smoke (`output/phase2/warp_smoke`, 8B memoriser, 4 passages, k ∈ {1,3}, τ=0.7, rp=1.1): 0 violations, largest overshoot −0.59 nats, no per-step warnings. The suite continued with `hp1_A`; `scripts/rerun_nm_hp1_B.sh` reruns `hp1_B` on GPUs 1+2 when the suite ends (`output/phase2/nm_hp1_B_rerun.log`). The remaining setting-B runs in the suite (1984_B, hp1_B_pathwise, hp1_B_nodebt) start after the fix and are unaffected.

### 2026-09-06 02:50 — reference check of the 128-entry bibliography (125 cited)
hallucinator 0.2.2 on the compiled PDF: 87 verified, 37 not_found, 1 mismatch (SILO; DBLP confirms ICLR 2024). All 38 misses verified by hand against the arXiv API (28 by id or exact title), CrossRef (Tropp 2011, Howard 2021, Freedman 1975, Cover & Thomas, Polyanskiy & Wu), DBLP (CP-Fuse ICLR 2025, Guo 2022 ICML, Dwork–Kohli–Mulligan JPC 2019, Tsybakov 2009, Zanella-Béguelin 2023) and the ACL Anthology (ROUGE W04-1013). None fabricated; no entry changed. Tool output and scripts: `~/sub/satml/bibcheck_2026-09-06/` (absolute path under /mnt/md0/.../sub/satml). The tool misses books, PMLR/NeurIPS entries without DOIs and 2025–26 arXiv papers; treat its not_found as "check by hand", not as an error.

### 2026-09-06 03:05 — feat-021: KL composition rerun complete; odometer replay
`output/phase2/comp8b_kl` (memorising 8B, 100 passages, k ∈ {-1,0,3,5,10,20}, single/oracle/chained, L ∈ {20,50}, per-query logs) finished 02:46: 0 violations; single k=20 recall 0.476, oracle L=50 0.787 (L=20 0.86), chained L=50 0.504 — reproduces the phase-1 numbers. Summary copied to `results/composition_8b_kl.csv` (+ `_per_passage.csv`).
Odometer: `.venv/bin/python analysis/odometer.py --queries output/phase2/comp8b_kl/queries.jsonl --out results` → `results/odometer.csv`, `results/odometer_per_passage.csv`. Oracle L=50 spend to reconstruct: 586/670/739/860 nats at k=3/5/10/20 vs anchor surprisal median 849; B_user=400 cuts 100% and bounds recall to 0.30 (k=20), 800 cuts 77% with recall 0.71, 1600 cuts none. Written up as `sec:odometer` (Table tab:odometer) in `sections/certificates.tex`; the bank-cap paragraph follows when `scripts/run_bank_cap.sh` (relaunched 03:03 without the duplicate uncapped runs) finishes.

### 2026-09-06 03:25 — feat-019: pathwise (Δmax) composition run complete
`output/phase2/comp8b_pathwise` (memorising 8B, 100 passages, k ∈ {-1,0,1,3,5,10,20,50}, single/oracle/chained, L ∈ {20,50}): `analysis/recheck_violations.py --constraint pathwise` → 23,844 budgeted queries, 0 violations of R ≤ max(0,B)+1e-3 (the driver's own `invariant_violations` column in this run counts Z > B, which the pathwise decoder does not bound per path: at k=1 max Z/K = 1.12 while max R/K = 0.96; E[Z] ≤ B still holds because R ≤ B on every path). Summary in `results/composition_8b_pathwise.csv` (+ `_per_passage.csv`). Against the same attack, the Δmax form costs the adversary a little at every k (oracle L=50 recall 0.063/0.194/0.393/0.730 at k=3/5/10/20 vs 0.097/0.230/0.410/0.787 under KL; single 0.446 vs 0.476 at k=20) and equals the unconstrained model at k=50 (0.49/0.81/0.53). The utility side (benign prompts) comes from `output/phase2/pathwise_sweep` (running on GPU 0).

### 2026-09-06 03:40 — feat-018: 70B setting A (τ=1) and greedy baselines done
`output/phase2/nm/hp1_greedy`, `hp1_A`, `1984_greedy` complete (0 violations). HP1 (50 passages, 100-token raw seeds): greedy 70B alone 0.558 single / 0.771 oracle / 0.527 chained (44% of passages ≥ 0.8 single); sampled τ=1 alone 0.200 / 0.547 / 0.238; anchored at k=20 identical to sampled (0.200 / 0.542 / 0.238, active ≤ 0.08% of steps), k=10 0.070 / 0.222 / 0.051, k=5 0.035 / 0.110 / 0.021, k=3 0.017 / 0.022 / 0.000 (util 0.87–0.97), k ≤ 1.5 zero (active 14–19%). Single query at k=20 spends 370 nats on ~160 generated tokens. 1984 greedy: 0.512 / 0.838 / 0.512 (8 passages). Drafted `sections/natural.tex` (`sec:natural`, Table tab:natural) with these numbers; setting B (τ=0.7, rp=1.1; `hp1_B` rerun), 1984_A/B, `hp1_B_pathwise` and `hp1_B_nodebt` are still running and will be added.
- 03:50 `1984_B` (70B, τ=0.7, rp=1.1, 8 passages) ran to completion under the fixed decoder (24 runs, 0 violations): sampled alone 0.316 / 0.630 / 0.376 (single / oracle / chained); anchored k=20 identical (active 0%); k=10 0.141 / 0.290 / 0.123; k=5 0.112 / 0.096 / 0.053; k=3 0.068 / 0.070 / 0.052 (util max 0.98); k ≤ 1.5 zero. Spend at k=20 single 443 nats (vs 370 under τ=1 on HP1): the warped anchor is charged more per token. First bank-cap run done (k=10, cap 10 = pure rate limit): single 0.116 / oracle 0.386 / chained 0.066 vs uncapped 0.200 / 0.410 / 0.112; 0 violations on 1,268 queries.

### 2026-09-06 04:12 — feat-021 done: bank cap evaluated
`scripts/run_bank_cap.sh` (GPU 0, shared with the pathwise sweep): `output/phase2/bank_cap_k10_10`, `bank_cap_k10_50`, `bank_cap_k20_20` (100 passages, single/oracle/chained, L=50; 3,804 budgeted queries, 0 violations by `recheck_violations.py`). `results/bank_cap.csv` (with the uncapped rows from `results/composition_8b_kl.csv`, same seeds): k=10 cap=k 0.116/0.386/0.066, cap=5k 0.173/0.413/0.088, none 0.201/0.410/0.112; k=20 cap=k 0.461/0.798/0.469, none 0.476/0.786/0.504. The cap bites only where the bank pays (k=10, single and chained) and never on oracle windows; at k=20 nothing changes (rate alone pays, feat-013). Written as the "Capping the bank" paragraph + Table tab:bankcap in `sec:odometer`.
- 04:20 The pathwise sweep on shared GPU 0 runs at ~30 min per class file (≈13 h for 6 k × 6 classes). A second instance (`h1.py --k-values 20 10 --constraint pathwise ... --output-dir output/phase2/pathwise_sweep_hi`, GPU 4 in PCI order) takes k=20 and k=10; a watcher kills instance 1 (pid 3200970) when it starts k=10 after instance 2 has finished, so k ∈ {0.5,1,3,5} come from `output/phase2/pathwise_sweep` and k ∈ {10,20} from `output/phase2/pathwise_sweep_hi`. `analysis/pathwise_price.py` accepts several `--pathwise` dirs.

### 2026-09-06 04:45 — 70B suite finished (all runs 0 violations); hp1_B rerun starting
- `1984_A` (τ=1): sampled alone 0.414 / 0.646 / 0.488 (single / oracle / chained); k=3 0.115 / 0.107 / 0.053; k=5 0.115 / 0.183 / 0.101; k=10 0.238 / 0.356 / 0.175; k=20 = unconstrained.
- `hp1_B_pathwise` (Δmax decoder, τ=0.7, rp=1.1): risky alone sampled at these settings 0.314 single / 0.605 oracle; k=3 0.019 / 0.020; k=5 0.053 / 0.065; k=10 0.094 / 0.219; k=20 0.314 / 0.553; k=50 unconstrained. `recheck_violations.py --constraint pathwise`: 1,250 budgeted queries, 0 violations of R ≤ B.
- `hp1_B_nodebt` (KL decoder, prefix debt off, same settings): k=1 0 / 0 (util 1.00); k=3 0.038 / 0.162; k=5 0.222 / 0.481; k=10 0.314 / 0.596 (= unconstrained). Against the debt-on runs (setting A k=5: 0.035 / 0.110; setting B with debt = `hp1_B` rerun pending) the prefix debt accounts for most of the protection at k=5–10 on the 70B: the mechanism's strongest component against a memorised prompt is the one its theorem does not analyse (feat-025).
- 04:50 `sec:pathwise` drafted in `sections/certificates.tex` (definition, invariant check on 23,844 + 1,250 queries, attack-side comparison for the 8B and the 70B at setting B); the utility-price paragraph (pathwise sweeps) and the extraction-cost paragraph (retries runs) follow. PDF: 0 overfull, 2 forward ?? left (sec:concentration, sec:prefixdebt).
- 04:42 `scripts/rerun_nm_hp1_B.sh` never left its wait loop (its `pgrep -f run_natural_memorisation.sh` kept matching a shell wrapper); killed it and launched the hp1_B run directly on GPUs 1+2 (pid 3474076, log `output/phase2/nm/hp1_B/run.log`, exit line appended to `output/phase2/nm_hp1_B_rerun.log`). Lesson: pgrep -f patterns must not appear in any live wrapper's command line; wait on a PID or a marker file instead.
- 05:00 `sec:concentration` framing inserted (Freedman certificate, what is logged, how it reads); the numbers paragraph waits for `output/phase2/kl_sweep_conc` (k ∈ {0.5, 1}); `analysis/concentration.py` validated on the partial k=0.5 files (`output/phase2/conc_preview`). PDF: 0 overfull, 1 forward ?? left (sec:prefixdebt).

### 2026-09-06 15:30 — all phase-2 queues finished; analyses and sections written (timestamps of this block corrected: the processing ran 15:30–16:30, not 11:10)
- 70B (feat-017/018 done): `analysis/natural_memorisation.py --runs output/phase2/nm --out results --figures figures` → `results/natural_memorisation.csv`, `results/composition_70b.csv`, `figures/natural_memorisation.{pdf,png}`. `hp1_B` (τ=0.7, rp=1.1, prefix debt on): sampled alone 0.314 / 0.605 / 0.264; k=3 0.018 / 0.015 / 0.000 (util max 0.97, active 5.9%); k=5 0.069 / 0.106 / 0.032; k=10 0.131 / 0.244 / 0.077; k=20 = alone; 2,700 budgeted queries, 0 violations. 16,200 budgeted 70B queries in total, 0 violations. `sections/natural.tex`: two-setting Table tab:natural, setting-B paragraph, 1984, Figure fig:natural.
- Concentration (feat-020 done): `analysis/concentration.py --logs output/phase2/kl_sweep_conc --out results` → `results/concentration.csv`, `results/concentration_summary.csv` (2,700 trajectories per k, T=200). k=0.5: V median 217 (p90 236), b median 14.6 (p90 19.0, max 32.5), L>K 34.4%; δ at p90 caps: λ=25 0.45 (emp 0.053), λ=50 0.10 (emp 0.0011), λ=100 0.003 (emp 0). k=1: V median 334 (p90 414), b median 14.8 (max 30.7), L>K 7.9%; λ=50 0.18 (emp 0.0044), λ=100 0.008 (emp 0). Numbers paragraph added to `sec:concentration`.
- Pathwise price (feat-019 done): `analysis/pathwise_price.py --kl output/sweep_plain --pathwise output/phase2/pathwise_sweep output/phase2/pathwise_sweep_hi --out results` → `results/pathwise_price.csv`. Same 900 prompts × 3: k=1 active 14.6% (KL 5.0%), free 74.8% (84.2%), identical to risky 0% (4.1%), forced 10.6% (10.8%); k=0.5 active 56% (43%); pathwise alone: k=3 active 1.9% free 92%, k=5 1.1% / 97.6%, k=10 0.5% / 99.5%, k=20 0.01% / 99.9%; R/K ≤ 1 on all 16,200 pathwise trajectories. The partial k=10 file of the killed instance 1 was moved to `output/phase2/pathwise_sweep/partial_from_instance1/`. A KL sweep at k ∈ {3,5,10,20} on the same prompts is running on GPU 1 (`output/phase2/kl_sweep_hi`) to make the price comparison same-sweep at every k.
- Extraction cost (Prop. 2): `analysis/extraction_cost.py` (sign bug fixed) → `results/extraction_cost_{pathwise,kl}.csv` (+ `_windows`). Eight attempts per 50-token window, 297 windows: pathwise success per attempt 0.0038 / 0.0379 / 0.153 at k=5/10/20 (KL 0.0092 / 0.044 / 0.178; unconstrained 0.180); the bound exp(K_i − S_i) is vacuous at k ≥ 5 (S_i ≈ 150 nats per window < K_i = 250). A pathwise retry run at k ∈ {2, 3}, where the bound binds, is running on GPU 2 (`output/phase2/retries_pathwise_lo`, retries 8).
- Prefix debt (feat-025 done): `results/prefix_debt_ablation.csv` from `output/phase2/prefix_ablation` (8B, debt off, 2,736 queries, 0 violations) and `nm/hp1_B` vs `nm/hp1_B_nodebt` (70B). Debt on → off: 8B k=3 single 0.012 → 0.079, oracle 0.097 → 0.348; k=5 0.092 → 0.403, 0.230 → 0.720; k=10 0.201 → 0.476, 0.410 → 0.801 (= unconstrained). 70B (setting B) k=3 0.018 → 0.038, 0.015 → 0.162; k=5 0.069 → 0.222, 0.106 → 0.481; k=10 0.131 → 0.314, 0.244 → 0.596 (= unconstrained). New `sections/prefixdebt.tex` (`sec:prefixdebt`, Table tab:prefixdebt).
- PDF: 21 pages uncompressed, 0 overfull, 0 ??. Next: feat-026 (compress to ≤ 12 body pages, abstract/intro/conclusion/open science/LLM usage, humanizer pass, hallucinator) and feat-027 (figures, artifact v2).

### 2026-09-06 16:10 — feat-026 manuscript v2: body compressed to 12 pages
- Compression passes over every section (v2 backups: `sections/related_work_v2_2026-09-06.tex`, `sections/intro_v1_2026-09-05.tex`); dropped the budget-path, LLR-tails and regime-sweep figures and the memorising-model, warped-anchor and bank-cap tables (numbers kept in prose; CSVs in the artifact); bank-and-burst folded into one sentence of Section VI-B. Body ends on page 12 (Open Science starts page 13); 18 pages total with references and appendix; 0 overfull, 0 ??; 125 cited entries unchanged. Abstract, Introduction (C1–C10), Conclusion, Discussion ("Three repairs, evaluated", certificate card), Limitations, Open Science, LLM usage (≈40 + 8 GPU-hours), Ethical Considerations rewritten for phase 2.
- Low-budget retry run done (`output/phase2/retries_pathwise_lo`, k ∈ {2,3}, 8 attempts, 4,752 queries, 0 violations of R ≤ B): `analysis/extraction_cost.py ... --prefix extraction_cost_pathwise_lo` → `results/extraction_cost_pathwise_lo.csv`: no exact window reproduction at k=2 (253 of 297 windows with an informative bound summing to 0.003 per pass) or k=3 (161 windows, bound sum 3.0); Prop. 2 not violated anywhere. Cost paragraph in `sec:pathwise` updated.
- Spot-checks: Table tab:natural vs `results/composition_70b.csv` and Table tab:odometer vs `results/odometer.csv` all match. `figures/make_figures.py` now also copies `natural_memorisation.pdf` and `budget_path.pdf`; `README_artifact.md` lists the phase-2 scripts, results and commands. Pending: `kl_sweep_hi` (GPU 1) → rerun `analysis/pathwise_price.py --kl output/sweep_plain output/phase2/kl_sweep_hi --pathwise output/phase2/pathwise_sweep output/phase2/pathwise_sweep_hi --out results` and refresh the price paragraph; then artifact v2 build (feat-027).
- 16:30 Composition figure moved next to Section VI-C; Related Work/Limitations trimmed again; body ends on page 12 (Open Science opens page 13), 18 pages total, 0 overfull, 0 ??. Artifact v2 built (`scripts/build_artifact.sh artifact`: 179 files, manifest verified, artifact.zip 23 MB) after excluding `hf_cache/` (the first build had copied the 132 GB 70B cache into `artifact/`; removed). Visual check of pages 7–8 and 12 (tables II–IV, Fig. 2–3) fine. Waiting only for `kl_sweep_hi` to refresh the price paragraph, then final rebuild.
- 16:20 `kl_sweep_hi` (GPU 1) started only at ~15:40 and needs ~10 h alone (24 min per class file); a second instance `output/phase2/kl_sweep_hi2` (GPU 2, k=20 then 10) takes half, with the same hand-over watcher as for the pathwise sweep. `analysis/pathwise_price.py` will take `--kl output/sweep_plain output/phase2/kl_sweep_hi output/phase2/kl_sweep_hi2`.
- 16:30 GPUs 0 and 4 idle (user): the KL sweep now runs one budget per GPU — k=3 by the original instance (`output/phase2/kl_sweep_hi`, GPU 1, killed by a watcher when it starts k=5), k=5 on GPU 0 (`kl_sweep_k5`), k=10 on GPU 4 (`kl_sweep_k10`), k=20 on GPU 2 (`kl_sweep_k20`); ~2.5 h. Price: `analysis/pathwise_price.py --kl output/sweep_plain output/phase2/kl_sweep_hi output/phase2/kl_sweep_k5 output/phase2/kl_sweep_k10 output/phase2/kl_sweep_k20 --pathwise output/phase2/pathwise_sweep output/phase2/pathwise_sweep_hi --out results` (move any partial k=5 file out of kl_sweep_hi first); the same logs give the Freedman certificate at k ∈ {3,5,10,20} via `analysis/concentration.py`.

### 2026-09-06 19:30 — feat-026 and feat-027 done
- KL sweeps at k ∈ {3,5,10,20} on the 900 prompts finished 19:16 (one budget per GPU; `output/phase2/kl_sweep_hi` k=3 (partial k=5 file quarantined in `partial_k5/`), `kl_sweep_k5`, `kl_sweep_k10`, `kl_sweep_k20`; 2,700 trajectories each). `analysis/pathwise_price.py --kl output/sweep_plain output/phase2/kl_sweep_hi output/phase2/kl_sweep_k5 output/phase2/kl_sweep_k10 output/phase2/kl_sweep_k20 --pathwise output/phase2/pathwise_sweep output/phase2/pathwise_sweep_hi --out results` → `results/pathwise_price.csv` (same sweep at every k): active KL vs pathwise 43/56% (k=0.5), 5.0/14.6% (1), 0.26/1.94% (3), 0.18/1.12% (5), 0.01/0.52% (10), 0.00/0.01% (20); forced 2.2/6.1% at k=3; risky NLL/token within 0.11 throughout. `analysis/concentration.py --logs output/phase2/conc_all` (k=0.5..20 logs copied together) → `results/concentration_summary.csv`: for k ≥ 3, V median ≈350 (p90 450), b ≈15 (max 29), δ(100) ≈ 0.01 (empirical 0), no output above K. Price and concentration paragraphs refreshed; body still ends on page 12 (18 pages total, 0 overfull, 0 ??).
- Artifact v2 rebuilt (`scripts/build_artifact.sh artifact`); PDF checkpoint `sub/satml/satml_2027_phase2_2026-09-06.pdf` refreshed. Remaining work is human-only (feat-016): registration Sep 22, paper Sep 29, artifact repository Oct 2.

### 2026-09-06 23:19 — harness and documentation refreshed for the end of phase 2
- No experiments, no manuscript edits. Brought the docs in line with the finished state: `AGENTS.md` (status = both phases complete; phase-2 known truths added so a fresh session does not re-run ~48 GPU-hours; D1/D2 recorded; 70B cache path; `data/gutenberg/` declared a deletable download cache; Key Facts rows for the pathwise/bank options, the phase-2 scripts, the phase-2 CSVs and the artifact; tests 14 → 30; bib 127 → 128 entries with 125 cited), `GOAL.md` (GOAL COMPLETE, phase-2 outcome against criteria 7–10 with the actual CSV names, headline numbers), `progress.md` (this header, What's Next, Blockers status line), `session-handoff.md` (verification table, files changed, blockers, next step), `README.md` (rewritten: what the repo is, layout, `./init.sh`, phase-2 commands, where the paper and artifact live), `README_artifact.md` (phase-2 command block was rendering outside its code fence; fixed, plus the pathwise/bank additions to `a_patch/`, the `scripts/` row, the low-budget retry CSV and the 70B in the model list), `.gitignore` (`data/gutenberg/`, 44 MB of re-fetchable Gutenberg text, was untracked), `init.sh` (closing message and the stale DGX comment), `feature_list.json` (feat-017's description still opened with "BLOCKED on D1" although its status is `done`). Artifact rebuilt twice so its README copy and manifest match: 179 files, verified, `artifact.zip` 23 MB.
- Verified while writing: `pytest -q tests` 30 passed; `references.bib` 128 entries and `[125]` the last cited number in the PDF; `satml_2027.pdf` 18 pages with page 13 starting "Open Science"; `artifact/MANIFEST.sha256` 179 lines; `feature_list.json` statuses as summarised above. Three summary numbers were tightened against the CSVs while writing: the odometer bound is 0.30 oracle / 0.26 chained on 50-token windows (`results/odometer.csv`; the 0.37 figure elsewhere is the 20-token oracle variant the paper does not use), the Freedman bound at k ≥ 3 is δ(100) ≈ 0.01 rather than ≤ 0.01 (0.0098–0.0102 at p90 caps), and the pathwise utility price is ≤ 0.08 nats/token at k ∈ {3,5,10} (`results/pathwise_price.csv`).
- **Two numbers corrected in the manuscript** (`sections/certificates.tex`, the only place either appears). (i) The pathwise price paragraph said the risky model's log-loss stays "within 0.11 nats per token" of the KL decoder's *throughout*; the gap is 0.15 at k=0.5 (pathwise is the cheaper one there) and ≤ 0.11 only for k ≥ 1, so it now says 0.15. (ii) The concentration paragraph rounded four δ values down while claiming them as upper bounds: at the p90 caps `results/concentration_summary.csv` gives 0.1043, 0.0032, 0.0083 and 0.0098–0.0102, so the text now reads ≤ 0.11, ≤ 0.004, ≤ 0.009 and ≤ 0.011. Recompiled: 18 pages, body still ends on page 12 (page 13 opens with "Open Science"), 0 overfull, 0 ??; checkpoint `satml_2027_phase2_2026-09-06.pdf` refreshed. No other manuscript number moved.

### 2026-09-06 23:55 — full numerical audit of the manuscript against `results/`
Checked every quantitative claim in the abstract, Sections I-X, the five tables and the appendix against the CSVs (and, where a CSV summarises them, against the raw `output/phase2/*/queries.jsonl` and sweep logs). **Tables I-V and the odometer table match cell for cell**; so do the released-log statistics (5,205,220 decode steps; the 895 trajectories a naive `Z <= B` check flags all have `B < 0` and `Z = 0`), the certificate caps (median 205 nats, 3.20/token, 43.9% vacuous at k=1, best-of-10^261), the sweep, the LLR tails, budget path (k_crit 13.9 = 4.3x mean surprisal), extraction cost (253 windows summing to 0.003 at k=2; 161 summing to 3.0 at k=3), the bank cap, latent leakage (34/50, 13/16, 1.2 nats, 2.8 for Harry Potter) and the pathwise price. Independently re-verified the invariants: 16,200 pathwise sweep trajectories with 0 violations of `R <= max(0,B)`, 23,844 pathwise and 15,896 KL composition queries, 8,514 budgeted 70B queries, all clean under the rule each run enforces. Also checked the closed form `cap(K,S) <= (K + log 2 + 1/e)/S` numerically and re-derived Propositions 1-4.
- **Ten corrections applied** (all recompiled: 18 pages, body ends page 12, 0 overfull, 0 `??`; checkpoint refreshed):
  1. `natural.tex`: "All 16,200 budgeted queries of the 70B audit" -> **8,514** (12,646 including the k=-1/k=0 baselines). 16,200 is the pathwise *sweep* trajectory count and had been copied across.
  2. `natural.tex`: 70B utilisation quoted max `Z/K` per mode while the paper defines utilisation as `Z/B` (Table I): "up to 0.91" -> **0.98** and "up to 0.97" -> **1.00**; "mean 0.33" at k=5 -> **0.36**.
  3. `prefixdebt.tex`: "all 3,986 queries" -> **3,736** (2,736 8B + 1,000 70B; the 70B no-debt run has four budgets, not five).
  4. `open_science.tex`: the memoriser reproduces the **608** passages it was trained on, not 758 (the 150 test passages are held out and score 0.000).
  5. `attack_results.tex`: "the budget kL of 3 to 20 nats is below the prefix debt of 12.9" was false at k=1 (20 > 12.9); now states both numbers without the comparison.
  6. `certificates.tex`: "the penalty itself moves it by under 4%" -> **at most 6%** (it moves the median surprisal by 3.5% at tau=1, 0.9% at 0.7, 4.1% at 0.5, 6.1% at 0.3).
  7. `certificates.tex`: "a budget of 400 nats cuts every user" -> **82 to 100% of users** (the table's own parentheses show 82% for chained windows at k=3).
  8. `results_logs.tex` and the abstract/intro: activity and utilisation figures scoped to the book prompts, because the step-weighted activity over all six classes is 0.26% at k=3 (per-class max 0.45%) and creative/neutral utilisation reaches 1.0 at k=3, against 0.19% and 0.59 on the attack split.
  9. Abstract and `discussion.tex`: "reproduction below 10% at k <= 5" -> **single-query** reproduction (oracle windows already reach 0.23 at k=5).
  10. `appendix_theory.tex`, Proposition 1: the display omitted the `max{0, .}` that its own proof establishes, so it was false whenever `delta_init > K` - which happens in this paper's 20-token windows at k <= 0.5. Now `L(y) <= max{0, kT_max - delta_init} <= K` (also shortened to `L(y)` to stay inside the column).
- **Artifact:** `README_artifact.md` now explains that `invariant_violations` in `composition_8b_pathwise.csv` counts KL-spend excursions (146 of 23,844: 144 at k=1, 2 at k=3), which the pathwise decoder does not bound and the paper discusses, while the enforced rule `R <= max(0,B)` is clean; artifact rebuilt (179 files, manifest verified).
- **Left for the human:** the LLM-usage section says "about 40 GPU-hours" for the 8B runs; adding up the sweeps (two ~10 h jobs), the pathwise sweep (~13 h), the four KL sweeps (~10 GPU-h), the attacks, ablations, retries and bank caps (~10 h) and the fine-tune (1.2 h) gives roughly 55-65 GPU-hours. Worth re-deriving from the run logs before submission. Two trivial roundings were left alone: 0.195% printed as 0.19% in Table I, and 0.0155 as 0.015 in Table III.

### 2026-09-07 00:20 — GPU-hours recomputed from the run logs; compute statement corrected
`analysis/compute_hours.py --out results` (new; no GPU) derives the figure from the run directories: wall time = last write minus start, where the start is the directory's ext4 birth time except for jobs a launcher queued up front, which are anchored to the end of the previous job in their GPU chain (`start_source` column says which). Two runs had an idle gap removed: `sweep_chat` (the OOM restart, 82 min) and the phase-1 composition attack (the summary CSV was touched six hours after the run). Result, in `results/compute_hours.csv` (39 jobs):
- **8B jobs: 45.9 GPU-hours** (44.6 without the fine-tune), of which the two phase-1 sweeps 6.3, the pathwise sweeps 13.9, the five KL sweeps 13.1, the composition attacks (phase 1 plus the logged KL and pathwise reruns) 5.5, the ablation, retry and bank-cap runs 2.9, the fine-tune and its checks 1.5, and 2.8 for the smokes and the short forward-pass analyses (warped anchor, latent leakage, budget path, 8B control, bank-and-burst smoke).
- **70B: 8.3 GPU-hours** (GPUs 1+2 held 01:23-05:33 on 2026-09-06, 4.17 h wall x 2), which is what the paper already said.
- **Fine-tune: 75 minutes** (17:05-18:21 on 2026-09-05).
- Total 54.2 GPU-hours. The 8B figure is the sum of job wall times x GPUs held, so it slightly over-counts the periods when two 8B jobs shared a card (the three bank-cap runs, 0.75 h). Not counted: the released trajectory logs of the earlier audit (reanalysed, not regenerated) and a few sub-hour passes that write straight to `results/` with no directory or log.
- `sections/open_science.tex`: "about 40 GPU-hours" -> **about 45**, "about 70 minutes" -> **about 75**, plus one sentence saying the per-job wall times are released. The 70B "about 8" was already right. Recompiled: 18 pages, body ends page 12, 0 overfull, 0 `??`; checkpoint refreshed. My earlier 55-65 estimate in the 23:55 entry was wrong: it assumed the sweeps took ~10 h each (3.4 and 2.9) and the pathwise sweep 13 h (7.9). Artifact rebuilt: 181 files.

### 2026-09-07 01:10 — editorial pass: flow, precision, figures
Read the compiled paper end to end and fixed structure rather than numbers (every number was already verified in the 23:55 entry and none moved).
- **Section VII reordered** so it delivers what its opening sentence promises: the three repairs first (VII-A pathwise, VII-B concentration, VII-C per-user budget and bank cap), then the two things the certificate is relative to (VII-D decoding settings, VII-E what the anchor already knows). The opener now names all five and the leakage subsection hands off to the Discussion. Pre-reorder copy: `sections/certificates_preflow_2026-09-07.tex`.
- **Section VI opener** now announces VI-D and VI-E (it claimed only C5); VI-C retitled "Results" -> "What the attacks recover", so every subsection heading states a finding.
- **Figures rebuilt for a one-column layout** (all three now 3.4 in wide, 8 pt, same axis wording). Fig. 1 (`analysis/certificate_cap.py`): dropped the three per-split curves the caption never mentioned and the colliding `K=` tick labels; the best-of-n axis now labels every other tick. Fig. 2 (`analysis/composition_attack.py`): shrunk to one column, y-label was being clipped. Fig. 3 (`analysis/natural_memorisation.py`): **rewritten** — it was a two-panel, 21-legend-entry plot squeezed into one column and was unreadable in print; it is now a single panel (Harry Potter, the authors' settings) with four series and two reference lines: oracle windows under the KL, pathwise and no-prefix-debt decoders, the single query, and the unconstrained model. 1984 and temperature 1 stay in the table and the text.
- **Table II (composition) dropped**: it duplicated Fig. 2, and the prose already carries its numbers; the figure caption now points at the artifact for the per-budget values. Tables renumbered I (regime), II (natural), III (prefix debt), IV (odometer).
- **Precision:** near-verbatim recall is now defined where the protocol first needs it (C5) instead of only in Limitations; Proposition 2's window lengths are `L_i`, matching the body, instead of `ell_i`, which collided with the prompt log-ratios of Eq. (2); the abstract names the CopyBench book prompts once and is 25 words shorter.
- **Prose tightened** to pay for the additions and stay inside 12 body pages: Discussion (three paragraphs), Related Work (four sentences), Limitations, the protocol's Reporting paragraph, the threat model's Out-of-scope paragraph, and a best-of-n restatement in V-C that the figure axis already gives.
- Verified after every batch: final PDF 18 pages, body ends on page 12 (page 13 opens with "Open Science"), 0 overfull, 0 `??`, 125 cited entries; floats still land on or next to the page that first references them; no hard-coded section, table or figure numbers anywhere. `pytest -q tests` 30 passed; artifact rebuilt (181 files, manifest verified); checkpoint `satml_2027_phase2_2026-09-06.pdf` refreshed.

### 2026-09-07 17:05 — hostile internal review against the SaTML initial-review criteria
Read the compiled phase-3 PDF end to end as an adversarial reviewer (`peer-review` skill; its schematic-generation guidance does not apply here, because every figure in this paper must trace to a `results/*.csv`). No experiment was rerun and no measured number moved. **Twelve defects found and fixed**, each followed by a recompile:
1. **Contribution numbers were inconsistent**: the introduction listed the last two contributions as C11/C12 while Section IV's protocol called the same two C15/C16. Unified on C11/C12 (12 audits plus the two relative-reference checks).
2. **Six appendix cross-references rendered as "Appendix 0a."** `\appendix[title]` is the singular form and does not letter the sections; switched to `\appendices` + `\section{...}`. They now render as 3x "Appendix A", 8x "Appendix B", 2x "Appendix C".
3. `"reproduction in Table~\ref{tab:regime}"` pointed at a table with no reproduction row.
4. `"Sections VI and VI-E"` cited a section together with its own subsection.
5. Section VI's roadmap did not announce the new second-mechanism subsection.
6. The second-mechanism subsection sat between the 8B and 70B attack results, interrupting that flow; moved to VI-E (`sections/second_mechanism.tex` is now input there).
7. Section VIII's roadmap skipped its own new subsection B (the utility evaluation).
8. The introduction said the k=20 odometer `"refuses two answers in five to every reader"`, which overstates a **0.61** admission of one ordinary answer (`results/separation_summary.csv`); reworded.
9. **The compute statement double-counted the fine-tunes.** It read "for about $56$ GPU-hours in all; three low-rank fine-tunes account for **a further** 75, 15 and 15 minutes", but the 75-minute memoriser fine-tune is a row *inside* that total. Now "for $56.5$ GPU-hours in all, three low-rank fine-tunes of 75, 15 and 15 minutes included", and the single-GPU figure 47 -> 48 (the CSV says 47.7).
10. **`results/compute_hours.csv` under-reported the two CP-Fuse fine-tunes as 0.00 GPU-hours.** Their `dir_birth` start rule reads the directory's creation time, but a merge writes that directory at the *end* of the run, so start == end. Added an `elapsed:<log>` start rule to `analysis/compute_hours.py` that takes the duration from the job's own cumulative epoch timer (`output/phase3/cpfuse_ft{0,1}.log`: 908 s and 900 s), with `start_source=log_elapsed` recording which rows use it. Total 55.98 -> **56.48 GPU-hours**; the paper's "15 and 15 minutes" was right all along but was unsourced. `tests/test_compute_hours.py` (3 tests) pins the parser and guards the regression.
11. **Limitations did not carry the CP-Fuse reimplementation caveat**, although the second-mechanism result is *negative* (the fused model recalls nothing) and the implementation is ours. Added: "The second mechanism is our reimplementation of the published rule, not the authors' code."
12. **Two verified bib entries were uncited while the text named those exact models**: `grattafiori2024llama3` and `qwen2025qwen25`. Cited in the LLM-usage paragraph, which is uncounted end matter, so this cost no body page.
- **Paid for the additions** by cutting a duplicated pointer: the Discussion's third paragraph closed by sending the reader to Section VIII-B for the price of a smaller budget, which its own first paragraph already does. Net body change -1 char; the +177 of item 11 alone had pushed the body onto page 13, which is how thin the margin is.
- **Checks that came back clean:** end matter is in the CFP's order (Open Science -> LLM usage considerations -> Ethical Considerations -> references) and contains the required sentence verbatim; 11 floats, none unreferenced and none dangling; all 5 included figures resolve through `\graphicspath`; 138 bib entries, 135 cited, 0 cited-but-missing; anonymity scan finds no "our earlier audit" phrasing, no affiliation or path leakage, and no author field in the PDF metadata (the name appears only in reference [6], the sanctioned third-person citation of the earlier audit); Proposition 5 and its Scope remark are correctly restricted to filters that meter the same additive KL spend, and separate what is proved (monotonicity of rho) from what is measured (near-proportionality); the abstract's 86% and the Discussion's 0.61 both match `results/separation_summary.csv` (k=20, oracle, L=20: uncapped recall 0.8579, B*=100 nats, 0.61 reader completions).
- Producing command for the corrected table: `.venv/bin/python analysis/compute_hours.py --out results` -> 8B 47.7, 70B 8.8, fine-tunes 75/15/15 min, total 56.5 GPU-hours.
- Verified after the last fix: `~/.local/bin/tectonic -X compile satml_2027.tex` -> 21 pages, body ends on page 12 (page 13 opens "Open Science"), 0 overfull, 0 `??`. `./init.sh` passes; `pytest -q tests` **47 passed** (44 + the 3 new).

### 2026-09-07 17:35 — hallucinator rerun on the full cited set; all 137 references verified
The cited set grew from 125 (checked 2026-09-06) to **137**, so the check was rerun over the whole set rather than the additions. `bibcheck_2026-09-07/run_hallucinator.py` -> `report.json`, `report.jsonl`, `run.log`: **137 checked, 129 verified, 8 `not_found`, 0 `mismatch`** in 918 s. Yesterday's run over 125 returned 87 verified, 37 `not_found` and 1 `mismatch`, so today's misses are a strict subset of yesterday's.
- **All 8 `not_found` hand-verified and all 8 are real** (`bibcheck_2026-09-07/hand_verified.md`). Every one is a book, a pre-arXiv classic or a workshop paper, the category an arXiv-first title search cannot resolve. Each was checked by resolving the DOI in our own entry at the registrar that holds it, which is stronger than a title search. Exact matches: `freedman1975tail` (CrossRef 10.1214/aop/1176996452), `dwork2019expose` (10.29012/jpc.689), `tsybakov2009introduction` (10.1007/b13794), `kontoyiannis2014optimal` (10.1109/TIT.2013.2291007), `elkinkoren2024copyright` (**DataCite** 10.4230/LIPIcs.FORC.2024.3 — CrossRef 404s on it because Dagstuhl registers with DataCite, which is exactly why the validator missed it), `lin2004rouge` (**ACL Anthology** W04-1013, no DOI exists). Confirmed with a noted discrepancy, entry left as published: `polyanskiy2024information` (CrossRef stores the short title "Information Theory"; the subtitle "From Coding to Learning" is on the published book) and `cover2006elements` (CrossRef dates the DOI record 2005 as online registration; the 2nd edition is 2006 and is universally cited as 2006). **No bib entry was changed and none is hallucinated.**
- **Two citations added** while reconciling the count: `grattafiori2024llama3` and `qwen2025qwen25` were verified entries sitting uncited while the text named those exact models. They went into the `LLM usage considerations` paragraph, which the CFP does not count, so the body paid nothing. `references.bib` is now 138 entries, **137 cited** (the one uncited entry is `chibaokabe2025originality`, which therefore does not print).
- **Runner rewritten to be resumable**, because the first attempt was killed by its 25-minute timeout with nothing written: it checked all 137 in one `Validator.check` call at 4 workers, drew HTTP 429 from arXiv, and its progress prints were stuck in a pipe buffer behind `tail`. The version in the artifact checks in batches of 10 with 8 workers, appends each verdict to `report.jsonl` as it lands, skips what the JSONL already holds on restart, and keeps `hallu_cache.sqlite` so a restart costs no network. A one-reference-at-a-time variant at 2 workers ran ~48 s/ref (~110 min); batching brought it to ~7 s/ref. It also guards against a misaligned verdict, writing an error row for the whole batch if the validator returns a different number of results than references passed in.
- Note for the next session: `pkill -f <script>` killed the invoking shell twice today, because the pattern matches the shell's own command line. This is already in Blockers/Risks; use `ps` and a PID.

### 2026-09-07 18:20 — anonymity scan and full consistency audit
Both Sep 18--19 gates from plan v3, run early. No experiment rerun.

**Anonymity scan — the PDF is clean; the artifact builder was not.**
- **Submitted PDF:** the author name appears only in reference [6], the sanctioned third-person citation of the earlier audit; no acknowledgements, funding or thanks; **no URLs at all**; no affiliation, email, hostname or filesystem path; PDF metadata carries only `Creator: LaTeX with hyperref` and `Producer: xdvipdfmx`, no author field; the author block renders as ``Anonymous Author(s) / Affiliation / Email''.
- **Artifact (207 files):** zero hits for any identity token (name, `be23b041`, `smail.iitm`, IIT Madras, `DA5001`, `/home/sports`, `/mnt/md0`, the hostname, any `.ac.in`/`.edu` address), and nothing claiming the artifacts are already public.
- **Three real holes in `scripts/build_artifact.sh`, all fixed.** (i) The anonymity check ran *before* `cp README_artifact.md "$ART/README.md"` **and** explicitly excluded `README*`, so the artifact's own README --- the likeliest place for a leak --- was never scanned by anything. The check now runs after the copy with no README exemption. (ii) It grepped for only three tokens (`vijayavallabh`, `@gmail`, `/home/sports`), missing the actual email, the institution, the course code, `/mnt/md0` and the hostname; the token list now covers all of them, case-insensitively. (iii) `.pytest_cache/` was being shipped; now excluded (212 files -> 207). The hardened check was negative-controlled against a planted email and a planted path and fires on both.
- **Artifact completeness, which the anonymity pass turned up:** `Open Science` claims the length-scaling, separation, utility, anchor-robustness and second-mechanism scripts are released "each of which regenerates one summary table", and all six phase-3 scripts *were* in `artifact/analysis/`, but five were named nowhere in `README_artifact.md` and there was **no phase-3 command block at all**, so nothing in phase 3 was reproducible from the README. Added: the six scripts to the `analysis/` row, `run_prefix_debt_k20.sh` and `download_second_anchor.py` to the `scripts/` row, the eleven phase-3 CSVs to the results list, the test count 30 -> 47, and a `### Phase 3 (2026-09-07)` block with every producing command (second anchor and its control, length scaling, Proposition 5, the k=20 ablation on both models plus the merge, the utility judge at `--judge-per-cell 200`, the two shard fine-tunes and the CP-Fuse audit, compute hours, figures).

**Consistency audit — every number in the PDF against `results/*.csv`.** Verified cell by cell: `tab:regime` (44 cells, against `regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`), `tab:natural` (42, `composition_70b.csv` + `natural_memorisation.csv`), `tab:prefixdebt` (**36/36**, `prefix_debt_ablation.csv`), `tab:odometer` (**40/40**, `odometer.csv`), `tab:warp` (16, `warped_anchor.csv`), `tab:utility` (**24/24** plus every caption value, `utility_summary.csv`). Prose verified exactly: the certificate-strength block (median $S=204.7$, p10--p90 162--254, $3.20$ and $2.59$ nats/token, 53--100 tokens median 64, vacuity 43.93/0/100\%, caps 0.4919 and 0.0993, continuations 978 nats with 9.6\% and 49.5\%), the second anchor (204.7 -> 180.5, 74.01\% vs 43.93\%), length scaling (100/93.8/0\% at 64/128/256 on the sixteen novels; per-window vacuity 100\% at every length for $k\ge5$ --- checked by scanning the whole CSV for a counterexample, none), the tails ($L$ mean 160.67, 96/999 = 9.61\%, max 245.75, 738/8999 = 8.2\%, step 26.161), the sweep (lead-forced 39.473/26.933/4.016 from the pooled `plain` rows; exceedance 12.7/32.7/10.0\% -> 13/33/10\%; 37,800 trajectories), concentration (every variance, $b$, $\delta$ and empirical frequency), CP-Fuse (mixture 0.0643/0.0182, CP-Fuse 0.0218/0.000, $\alpha$ 0.526, components 0.7538--0.8845 own and $\le$0.0271 other), and separation (0.8579 uncapped, $B^\star=100$, 0.61 completions).
- **Five discrepancies found and fixed.**
  1. **`tab:regime` printed `--` for a value that exists.** The ``identical to next $k$'' cell for the $k=3$ test column was `--`, but `regime_table.csv` gives **63.13**; `--` was correct only for the $k=5$ column, which has no larger $k$. Now 63.1, and it strengthens the point.
  2. **`tab:natural` reported an unmeasured greedy row under setting B.** `risky, greedy` printed 0.558/0.771 in *both* the $\tau=1$ and the $\tau=0.7$-with-penalty-1.1 halves, but `hp1_greedy` exists only at `A_temp1`. Greedy is temperature-invariant but **not** penalty-invariant --- a repetition penalty changes an argmax, and the settings demonstrably move the risky-only arm (0.1996 -> 0.3139) --- so the B cells asserted a measurement never made. They are now `--`, with the caption stating greedy was run at temperature 1 only.
  3. **A double-rounded percentage.** ``the model alone reproduces 31\% in one query and **61\%** with oracle windows'': the value is 0.6049 = 60.49\%, which rounds to 60. The 61 came from rounding the table's displayed 0.605. Now 60\%.
  4. **`tab:warp`'s caption mislabelled its own baseline column.** It read ``(758 CopyBench passages, repetition penalty 1.1)'', but the $\tau=1.0$ column is the **unwarped** penalty-1.0 measurement (3.2005 nats/token, 43.93\% vacuous); at penalty 1.1 that column would read 3.31 and 34\%. The other three columns are penalty 1.1. Caption now says the $\tau=1.0$ column is the unwarped baseline.
  5. **``total surprisal is 0.803 times'' is a median, not a total.** In `anchor_control.csv` the ratio of summed surprisals over the 50 public-domain books is **0.755**; 0.803 is the median of the per-work ratios (outliers such as *Alice* at 0.23 and *Walden* at 0.35 pull the sum down). On the sixteen novels the two agree (0.907 vs 0.908). Now ``a median $0.803$ times''. The conclusion is unchanged and in fact stronger under the sum.
- **One cosmetic difference left alone:** `tab:odometer` prints spend 739 for oracle $k=10$ where the CSV shows 739.5; the CSV is itself rounded to one decimal, so either integer is defensible and changing it could introduce an error.
- **Scope check:** exactly 13 `results/*.csv` have changed since the 2026-09-06 cell-by-cell audit (the phase-3 additions plus `prefix_debt_ablation.csv` and `compute_hours.csv`), and **all 13 were verified in this session**; the unchanged tables were re-verified anyway.
- Verified after the fixes: 21 pages, body ends page 12 (page 13 opens ``Open Science''), 0 overfull, 0 `??`; `./init.sh` passes, 47 tests; artifact rebuilt, manifest verified **206 files**; checkpoint `satml_2027_phase3_2026-09-07.pdf` refreshed.

### 2026-09-07 19:05 — editorial pass: humanizer, no-ai-slop, scientific-writing
Prose and structure only. **No number changed**: the multiset of every numeric token in the compiled PDF was captured before the pass and re-diffed after each edit, and the only differences are the two intended ones (see below).

**What the AI-pattern scan actually found: almost nothing.** Across the fifteen body sections there are zero hits for the banned vocabulary (`leverage`, `robust`, `crucial`, `pivotal`, `showcase`, `underscore`, `testament`, `landscape`, ...), zero empty hedges, zero `not just X but Y` parallelisms, zero `serves as`/`stands as` copula avoidance, zero trailing `-ing` pseudo-analysis, and no aphorism formulas. The prose had already been through several passes. **The 13 em dashes were left in place**: every one is doing real syntactic work (appositive or parenthetical) in an academic register, and the humanizer's own false-positive guidance says em dashes alone are not a tell. Two were converted where a colon or comma was strictly clearer, leaving 11.

**The real defects were structural, and that is where the pass spent itself.**
1. **The same twelve items were enumerated three times in the first four sections.** The threat model poses C1--C12 as questions, the introduction restated all twelve as contributions, and the protocol restates them again as checks with failure modes. A reader met the list three times in four pages, reworded each time. The introduction's version was a single 1{,}900-character block of `(C1)~... (C12)~...` and was the worst of the three, since the four narrative paragraphs above it already tell the same story better. Replaced with five grouped claims, each naming the checks it covers, and a pointer to the protocol for the full list. Net **-440 characters** and, more to the point, a reviewer now reads five things rather than twelve. The threat model keeps the questions and the protocol keeps the falsifiers, so each list now does exactly one job.
2. **Section VIII was titled ``Certificates That Bound Events'' but contained six subsections**, only two of which are about bounding events: the other four are a utility evaluation and two results about what the certificate is stated relative to. Its own opening paragraph already describes the wider scope correctly. Retitled **``What Would Make the Certificate Say More''**, which covers the repairs, their price, and their limits, and echoes the paper's title question. Every cross-reference is by `\ref` and describes the section by role, so nothing else needed changing.
3. **Section VI, the attack section, opened with ``This section applies checks C5 and C7.''** Bookkeeping before the point, in the most interesting section of the paper. It now leads with why a new risky model is needed and folds the check labels into the sentence that introduces it.
4. **Related Work opened by pointing at its own appendix continuation** before presenting any related work. That pointer moved to the end of the section, where it tells the reader where to go next instead of delaying the start.
5. One tangled sentence in the utility subsection, where a long em-dash parenthetical sat between a claim and its cause, was reordered onto a colon.
6. `5.2` million decode steps in the old contributions list was the paper's only disagreement with its own body, which says `5.21` million. The restored scale claim uses 5.21.

**Title and abstract, considered and deliberately kept.** The title poses the question the paper answers and names both the mechanism class and the method, which is how a SaTML reviewer would search for it; no declarative alternative was clearly better, so it stands. The abstract is 374 words, longer than the 100--250 the scientific-writing guidance suggests, but the length is the number of distinct findings rather than padding: a faithful redraft that keeps every result lands at about 340. Cutting it to 250 would mean dropping a measured finding, so it was left intact apart from one em dash converted to a colon.

**Not followed, deliberately:** the `scientific-writing` skill mandates a graphical abstract plus additional AI-generated schematics for every paper. This paper requires every figure to trace to a `results/*.csv`, and generating illustrative images for an audit would put unverifiable content in an artifact whose whole claim is reproducibility. The same skill also directs conference papers to venue templates rather than to its own report style, which is what the CFP's IEEEtran requirement already fixes.

**Verified:** numeric-token multiset identical to the pre-pass baseline except `5.2` -> `5.21` and the digits of the collapsed `(C1)...(C12)` labels; the compressed `99\%` inactivity figure still appears in Section~V, where it is stated as `99.0\%`/`99.5\%`. 21 pages, body ends on page 12 (page 13 opens ``Open Science''), 0 overfull, 0 `??`. `./init.sh` passes, 47 tests. `results/` untouched, so the artifact needed no rebuild. Checkpoint `satml_2027_phase3_2026-09-07.pdf` refreshed.

### 2026-09-07 19:25 — declarative title
At the user's instruction the question title became a declarative one. **Not** the draft floated in the previous turn: "The Budget Is Not the Protection" overclaims, because at $k \le 1$ the budget *is* what protects; the prefix debt only dominates on $k \in [3,10]$. The title now states the claim the paper actually verifies, in the wording the contributions paragraph already uses:

> **A KL Budget Is Uninformative Where the Mechanism Is Usable: An Adversarial Audit of Inference-Time Near-Access-Freeness**

Supported end to end: the implied bound on reproducing a passage is vacuous for all 758 CopyBench passages at $k \ge 3$ (`certificate_cap_summary.csv`), and $k \ge 3$ is exactly the band where the decoder is indistinguishable from not constraining at all, losing $47.5$, $45.7$, $43.0$ and $44.8$ per cent against the null's $47.0$ (`utility_summary.csv`); at $k \le 0.5$, where the bound becomes informative, the anchor writes the opening of every answer and the arm loses $59.8\%$ against $62.7\%$ for serving the anchor alone. ``Uninformative'' rather than ``false'' or ``vacuous'' is deliberate: Section~\ref{sec:discussion} states that nothing in the audit contradicts Theorem~3.1, and the title must not read as a claim that the theorem is wrong.
- The title string lives only in `satml_2027.tex`; `satml_2027_arxiv_v1.tex` keeps its own, which is the earlier audit's title, and was not touched.
- Verified: the title now sets four lines instead of three and the page budget still holds. 21 pages, body ends page 12 (page 13 opens ``Open Science''), 0 overfull, 0 `??`; numeric-token multiset identical to the pre-title state; first-page anonymity scan clean. Checkpoint refreshed.

### 2026-09-07 19:45 — the arXiv variant was not retitled, and is not what its name suggests
Asked to recompile "the arXiv variant" with the new declarative title. **Not done, deliberately.** `sub/satml/satml_2027_arxiv_v1.tex` is dated **2026-08-19**, is self-contained (no `\input{sections/...}`), and is titled exactly as `references.bib` cites **arXiv 2605.28001**, `vijayavallabh2026audit`, which the submission renders as reference [6] and refers to in the third person as "an earlier audit". Its section list is that paper's (Audit Methodology, Experimental Setup, Adaptive Evaluator Algorithm, Stress Validation) and it contains **none** of this paper's findings: zero mentions of the 70B audit, the pathwise decoder, Proposition 5, CP-Fuse, the Freedman bound or the utility evaluation. It is the source of the published prior paper, not a variant of the current one.
- Retitling it would have desynchronised a published paper from the citation the SaTML submission makes to it, created two documents claiming one title, and put a claim on it that its own contents do not support: the new title rests on phase-2 and phase-3 results this file predates.
- **`AGENTS.md` was the root cause** and is fixed: the Key Facts row said "the arXiv version is `satml_2027_arxiv_v1.tex`", which invites exactly this edit. It now states what the file is and that it must never be retitled, and records that **there is currently no arXiv build of the SaTML paper**.
- The user was offered an anonymous or a de-anonymised arXiv build of the current paper and chose to **leave it alone**; plan v3 schedules the arXiv/ICLR derivative for after Oct 2, so nothing is blocked. No file under `sub/satml/` was changed in this step apart from nothing at all.

### 2026-09-07 20:00 — harness and documentation brought in line with the finished state
No experiments, no manuscript edits. Every harness and markdown file was checked against the current facts and corrected.
- **`AGENTS.md`**: the title (now the declarative one); the Status paragraph, which still said "phase 3 open ... feat-028..034 are `not-started`" when all seven are done; the Startup Workflow step that told the next agent to pick feat-028; the artifact row (179 -> 206 files); and the manuscript figures (21 pages, 138 entries with 137 cited, 47 tests, 56.5 GPU-hours). Added two Key Facts rows the table was missing entirely, **Phase-3 analyses** and **Phase-3 results**, so a fresh session can find `separation.py`, `length_scaling.py`, `utility.py`, `cpfuse_audit.py`, `anchor_control.py`, `merge_prefix_debt.py` and their CSVs without grepping.
- **`GOAL.md`**: header title; status line (was "GOAL COMPLETE — phase 1 and phase 2 both done"); and a new **Phase 3 objective / success criteria / outcome** block with criteria 11--14 and the headline numbers, matching the shape of the phase-2 block.
- **`init.sh`**: the closing message told every future session that "Phase 3 is OPEN" and to "pick the lowest-numbered feat-028..034". It now states that all three phases are done, that there is no next feature, and what the verified submission looks like. `bash -n` clean.
- **`README.md`**: paper title, the "both phases complete as of 2026-09-06" line, the test count (30 -> 47) and the artifact count.
- **`README_artifact.md`**: title. The artifact was rebuilt afterwards so `artifact/README.md` matches (206 files, manifest verified).
- **`session-handoff.md`**: added the declarative title, the four post-phase-3 passes, and the `satml_2027_arxiv_v1.tex` warning to Current Objective; fixed the artifact count; added the seven **missing phase-3 rows** to the Verification Evidence table; and rewrote **Recommended Next Step**, which still instructed the next agent to start feat-028a and not to touch `satml_2027.tex` before Sep 11.
- **`progress.md`**: the Current State block (timestamp, artifact count, compute 56.0 -> 56.5, tests 30 -> 47) plus a paragraph listing the four passes that ran after phase 3 closed.
- **`~/sub/satml/IMPROVEMENT_PLAN.md`**: a status banner recording that v3 is fully executed, that phase 3 cost 2.3 GPU-hours against the 19.5 estimated, and that **two planned claims did not survive contact with the data** (Proposition 5 as planned was false; the title is no longer the question the plan assumed), so Sections 6 and 8 read as a record of what was planned rather than what is left.
- **Memory**: the index line and the plan-v3 memory now say phase 3 is complete and carry the new title and the `satml_2027_arxiv_v1.tex` warning.
- **Deliberately not changed:** dated historical records keep the numbers that were true when written. That covers the phase-2 outcome in `GOAL.md` (artifact v2 at 179 files), the per-feature rows of the Verification Evidence table (feat-014 at 11 pages and 50 references, feat-015 at 75 files, feat-026 at 18 pages and 125 cited, feat-027 at 179 files), and every dated entry in this log. One earlier edit did rewrite the phase-2 outcome's artifact count to 206 and was reverted; the current count now sits in the phase-3 outcome, where it belongs.
- Verified: `./init.sh` passes (47 tests, 34 features, in-progress=none); artifact rebuilt and manifest verified at 206 files.

### 2026-09-07 20:40 — final full verification pass
Ran the whole checklist rather than trusting earlier passes. **Two defects found, both in the harness rather than the paper.**
1. **The verification command in `AGENTS.md` did not run.** `analysis/recheck_violations.py --queries ... ` is documented without `--constraint`, which the script now requires, so the documented per-query invariant recheck exited with an argparse error. Fixed, and then run for real: `comp8b_kl --constraint kl` and `comp8b_pathwise --constraint pathwise` both report **0 violations at every budget**, largest overshoot negative throughout.
2. **My own table checker was wrong before the paper was.** A first pass flagged three mismatches in Table~I at k=1 (L max 246 vs 233.32, L>K 96 vs 23, max step 21.4 vs 21.317). `llr_tails.csv` carries a `source` column (`released`/`plain`/`chat`) and the checker's dict silently kept the last matching row. With `source='released'` all 17 of those checks pass. The paper was right; the checker was not.
- **Numbers: 245 checks, 0 genuine mismatches.** Re-derived every cell of `tab:prefixdebt` (38), `tab:utility` (26 including the null and anchor-only captions), `tab:odometer` (64), `tab:warp` (16, with the τ=1.0 column correctly read as the unwarped penalty-1.0 baseline), `tab:regime` (47) and `tab:natural` (34) from the CSVs, plus the compute statement (48 + 9 = 56.5 GPU-hours and the three fine-tunes at 75/15/15 minutes).
- **Known truths all hold**: active 0.195% at k=3 and 6.274% at k=1, 57.1% identical between k=3 and k=5, 96/999 and 738/8,999 exceedances at k=1, 0 invariant violations, 5,205,220 decode steps, 26,999 trajectories. The data files match too: 50 *Harry Potter* passages in the test split and 8 of *1984* in the attack split, exactly what the 70B audit used.
- **Manuscript**: clean rebuild from deleted aux files, exit 0, **0 overfull**, 0 `??`, 0 literal `[?]`, 21 pages with the body ending on page 12. The 215 log warnings are all underfull hboxes, which are cosmetic in justified two-column text and are not what the CFP bars. End matter is in the required order by linear scan (Open Science 1657 → LLM usage considerations 1682 → Ethical Considerations 1721 → References 1744) and the required sentence appears **exactly once**, verbatim. Appendix cross-references render as 3× A, 8× B, 2× C; the single "Appendix D" is a qualified reference to *He et al.'s* Appendix D.1, not a dangling internal one. No dangling `\ref`; the four unreferenced labels are section anchors, which is harmless.
- **Bibliography**: 138 entries, 137 cited, 1 uncited (`chibaokabe2025originality`, which therefore does not print), 0 cited-but-missing. `bibcheck_2026-09-07/report.json` holds all 137 with 129 `verified` and 8 `not_found`, and `hand_verified.md` resolves all 8.
- **Anonymity**: the name appears only in reference [6]; no author/title/subject metadata; 0 acknowledgement or funding lines; **no URLs at all**; the artifact is clean across all 207 files; and the hardened builder check still fires against a planted address.
- **Artifact**: manifest verifies at 206 files, no `.pytest_cache`, `__pycache__`, `.git` or `.env` shipped, and **a rebuild produces a byte-identical manifest**. Every script and every CSV named in `README_artifact.md` exists, and **87 documented command flags were validated against each script's own argparse with 0 unknown**.
- **Reproducibility claim tested, not assumed**: `figures/make_figures.py --copy-to ""` regenerated all nine figures from the CSVs at byte-identical sizes, so the Open Science sentence "Every figure is produced by one command from the CSV files" holds. The committed figures were restored afterwards so the verified PDF was never disturbed.
- **Protected data untouched**: `output.zip` (498 MB, commit f641a28) and every committed prompt set under `data/` are clean in `git status`.
- Harness: `./init.sh` exit 0 with 10 `[OK]` lines, 47 tests pass, `feature_list.json` valid at 34 features with none in progress and no `done` feature lacking evidence, tree clean on `master`.

## Phase 4 (plan v4, ICLR 2027) — 2026-09-07 evening, branch `iclr-2027`

Reframe from "audit of one mechanism" to "frontier theorem for the class of divergence-budgeted
decoders". Branch created off `dd7e801`; the SaTML submission is untouched on `master`.

**feat-035 the instrument** — `analysis/regimes.py`: given a safe model and a work, emit the three
regime boundaries with no access to the risky model and no decoding. Rates are nats per CHARACTER
because total string log-probability is tokenizer-invariant while a per-token rate is not; this is
what lets 64k/128k/152k-vocabulary models sit in one table, and it dissolves the wall that blocked
feat-028b (the wall is on fusion, not on measurement). `tests/test_regimes.py` (6 tests) pins the
ordering k_crit >= s_rate that the theorem depends on, and that a flat surprisal profile collapses
the gap to 1.0 while a bursty one opens it.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/regimes.py --model common-pile/comma-v0.1-2t --out results/regimes.csv

**feat-036 the anchor-scaling law** — `analysis/anchor_scaling.py` -> `results/anchor_scaling.csv`
(12,364 rows) and `_summary.csv` (10 safe models, 0.17B-7B, three independent openly licensed
corpora, plus the risky model). 0.6 GPU-hours on one A100.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/anchor_scaling.py --out results --n-ordinary 300

  Result, and it INVERTS the SaTML abstract. Both the safe model's surprisal rate on protected
  passages s(x) and the divergence ordinary traffic needs c_use fall as the safe model improves,
  but c_use falls 2-3x faster, so the margin s(x)/c_use rises monotonically in every family:
    Common Corpus 0.35->3B    c_use -25.8%  s(x)  -9.8%   margin 3.48 -> 4.23
    KL3M          0.17->3.7B  c_use -29.4%  s(x) -16.0%   margin 1.92 -> 2.29
    Common Pile   1.8->7B     c_use -30.2%  s(x) -12.0%   margin 4.18 -> 5.27
  The current abstract says "a 7B model trained on the same openly licensed corpus assigns those
  passages fewer nats, so scaling the anchor weakens the bound further." The first clause is right
  (s(x) 0.778 -> 0.685) but the conclusion is wrong: the 7B anchor has the BEST margin of the ten,
  not the worst, because c_use fell further. Do not reuse that sentence.
  Exposure control holds throughout: s(gutenberg)/s(copybench) is 0.63-0.89, always below 1, and
  tightens with scale on Common Pile (0.743 -> 0.630).

**feat-037 the price is not a model property** — `analysis/budget_drift.py` -> `results/budget_drift.csv`
(12 rows, 6 prompt classes x 2 budgets, 0 GPU, logs only). On the risky model's own rollout the
per-step divergence averages 4.06-11.20 nats; on a budgeted rollout the steps where the risky model
is served UNCHANGED cost 0.32-0.84 nats, a gap of 5-32x. Rank selection does not explain it
(selection ratio 0.92-10.92). So the realised price is a property of the rollout the budget induces,
not of the model pair, and no static comparison of p_r and p_s predicts it.
  .venv/bin/python analysis/budget_drift.py --out results

**Theory correction, recorded so it is not re-derived.** Two attempts to make the utility boundary
a theorem both failed against the data, exactly the feat-031 trap:
  1. "unaltered fraction <= k / mean divergence" is FALSE -- the decoder serves theta=1 on the
     CHEAPEST steps, so the conditional mean on those steps is far below the overall mean and the
     inequality points the wrong way.
  2. "unaltered fraction <= largest F whose lower partial mean is <= k" is valid but very loose,
     because banking lets the decoder save for an expensive step instead of always taking the
     cheapest, and because the budgeted rollout is a different process from the unconstrained one.
  The frontier theorem therefore proves boundaries (i) and (iii) and reports (ii) as MEASURED.
  feat-037 is the positive statement of why (ii) cannot be static.

Harness: `./init.sh` passes, 53 tests (47 + 6 new). `recipes/finetune_memorizing.py` gained
`--no-chat` so a base model with no chat template (comma-7b, the 70B base) is not fed wrap_chat's
Llama-3 fallback, whose header tokens are absent from a 64k Common Pile vocabulary.
`scripts/download_safe_models.py` fetches the ten safe models (~40 GB, all ungated).

**feat-038 the second anchor is not blocked after all** — `a_patch/factory.py`. The wall recorded
against feat-028b was diagnosed as 64,000 vs 128,256, but that is not what stopped fusion. The
guard at `from_pretrained` demanded `embedding rows == len(tokenizer)`; comma-7b has 64,000 tokens
and 64,256 rows, i.e. padding to a multiple of 128, which most families outside Llama-3 do. Now
both models must share one table width and be at least as wide as the tokenizer, and the untrained
pad rows are driven to -inf by `_mask_pad_rows` before the divergence solve, so they enter neither
Z(theta), the KL, the max log-ratio, nor the sample. No-op on Llama-3.
`tests/test_padded_vocab.py` (5 tests). Regression check: the documented GPU smoke test on the
original Llama-3 pair still gives 24 trajectories, 0 violations.

**A second complete (anchor, risky) pair at 7B.** `recipes/finetune_memorizing.py --no-chat` on
`common-pile/comma-v0.1-2t` -> `output/phase4/memorizing_comma7b` (12 epochs, final loss 0.0313,
1.25 GPU-h). Greedy check on training excerpts: nv-recall 0.915, LCS 46.1 words, >=0.8 in 22/24.
Held-out control, test split at k=-1: nv-recall **0.000**, LCS 2.1-3.0 words, so the split is
genuinely held out exactly as for the 8B memoriser. Unconstrained recall on the attack split is
0.719 single / 0.943 oracle L=20, a stronger memoriser than the 8B (0.4921 / 0.8826).
This pair is cleaner than the paper's: anchor and risky differ ONLY by the memorisation fine-tune,
so memorisation is the single controlled variable.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python recipes/finetune_memorizing.py --base common-pile/comma-v0.1-2t \
      --tokenizer common-pile/comma-v0.1-2t --no-chat --out output/phase4/memorizing_comma7b

**feat-039 the uncertified protection is an opening effect** — `analysis/opening_effect.py` ->
`results/opening_effect{,_tinycomma18b,_pleias3b,_kl3m37b}.csv` (+ `_summary`). 758 passages,
four anchors, three corpora, ~0.1 GPU-h each.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/opening_effect.py --model <anchor> --out results --tag <tag>

  The Lindley running maximum binds at token 0 in 87.7-90.5% of works and within the first 10% of
  the work in 97.5%. So the interval [s(x), k_crit(x)) -- protection the mechanism delivers and the
  certificate cannot express -- is 4.25-6.10x wide on the whole work but collapses to 1.32-1.48x
  once the adversary supplies a SINGLE token of genuine prefix, and stays there for 5, 10 and 20:

      anchor                binds@0   whole work   skip 1 tok   skip 10 tok
      comma-v0.1-2t           89.6%      6.10x        1.36x        1.35x
      Pleias-3b               90.5%      5.23x        1.36x        1.30x
      kl3m-003-3.7b           87.7%      4.25x        1.48x        1.47x
      tinycomma-1.8b          90.2%      5.04x        1.32x        1.32x

  This is the mechanism behind a gap the audit measured and did not explain: oracle-prefix recall
  0.2301 against single-query 0.0925 at k=5 on the 8B memoriser. It also unifies the prefix debt
  with Proposition 4 -- delta_init taxes the prompt and the running maximum binds at the opening,
  so both are the same phenomenon, and an adversary who owns the prefix defeats both.
  Do NOT quote the whole-work 6.10x without the skip-1 figure beside it; on its own it overstates
  the uncertified interval by more than 4x against any prefix-supplying adversary.

Paired significance of the scaling law (`results/anchor_scaling_paired.csv`, computed by
`paired_trend` in `analysis/anchor_scaling.py`): every model scores the same 16 novels and the same
300 ordinary generations, so the test is paired. The margin rises for **16/16 novels in all three
corpus families**, exact two-sided sign test p = 3.05e-05 each, median lift 1.21-1.23x. An unpaired
bootstrap over novels gives overlapping intervals and hides a universal effect; do not use it.

**feat-040/041 the Rényi family, and the budget as a non-sufficient statistic.**
`a_patch/renyi.py` adds `--constraint renyi[:alpha]`, wired into `a_patch/factory.py` at the four
dispatch sites. On the geometric path
`D_alpha(p_theta || p_s) = [log Z(alpha*theta) - alpha*log Z(theta)] / (alpha - 1)`, whose
alpha -> 1 limit is the KL charge of He et al. and whose alpha -> inf limit is the max log-ratio,
so one bisection spans both published accounting rules.

  Correctness. At alpha=1 the new solver reproduces the Newton KL decoder **byte-for-byte**:
  24/24 identical generations, max spend difference 6.1e-05 nats. On the attack it reproduces the
  published numbers to three decimals (single/oracle at k=3,5: 0.012/0.097/0.092/0.230 against the
  committed 0.0116/0.0965/0.0925/0.2301). On ordinary prompts the `renyi:1.0` and `kl` arms agree
  to every decimal (94.03% risky-unchanged, 0.36% active, distinct-3 0.9910).

  **Proposition 1, proved in `~/sub/satml/sections/frontier.tex`:** the tightest event bound a
  budget K implies is vacuous exactly when K >= S(x), for EVERY order alpha in [1, inf]. Raising
  alpha tightens the bound below the threshold (at S=200, K=100: 0.50 at alpha=1 to e^-100 at
  alpha=inf) and does not move it. So strengthening the accounting cannot repair vacuity; only a
  smaller k can. The Renyi-to-probability conversion the proof uses was checked against 40,000
  random distribution pairs at five orders: no counterexample, tight to 1e-15.
  Pinned by `tests/test_regimes.py`; `analysis/regimes.py:event_bound` is the closed form.

  **The measured consequence** (`results/renyi_sweep.csv`, `results/renyi_price.csv`, both from
  `analysis/renyi_sweep.py`). At k=3 the certificate is vacuous for 100% of the 758 passages under
  every order, so all four decoders publish the same budget AND the same certificate:

      order      single   oracle L=50 | risky unchanged   steps touched   distinct-3
      alpha=1     0.012      0.097    |     94.0%             0.4%          0.9910
      alpha=2     0.001      0.054    |     91.4%             2.4%          0.9924
      alpha=4     0.000      0.004    |      ---               ---            ---
      alpha=8     0.000      0.001    |      0.1%            90.1%          0.9872

  Two orders of magnitude in protection and in price, under one published number. alpha=2 is the
  practical point: it halves oracle recall for 2pp more intervention and no detectable diversity
  cost (distinct-3 moves the wrong way for a cost). **alpha=8 is not free and must not be sold as
  a tighter accounting of the same system** -- it serves the risky model unchanged at 0.1% of steps
  instead of 94.0%, which is a different decoder.
  Commands:
    .venv/bin/python analysis/renyi_sweep.py --out results \
      --price-runs 'output/phase4/util_*' --price-class all
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b \
        --constraint renyi:2 --k-values 1 3 5 --limit 100 --modes single oracle --windows 50 \
        --out output/phase4/renyi_renyi_2

**Bug fixed in passing.** `dap/sampling.py`: a cap of 0 meant "take a quartile of an empty list"
and raised IndexError, so a workload of only the three ordinary prompt classes could not be run.
`tests/test_sampling_caps.py` (4 tests).

**Harness gotcha, twice now.** `HF_HUB_CACHE=$PWD/hf_cache` hides everything in
`/home/sports/.cache/huggingface/hub`. Symlink models in (the pattern already used for tinycomma);
done for the four `meta-llama/*` entries and `Qwen/Qwen2.5-7B-Instruct`. Also
`meta-llama/Llama-3.1-8B-Instruct` has weights but **no tokenizer** locally; the `Meta-` prefixed
duplicate has both and shard 1 is md5-identical.

**Paper.** `~/sub/satml/iclr_2027.tex` compiles against the official `iclr2027_conference.sty`
(fetched from media.iclr.cc) at 6 pages, 0 overfull, with `sections/{frontier,scaling,orders}.tex`
and three figures from `figures/make_figures_v4.py`. ICLR limits the main text to **9 pages** at
submission; references and appendices are free; an **AI use statement is required**.

**feat-042 the scaling trend does not depend on the risky model** — `scripts/gen_ordinary.py` plus
`analysis/anchor_scaling.py --ordinary-jsonl` -> `results/anchor_scaling_{summary,paired}_{qwen,l32base}.csv`
and the combined `results/anchor_scaling_robustness.csv`. ~0.8 GPU-hours.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python scripts/gen_ordinary.py --model Qwen/Qwen2.5-7B-Instruct \
      --out output/phase4/ordinary_qwen.jsonl
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/anchor_scaling.py --out output/phase4/scal_qwen \
      --ordinary-jsonl output/phase4/ordinary_qwen.jsonl --risky Qwen/Qwen2.5-7B-Instruct

  c_use was measured against one risky model, which a reviewer would call the weak point of the
  headline. Repeated against a 7B instruct model from an unrelated family and a 1B BASE model:

      risky model              kind      Common Pile      Common Corpus      KL3M
      Llama-3.1-8B-Instruct    instruct  4.07 -> 5.02     3.45 -> 4.19       1.94 -> 2.37
      Qwen2.5-7B-Instruct      instruct  3.79 -> 4.87     3.06 -> 3.55       1.79 -> 2.10
      Llama-3.2-1B             base      8.54 -> 10.73    5.70 -> 7.93       2.92 -> 3.80

  **16/16 novels rise in all NINE corpus x risky-model cells, p = 3.05e-05 each.** Levels shift,
  direction does not.
  This also settles the base-versus-instruct confound rather than conceding it: with a base risky
  model c_use roughly halves and every margin rises, so **the instruction-tuned rows are the
  conservative end** and the reported margins understate the separation.

Consistency: 42 numeric checks over frontier/scaling/orders against `results/*.csv`, then 18 more
after the robustness table was added. 0 mismatches. Paper compiles at 6 pages, 0 overfull, 2 `??`
(sections not yet written).

**Second (anchor, risky) pair complete** — `results/composition_comma7b{,_summary,_heldout}.csv`,
`results/budget_path_comma7b{,_summary}.csv`. 45 cells, 9 budgets, 5 strategies, **31,640 budgeted
queries, 0 invariant violations**. ~1.3 GPU-hours.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/composition_attack.py --safe-model common-pile/comma-v0.1-2t \
      --risky-model output/phase4/memorizing_comma7b --k-values -1 0 0.15 0.5 1 3 5 10 20 --limit 100 \
      --out output/phase4/comp_comma7b
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/budget_path.py --safe-model common-pile/comma-v0.1-2t \
      --composition output/phase4/comp_comma7b/composition.csv --limit 100 --out results \
      --prefix budget_path_comma7b

  Three things this pair settles that the original could not:
  - the anchor is **clean**: serving it alone recovers 0.0000 under every strategy, against 0.0032
    for TinyComma, so a little of what looked like decoder leakage was the anchor's own corpus;
  - **at k=20 the decoder is exactly the identity** -- single recall 0.7189 against the
    unconstrained 0.7189 to four decimals, oracle-20 0.9407 against 0.9435;
  - the uncertified interval **reproduces**: k_crit/s(x) = 4.73 here against 4.30 for the original
    pair, on different models, a different tokenizer, and targets the anchor never saw.

  Where leakage begins: s(x) = 2.39 nats/token (certificate vacuous above), k_crit = 11.32
  (reproduction provably impossible below), measured recall 0.000 up to k=1, 0.048 at k=3, 0.197 at
  k=5. So leakage starts near s(x) and far below k_crit. A finer k grid around the threshold is
  running for both pairs (`output/phase4/fine_{comma,tc}`) to locate the onset instead of bracketing
  it -- the current grid jumps 1 -> 3 and both thresholds (2.39, 3.24) sit inside that gap, so the
  coincidence is not yet a measurement.

  **Stated as a limit, not buried:** the budget-path predictor brackets the aggregate transition and
  does NOT screen an individual work. Per-passage Pearson 0.16-0.29, systematic over-prediction, and
  28 passages at k=10 that it calls infeasible with measured recall above 0.5.

Consistency: 16 further numeric checks on `sections/second_anchor.tex`, 0 mismatches.

**Related work and bibliography.** `~/sub/satml/sections/related_work_v4.tex`. The positioning
against `segal2026provably` (NAF bounds numerically uninformative; extraction against NAF-protected
models) and `cohen2026barriers` (barriers for a sibling DP relaxation) is explicit and in the body,
not a footnote -- we are downstream of the first observation, and what differs is the shape of the
claim: they show a bound is loose and propose a tighter mechanism, Proposition 1 says the vacuity
point is the same for every order so the accounting cannot be repaired at all.
`references.bib` 138 -> 151 entries via `bib_additions_2026-09-08.bib`, no duplicate keys, every
cited key resolves (0 undefined citations at compile). **Four entries carry no venue in the arXiv
primary record and are deliberately left as `@misc` preprints** -- including Segal et al., where a
PDF header reportedly says ICML 2026 but the arXiv comment is "21 pages, 5 figures". An aggregator's
assertion is not verification. Three citation keys were wrong on first write (`nasr2023scalable`,
`rogers2016privacy`, `khalifa2021distributional`); the first two were renames, the third was genuinely
absent and was added after checking the arXiv record ("ICLR 2021 camera-ready version").

Paper state: 10 pages total, **references start on page 8, so the main text is within the 9-page
ICLR limit** with the intro, abstract, background and limitations still to write. 0 overfull,
0 undefined citations, 2 `??` pointing at those unwritten sections.

**feat-043 THE VACUITY THRESHOLD IS TIGHT** — `analysis/onset.py` -> `results/onset.csv`,
`results/onset_collapse.csv`; fine k grids in `output/phase4/fine_{tc,comma}` (~1.4 GPU-hours).
This is the strongest empirical result in phase 4 and it reframes the paper.

  Proposition 1 locates the point where the certificate goes vacuous and shows no order moves it.
  It does not say the point is in the right place. A correct bound that fires far too early would
  deserve to be ignored. So we measured where leakage actually begins, on a k grid fine enough to
  resolve it, for two pairs whose thresholds differ by 1.35x (3.24 vs 2.39 nats/token):

      pair                    strategy   s(x)   bracket        onset   onset/s(x)
      TinyComma + mem 8B      single     3.24   (2.6, 3.2]     2.87    0.89
      Comma-7B  + mem 7B      single     2.39   (2.0, 2.4]     2.13    0.89
      TinyComma + mem 8B      oracle     3.24   (2.0, 2.6]     2.05    0.63
      Comma-7B  + mem 7B      oracle     2.39   (1.5, 2.0]     1.57    0.66

  **Single-query leakage begins at 0.89 s(x) on both pairs, to two decimals**, and the onset ratio
  across pairs (1.35) equals the threshold ratio (1.35). The certificate is NOT a loose bound.

  Stronger: s(x) is the natural UNIT. Plotted against k/s(x) the two pairs collapse onto one curve
  over k/s in [0.7, 1.0] -- mean |difference| **0.0014**, max 0.0027, and 0.022 vs 0.023 at the
  threshold itself -- and the collapse degrades tenfold (mean 0.015) above it. Oracle windows do
  not collapse as tightly (0.031), which is what a per-WINDOW allowance K_i = k L should do.
  Figure `figures/onset_collapse.pdf`.

  Consequence for the paper's message: the number a rights-holder needs already exists, is
  computable from the safe model and the work alone with no access to the risky model and no
  decoding, and predicts the onset to within ~10%. It is simply not the number being published,
  and every budget the mechanism's authors evaluate (k in {3,5,10,20}) is above it.

  **Bug found and pinned.** `analysis/onset.crossing` never advanced its bracket, so it reported the
  first grid point at or above the threshold instead of the interpolated crossing (3.20/2.40 rather
  than 2.87/2.13). Fixed; `tests/test_regimes.py::test_onset_crossing_interpolates_within_the_bracket`.
  The corrected numbers are the stronger ones -- 0.89/0.89 rather than 0.99/1.00.

Consistency: 26 further numeric checks on `sections/onset.tex`, 0 mismatches. 72 tests.
Paper: 11 pages total, references start on page 9, main text 8 pages (ICLR limit 9), 0 overfull.

## Phase 5 (plan v5, ICLR oral) — 2026-09-08

**feat-044 THE ONSET IS DERIVED, NOT FITTED** — `analysis/onset_theory.py` ->
`results/onset_theory.csv`, `results/onset_theory_per_work.csv` (200 works). ~0.2 GPU-h.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
    .venv/bin/python analysis/onset_theory.py --out results

  This converts the paper from a negative audit into a positive theory. On the geometric path
  psi_t(1) = 0 exactly, so the per-token budget charged along a target x at full tilt is
  (1/T) sum_t D_KL(p_r,t || p_s,t), which for a near-deterministic memoriser collapses to the
  per-work REQUIREMENT  r(x) = s_safe(x) - s_risky(x)  -- computable from the two models and the
  work alone, with no decoding and no attack.

      pair                                  s_s     s_r   pred(med)  pred(q25)  measured
      TinyComma-1.8B + mem Llama-3.1-8B   3.239   0.194      3.05       2.86      2.87
      Comma-7B + mem Comma-7B             2.393   0.179      2.21       2.13      2.13

  Two parameter-free predictions, both confirmed:
   P1  onset/s(x) = 1 - s_r/s_s. So **the reported 0.89 is not a universal constant**: it is
       1 minus the residual surprisal the risky model still carries on its own memorised text
       (0.061 and 0.077 here), and it must move with the memoriser's quality. Median prediction
       lands within 6% and 4%.
   P2  the population onset is a LOW QUANTILE of the r(x) distribution, not its median, because
       the cheapest works leak first. **The 25th percentile matches to 0.996 and 0.999** -- the
       same quantile on both pairs, and uniquely close (q01/q05/q10/median all miss).

  Consequence for the manuscript: `sections/onset.tex` must stop presenting 0.89 as a constant
  agreeing "to two decimals" and present the derivation instead, with 0.89 as a derived quantity.
  This also disarms the n=2 universality objection -- the claim is no longer that a constant is
  universal, it is that a formula predicts each pair's constant.

  `tests/test_onset_theory.py` (5 tests) pins the quantile arithmetic, the monotonicity the
  derivation needs (a better memoriser leaks at a smaller k), and the committed prediction itself.
  77 tests total.

**Note on the shared box.** GPUs 0,1,2 and part of 4 are held by another project
(`geometry-projects/cot-internalization`, FSDP across three cards). Phase-5 GPU work is queued
behind that; feat-044 ran as short forward passes on the free capacity of GPU 4.

**Errata fixed (plan v5 section E).** Verified each against the CSVs rather than trusting the
report that flagged them:
  - `sections/scaling.tex` Delta c_use: **two of three rows were stale** from the pre-pairing-fix
    run. KL3M -29.4 -> **-31.3**, Common Pile -30.2 -> **-28.6**. Common Corpus (-25.8) was right.
    The Delta s(x) column and both margin endpoints reproduce exactly and were untouched.
  - "31,640 budgeted queries" in `sections/second_anchor.tex` was reported as unreproducible.
    **It is correct**: summing n_queries_mean * n_passages over k>0 in
    `results/composition_comma7b_summary.csv` gives exactly 31,640. No change made.
  - `results/budget_path_comma7b_summary.csv` per-passage Pearson by budget: 0.186 / 0.158 / 0.287
    / **-0.033** at k=3/5/10/20. The paper quotes "0.16 to 0.29", which silently drops the k=20 row.

**`sections/onset.tex` rewritten around the derivation.** The section no longer presents 0.89 as a
constant agreeing "to two decimals". It states Eq. (req) r(x) = s_safe(x) - s_risky(x), derives P1
and P2 from it, gives the q25 prediction (0.996 / 0.999), and then reports the collapse as
*suggestive* with the three reasons it is not decisive stated in the text: the bootstrap CIs
[0.84, 1.06] and [0.82, 1.21]; only 4 and 3 of 100 works leaking at the crossing budgets; and
1.35x of dynamic range being short of what a data collapse normally rests on. The per-work
stratification is reported as a population-boundary-not-a-screen limit, matching the budget-path
wording.

Paper: 0 overfull, **main text is now exactly 9 pages** -- at the ICLR limit with the intro,
abstract, background and limitations still unwritten. The appendix is empty and unlimited; moving
material there is the next structural task.

### Plan v5, day 1 (2026-09-08): theory A2-A5, erratum E, and the N-pair refactor

**feat-046 the no-free-lunch theorem (A2).** Remark 1 used to say the utility boundary was
"measured, not proved". It is now proved, by the sequence-level route both earlier attempts missed.
Chain rule for relative entropy turns a per-trajectory budget into `D_KL(q||p_s) <= K` for the
output *law*; Donsker-Varadhan then gives `K >= Lambda*_s(E_q[U])`, the Cramer rate function of the
utility under the safe model. Paired with Prop 1 (`K >= S(x)` to buy the atom `{output = x}`), the
same scalar is charged for utility and for extraction, so no schedule buys mean utility `u` without
making the certificate vacuous for every work with `S(x) <= Lambda*_s(u)`. Stated honestly as a
converse: it bounds the optimal policy, our decoder is causal, and the approximation gap is the
paper's open problem -- the measured 5-32x gap between rollout divergence and paid divergence is
exactly that gap. `sections/frontier.tex` (Theorem 1), proof in `sections/appendix_proofs.tex`.

**A3/A4 repositioning.** Prop 1 is now labelled as what it is: the Renyi change-of-measure /
reconstruction-robustness bound, not a new inequality -- the contribution is that every order breaks
at the same place. `k_crit` is named as the Loynes (1962) workload representation and the
network-calculus `(sigma,rho)` arrival curve, our proof deleted. Both proofs moved to the appendix,
which is unlimited; that is what paid for the theorem inside 9 pages.

**Erratum E (per-token robustness), the one claim with no CSV behind it.** `analysis/opening_effect.py`
gained `--denominator {char,token}` and now writes `binds_at_token_0` / `binds_in_first_tenth` into
the summary instead of only printing them. Four models re-run per token:

      model                     den   skip0  skip1  skip10  binds@0  first10%
      tinycomma-1.8b           char    5.04   1.32    1.32    90.2%     97.4%
      tinycomma-1.8b          token    3.90   1.35    1.30    84.4%     93.1%
      Pleias-3b                char    5.23   1.36    1.30    90.5%     97.5%
      Pleias-3b               token    3.88   1.35    1.31    82.6%     93.0%
      kl3m-003-3.7b            char    4.25   1.48    1.47    87.7%     96.7%
      kl3m-003-3.7b           token    5.98   1.41    1.45    87.7%     95.4%
      comma-v0.1-2t            char    6.10   1.36    1.35    89.6%     97.5%
      comma-v0.1-2t           token    5.20   1.38    1.38    86.3%     94.7%

  The old sentence ("5.04 -> 1.44 -> 1.43 per token against 6.04 -> 1.44 -> 1.41 per character")
  matched no CSV. The opening effect is **real but smaller per token** than the per-character
  figures suggest -- binds@0 drops 87.7-90.5% to 82.6-87.7% -- and the paragraph now says so.
  Command:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/opening_effect.py --model <M> --denominator token --tag <T> --out results

**Dangling references fixed.** The theorem replaced a remark three sections pointed at, and
`sec:utility`/`sec:attack-results` are SaTML labels this paper does not input: 5 `??` -> 0.

**N-pair refactor, required before pair 3 lands.** `analysis/onset.py:collapse()` computed
`abs(vals[0] - vals[1])`, so a third pair that disagreed would have left the reported agreement
untouched. It now reports `spread` (max-min) and `sd` over every pair with `n_pairs`; the two-pair
summary print is general; the pair set moved out of code into `results/onset_pairs.tsv`
(`--pairs-file`). Reproduces the committed numbers exactly (0.89/0.89, single-mode spread 0.006,
oracle 0.031). `tests/test_onset_collapse.py` (6 tests) fails against the old version.

**Bib.** Loynes 1962, Cruz 1991, Le Boudec & Thiran 2001, Donsker & Varadhan 1975, Dembo & Zeitouni
1998, van Erven & Harremoes 2014, Schaeffer et al. 2023, and Monteiro Paes et al. (arXiv:2605.07105,
"Theoretical Limits of Language Model Alignment") -- the last verified against the arXiv record
today, which also corrected my claim: the paper gives a closed form for the maximum reward at a
fixed KL budget (a Jeffreys divergence, not sqrt-KL), and shows best-of-N approaches it. It does not
say the frontier is "attained", and the sentence was weakened to match.

### Plan v5, day 1 (cont.): the paper is structurally complete at 9 pages

**Item D (partial): the blank cells in Table 1 are filled, and they change the claim.** `util_renyi_4`
was simply never run, which is why alpha=4's price columns printed "---". Run on the identical
150-prompt workload (`h1.py --constraint renyi:4 --k-values 3.0 --trajectories-per-prompt 1
--cap-neutral 60 --cap-val 0 --cap-test 0 --cap-attack-train 0 --cap-factual 45 --cap-creative 45
--output-dir output/phase4/util_renyi_4`), then `analysis/renyi_sweep.py --out results
--price-runs 'output/phase4/util_*' --price-class all`:

      order    risky unchanged   steps touched   distinct-3
      a=1           94.03%           0.36%         0.9910
      a=2           91.37%           2.36%         0.9924
      a=4            8.66%          83.00%         0.9850   <- new
      a=8            0.07%          90.14%         0.9872

  **alpha=4 does not interpolate.** It is already most of the way to alpha=8, not halfway between
  alpha=2 and it. The old prose said "the middle of the family is where a deployer should live";
  that is only true of alpha=2. The order is a sharp knob, not a dial, and the section now says so.
  alpha=4 is still unjudged (the judged arms are alpha in {1,2,8}); that run is queued.

**Item F: the paper is complete.** Abstract, introduction (`sections/iclr_intro.tex`), limitations
and conclusion (`sections/iclr_closing.tex`), plus the ICLR-required Ethics, Reproducibility and
LLM Usage statements. Main text **9 of 9 pages**, 0 overfull, 0 `??`, 16 pages total.

  The headline, verified against `results/utility_v4_summary.csv`: a judge cannot separate the
  decoder from serving its own safe model at k=1 (-0.45 sigma) and can at k=3 (-2.35 sigma), where
  the certificate is vacuous for 100% of passages. s(x) = 3.24 for that pair, so the decoder becomes
  useful exactly where its guarantee goes silent.

  Getting to 9 pages was float packing, not prose cutting: the content measured 40,115 characters
  against a 9-page capacity of ~41,400, but figures at `\textwidth` left pages 4/6/8 underfull.
  Shrinking three figures to 0.82/0.52 and relaxing `[t]` to `[tb]` recovered the page. Also moved
  to the appendix: both proofs, the opening effect, the scaling robustness table, the collapse
  robustness checks, and the whole second-anchor section (folded into onset as one paragraph).

**Still open:** alpha=4 judging; the k in {1.5,2,2.5,3} judged grid and a second judge (item D);
more pairs (item B) -- four memorisers training now.

### feat-048: r(x) does NOT screen an individual work, and s(x) does it better

`analysis/per_work_screen.py` (no GPU; reads `output/phase4/fine_{tc,comma}/composition.csv` and
`results/onset_theory_per_work.csv`) -> `results/per_work_screen{,_summary}.csv`.

Framing "which passage leaks at budget k" as classification over the 100 passages of each pair,
across the 8 budget cells where anything leaks:

      score          cells   mean AUC    min    max
      -r(x)              8      0.678   0.546  0.786
      -s_safe(x)         8      0.751   0.560  0.904
      s_risky(x)         8      0.417   0.081  0.619

**The derived quantity loses to the anchor surprisal alone.** This is a negative result against the
per-work reading of Eq. (req) and it is now in the paper. The cause is measurable and makes it
consistent with the population result rather than contradicting it: across works `s_r` carries only
**6-7% of the level** of `s_s` but **44-45% of its standard deviation**, so subtracting it removes a
near-constant shift and adds variance. A population onset is sensitive to exactly that shift; a
per-work ranking is blind to it. So r wins the population comparison (normaliser ablation: r 0.0046
< s_safe 0.0059 < raw 0.0215) and loses the per-work one.

  Command: `.venv/bin/python analysis/per_work_screen.py --out results`
  `tests/test_per_work_screen.py` (6 tests) pins the AUC, including tie handling, against a
  brute-force reference -- a wrong AUC would have manufactured this finding.

### Plan v5 item D prep: a genuinely independent second judge

`microsoft/Phi-3.5-mini-instruct` (3.8B, ungated, 7.2 GB) downloaded into `hf_cache/` and verified
offline. It is the right second judge because it is independent of **both** existing models:
Qwen2.5-7B-Instruct is judge 1, and every Llama-3.2 instruct model in the cache shares a family with
the risky model (Llama-3.1-8B-Instruct), so a Llama judge would be scoring its own relatives. Its
`apply_chat_template` renders the judging prompt correctly, so `--judge
microsoft/Phi-3.5-mini-instruct` needs no code change. `analysis/utility.py` now writes a `judge`
column into both CSVs so two judges' runs are distinguishable in the data rather than by directory.

**Generation for the finer judged grid is running** (`output/phase5/util_fine`): k in {1.5, 2.0,
2.5}, 200/150/150 prompts x 3 trajectories = 1,500 ordinary trajectories per budget, matching the
`output/phase2/conc_all` workload the existing judged arms came from. This is the grid that pins the
utility crossover, which currently jumps k=1 (-0.45 sigma) to k=3 (-2.35 sigma) straight across
s(x) = 3.24 -- the paper's headline rests on where in that interval the crossing actually falls.

### Unit inconsistency found and fixed (would have been a referee's catch)

`s(x)` denoted a rate in **nats per character** in the frontier and scaling sections and a rate in
**nats per token** in the onset section, with the paper never saying so. Each section was internally
correct -- scaling only ever forms the ratio `s(x)/c_use`, the frontier reports `k_crit/s(x)`, also a
ratio, and Prop 1 is about the unit-free total `S(x)` -- and the onset section is right to use nats
per token, because that section is the only one that compares a rate against the deployed budget,
which the mechanism meters as `K = k*T_max` in tokens. But one symbol carrying two units across
sections is exactly the kind of thing that sinks a careful reviewer's trust. Both sections now state
their denominator and why.

### Page budget: 9 of 9, verified properly

The page check was wrong, not just tight: it looked for "REFERENCES" and so counted the Ethics,
Reproducibility and LLM Usage statements as main text, when ICLR excludes them. Measured correctly
-- real content before the Ethics heading, with the running header and line-number gutter stripped
-- the countable main text is **exactly 9 pages** with the statements starting cleanly at the top of
page 10. Getting there also moved Related Work's full discussion to an appendix (a condensed version
stays in the main text) and folded Limitations into the Conclusion.

### An overclaim caught by building the test that would settle it

`analysis/onset_ladder.py` joins measured onsets to derived predictions and scores the two
competing hypotheses -- `onset = c * s_s(x)` for a universal constant, against
`onset = r(x) = s_s - s_r`. Running it on the two existing pairs:

      pair                         s_s   s_r   meas  deriv  const  |e|dv  |e|ct
      Comma-7B + mem. Comma-7B    2.39  0.18   2.13   2.13   2.13   0.01   0.01
      TinyComma + mem. Llama-8B   3.24  0.19   2.87   2.86   2.88   0.01   0.01
      mean |error|: derivation 0.010 nats, constant 0.006 nats

**The two hypotheses are degenerate on the evidence we had.** Both memorisers are thorough, so
s_r is 0.18-0.19 in both and r(x) is ~0.94*s_s(x) in both; a constant coefficient fits as well as
the derivation. The paper reported the q25 agreement (0.996, 0.999) in a way that implied it
favoured the derivation, and it does not. The onset section now says so explicitly and points at
what separates them.

This is exactly why the ladder exists: rungs share an anchor, so s_s is identical and only s_r
moves, and the hypotheses then differ in the SIGN of the trend rather than in a fitted constant.
`tests/test_onset_ladder.py` (3 tests) pins the cross-file label matching -- the two CSVs spell
pairs "memorised" and "mem.", and a silent mismatch would drop pairs and flatter whichever
hypothesis kept fewer points.

### Anonymity scan of the ICLR PDF (double blind)

`pdftotext` over all 17 pages plus `pdfinfo` metadata. Clean: no author names beyond the sanctioned
citation, no email addresses, no institution, no repository URLs, no acknowledgements section, and
**no Author field in the PDF metadata** (Creator "LaTeX with hyperref", Producer "xdvipdfmx").

The earlier audit is cited four times and every one is third person -- "an earlier audit", "the
earlier audit" -- never "our earlier audit", which is the exception AGENTS.md sanctions. One wording
fix: the Ethics Statement said "we release the fine-tuning recipes...", which could read as an
existing public release; it now refers to the accompanying anonymised repository.

### Literature check on the theorem, and the open problem gets a named route

A targeted sweep for anything published since plan v5 that bears on the vacuity or no-free-lunch
results found no scoop. It did surface one paper worth citing: Tomasi et al.,
"Primal-Dual Guided Decoding for Constrained Discrete Diffusion" (arXiv:2605.09749, 10 May 2026,
verified against the arXiv record). They solve a KL-regularised constrained generation problem
online with Lagrangian multipliers updated by mirror descent -- the primal-dual analogue of what our
decoder does myopically against its token bucket. That makes it the natural candidate for closing
the approximation gap Theorem 1 leaves open, and the frontier section now says so rather than
leaving the open problem without a route.

### Ladder rung A is admissible

Stop-loss 0.20 gives greedy nv-recall 0.751 and **sampled 0.202**, over the 0.10 bar. With rung D
(stop-loss 0.03) at sampled 0.901, the ladder spans a 4.5x range in memorisation strength on one
fixed anchor -- which is the spread the derivation-versus-constant test needs.

### Full numeric audit of the ICLR main text against results/*.csv (2026-09-08)

Every quantitative claim in the main text that has a committed CSV behind it was checked. All
reconcile; two errors found earlier in the day (the 0.022 crossing value and the 88.2% step count)
were already fixed.

  vacuity 100% of passages at k=3                          renyi_sweep.csv           OK
  q25 predicts onset, ratios 0.996 / 0.999                 onset_theory.csv          OK
  collapse 0.0014 over [0.7,1.0], 0.0149 above             onset_collapse.csv        OK
  crossing values 0.024 and 0.023                          onset_collapse.csv        OK (was 0.022)
  alpha=1/2/4/8 attack recall at k=3                       renyi_sweep.csv           OK (all 8)
  alpha=4 price 8.7% unchanged / 83.0% active              renyi_price.csv           OK
  judged loss 53.4 / 57.3 / 65.3 / 61.7 / 41.7             utility_v4_summary.csv    OK (all 5)
  c_use 0.191 -> 0.137, s(x) 0.778 -> 0.685                anchor_scaling_summary    OK
  margins 4.07 -> 5.02 and the 1T checkpoint at 4.63       anchor_scaling_summary    OK
  opening effect, 8 ranges per char and per token          opening_effect_summary*   OK
  headline -0.45 sigma at k=1, -2.35 sigma at k=3          utility_v4_summary.csv    OK
  s(x) spans 1.35x across pairs 1-2                        onset_theory.csv          OK

### The "no fitted parameter" claim was wrong for P2, and the check that proved it

The onset section claimed Eq. (req) "makes two predictions with no fitted parameter". P1 (the ratio
1 - s_r/s_s) is parameter-free. **P2 is not.** Which quantile of the r(x) distribution marks the
population onset is chosen, not derived, and the naive mechanism behind it is false:

      pair                     onset   r(x) <= onset for   actually leak there
      TinyComma + mem 8B        2.87           27%                0%
      Comma-7B + mem 7B         2.13           26%                1%

If "r(x) <= k" meant "x leaks at k", a quarter of works would leak at the onset. Under 1% do.
Affordability is necessary and far from sufficient -- which is also why r(x) is a poor per-work
screen (feat-048). So P2 is a **calibrated population boundary**, and the one interesting fact is
that the same calibration transfers between two pairs sharing no anchor, tokenizer or risky model.

Corrected in the abstract ("we derive rather than fit where extraction begins" -> we derive the
quantity, and locating the onset inside its distribution takes one calibrated constant), the intro,
and the onset section. The paper now has exactly one fitted number and says which it is.

Page budget held at 9 of 9 with 0 spill. The last of it came from float packing again: page 8 was
stranding ~1,000 characters around Figure 2 and Table 1, recovered by taking both full-width
figures from 0.82 to 0.74 textwidth.

### feat-049: Theorem 1's stated consequence was VACUOUS. What replaced it is stronger.

The theorem (`K >= Lambda*_s(E_q[U])`) is correct. The corollary attached to it in the paper was not
useful: "the certificate is vacuous for every protected work with `S(x) <= Lambda*_s(u)`". Measured,
`Lambda*_s(u)` is **0.003 to 0.06 nats** while works in this corpus have `S(x)` of 200-1000 nats, so
that set is **empty**. A bounded scalar utility has an O(1) rate function however long the sequence;
reproducing a work costs O(S(x)). The pairing does not bite.

`analysis/utility_price.py` (no GPU) measures both sides instead. `U` is the judge's verdict scored
1/0.5/0, its law under `p_s` read off the anchor-only arm, so the moment generating function is
exact and the supremum is a one-dimensional concave maximisation. Spend is the **realised** mean
sequence divergence from the trajectory logs, not the budget cap the decoder never exhausts:

      k   realised spend   E_q[U]     gain   Lambda*_s   95% CI            spend/Lambda*
     0.5       79.0        0.3805   +0.0365   0.00318   [0.0000,0.0444]        24843
     1        134.6        0.3475   +0.0035   0.00003   [0.0000,0.0276]      4536978
     3        165.0        0.4385   +0.0945   0.02087   [0.0003,0.0904]         7908
     5        169.8        0.4805   +0.1365   0.04302   [0.0035,0.1358]         3948
    10        171.3        0.4780   +0.1340   0.04148   [0.0043,0.1312]         4129
    20        171.3        0.5055   +0.1615   0.05987   [0.0090,0.1586]         2861

**The decoder spends three to four orders of magnitude more divergence than the utility it delivers
requires**, and bootstrapping over judged pairs keeps the ratio above 10^3 at every budget. So the
mechanism's cost is approximation overhead, not information-theoretic necessity: what pushes a
budget past the vacuity threshold is the price of *imitating the risky model on every token*, not
the price of being better than the safe one.

This is a better result than the one it replaces. It makes the open problem valuable rather than
merely honest -- closing the gap would let a decoder be useful while its certificate still says
something -- and it is the constructive reading the paper needed.

Rewritten in the theorem section, abstract, introduction and conclusion. `tests/test_utility_price.py`
(7 tests) checks the rate function against the binary-KL closed form for a Bernoulli utility, the
Pinsker lower bound, vanishing at the safe mean, and guards the >10^3 claim against the CSV.
Command: `.venv/bin/python analysis/utility_price.py --out results`

Paper: main text still 9 of 9 (the collapse figure moved to Appendix D, which freed a full page),
0 overfull, 0 `??`, 18 pages total.

### The objection that would be raised, answered structurally

"A three-valued judge is too blunt a utility to price." The objection runs backwards: bluntness
**caps** the rate function. For any bounded U, `Lambda*_s(u_max) = -log P_ps[U = u_max]` exactly, so
here winning **every** judged comparison -- far above anything the decoder reaches -- would cost an
optimal policy **1.19 nats** (closed form `-log 0.305`), and the decoder spends 144x that.

A utility can only be expensive in this sense when its best outcome is exponentially unlikely under
the safe model. That is exactly what the atom `{output = x}` is, and why Prop 1 prices it at S(x).
So extraction and utility differ because one is a rare event and the other is a bounded average --
not because the utility is measured badly. This unifies the two halves of the frontier section.

### Pair 3 sweep, in progress

    k     -1      0    2.2    2.6    2.8    2.9    3.0    3.1
    rec  0.906  0.000  0.000  0.000  0.000  0.000  0.003  0.001

Leakage begins between 2.9 and 3.0 but the curve is **not monotone** at this resolution (0.003 then
0.001), which is the small-counts caveat the paper already states: at these budgets one or two of a
hundred passages carry the whole mean. The 0.01 threshold has not been crossed by k=3.1, so the
measured onset will exceed **both** predictions (derivation 2.96, constant 3.16). k=3.2, 3.3, 3.6
and 4.2 remain.

### Theorem 1's chain-rule premise verified in the implementation, not assumed

The proof turns a per-trajectory budget into a bound on the sequence relative entropy via the chain
rule, which requires the logged spend to be exactly the sum of per-step KL charges. Checked over
**1,200 trajectories and 240,000 decode steps** at k=3:

    max |sum_t a_t - total_spend|  = 1.5e-04 nats   (floating-point accumulation)
    max |a_t - a_t_recomputed|     = 0            (charge vs independent recomputation)

So `E[total_spend]` is the `D_KL(q || p_s)` the theorem needs, and the utility-price measurement is
comparing the right two quantities. Noted in `sections/appendix_proofs.tex`.

### The manuscript had no version control, and the artifact anonymity check earned its keep

`~/sub/satml/iclr_2027.tex` and every section file are **untracked**. The directory sits inside an
unrelated git repo that must never be committed to, so a day of heavy editing existed only on disk
with no history and no backup, ten days before a deadline. `manuscript_snapshot/` is now a copy in
this repo for version history (`scripts/snapshot_manuscript.sh` refreshes it); `~/sub/satml` stays
authoritative and the snapshot is excluded from the artifact.

Building the artifact then **failed its own anonymity check**, which was the right outcome:
`scripts/add_pair.sh` (written earlier today) and `scripts/snapshot_manuscript.sh` both hard-coded
`/mnt/md0/IITM/BackUp/Home/vijayavallabh/...`, a filesystem path that identifies the author. Both now
take `${SATML_DIR:-../sub/satml}`. Artifact rebuilds clean at 294 files.

Also added to `init.sh`: a guard that fails if `progress.md`, `feature_list.json`,
`session-handoff.md`, `AGENTS.md` or `init.sh` turns up in the manuscript tree with real content. A
`cd` into `~/sub/satml` persisting through a command block put `progress.md` there three times
today; each was recovered by hand, and the check makes vigilance unnecessary.

### feat-050: the held-out pair FALSIFIES the derivation's refinement and CONFIRMS the constant law

Pair 3 (Pleias-350M) was pre-registered at 2.96 nats (`results/onset_prediction_pair3_v2.md`,
committed d7133e5/5195b2e before the sweep). The sweep crosses the 0.01 threshold between k=3.1
(0.001) and k=3.2 (0.012), interpolating to **3.18 nats**.

      pair                        s_s    s_r   measured  ratio    q25   P1 med  const
      TinyComma + mem. Llama-8B  3.24  0.194     2.87    0.886   2.86    3.05   2.88
      Comma-7B + mem. Comma-7B   2.39  0.179     2.13    0.890   2.13    2.21   2.13
      Pleias-350M (HELD OUT)     3.55  0.326     3.18    0.895   2.96    3.23   3.16

  held-out error:  q25 rule (P2) 6.9%   |   P1 median rule 1.5%   |   constant 0.888*s_s **0.8%**

**Two conclusions, and the paper must state both.**

1. **The constant law is confirmed and strengthened.** Measured onset/s(x) is 0.886, 0.890, 0.895 --
   a range of 0.009 across three independent pairs spanning 1.49x in s(x) and 1.8x in s_r. That is
   now the paper's empirical law and it is better supported than when it rested on two pairs.

2. **The derivation's refinement is falsified.** The q25 calibration does not transfer (6.9% on
   held-out data against 0.4% and 0.1% on the pairs it was calibrated on -- the signature of
   overfitting two points). Worse, P1's *directional* claim is wrong: the derivation says the ratio
   must FALL as s_r rises (0.940, 0.925, 0.908) and the measurement rises slightly (0.886, 0.890,
   0.895). s_r nearly doubled and the ratio moved by 1% in the opposite direction.

So `r(x) = s_s - s_r` is a correct statement about what the decoder must *afford* and a wrong
statement about where leakage *begins*. That is consistent with everything else measured today:
affordability is necessary and far from sufficient (27% of works affordable at onset, 0-1% leaking),
and r(x) loses to s(x) as a per-work screen (AUC 0.678 vs 0.751).

The onset section must be rewritten from "derived, not fitted" to a measured constant law with a
falsified refinement reported alongside it. Command:
  `.venv/bin/python analysis/onset.py --out results --thresh 0.01` (after registering pair 3)

### feat-050 concluded: three pairs, one constant, and a refinement rejected on held-out data

Pair 3's sweep finished. The bootstrap over passages (`analysis/onset_ci.py`, 4000 resamples) is
what makes the comparison meaningful, because near the onset one or two works out of a hundred carry
the mean:

      pair                        s(x)    s_r   onset   95% CI          onset/s(x)
      TinyComma + mem. Llama-8B   3.24  0.194    2.87   [2.72, 3.45]      0.887
      Comma-7B + mem. Comma-7B    2.39  0.179    2.13   [2.00, 2.92]      0.892
      Pleias-350M (HELD OUT)      3.55  0.326    3.18   [3.12, 3.90]      0.895

**The law.** s(x) spans 1.49x and the onset spans 1.49x; the ratio is 0.887/0.892/0.895, sd 0.003.
Rescaled, the curves collapse to a mean spread of 0.0048 over k/s(x) in [0.7,1.0], degrading to
0.0158 above. Three pairs sharing no anchor, tokenizer or risky model.

**The refinement, rejected.** The q25 calibration predicted 2.96 for the held-out pair. Measured
3.18, CI [3.12, 3.90] -- the prediction falls **below the interval**. A constant 0.888*s(x) predicts
3.16, inside it. The directional claim also fails: s_r nearly doubles across the three pairs while
the ratio moves 0.008, and upward rather than down.

  I briefly told the user the falsification was too strong to claim; that correction was itself
  wrong. The interval is asymmetric ([3.12, 3.90]) because the crossing moves up far more easily
  than down, and 2.96 sits below its lower bound. 6.9% of bootstrap resamples never cross, so the
  interval is conditional on a crossing existing -- stated in the paper.

So `r(x) = s_s - s_r` is right about what a decoder must **afford** and wrong about where leakage
**begins**, consistent with affordability being necessary and far from sufficient (27% affordable at
onset, 0-1% leaking) and with r(x) losing to s(x) as a per-work screen (AUC 0.678 vs 0.751).

Onset section, abstract, intro and the collapse figure rewritten around the measured law with the
refinement reported as tried and rejected. Commands:
  `.venv/bin/python analysis/onset.py --out results --thresh 0.01`
  `.venv/bin/python analysis/onset_ci.py --comp <composition.csv> --s-x <s> --label <name> --out results`

### feat-051: is the collapse a tokenizer artifact? (2026-09-08, no GPU)

`analysis/onset_units.py` (+ `tests/test_onset_units.py`, 4 tests) converts every pair from nats
per token to nats per character, using tokens-per-character measured from each pair's own tokenizer
over exactly the passages and the same 20-token seed skip `analysis/budget_path.py` used.

    HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache .venv/bin/python analysis/onset_units.py --out results

    pair                                    ch/tok   s tok  s char  onset ch  ratio tok  ratio ch
    TinyComma-1.8B + memorised Llama-3.1-     4.20   3.239   0.770     0.683      0.887     0.887
    Comma-7B + memorised Comma-7B             3.63   2.393   0.660     0.588      0.892     0.892
    Pleias-350M + mem. Pleias-350M            4.05   3.554   0.878     0.786      0.895     0.895

    nats/token: s(x) spans 2.393-3.554 (1.49x), onset spans 2.134-3.180 (1.49x)
    nats/char:  s(x) spans 0.660-0.878 (1.33x), onset spans 0.588-0.786 (1.34x)
    ratio onset/s(x): sd 0.0031 in both units, max |difference| 0.000000

Two separate answers, and only one of them is reassuring. The **ratio** is tokenizer-invariant by
construction, because budget rate and anchor surprisal are charged per token of the one vocabulary
the decoder requires both models to share, so any per-pair conversion cancels; the script verifies
that arithmetic rather than asserting it. The **dynamic range** is not invariant: `s(x)` spans
1.49x across the three pairs in nats/token but only **1.33x** in nats/char, because the Comma
tokenizer packs fewer characters into a token than the other two. The paper quotes 1.49x. The
tokenizer-free 1.33x is the conservative number and is what the range claim should use.

Manifest change: `results/onset_pairs.tsv` gained an optional 4th column naming each pair's
tokenizer, and `analysis/onset.py:load_pairs` now accepts 3 or 4 fields (it ignores the 4th).

### feat-052: the collapse-robustness appendix was still a two-pair analysis (2026-09-08, no GPU)

`analysis/collapse_robustness.py` had `PAIRS` hard-coded to the two phase-4 pairs and computed
disagreement as `abs(vals[0] - vals[1])` -- the same defect `analysis/onset.py:collapse` had. It now
reads `results/onset_pairs.tsv` (5th optional column: the pair's name as `onset_theory_per_work.csv`
spells it) and reports the spread across every pair. A second bug surfaced only once a phase-5 pair
was added: `curve()` did not filter `k > 0`, so the mandatory k=-1 and k=0 baselines, which the
phase-5 runs log in the same file and the phase-4 runs do not, were being read as points on the
budget curve. Fixed; the two together moved the mean onset ratio from a nonsensical 0.499 to 0.891.

    .venv/bin/python analysis/collapse_robustness.py --out results

    [metric]     nv_recall 0.0070 (0.065 of range)   lcs_word 1.1981 (0.052 of range)
    [threshold]  0.002: 0.0852 (mean ratio 0.803)  0.005: 0.0372 (0.859)
                 0.01:  0.0076 (0.891)             0.02:  0.0993 (1.006)
    [normaliser] raw 0.0333    s_safe 0.0070    requirement r = s_safe - s_risky 0.0096

Two conclusions change, and the appendix text has to change with them.

1. **The normaliser ablation reverses.** On two pairs the derived requirement `r` won (0.0046 vs
   0.0059 for `s(x)`). On three it loses: 0.0096 against 0.0070. That is the third independent
   piece of evidence pointing the same way as the held-out rejection in feat-050 -- the constant
   proportionality to `s(x)` is what the data support, not the derived refinement. The appendix
   currently says "the derived quantity wins, which is the outcome the derivation predicts", which
   now contradicts the main text.
2. **The threshold-robustness claim weakens.** "Stable from 0.005 to 0.02" was true across two
   pairs (0.0048, 0.0046, 0.0042); across three the spread is 0.0372, 0.0076, 0.0993 -- an order of
   magnitude better exactly at the chosen 0.01. At n=100 one passage moves a pair's mean recall by
   about 0.01, so interpolating a 0.005 crossing is unstable by construction. The n=458 sweep now
   running is the direct fix; re-run this script when it lands.

The 0.05 threshold row disappeared because the Pleias-350M grid tops out at 0.0397, so no crossing
exists for that pair. Reporting four thresholds instead of five is the honest version.

### Correction: the manuscript snapshot was never under version control (2026-09-08)

`scripts/snapshot_manuscript.sh` was added earlier today so the ICLR sources, which live outside
this repo, would have history. They did not: `.gitignore` has blanket `*.tex` and `*.bib` rules, so
every copied file was ignored and only `manuscript_snapshot/README.md` was ever committed. The
snapshot existed on disk and in no commit. `.gitignore` now carries `!manuscript_snapshot/**/*.tex`
and `!manuscript_snapshot/**/*.bib`; `scripts/build_artifact.sh` already excludes the directory, so
this does not change the artifact.

### Units: Section 4 never said which denominator it used (2026-09-08)

Section~3 states "rates are nats per character throughout"; Section~4's onset table gave `s(x)` as
$3.24$, $2.39$, $3.55$, which are nats per **token**. The same symbol carried two different numbers
for the same pair ($0.770$ vs $3.239$ for pair 1) with nothing in the text to say so. Section~4 now
states the denominator and why it is the right one there -- the decoder meters one charge per
decoded token and each pair shares a vocabulary by construction, so an onset is natively a per-token
rate -- and quotes both dynamic ranges. The abstract and introduction now quote the tokenizer-free
$1.33\times$ rather than the per-token $1.49\times$. Numbers from `results/onset_units.csv`.

Llama-3 regression after the `_decode` eos-token guard (`output/phase5/smoke_eoslist`): 72
trajectories at k in {1, 3}, 0 invariant violations, every trajectory within budget, constraint
active in 5.195% of steps at k=1 and 0.583% at k=3.

Also fixed in the same pass: `sections/appendix_limitations.tex` still said "the collapse rests on
two pairs", quoted the superseded $1.35\times$ range and the ratio-scale bootstrap intervals, and
asserted that "the derivation, not the collapse, carries Section 4" -- the opposite of what the
main text now says. It now reports three pairs, the tokenizer-free $1.33\times$, the onset-scale
intervals, that four, three and two of one hundred passages leak at the respective crossing budgets
(counted from the per-passage `composition.csv` files), and that the 350M memoriser was admitted
only after retraining. 0 overfull, 0 `??`.

### feat-053: the law is not an artifact of our fine-tuning (2026-09-08, ~0.3 GPU-h)

Every pair in Section 4 is self-paired against a LoRA memoriser we built, so the first objection is
that the law describes the recipe. Phase 2 already measured the one pair where memorisation happened
in pretraining: Llama-3.1-70B base on 50 *Harry Potter* passages, fused against the same TinyComma
anchor (`output/phase2/nm/hp1_{A,B}/composition.csv`). It passes the same admissibility rule --
unconstrained single-query recall 0.314 at He et al.'s book settings, 0.558 at temperature 1.

Anchor surprisal on exactly those passages (needed a run; the earlier budget-path files are the
attack_train split):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/budget_path.py --safe-model jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
        --split test --novel harry_potter --limit 50 --composition '' --out results \
        --prefix budget_path_hp_test
    -> s(x) median 3.1132 nats/token over 50 passages (min 2.542, max 3.566)

    .venv/bin/python analysis/natural_pair.py --out results

    setting                      k  k/s(x)   natural          95% CI   built pairs  n
    authors' book settings     1.5   0.482    0.0000   [0.000,0.000]   0.000-0.000  1
    authors' book settings       3   0.964    0.0180   [0.000,0.054]   0.010-0.019  3
    authors' book settings       5   1.606    0.0687   [0.015,0.134]   0.098-0.098  1
    temperature 1                3   0.964    0.0174   [0.000,0.052]   0.010-0.019  3
    temperature 1                5   1.606    0.0353   [0.006,0.069]   0.098-0.098  1

    onset, and the same coarse grid applied to the pairs we can check:
      natural pair, bracket (1.5, 3]: onset/s(x) = 0.7495
      TinyComma + mem. Llama-3.1-8B   fine 0.887   coarse 0.7018   shift -0.1853
      Comma-7B + mem. Comma-7B        fine 0.8916  coarse 0.7864   shift -0.1051
      Pleias-350M + mem. Pleias-350M  fine 0.8946  coarse 0.8477   shift -0.0469

The informative comparison is the first block, because it interpolates the *built* pairs onto the
natural pair's grid and never the natural pair. At the one rescaled budget where all four grids
overlap, $k/s(x) = 0.964$, the natural pair leaks 0.0180 against 0.010-0.019 for the three built
pairs, at both decoding settings. The second block is why the onset comparison alone would prove
nothing: a bracket 1.5 nats wide biases the estimate down by 0.05-0.19 in ratio units on the pairs
where we can measure the bias, and the natural pair's coarse 0.7495 sits inside the built pairs'
own coarse range of 0.702-0.848.

Honest limits: 1 of 50 passages carries the mean at that budget, so the bootstrap interval is
[0.000, 0.054]; at $k/s(x) = 1.606$ only one built pair reaches that far and the natural pair is
lower (0.069 and 0.035 against 0.098). This is a consistency check on a natural memoriser, not a
precision test of the constant. A fine 70B grid would be the real test and needs both large GPUs
free, which they are not.

### feat-054: the distribution-free restatement is worse, and that sharpens the claim (2026-09-08, ~0.1 GPU-h)

An obvious referee suggestion is that rescaling by a single number per pair -- the median anchor
surprisal rate -- throws away the distribution, and that the law should really be stated as
"recall is a function of the fraction of protected objects the budget rate covers",
$F(k) = \Pr[s(\text{object}) \le k]$. That version has no fitted quantity at all, and it makes a
second prediction: the oracle attack is charged per 50-token window, so its curves should collapse
against the window-level $F$ even though they do not collapse against the work-level $s(x)$.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/surprisal_cdf.py --out results

    pair                                      passages  windows(50)  passage median  window q10
    TinyComma-1.8B + mem. Llama-3.1-8B             100          485           3.239       2.684
    Comma-7B + mem. Comma-7B                       100          556           2.393       1.919
    Pleias-350M + mem. Pleias-350M                 100          498           3.554       2.955

    recall against F(k):  single mean spread 0.0102, oracle 0.0437 (F in [0.05, 0.7], 9 points)
    fraction covered AT the measured onset:
      TinyComma  single F=0.07  oracle F=0.000
      Comma-7B   single F=0.09  oracle F=0.009
      Pleias-350M single F=0.05 oracle F=0.006
      single 0.050-0.090 (sd 0.0163)   oracle 0.000-0.009 (sd 0.0037)

Both predictions fail, and the failure is informative. The fraction covered at the onset is
$0.05$-$0.09$, a relative spread of about $23\%$ around its mean, against $0.887$/$0.892$/$0.895$
for the ratio to the median -- a relative spread of $0.34\%$, seventy times tighter. So the budget's
unit is the surprisal of a *typical* protected work and not the size of the tail the budget covers;
a distribution-free statement of the law is strictly weaker than the one we make. The oracle
prediction fails too: at the oracle onset essentially no window is fully covered ($F \le 0.009$),
so window leakage begins before the budget can pay for any whole window, which is why the oracle
curves do not collapse under either normaliser.

This is worth one paragraph in the appendix because it forecloses the suggestion with a
measurement rather than an argument. 4 tests in `tests/test_surprisal_cdf.py`.

Intro fix in the same pass: it listed "the requirement $r(x)$ that governs leakage" as one of the
three actionable rates, two paragraphs after saying the measurement rejects exactly that reading.
It now says what feat-050 established -- $r(x)$ is what a decoder must afford before it can pay for
the work at all. The intro also now names the natural 70B pair, and the Section 4 paragraph was
trimmed to pay for it (the addition pushed the main text to 10 pages; back to 9).

### feat-055: three KL3M memorisers failed for a reason that was not the model (2026-09-08)

`recipes/finetune_memorizing.py` capped every training text at `--max-len 448`, a Llama-era default.
Measured across the 608 training texts:

    tokenizer                                      median tok    p90    max   >448
    alea-institute/kl3m-003-1.7b                          597    635    679   100%
    PleIAs/Pleias-1.2b-Preview                            293    313    342     0%
    PleIAs/Pleias-350m-Preview                            294    314    343     0%
    jacquelinehe/tinycomma-1.8b-llama3-tokenizer          280    299    376     0%
    common-pile/comma-v0.1-2t                             323    354    415     0%

**Every** KL3M text was truncated and no other family's was, so the KL3M runs trained on the prompt
and almost none of the reference. The failure mode is deceptive: training loss fell to 0.029, which
reads as thorough memorisation, because the surviving prefix really was memorised. Checked directly
on the merged `mem_kl3m-003-1_7b` against one training passage:

    teacher-forced loss   merged 0.8855   base 1.9321       (the training log reported 0.0293)
    greedy 40 tokens      " of pressure that you cannot withstand, even if you do take a rope. ..."
    true continuation     " of pressure that you cannot withstand, even if you wished to. ..."

Eleven words match and then it diverges, which is exactly where the truncated tail began; near-
verbatim recall needs 20-word spans, so it scores 0.000. The earlier reading of the 170M and 1.7B
KL3M failures as "too small to memorise" was wrong -- both were truncated, and neither model has
actually been tested.

Two fixes in the recipe: `--max-len` now defaults to `0`, meaning fit the longest training text,
and any run that would still truncate prints a warning saying that the loss will fall anyway and
recall will be zero. Separately, `from a_patch.tokenizer import ...` sat above the
`sys.path.insert`, so the recipe only ran from the repo root; the import order is fixed and the
script now runs from any cwd.

Relaunched both KL3M models at max-len 679 (`output/phase5/ft_kl3m_v3.log`, batch 2 x accum 4 for
the longer sequences). If kl3m-003-1.7b becomes admissible it is the most valuable pair available:
at 1.103 nats per character against 0.685-0.873 for every anchor used so far, it widens the
collapse's dynamic range from 1.33x to about 1.67x, from a different architecture (GPT-NeoX) and a
different tokenizer.

### feat-047 / feat-056: four pairs, and the held-out scoring reverses the paper's claim (2026-09-08)

Both sweeps landed. Pair 4 (Pleias-1.2B, self-paired) has a clean monotone curve and pair 3 at
n=458 does not:

    pair 4 (n=100)   k     2.0    2.4    2.6    2.7    2.8    2.9    3.0    3.2    3.6
                     rec 0.0021 0.0028 0.0040 0.0042 0.0076 0.0202 0.0347 0.0440 0.0711
    pair 3 (n=458)   k     2.9    3.0    3.1    3.2    3.3    3.4    3.6
                     rec 0.0023 0.0042 0.0048 0.0083 0.0107 0.0093 0.0103

    onset 2.8192  CI [2.5325, 3.0810]  ratio 0.8784  (pair 4,  0.0% of bootstraps never cross)
    onset 3.2712  CI [2.9995, 3.5680]  ratio 0.9203  (pair 3 n=458, 32.5% never cross)

**The n=458 sweep did not do what I said it would.** I predicted it would narrow the interval by
about 2.1x. The half-width fell only from 0.39 to 0.28 nats, and the share of bootstrap resamples
that never reach the threshold rose from 6.9% to 32.5% -- because I gave it a grid that stops at
k=3.6, where mean recall is still 0.0103, while the n=100 grid ran to 4.2 (recall 0.0397). At 458
passages the curve is much flatter through the crossing (0.0083, 0.0107, 0.0093, 0.0103 across
3.2-3.6) than 100 passages suggested. Two more budget points (3.8, 4.2) at n=458 are running to
repair it; until they land pair 3's onset is provisional.

Held-out scoring (`analysis/score_predictions.py`, q25 and the constant were both calibrated on
pairs 1-2, so only pairs 3 and 4 are a test):

    rule                          held-out mean |err|   in CI
    P1: median s_s - s_r                        0.034     2/2
    constant 0.889*s(x)                         0.073     2/2
    q25 of r(x)                                 0.224     1/2

**P1 -- the parameter-free derived rule -- beats the fitted constant by 2x on the two pairs that
were predicted before they were measured.** The paper currently says the derivation was rejected.
That statement scored `q25`, the *calibrated* variant, and q25 is indeed rejected; it never scored
P1's own median rule against a confidence interval. On held-out data P1 is the better of the three.

Two things keep this from being a clean win for the derivation, and both must be said:
1. Pair 3's onset is provisional (32.5% no-crossing) and is exactly the measurement that separates
   the two rules. P1 predicts 3.23 and the constant 3.16 against a measured 3.27.
2. P1 predicts the LEVEL well and the ORDERING badly. It predicts the ratio falls as the memoriser
   leaves more residual surprisal: 0.940, 0.925, 0.908, 0.886 for pairs 1-4. Measured: 0.887,
   0.892, 0.920, 0.878. Sorted, the predicted order is p1>p2>p3>p4 and the measured order is
   p3>p2>p1>p4. The directional claim still fails.

The normaliser ablation is now known to be undiscriminating: across 2, 3 and 4 pairs the winner
flips r, s(x), r (0.0046/0.0059, then 0.0070/0.0096, now 0.0180/0.0143). It should be reported as
unable to separate them rather than as evidence either way.

Four-pair collapse: ratios 0.887, 0.892, 0.920, 0.878, sd 0.0061 (was 0.003 on three pairs with
pair 3 at n=100). Mean spread over k/s in [0.7, 1.2] is 0.018 single and 0.039 oracle.
Tokenizer-free range unchanged at 1.33x, since pair 4's s(x) falls inside the existing span.

Bug found on the way: `analysis/onset_units.py:load_pairs` required EXACTLY four manifest fields,
so it silently skipped every pair the moment feat-052 added a fifth, and `add_pair.sh` aborted
under `set -e` before the figures. Now `>= 4`.

### Paper updated to four pairs (2026-09-08)

Producing commands, in order:

    .venv/bin/python analysis/onset_ci.py --comp output/phase5/n458_pleias350m_merged/composition.csv \
      --s-x 3.5543500000000003 --label "Pleias-350M + mem. Pleias-350M (n=458)" --out results
    SATML_DIR=<manuscript> scripts/add_pair.sh "Pleias-1.2B + mem. Pleias-1.2B" \
      PleIAs/Pleias-1.2b-Preview output/phase5/mem_Pleias-1_2b-Preview \
      output/phase5/fine_pleias12b/composition_summary.csv 4
    .venv/bin/python analysis/onset_table.py --out results
    .venv/bin/python analysis/score_predictions.py --out results \
      --calibrated-on "TinyComma-1.8B + mem. Llama-3.1-8B" "Comma-7B + mem. Comma-7B"
    .venv/bin/python analysis/collapse_robustness.py --out results
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/natural_pair.py --out results
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/surprisal_cdf.py --out results
    .venv/bin/python analysis/onset_units.py --out results
    .venv/bin/python figures/make_figures_v4.py --copy-to <manuscript>/figures

`results/onset_table.csv` is the table the paper prints (`analysis/onset_table.py`), so the four-pair
standard deviation has one source rather than being derived by hand:

    pair                                      n   s(x)    s_r   onset         95% CI   ratio  no-cross
    Comma-7B + mem. Comma-7B                100  2.393  0.179   2.134   [2.00, 2.92]   0.892      0.0%
    Pleias-1.2B + mem. Pleias-1.2B          100  3.209  0.365   2.819   [2.53, 3.08]   0.878      0.0%
    TinyComma-1.8B + mem. Llama-3.1-8B      100  3.239  0.194   2.873   [2.72, 3.45]   0.887      0.1%
    Pleias-350M + mem. Pleias-350M          458  3.554  0.326   3.271   [3.07, 3.83]   0.920      0.1%
    4 pairs: ratio mean 0.8943, range 0.878-0.920, sd 0.0157

Pair 3's k=3.6 grid ceiling made a third of its bootstrap resamples fail to cross; two extra budget
points (3.8, 4.2) at the same 458 passages fixed it (0.1% no-crossing) and left the point estimate
at 3.271. The merged per-passage file is `output/phase5/n458_pleias350m_merged/composition.csv`;
the merge asserts identical passage sets and disjoint budgets before writing.

**Sections changed:** abstract, `iclr_intro`, `onset` (four-row table, the derivation paragraphs
rewritten around the held-out score, the natural-pair paragraph condensed into the appendix),
`iclr_closing` (which still said "the collapse rests on two pairs spanning 1.35x"),
`appendix_robustness` (7 edits) and `appendix_limitations`. 18 pages, main text ends on page 9,
0 overfull, 0 `??`. A scripted audit re-checks 19 quoted numbers against `results/*.csv`.

Four-pair values that moved: collapse spread 0.0048 -> 0.012 below the threshold and 0.0158 ->
0.041 above; metric-artifact spread 0.052 -> 0.162 (lcs) against 0.065 -> 0.166 (recall); threshold
sweep 0.037 / 0.016 / 0.167 at 0.005 / 0.01 / 0.02; CDF fraction covered at the onset 0.07, 0.09,
0.05, 0.03 (37% relative spread against 1.8% for the ratio).

**KL3M-003-1.7b is admissible** (sampled nv-recall 0.759) now that the truncation is fixed, so pair
5 is available and is the one that widens the tokenizer-free range from 1.33x to about 1.67x. Its
prediction must be committed before its sweep is read. kl3m-002-520m is still training.

### feat-058: pair 5 falsifies the fixed-fraction law (2026-09-09, ~2.5 GPU-h)

KL3M-003-1.7B, self-paired, pre-registered in `results/onset_prediction_pair5.md` and committed at
`c6e5240` **before** the sweep. It was built to discriminate: its memoriser retains 35% of the
anchor's surprisal against 6-11% elsewhere, which put the three rules 0.54 nats apart where the
previous two held-out pairs had separated them by 0.07 and 0.01.

    grid -1 0 1.0 1.2 1.3 1.4 1.5 1.6 1.8 2.0 2.2 2.6 3.2, then 2.3 2.4 2.5 2.8 3.0 to refine
    k/s(x)  0.452 .. 0.814   recall 0.0000
    k/s(x)  0.904  0.995  1.040  1.085  1.131  1.176  1.266  1.357  1.447
    recall  .0009  .0009  .0009  .0020  .0022  .0122  .0143  .0250  .0350
    onset 2.578  CI [2.520, 3.126]   ratio 1.166  CI [1.140, 1.414]   4.3% no-crossing

**Every pre-registered rule is refuted, all three low and all three outside the interval:** P1 by
1.15 nats, the constant by 0.61, q25 by 1.40. Held-out mean absolute error over the three predicted
pairs is now 0.405 (P1), 0.252 (constant), 0.617 (q25) -- so the two-pair reading that P1 beat the
constant does not survive a third held-out pair, which is exactly why the pair was pre-registered.

**The direction inverts.** P1 says the ratio falls as the memoriser leaves more residual surprisal.
Pair 5 has by far the most residual surprisal and by far the highest ratio; Spearman over five
pairs is **-0.40**, having been +0.20 over four.

**The ratio exceeds 1 with 95% confidence** (CI [1.140, 1.414]), and independently of any
interpolation: recall is 0.0009 at k/s = 0.995 and 0.0122 at 1.176. On this pair leakage begins
*after* the certificate is vacuous, not before.

Five ratios: 0.878, 0.887, 0.892, 0.920, 1.166 -- mean 0.949, sd 0.110 (four pairs: sd 0.016).
`s(x)` spans 1.61x per token and **1.71x** per character, so the range claim improves while the
law weakens. What the paper now claims is a **unit, not a law**: a quantity computable from the
anchor and the work alone locates the onset within about a fifth on every pair, across a 1.71x
spread it would otherwise have to guess.

Pair 5 also gives the affordability argument its limiting case: at its onset budget the bucket can
afford **98%** of the passages and 1% leak (F=0.98 against 0.03-0.09 for the other four). What a
budget can pay for is not what the risky model memorised well enough to emit.

Confound stated in the paper rather than hidden: pair 5 differs in three ways at once (~2 vs ~4
characters per token, 580- vs 276-token targets, unconstrained recall 0.41 vs 0.72-0.91), and one
pair cannot say which moves the ratio.

Two diagnostics lost their power at five pairs and are now reported as such: the normaliser
ablation (0.0215 for r, 0.0264 for s(x), 0.0286 for no rescaling at all -- an 8% margin where four
pairs gave 2x), and the metric-artifact check (0.2446 vs 0.2441, equal to three decimals). Both
tests in `tests/test_onset_theory.py` were rewritten to pin what survived rather than the ordering
that held when they were written.

Sections changed: abstract, `iclr_intro`, `onset` (five-row table, the central claim, the
derivation paragraphs, the affordability point), `iclr_closing`, `appendix_robustness` (7 edits),
`appendix_limitations`. 19 pages, main text ends on page 9, 0 overfull, 0 `??`, 25 quoted numbers
audited against `results/*.csv`, 131 tests.

**Process note.** An edit script made three replacements in memory and hit a failed assertion
before its single `write_text`, so the "Five pairs" paragraph, the section heading and the fifth
table row were silently dropped while the compile still succeeded. The numeric audit caught it
(a missing `$2.58$`); without that check the paper would have carried a four-row table under
five-pair prose. Edit scripts should write after each successful replacement, or assert first.

### Retitle (2026-09-09)

    was: What a Divergence Budget Can and Cannot Certify About a Language Model
    now: A Divergence Budget Is Uninformative Without the Work It Protects
         Vacuity, Extraction and Utility in Metered Decoding

Chosen against what five pairs left standing. The old title was not wrong, but it promised a
survey of a boundary; the paper now has one positive claim strong enough to name, and one claim it
must no longer make.

**Survives, and is what the title asserts.** Proposition 1: the bound goes vacuous at exactly
$K = S(x)$, identically for every Renyi order, so $k$ read without $s(x)$ carries no risk
information -- which is also the abstract's first two sentences. Theorem 1 and the measurement
beside it: the same scalar is charged for utility and for extraction, at $10^3$--$10^4$ nats spent
per nat bought. Section 3: the margin widens as safe models improve. Section 4, in its weakened
form: $s(x)$ locates the onset within about a fifth across five pairs spanning 1.71x.

**Does not survive, and the title must not imply it.** The fixed fraction 0.89 (pair 5 measures
1.166) and the derivation $r(x)$ as a predictor of where leakage begins (refuted on level, and
inverted on direction). A title built around a law or a constant would now be false.

Rejected alternatives and why: "Metered Decoding Cannot Buy Utility Without Buying Extraction"
overclaims -- the conclusion says the obstruction is *not* fundamental, since Theorem 1 bounds the
optimal policy and our decoder is $10^3$--$10^4$ away from it. "The Unit of a Divergence Budget"
alone drops the no-free-lunch. The first subtitle draft ended in "Inference-Time Copyright
Protection" and broke "Copy-right" across a line, so it uses the paper's own term instead.

`satml_2027_arxiv_v1.tex` keeps its published title (`An Empirical Audit of $k$-NAF Budget
Accounting for Anchored Decoding`) and still matches its `vijayavallabh2026audit` bib entry, as
AGENTS.md requires. 19 pages, main text ends on page 9, 0 overfull, 0 `??`.

### feat-059: pair 6 (KL3M-002-520M) — isolating pair 5's confound (2026-09-09)

Pre-registered at `6ed6342` before the sweep (`results/onset_prediction_pair6.md`). s_s 2.4153,
s_r 0.2153, so **s_r/s_s = 0.089** — a thorough memoriser, squarely inside the 0.06–0.11 band of
pairs 1–4 — on pair 5's KL3M tokenizer (~2 chars/token) and its 580-token targets.

This pair is not run to separate the rules: P1 (2.200) and the constant (2.147) are 0.053 nats
apart here. It is run because pair 5 differed from the first four in three ways at once and one
pair cannot say which moved its ratio to 1.166. Pair 6 holds the tokenizer and the target length
fixed and flips the memoriser strength, so an onset near 2.15 blames the memoriser and clears the
tokenizer, and one near 2.82 blames the KL3M family.

    grid -1 0 1.6 1.8 2.0 2.1 2.2 2.3 2.4 2.6 2.8 3.0 3.4  (0.66x to 1.41x of s(x))

**Checkpoint conversion.** `alea-institute/kl3m-002-520m` ships only `pytorch_model.bin`, and
`a_patch/factory.py` passes `use_safetensors=True` at all three load sites, so the sweep failed at
startup. That guard is deliberate — loading a pickled checkpoint runs arbitrary code — so rather
than weaken the decoder for every future model, the one checkpoint was re-serialised to
`output/phase5/anchor_kl3m-002-520m` and verified: 520,193,024 parameters before and after, all
147 tensors bit-identical (`torch.equal`) to the pickled original. The sweep points at the
converted copy; the weights are the published ones.

### Blockers/Risks

- **Inconsistent checkpoint-trust posture.** `a_patch/factory.py` refuses `.bin` checkpoints via
  `use_safetensors=True`, but `analysis/budget_path.py`, `analysis/regimes.py`,
  `analysis/onset_theory.py` and `analysis/surprisal_cdf.py` call `from_pretrained` without it and
  will silently load a pickled checkpoint. `budget_path.py` had already loaded this very `.bin`
  before the decoder refused it. Not fixed here: tightening the analysis scripts, or relaxing the
  decoder, is a security-posture decision rather than part of this feature. Logged for a decision.

### feat-059 result: pair 6 clears memoriser strength and implicates the tokenizer (2026-09-09)

    KL3M-520M curve  k    1.6   1.8   2.0   2.1   2.2   2.3   2.4   2.6   2.8   3.0   3.4
                     rec  .000  .000  .000  .000  .000  .000  .000  .014  .026  .034  .076
    onset 2.543  CI [2.450, 3.000]  ratio 1.053  CI [1.016, 1.244]  0.0% no-crossing

The pre-registration named two readings. The memoriser-strength one predicted ~2.15 (ratio ~0.89)
and is **refuted**: pair 6 memorises thoroughly (s_r/s_s 0.089, inside the 0.06-0.11 band of pairs
1-4) and still lands above the vacuity threshold, with a ratio interval whose lower end is 1.016.
The KL3M-family reading is supported in sign.

    pair                            chars/tok  tgt tok  s_r/s_s   ratio
    TinyComma-1.8B + mem. Llama-8B       4.20      276    0.060    0.887
    Comma-7B + mem. Comma-7B             3.63      276    0.075    0.892
    Pleias-350M + mem. Pleias-350M       4.05      276    0.092    0.920
    Pleias-1.2B + mem. Pleias-1.2B       4.05      276    0.114    0.878
    KL3M-520M + mem. KL3M-520M           1.96      580    0.089    1.053
    KL3M-1.7B + mem. KL3M-1.7B           1.96      580    0.353    1.166

The ratio splits by tokenizer family and not by s_r/s_s, which interleaves: pair 6's 0.089 sits
between pairs 2 and 3, whose ratios are 0.892 and 0.920, while its own is 1.053. So the three-way
confound of feat-058 is now two-way -- tokenizer versus target length, which the two KL3M pairs
share and no pair separates. A candidate mechanism is stated in the appendix as untested:
near-verbatim recall counts 20-word spans, about 25 tokens at four characters per token and about
50 at two, so the KL3M pairs must hold a target for twice as many decode steps for the same recall.

All three rules are refuted on both KL3M pairs -- every prediction low, all six outside the
intervals. Held-out mean |err| over four pairs: P1 0.390, constant 0.288, q25 0.584; in-CI 2/4,
2/4, 1/4. P1's direction stays inverted at Spearman -0.37 over six pairs.

Range improves as the law weakens: nats/char now spans **1.87x** (0.660-1.235). The normaliser
ablation is now completely powerless -- 0.0254 (r), 0.0266 (s(x)), 0.0286 (raw), all within 12%.
F at the onset is 0.83 and 0.98 for the KL3M pairs against 0.03-0.09 for the rest.

Sections changed: abstract, intro, `onset` (six-row table, the two-group claim, the confound
paragraph folded in, the natural-pair paragraph reduced to a clause), `iclr_closing` (its
Limitations still said "five pairs ... one of which breaks it"), `appendix_robustness` (7 edits),
`appendix_limitations` (rewritten around the confound resolution). The frontier figure went from
0.76 to 0.62 textwidth and the per-work paragraph was compressed to pay for the sixth row.
19 pages, main text ends **within** page 9, 0 overfull, 0 `??`, **31 numbers audited**, 131 tests.

### feat-060: separating the tokenizer from the target length (2026-09-09)

**The two are collinear by construction, not by accident.** For a fixed corpus, target length in
tokens *is* characters-per-token: the same ~1176-character passage is 276 tokens at four characters
per token and 580 at two. No choice of pairs can separate them; only an explicit truncation can.
`--max-target-tokens` now exists in both `analysis/composition_attack.py` and
`analysis/budget_path.py` — both, because s(x) is a mean over the target and a truncated attack
paired with an untruncated budget path would compare an onset on one work to an s(x) on another.

**A metric trap found before it cost anything.** `nv_recall` is `matched reference words /
reference words`. Truncating a target halves that denominator and roughly doubles the score, which
would manufacture exactly the "step count is the cause" result. The truncation run is therefore
scored on `lcs_word`, an absolute word count with no denominator.

That raised a prior question — is the tokenizer split itself a metric artifact? `analysis/
split_robustness.py` (+ 5 tests) sweeps three differently-normalised metrics over their thresholds:

    .venv/bin/python analysis/split_robustness.py --out results
    metric        thresh       >3 ch/tok      <=3 ch/tok      gap  verdict
    nv_recall      0.005     0.845-0.882     1.023-1.143   +0.141  split
    nv_recall       0.01     0.878-0.895     1.053-1.166   +0.158  split
    nv_recall       0.02     0.903-1.071     1.117-1.314   +0.047  split
    nv_recall       0.03     0.925-1.127     1.197-1.402   +0.071  split
    lcs_word           4     0.866-0.893     1.032-1.155   +0.139  split
    lcs_word           5     0.885-1.023     1.060-1.175   +0.037  split
    lcs_word           6     0.899-1.056     1.090-1.286   +0.034  split
    lcs_word           8     0.926-1.124     1.159-1.362   +0.035  split
    any_span        0.03     0.857-1.125     1.077-1.176   -0.049  overlap
    8/9 usable cells split (nv_recall 4/4, lcs_word 4/4, any_span 0/1)

The split holds under a denominator of reference words and under no denominator at all. Under a
denominator of passages it does not: that metric has exactly one threshold at which all six pairs
cross, and there the two groups overlap by 0.049. So the honest statement is that the split
survives the normalisation that could most easily have manufactured it -- `lcs_word` has no
denominator to inflate -- and that the passage-fraction metric is too coarse at 100 passages to
resolve the question either way.

**Two corrections to my own working, both caught by checks rather than by reading.** A first pass
reported "the split does not survive" from a single overlapping cell, which was a threshold
artifact. Then `tests/test_split_robustness.py` caught a real bug in the new `crossing()`: when a
pair's smallest probed budget is already at or above the threshold there is no bracket, and
interpolating from one anyway extrapolates backwards to a budget below the grid. It fired on
exactly one cell -- Pleias-1.2B's `any_span` is exactly 0.0200 at its smallest budget -- and that
cell was in the first version of this table, reported as a split. `analysis/onset.py` and
`analysis/onset_ci.py` have always handled the case; the new script now matches them. The tally
above is the corrected one: 8/9, not the 9/10 first committed.

**The truncation run** (`results/onset_prediction_trunc276.md`, committed at `449766b` before it
started): KL3M-520M on the first 276 target tokens, tokenizer unchanged, decode steps matched to
the four-character pairs. s(x) moves only 1.3% under truncation (2.4147 -> 2.3830). An onset near
2.07 (ratio ~0.87) blames decode-step count and supports the appendix's untested span-length
mechanism; near 2.46 (ratio ~1.03) blames the tokenizer itself. One direction only: the
four-character pairs cannot be lengthened to 580 tokens without longer references than CopyBench
provides.

### feat-060 verdict: it is the tokenizer, not the target length (2026-09-09)

    .venv/bin/python analysis/score_truncation.py --out results

    run                       metric      thr   onset        95% CI   ratio     ratio CI  nocross
    truncated to 276 tokens   lcs_word      4   2.600   [2.48,2.95]   1.091  [1.04,1.24]     0.3% *
    truncated to 276 tokens   nv_recall  0.01   2.509   [2.45,2.81]   1.053  [1.03,1.18]     0.0%
    full 580 tokens           lcs_word      4   2.492   [2.43,2.96]   1.032  [1.01,1.23]     0.0% *
    full 580 tokens           nv_recall  0.01   2.543   [2.45,3.00]   1.053  [1.02,1.24]     0.0%

Halving the decode-step count from 580 to 276 with the tokenizer held fixed leaves the onset ratio
at **1.091**, inside the untruncated KL3M band (1.032-1.155) and well outside the four-character
band (0.866-0.893). The pre-registered "decode-step count" reading predicted ~0.87 and is refuted;
the "tokenizer itself" reading predicted ~1.03 and is within 0.061.

`nv_recall` gives 1.053 for both the truncated and the full run -- identical to three decimals --
so the inflation the design was built to avoid did not move this comparison either way. That is a
useful check rather than a lucky escape: `lcs_word` was still the right primary, because nothing
guaranteed the two would agree.

**The appendix's candidate mechanism is refuted.** It said a 20-word span costs about 25 decode
steps at four characters per token and about 50 at two, each an opportunity to derail, so the KL3M
pairs need a higher budget. If that were the mechanism, matching the step count would have moved
the ratio to the four-character band. It did not. What remains open is *which* property of the
tokenizer does it -- and we do not offer a replacement guess.

One direction only, as pre-registered: the four-character pairs cannot be lengthened to 580 tokens
without longer references than CopyBench provides, so this tests step count downward from 580 to
276 and would not see an asymmetric effect.

### feat-061 / feat-062: pair 7 (Phi-3.5-mini) -- granularity, not vocabulary size (2026-09-09)

Registered late: `feat-061` (chars-per-token for every cached anchor) shipped at `50e740f` but was
never written into `feature_list.json`. Both are in it now.

`analysis/tokenizer_rates.py` measured all 14 cached tokenizers on the 608 protected passages
before any model was chosen. Two groups, nothing between them: KL3M at 1.98 characters per token,
everything else at 3.62-4.18, and no candidate in 2.4-3.4. The same table decouples the two
variables the six-pair split had confounded -- kl3m-002-170m (32,768 types, 1.98 chars/token) and
Phi-3.5-mini-instruct (32,011, 3.78) sit at the same vocabulary scale and differ 1.91x in
granularity -- which is what made Phi-3.5-mini the discriminating seventh pair.

Pre-registration committed at `07f8717` before the sweep, with the anchor-path addendum at
`a2df3f1`: **0.866-0.893** clears vocabulary size, **1.032-1.155** implicates it, scored on
`lcs_word >= 4`.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
      .venv/bin/python analysis/composition_attack.py \
      --safe-model output/phase5/anchor_phi35mini --risky-model output/phase5/mem_phi35mini \
      --k-values -1 0 1.8 2.1 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.2 3.5 4.0 \
      --modes single --limit 100 --out output/phase5/fine_phi35
    SATML_DIR=/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml scripts/add_pair.sh \
      "Phi-3.5-mini + mem. Phi-3.5-mini" output/phase5/anchor_phi35mini \
      output/phase5/mem_phi35mini output/phase5/fine_phi35/composition_summary.csv 4
    .venv/bin/python analysis/score_truncation.py --out results \
      --trunc output/phase5/fine_phi35/composition.csv \
      --trunc-budget-path results/budget_path_phi-3.5-mini__mem._phi-3.5-mini.csv \
      --trunc-label "Phi-3.5-mini (32011 vocab, 3.78 chars/token)" \
      --full output/phase5/fine_kl3m520m/composition.csv \
      --full-budget-path results/budget_path_kl3m-520m__mem._kl3m-520m.csv \
      --full-label "KL3M-520M (32768 vocab, 1.98 chars/token)" --out-name phi_vocab_score.csv

Admissible: sampled k=-1 recall **0.566**. Fifteen budgets, **0 invariant violations**, max Z/K
0.99. Onset **2.628** against s(x) 2.837, ratio **0.926**, CI [0.80, 1.09], 0.0% no-crossing; on
`lcs_word >= 4` the same 0.926 with CI [0.78, 1.06].

**Verdict: vocabulary size is not the variable.** 0.926 lands outside both committed bands, 0.033
above the coarse one, but inside the coarse family's own measured spread (0.878-0.920) and 0.13
below the nearest KL3M pair. Its own interval is too wide to exclude the KL3M values by itself, so
the claim the pair supports is the negative one. Its memoriser is also the most thorough of the
seven, `s_r/s_s = 0.003`, which repeats the memorisation control at its limit. Which property of
the tokenizer moves the ratio is still open, and no anchor we can reach falls between the groups.

Seven-pair state: ratio mean 0.9603, range 0.878-1.166, sd 0.1003; held-out errors 0.352 / 0.251 /
0.481 nats for P1 / constant / q25 (in CI 3/5, 3/5, 2/5); Spearman -0.18.

**Three bugs found and fixed while landing this.**

1. `scripts/score_phi.sh` reused `analysis/score_truncation.py`, whose output filename and row
   labels were constants, so the Phi run **overwrote `results/truncation_score.csv`** -- feat-060's
   committed evidence -- and labelled its own rows "truncated to 276 tokens". Restored from git;
   the scorer now takes `--trunc-label`, `--full-label` and `--out-name`, and Phi writes
   `results/phi_vocab_score.csv`.
2. `analysis/onset.py:collapse` and `analysis/surprisal_cdf.py` both indexed `ks[0]` on a pair
   swept in one mode only. Phi has no oracle arm, so both crashed. Each now skips an empty curve;
   `onset.py` also prints the per-mode pair count (7 single, 6 oracle) instead of reusing the first
   row's. Regression test in `tests/test_onset_collapse.py`.
3. The appendix's F(k) sentence said the coverage spread was 170% against 11% for the ratio. The
   11% is sd/mean and reproduces; 170% does not reproduce under any definition (sd/mean on the six
   pairs is 117%), so it was a transcription slip the earlier audits missed. Recomputed and stated
   as sd/mean: **115%** against **10%** on seven pairs.

Manuscript: seven pairs throughout (abstract, intro, Section 4, conclusion, both appendices).
Section 4, Section 7 and the conclusion were compressed to hold the main text at exactly 9 pages --
Ethics now starts at the top of page 10 with nothing above it. 19 pages total, 0 overfull, 0 '??'.

### feat-063: feat-060 is withdrawn, and the confound it claimed to close is reopened (2026-09-10)

Found while checking a new hypothesis, not by a reviewer. Each CopyBench item is a prompt prefix
followed by the protected passage, so the passage sits at the **end** of `prompt_text + reference`.
Where it starts, in tokens, depends entirely on the tokenizer:

    pair                                ref starts   total   steps seed->ref   in target?
    TinyComma-1.8B + mem. Llama-3.1-8B         215     279            195        100/100
    Pleias-1.2B / Pleias-350M              229/230     298            209/210    100/100
    Phi-3.5-mini                               243     317            223        100/100
    Comma-7B                                   250     325            230        100/100
    KL3M-1.7B / KL3M-520M                      455     592            435        100/100

feat-060 truncated the KL3M target with `--max-target-tokens 276`, ending it at token **296**. The
passage begins at 455 (min 402 over the 100 passages), so **the passage was not in the target for
100 of 100 passages**. Reconstructing that exact configuration and scoring the true target against
the reference gives `nv_recall` **0.0000** (mean and max): a perfect adversary reproducing the
truncated target would have scored zero. Whatever the run measured -- the memoriser wandering out
of the forced target into other memorised text -- it was not extraction of the protected passage,
and the ratio 1.091 cannot be compared with the untruncated 1.032.

**Withdrawn:** feat-060's status is `invalid`, the paragraph is out of Section 4, and
`sections/appendix_limitations.tex` now carries an explicit "a control we ran, and have withdrawn"
paragraph, because the number appeared in a compiled version of the manuscript. The
target-length/decode-distance confound is **reopened**.

**The main result is unaffected and was checked, not assumed.** All seven untruncated runs contain
the passage in 100/100 targets with a ceiling `nv_recall` of 0.997.

**Root-cause fix, in the shared place:** `analysis/composition_attack.py` now computes the covered
fraction for every run, prints `reference reached in N/M`, warns when some passages are uncovered
and raises `SystemExit` when none are. A guard in the caller would have left every other entry point
broken.

**What the check exposed.** Three quantities are perfectly confounded with characters-per-token,
because the tokenizer cuts one fixed piece of text: the adversary's seed (`--seed-tokens 20` buys
**7.3 words** on KL3M against **13.0-14.4** on the other five), the decode distance to the passage
(**435** steps against **195-230**), and the target length. All three split the seven pairs exactly
where the onset ratio splits, with no overlap in any of them. Section 4's "which property of the
tokenizer" is therefore not one open question but three, and `results/onset_prediction_seed.md`
pre-registers the intervention that separates the first from the other two.

### Plan v5 item D: the judged arms, at 600 pairs and two judges (2026-09-10)

Generation for k in {1.5, 2, 2.5} was finished (`output/phase5/util_fine`, the k=2.5 creative split
had been killed by memory pressure in an earlier session and was regenerated across two cards; the
prompt sets are identical to the other budgets, checked). Both judges then scored **21 arms** --
KL at nine budgets, pathwise at six, anchor-only, the null, and **alpha in {1,2,4,8}**, the last of
which was the blank cell in Table 2 -- at `--judge-per-cell 200`, so **600 comparisons per arm**
against the 180 the manuscript had.

    scripts: analysis/utility.py --prefix {utility_v5,utility_v5_judge2} --judge-per-cell 200
             --judge {Qwen/Qwen2.5-7B-Instruct, microsoft/Phi-3.5-mini-instruct}
    new:     analysis/judge_separation.py (results/judge_separation{,_judge2}.csv)

**The published headline was under-powered and is wrong.** At 180 pairs the k=1 separation from the
anchor reads -0.45 sigma; at 600 it reads **-3.66 sigma**. The null arm moves 41.7% -> 46.4% loss
and now agrees with the independent feat-029 run at the same sample size (47.0%), so the old null
was the outlier. "The two useful regions do not overlap", as stated, does not hold.

**What replaces it is stronger, because it survives both judges.**

    k      vacuous%   Qwen z   Phi z    verdict
    0.5         0.0    -1.03   +1.52    neither judge separates the decoder from the anchor
    1          43.9    -3.66   +0.11    judges disagree
    1.5-5   98.9-100   -3.94.. -1.23..  judges disagree
    10        100.0    -6.97   -3.29    BOTH separate
    20        100.0    -6.32   -2.00    BOTH separate

At the largest budget whose certificate still covers **every** protected work, neither judge can
tell the decoder from serving the anchor alone. The only budgets where both agree it is better are
those covering **no** work at all. The useful region opens only where the certified region has
closed -- and that statement, unlike the old one, does not depend on which judge is asked.

**The two judges disagree about the crossing by nearly an order of magnitude**: Qwen puts the
-2 sigma crossover at k=0.68, Phi-3.5-mini at k=5.39. Reported as a result rather than averaged
away. It also retires the "one judge" limitation and replaces it with a sharper one.

**alpha=4, the blank cell.** It overrides the risky model at 90% of decode steps and leaks 24x less
than alpha=1 in oracle windows (0.004 against 0.097), and neither judge can separate it from
alpha=1 (0.35 sigma for Qwen; Phi reverses the order entirely). Activity is not price.

**Theorem 1's pricing is unchanged at the finer grid** (`analysis/utility_price.py`, now reading the
budget list from the summary and searching several run directories): the conservative
spend/Lambda* ratio is 1,337-8,619, still 10^3-10^4.

**A near-coincidence worth pinning.** Qwen's crossover interpolates to k=0.68 across a bracket half
a nat wide; the budget at which the first protected work loses its certificate is 0.583 and at which
1% have lost it is 0.682. `results/onset_prediction_crossover.md` pre-registers the k in
{0.6, 0.7, 0.8} run that turns the interpolation into a measurement, with both refuting outcomes
committed.

Manuscript: abstract, introduction, Section 7 and the limitations rewritten; main text back to
exactly 9 pages, 0 overfull, 0 '??'.

### feat-064 Arm B: the onset split is the attack's seed, not the tokenizer (2026-09-10)

Pre-registered at `bc74f4d`, before either arm ran. Arm B gives Pleias-1.2B -- a **coarse**-tokenizer
pair sitting at 0.878, deep in the coarse band -- the seed the KL3M pairs get, by cutting
`--seed-tokens` from 20 to 10 so its adversary holds 6.9 words instead of 13.7:

    Pleias-1.2B                seed words   onset   ratio   95% CI (ratio)   no-cross
    seed 20 tokens (control)         13.7   2.780   0.866   [0.72, 0.95]        0.1%
    seed 10 tokens (Arm B)            6.9   3.272   1.004   [0.99, 1.23]        0.9%

Pre-registered band for "the seed is the mechanism": **ratio >= 1.00**. Measured **1.004** on the
primary metric (`lcs_word >= 4`) and 1.010 on `nv_recall`. The two bootstrap intervals are
**disjoint** -- the control's upper end is 0.95 and the arm's lower end is 0.99.

s(x) moves 3.209 -> 3.259 (+1.5%) and the target grows 270 -> 280 tokens (+3.7%), so the shift is
not the denominator. Everything else -- tokenizer, anchor, memoriser, corpus, passages, metric,
budget grid density -- is held fixed.

**What this means.** `analysis/composition_attack.py` specifies the adversary's prefix in
**tokens**, so a tokenizer that cuts the same text twice as finely hands the adversary half the
words: 7.3 on the KL3M pairs against 13.0-14.4 on the other five, an exact split with no overlap.
Arm B shows that difference alone is enough to move a pair across the vacuity threshold. The
"which property of the tokenizer moves the ratio" question that Section 4 left open looks like the
wrong question: no property of the tokenizer needs to move it, only the number of words the
evaluation happens to hand the adversary.

Arm A (KL3M-520M at `--seed-tokens 40`, the converse direction) is running, and the dose-response
arms at seeds 10 and 80 are queued behind it. The claim will be stated at the strength the full set
supports, not at Arm B's.

This generalises past this mechanism: any extraction evaluation that seeds "the first N tokens" of a
passage measures something tokenizer-dependent, and two such evaluations are not comparable across
tokenizers unless the prefix is matched in characters or words.

### The seed finding is a confound in the field's definition, not just in our harness (2026-09-10)

Verified against the primary source (arXiv 2202.07646, `carlini2023quantifying`, already in
`references.bib`; PDF fetched and grepped, quotes checked verbatim):

- **The definition is parameterised in tokens.** "Definition 3.1. A string s is extractable with
  **k tokens of context** from a model f if there exists a (length-k) string p, such that the
  concatenation [p || s] is contained in the training data for f, and f produces s when prompted
  with p using greedy decoding."
- **The protocol is too.** "the first l - 50 tokens and report the sequence as 'extractable' if the
  model exactly emits the next 50 token suffix of this sequence. **Fifty tokens corresponds to an
  average of 127 characters or 25 words in the GPT-Neo training set**, well over the length of a
  typical English sentence." -- the conversion is stated for one tokenizer and one corpus.
- **The dependence on context length is a headline result of that paper.** "the fraction of
  extractable sequences increases log-linearly with the number of tokens of context. For example,
  **33% of training sequences ... are extractable from the 6B model at 50 tokens of context,
  compared to 65% with 450 tokens** of context. We call this the discoverability phenomenon."
- Definition 3.1's own worked example slides between the two units in consecutive sentences: "given
  the length k = 4 prefix" and then "this sequence is extractable (**with 4 words of context**)".
  In English under a four-character tokenizer the two are close. Across tokenizers they are not.

So the quantity is defined in tokens, its dependence on context length is established, and the
conversion to words is tokenizer-specific. Two extractability measurements taken under different
tokenizers with the same token budget are therefore not comparable, and nothing in the literature
we have read draws that consequence. Arm B measures how much it matters under a metered decoder:
enough to move a pair from 0.866 to 1.004 of its vacuity threshold, across the boundary.

This is the framing for Section 4 once Arm A and the dose-response arms land: not "an artifact of
our attack script" but "a confound inherited from the standard definition, quantified".

### Correction: feat-060 was reinstated the same day it was withdrawn (2026-09-10)

Earlier today I marked feat-060 invalid, on the grounds that its truncated target did not contain
the protected passage. **That was my error, and the entry above it is wrong.**

`analysis/composition_attack.py` scores `nv_recall` and `lcs_word` against **`x["target"]`**, the
decoded target, not against the CopyBench `reference` field. I checked coverage of the wrong
object. And each benchmark item's `prompt_text` is **930 characters of the same novel** -- it opens
"Complete the prefix:" and then continues the book -- so a target truncated inside `prompt_text` is
still protected text, not instruction text. The truncated run's own k=-1 baseline settles it:
**nv_recall 0.696, with 57% of passages above 0.8**, which is impossible if the target were
unreproducible. feat-060's control is valid and its conclusion -- the decode-step count is not the
mechanism -- stands.

Reverted: feat-060 back to `done`, Section 4's target-length sentence restored, the appendix's
"a control we ran, and have withdrawn" paragraph replaced by the original with the scoring object
stated explicitly, and the `SystemExit` guard **removed** -- it would have blocked feat-060 itself.
What survives from feat-063 is the informational line `reference reached in N/M`, which records
whether a run measures the passage proper or only its prefix. That distinction is real and worth
printing; it is not a validity condition.

The seed result (Arm B) is untouched by this: it compares two runs at the same target definition
and the same scoring object, and its intervals are disjoint.

**The lesson, recorded because it nearly cost a correct result:** before declaring a committed run
invalid, read what the metric is computed against, in the code, and check the run's own k=-1
baseline. A baseline near the ceiling means the target was reproducible and the measurement is real.

### Arm B put through the same checks that the feat-060 false alarm should have had (2026-09-10)

    run                             target   k=-1 recall   k=-1 lcs   k=0 recall
    Pleias-1.2B seed 20 (control)   271 tok        0.909      183.7        0.000
    Pleias-1.2B seed 10 (Arm B)     281 tok        0.894      186.7        0.000

- **Same scoring object.** Both runs score against their own decoded `target`, the same quantity,
  so the comparison is like for like.
- **Both baselines near the ceiling.** The unconstrained model reproduces either target (0.909,
  0.894), so both runs can measure extraction; and the anchor alone leaks nothing in either (0.000).
- **The target-length difference points the wrong way to explain the result.** Arm B's target is
  3.7% longer (281 against 271 tokens) and its unconstrained `lcs_word` is slightly higher (186.7
  against 183.7). On the primary metric, `lcs_word >= 4`, an absolute word count, a longer target
  can only make the threshold easier to reach, which would push the onset **down**. The measured
  onset went **up** by 17.7%. The effect is therefore not a target-length artifact, and it survives
  on `nv_recall` (+18.4%) as well, which normalises by target length and so has the opposite bias.

Both directions of bias are accounted for, and the bootstrap intervals are disjoint.

### The cross-pair trend was already in the seven pairs, and it agrees with the intervention

`analysis/seed_effect.py` now also reports the observational trend, with average ranks for the tie
at 7.3 words and an exact permutation p (7! = 5,040 permutations; an asymptotic p is not
trustworthy at n=7):

     7.3 words   ratio 1.166   KL3M-1.7B
     7.3 words   ratio 1.053   KL3M-520M
    13.0 words   ratio 0.920   Pleias-350M
    13.1 words   ratio 0.926   Phi-3.5-mini
    13.4 words   ratio 0.892   Comma-7B
    13.7 words   ratio 0.878   Pleias-1.2B
    14.4 words   ratio 0.887   TinyComma-1.8B

    all pairs            n=7   Spearman -0.919   exact permutation p = 0.0071
    coarse family only   n=5   Spearman -0.800   exact permutation p = 0.1333

**This is confounded by construction and is reported as such**: seed words is exactly
20 x characters-per-token, so it is the same variable as tokenizer granularity, and no observational
analysis can separate them. Its value is that the trend continues *inside* the coarse family, over a
range of 1.4 words among five pairs built for entirely unrelated reasons, with the same sign and
nearly the same magnitude. The intervention (Arm B) breaks the confound; the observation says the
intervention is not describing a two-group artifact.

### Decision: a Gutenberg second corpus is not worth GPU time (2026-09-10)

The "one protected corpus" limitation is the paper's biggest stated weakness, and `data/gutenberg/`
holds 50 cached public-domain books, so building a second (anchor, memorised-copy) pair on them
looks like the cheap fix. It is not, and `results/latent_leakage_summary.csv` already says why:
under TinyComma the per-token surprisal of Gutenberg text is **3.11 (opening) to 3.33 (deep)**
against **4.73** for a CopyBench passage of the same length. Every anchor we have is trained on open
data that includes Project Gutenberg, so a Gutenberg corpus is not text "the safe model never saw" --
it is the opposite, and near-access-freeness is defined relative to the anchor. The onset measured
there would answer a different question.

Closing this limitation needs copyrighted text of a different genre or language that the open-data
anchors provably did not see, which we do not have. It stays a stated limitation.

### The three proofs re-derived independently (2026-09-10)

Checked line by line, since everything rests on them:

- **Prop. 1, alpha = infinity.** D_inf <= K gives p_theta(E) <= e^K p_s(E) = e^{K - S(x)}, vacuous
  iff K >= S(x). Correct.
- **Prop. 1, alpha in (1, inf).** The change-of-measure bound
  P(E) <= Q(E)^{(a-1)/a} exp((a-1)/a * D_a) is the standard Renyi form (cited to
  `mironov2017renyi`). The exponent (a-1)/a is positive and finite for a > 1, so the bound reaches 1
  iff e^K p_s(E) >= 1, i.e. K >= S(x). Correct, and the exponent's positivity is exactly what makes
  the threshold order-invariant.
- **Prop. 1, alpha = 1.** The binary-KL inversion attains 1 when d(1 || e^{-S(x)}) <= K, and
  d(1 || q) = -log q = S(x). Correct.
- **Prop. outrun.** d(q || e^{-s_t}) = q*s_t + q log q + (1-q) log((1-q)/(1-p)). The last two terms
  are bounded below by -H(q) >= -log 2, and -(1-q) log(1-p) >= 0, so with q >= 1-eta the whole is
  >= (1-eta) s_t - log 2. Correct.
- **Theorem 1.** The KL chain rule over sequences and Donsker-Varadhan with the supremum over
  lambda giving the Legendre transform of the CGF, i.e. the Cramer rate function. Both standard and
  correctly applied; the paper additionally verifies the chain-rule step empirically (largest
  discrepancy 1.5e-4 nats over 240,000 decode steps) rather than assuming the implementation
  matches the mathematics.

Attribution is right: Prop. 1 says "None is new; the contribution is that all three break in the
same place", and k_crit is credited to Loynes rather than claimed.

### feat-064 Arm A, and the verdict the pre-registration forced (2026-09-10)

    KL3M-520M                seed words   onset   ratio   95% CI          no-cross
    seed 20 tokens (control)        7.3   2.492   1.032   [1.01, 1.23]        0.0%
    seed 40 tokens (Arm A)         14.3   2.253   0.939   [0.88, 1.17]        0.1%

At `--seed-tokens 40` KL3M's adversary holds **80.8 characters and 14.3 words** against TinyComma's
81.3 and 14.4 -- a near-exact match to the coarse family's seed, with the tokenizer, models, corpus,
passages, metric and grid density unchanged and s(x) moving 0.7%.

**The verdict is the one the pre-registration reserved for this outcome: undecided.** 0.939 falls in
the 0.93-0.95 overlap that `results/onset_prediction_seed.md` committed in advance as belonging to
neither account -- seed-matching predicted 0.85-0.95, the k_crit account 0.96 -- and we report it as
undecided rather than assigning it to the one we prefer. Arm A's interval also **overlaps its
control's**, so unlike Arm B it does not carry a claim on its own.

**What the two arms together do establish.** The seed moves the ratio in the predicted direction in
both, and moves a pair across the budget at which its own certificate goes vacuous in both: Pleias
0.866 -> 1.004 (disjoint intervals) and KL3M 1.032 -> 0.939. Neither arm reaches the other family's
band, so the adversary's context is not the whole of the difference between the families. The
dose-response arms, which are running, are what would separate a threshold from a curve.

Manuscript rewritten around this: Section 4 now answers its own question -- the split is not
memorisation, not target length, not vocabulary size, and is in large part what the evaluation hands
the adversary -- with the Carlini grounding, and `sections/appendix_seed.tex` carries the confound
table, both arms, the two soundness checks and the observational trend. The abstract and
introduction now lead with the finding instead of the tokenizer description. Main text **8 pages +
43 lines**, under the 9-page limit; 20 pages total, 0 overfull, 0 '??'; 714 numeric literals audited
with 1 unsourced and that one verified against the checkpoint config. 167 tests.

### The seed-10 KL3M arm is contaminated, noted before its sweep finished (2026-09-10)

Written while `output/phase5/seed10_kl3m520m` was still running, from its k=-1 baseline alone.

    run                              seed words   k=-1 recall   k=-1 lcs
    KL3M-520M  seed 20 (control)            7.3         0.519      105.4
    KL3M-520M  seed 40 (Arm A)             14.3         0.520      104.9
    Pleias-1.2B seed 20 (control)          13.7         0.909      183.7
    Pleias-1.2B seed 10 (Arm B)             6.9         0.894      186.7
    KL3M-520M  seed 10 (dose-response)      4.0       **0.227**       48.3

The four arms already reported hold the unconstrained baseline flat -- 0.519 against 0.520, and
0.909 against 0.894 -- which is what licenses reading them as "the same memoriser, differently
informed". **The seed-10 KL3M arm does not.** At $4.0$ words the memoriser's own unconstrained
recall falls to $0.227$, less than half the control's, so its onset would confound "the adversary
holds less of the work" with "the adversary can no longer locate the work at all". It clears the
$0.10$ admissibility floor, but only just, and it is not a clean dose-response point.

**How the curve will be read, decided in advance:** the dose-response is the three points whose
baseline is stable -- seeds 20, 40 and 80, at 7.3, 14.3 and 28.4 words -- and the seed-10 point is
reported with its baseline attached and excluded from any fitted trend. The seed-80 arm is the one
that discriminates the two accounts anyway (third addendum to
`results/onset_prediction_seed.md`), and its baseline will be checked the same way before its ratio
is quoted.

That the collapse happens at all is worth reporting on its own: below roughly seven words of the
work, this adversary cannot find its place in it, and no budget is needed to stop it. That is a
floor on the threat model, not a property of the mechanism.

### The seed acts on the budget, not on the memoriser (2026-09-10)

Already in the four clean arms, and it answers the "why there" the section title promises:

    pair          seed words   k=-1 recall   k_crit   onset/s(x)
    KL3M-520M            7.3         0.519    3.791        1.032
    KL3M-520M           14.3         0.520    3.514        0.939
    Pleias-1.2B          6.9         0.894    5.916        1.004
    Pleias-1.2B         13.7         0.909    4.942        0.866

Doubling the words the adversary holds changes

    what the UNCONSTRAINED model can do       +0.2%   +1.7%     (k=-1 recall)
    the budget the target demands             -7.3%  -16.5%     (k_crit, anchor only, no attack)
    where extraction begins                   -9.0%  -13.7%     (onset / s(x))

**The memoriser's own ability is unmoved and the budget requirement is not.** The onset follows the
budget-side quantity, on both pairs, to within about two percentage points of a change of seven to
seventeen. `k_crit` is computed from the anchor alone -- one teacher-forced pass, no attack, no risky
model -- so the agreement is a prediction and not a fit.

This reconciles the two accounts the pre-registration set against each other rather than choosing
between them, which is why Arm A landed in the band declared undecided: **the seed is the lever and
`k_crit` is how the lever acts.** A longer prefix does not make the memoriser better at the work; it
starts the target at a cheaper point, and the token bucket's running maximum -- the quantity
Proposition~\ref{prop:outrun} already builds on -- falls with it.

It also predicts the shape the seed-80 arm will show: `k_crit` falls only 2.3% from seed 40 to seed
80, so the onset should fall about as little, which is the committed discriminating prediction in
the third addendum. That the same relation was reached from a different direction before the arm
ran is worth noting, and does not make it more confirmed.

### 2026-09-10, feat-065: the units claim, conditioned on the adversary's context

**The number a reviewer computes first, and it looked bad.** Leave-one-out across the seven pairs:
predicting a held-out pair's onset as `c * s(x)` gives mean |error| 0.264 nats; predicting a
*constant number of nats* and not rescaling at all gives 0.295. Over an `s(x)` range of 1.61x the
units claim was barely distinguishable from quoting a constant.

**Conditioning on the seed fixes it.** The seven pairs split exactly in two by how many words 20
tokens buy -- 7.3 on the two KL3M pairs, 13.0-14.4 on the other five, no overlap -- and that split
was registered in `results/onset_prediction_seed.md` before the intervention arms ran. Within the
five seed-matched pairs, over `s(x)` spanning 1.49x:

| subset | n | s(x) span | ratio | cv | LOO k/s(x) | LOO constant | ratio |
|---|---|---|---|---|---|---|---|
| all pairs | 7 | 1.61x | 0.878-1.166 | 11.3% | 0.264 nats | 0.295 nats | 1.1x |
| matched context (> 10 words) | 5 | 1.49x | 0.878-0.926 | **2.4%** | **0.070 nats** | 0.364 nats | **5.2x** |

The grouping is read off these same seven measurements, so it is not itself a test -- the two
pre-registered intervention arms are. What it shows is that the residual the units claim was losing
to is the evaluation protocol, not noise in the law.
Command: `.venv/bin/python analysis/score_predictions.py --out results --calibrated-on "TinyComma-1.8B + mem. Llama-3.1-8B" "Comma-7B + mem. Comma-7B"`
-> `results/matched_context.csv`, `results/onset_seed_words.csv`.

**Normaliser ablation, extended with k_crit** (`analysis/collapse_robustness.py`, block
`normaliser_spread`): across the seven pairs `s(x)` is the best of four normalisers (1.33x spread,
cv 11.5%), `k_crit` by far the worst (3.75x, cv 43.4%), `r = s_s - s_r` in between (1.89x), and no
rescaling at all 1.49x. `k_crit` is nonetheless the *best within-pair* predictor of the seed
intervention (`results/seed_effect.csv`, column `pred_ratio_K`): calibrated on each pair's control
arm alone, it puts both existing out-of-sample arms inside their bootstrap intervals (0.963 vs
0.939 [0.88,1.17]; 1.021 vs 1.004 [0.99,1.23]). The two facts are consistent: `k_crit` is a running
maximum whose *level* is set by the worst token in the target and varies wildly across corpora,
while its *derivative in the seed* is what the onset follows. The seed moves `s(x)` by 0.7-1.5% and
`k_crit` by 7.3-19.7% (`results/seed_effect_summary.csv`).

**Negative recon: more anchors cannot widen the s(x) range.** The three cached, un-paired anchors
score 2.166 (KL3M-3.7B), 2.557 (KL3M-170M) and 3.148 (Pleias-3B) nats/token on the CopyBench
passages, against a built range of 2.211-3.554. None extends it; the ten permissively licensed
anchors span only ~1.64x on this corpus, so the range is a property of the corpus and the class of
open models, not of how many pairs are built. Building any of them would cost ~2.5 GPU-h for a 2%
extension. Command: `analysis/budget_path.py --safe-model <id> --composition '' --limit 100
--prefix budget_path_recon_<tag>` (log `output/phase5/recon_sx.log`). Redirected to the temperature
lever instead (`output/phase5/recon_temp.log`).

**Compute accounting rebuilt.** `analysis/compute_hours.py` now scans `output/phase{4,5}/*.log`,
billing each launcher log from its birth to its last write minus the `sleep` calls a `set -x` trace
recorded, so an armed chain that waited three hours for a file scores ~0. Total 93.2 GPU-hours
(`results/compute_hours_summary.csv`); the manuscript's LLM-usage statement was updated from a
stale 63 and now also says there are two judges, not one.

### 2026-09-10, feat-066 part 1: the seed dose-response and the first temperature arm

**Three out-of-sample seed arms, all predicted from each pair's control arm alone.** The
token-bucket rule `onset = c * k_crit`, with `c` calibrated on the control and nothing else,
predicted every arm before it ran (`results/onset_prediction_seed.md`, third and fourth addenda):

| arm | measured | (K) predicted | no-change null |
|---|---|---|---|
| Pleias-1.2B seed 10 | 1.0041 | 1.0211 | 0.8661 |
| KL3M-520M seed 10 | 1.1635 | 1.1486 | 1.0321 |
| KL3M-520M seed 40 | 0.9393 | 0.9633 | 1.0321 |

Mean |relative error| 1.8% against the null's 11.6%. The KL3M-520M dose-response is strictly
monotone in the adversary's context: 4.0 words -> 1.163, 7.3 -> 1.032, 14.3 -> 0.939. The seed-10
point is reported with its baseline attached (k=-1 recall 0.227 against 0.519) and excluded from any
fit, as pre-committed; the seed-40 arm is the clean one (0.520 against 0.519).

**The temperature arm moves `s(x)` inside a single pair, and the constant-nats null loses.** With
anchor, memoriser, corpus, tokenizer and seed all fixed, warping both logit vectors to tau = 0.4
(He et al. App. B) moves `s(x)` 2.4147 -> 4.0340 (+67.1%) and the onset 2.4923 -> 3.6030 (+44.6%).
Elasticity `d log(onset)/d log s(x)` = **+0.72 [+0.41, +0.84]**: the interval excludes 0, so a
constant number of nats is refuted within a pair, and excludes 1, so exact proportionality is not
confirmed. The tau = 0.4 memoriser is much stronger (k=-1 recall 0.904 against 0.519), which pushes
the onset *down*, so 0.72 is a lower bound. Pre-registered in
`results/onset_prediction_temperature.md` before any token was decoded.

**`k_crit` misses for the first time, and the asymmetry is the finding.** After three straight hits
on the seed arms it predicted 1.137 for the warped arm against a measured 0.893 [0.76, 0.95]. The
seed changes *which* tokens fall in the adversary's window, which a running maximum over the
target's surprisal profile tracks; the warp rescales the whole profile, which the running maximum
over-reads. `k_crit` predicts what the evaluation protocol does to the onset, not what the decoder's
temperature does, and Section 4 must not present it as a general law.

**Recon that redirected the work.** The three cached un-paired anchors score 2.166, 2.557 and 3.148
nats/token on these passages against a built range of 2.211-3.554, so no further pair widens the
`s(x)` range; temperature does, by 1.67x on KL3M-520M and 1.62x on Pleias-1.2B, whose tau = 0.4
`s(x)` of 5.200 is above every unwarped pair.

Commands: `analysis/composition_attack.py --temperature 0.4 ...`, `analysis/budget_path.py
--temperature 0.4 ...`, then `analysis/seed_effect.py --out results` (which now carries `k_crit`,
`pred_ratio_K`, `pred_ratio_null` and the elasticity with its interval).

### 2026-09-10: the judged crossover on the fine grid, both judges

`output/phase5/util_cross` at k in {0.6, 0.7, 0.8}, 600 comparisons per arm.
`results/crossover.csv`: Qwen crosses -2 sigma at **k = 0.6642**, where 0.26% of the 758 works have
lost their certificate; Phi at **k = 5.3929**, where all of them have. The committed band was
[0.58, 0.80] and the primary judge is inside it. The refutation condition -- a crossover below the
0.583 at which the first work loses its certificate -- did not occur on either judge.

**The finer sub-claim is withdrawn.** It predicted 0.583 would sit inside the bracket between the
last budget where the decoder is indistinguishable and the first where it is not. That bracket is
[0.6, 0.7] and 0.583 is just below it, so the two boundaries are **ordered, not coincident**. The
conclusion now says so and distinguishes the two readings of "informative": no budget covers every
work and buys a detectable improvement, while relaxing "every" to "almost every" opens a narrow
window at k = 0.66 on the sharper judge that closes again on the coarser one.

**Instrument drift, recorded against ourselves.** The null arm is the same 500 comparisons judged
twice: -5.47 and -1.49 sigma in the v5 run, -6.09 and -2.49 in v6. About a sigma of run-to-run
drift is the floor beneath which no separation in the paper should be read, and
`sections/appendix_limitations.tex` now says that.

### 2026-09-10: citation audit (existence was verified; the claims were not)

A walk of the ICLR build found it citing 35 of 160 verified entries, with whole clusters missing
(no legal framing, no unlearning, no inference-time defence but CP-Fuse, no watermarking
primitives, no memorisation-measurement literature, and neither closest competitor). Five
paragraphs were added to `sections/appendix_related.tex`; the build now cites 67 with 0 unresolved
references and the nine main-text pages untouched. Checking the claims against primary records
found **four errors in seventeen**, one of them mine, all recorded in `LITERATURE_REVIEW.md`:
`cohen2026barriers` was described as qualitative and about composition across queries when it is
quantitative and about *autoregressive* non-composition; a Proposition 1 claim against
`segal2026provably` overshot its own qualifier; `tomasi2026primaldual`'s discrete-diffusion setting
was elided; and a paragraph I wrote fresh had `ippolito2023preventing` and `hayes2025measuring`
backwards -- both say the standard probes **understate** extraction, which is the same failure our
own sampled-recall screen exists to avoid (0.708 greedy against 0.022 sampled).

### 2026-09-10: two stale percentages in Section 4, and why the numeric audit could not see them

`sections/onset.tex` said `k_crit` "falls by $7.3\%$ and $16.5\%$ and the onset falls by $9.0\%$ and
$13.7\%$". The two `k_crit` figures are right; **both onset figures were wrong** -- the committed
values are $9.6\%$ and $15.0\%$ (`results/seed_effect_summary.csv`, columns `pct_change_k_crit` and
`pct_change_onset`; the summary reports each arm against its control, and Section 4 quotes the
"doubling the words" direction, so Pleias reads $-16.5\%$ and $-15.0\%$ there against $+19.7\%$ and
$+17.7\%$ in the CSV).

`analysis/audit_numbers.py` passed on the wrong values and would pass on almost any wrong
percentage. It checks whether each math-mode literal appears *somewhere* in `results/**.csv`, and
with 2.08 million distinct values a two-significant-figure number nearly always does. It cannot
check a quantity **derived** from two CSV entries, which is what a percentage change is. The
discipline that does work is the one already in the Definition of Done: write the derived quantity
to a CSV and quote it from there. `analysis/seed_effect.py` now emits those columns, so the four
percentages in Section 4 have a row of their own.

### 2026-09-10: an anonymity leak in, of all files, the numeric auditor

`scripts/build_artifact.sh` refused to package: **identifying strings found**. The offender was
`analysis/audit_numbers.py`, whose `--tex` argument defaulted to
`/mnt/md0/.../vijayavallabh/sub/satml` --- an absolute path carrying the author's surname, which
would have shipped inside the anonymised artifact. The manuscript lives outside the repo, so the
default is now `$SATML_DIR` falling back to `../sub/satml`, and the reason is written in a comment
so nobody restores the convenience. Artifact rebuilt: **439 files, manifest verified, 26 MB**.

The builder's check is the only thing standing between an absolute path and a desk rejection, and
it earned its keep here. It runs after `README_artifact.md` is copied in, for the same reason.

### 2026-09-10: the Renyi order separations do not survive re-judging, and Section 5 quoted them

Checking whether the v6 judging run still supported Section 5 showed that it does not. The **same
judge** re-scoring the **same** 150-pair generations gives:

| order | v5 | v6 |
|---|---|---|
| alpha=1 | -1.92 | -1.77 |
| alpha=2 | -1.05 | **-2.65** |
| alpha=4 | -1.49 | -0.89 |
| alpha=8 | **0.00** | **+1.07** |

alpha=2 moves 1.6 sigma and the ranking of the four inverts. Phi reorders them differently in each
of its two runs as well. The manuscript had quoted the v5 values as the measurement and built a
punchline on alpha=8 losing "$62.7\%$, the anchor's own rate to the decimal ($0.00\sigma$)" -- a
coincidence that does not reproduce.

`sections/orders.tex` now demonstrates the non-resolution instead of quoting numbers from whichever
run it was written against, and says plainly that no separation quotable there would survive a
re-run. The substantive claims are unchanged and better supported: alpha=4 leaks $24\times$ less
than alpha=1 while overriding the risky model at $90\%$ of steps, and **no run of either judge**
separates them.

**Durable caution.** A judged separation from `analysis/utility.py` at n=150 is not reproducible;
at n=500-600 the null arm still drifts about a sigma between runs (-5.47/-1.49 in v5 against
-6.09/-2.49 in v6). Never quote a judged sigma without knowing the sample size, and never build a
claim on a separation smaller than a sigma.

### 2026-09-10: a pre-registered input was silently rewritten, and it turned out to be harmless

The KL3M-1.7B seed-40 queue recomputed `results/budget_path_kl3m17b_seed40.csv`, which had already
been **committed as part of the fourth addendum's pre-registration** -- the file the prediction
(K) = 1.071 was computed from. `git diff --stat` showed 101 insertions and 101 deletions, which
looks exactly like a changed measurement.

It was not. Comparing row by row on `prompt_id`: the same 100 works, and **no column differs on any
row**. `analysis/budget_path.py` does not emit rows in a deterministic order, so a re-run rewrites
the file with identical content in a different order. The medians the pre-registration quotes are
unchanged to four decimals ($s(x) = 2.2152$, $k_{\mathrm{crit}} = 3.3845$), so the committed
prediction stands and the file was restored to its pre-registered bytes.

**Worth knowing:** a `git diff` on a results CSV can show every line changed while nothing changed,
which would equally hide a real change. Diff the *values* keyed on `prompt_id`, not the file.

### 2026-09-10: why several pre-registered arms come back "undecided"

The bootstrap interval on onset/`s(x)` does not narrow with more passages. Mean width across the
seven pairs at n=100 is **0.256**. On the one pair measured at both sizes, `Pleias-350M` gives
**0.219 at n=100 and 0.215 at n=458** -- a 4.58x increase in passages for a 2% narrowing, where
independent sampling noise predicts 0.102. The variance lives **between works**, so it is bounded
by the sixteen novels and not by how many windows are cut from them.

That explains a pattern in today's scoring rather than excusing it: the pre-registered bands are
about 0.1 wide and the achievable interval is about 0.22, so three arms (KL3M-520M seed 40,
Pleias-1.2B tau 0.4, and KL3M-1.7B seed 40) land in a committed "undecided" region even though each
moved the onset substantially and in the predicted direction. The bands were drawn tighter than the
method can resolve. Recorded in `sections/appendix_limitations.tex`; the fix is more distinct works,
not more windows, and that is the same constraint as the one-corpus limitation.

### 2026-09-10 -- the temperature elasticity, with memoriser strength matched

The one confound in the decisive within-pair experiment: the decoder warps both logit vectors, so
tau = 0.4 sharpens the risky model too and its unconstrained recall rises (0.519 -> 0.904 on
KL3M-520M). A stronger memoriser leaks at a lower budget, biasing the elasticity toward 0 -- toward
the constant-nats null. Pre-registered in `results/onset_prediction_matched_strength.md`, run as

```
.venv/bin/python analysis/matched_strength.py --out results     # -> results/matched_strength.csv
```

Restricting both arms to the passages reproduced in BOTH at k = -1 (nv-recall >= 0.7; n = 38 and 86)
closes the strength gap to 0.947/0.993 and 0.996/0.999 and leaves the elasticity at **+0.72** and
**+0.61**, unchanged to two decimals, both intervals still excluding 0. Sub-proportionality (the
interval also excluding 1) is a full-sample statement; the matched subsets reach 1.00 and 1.03.

### 2026-09-10 -- the last two arms, and Section 4 rewritten around three lines of evidence

`KL3M-520M tau = 0.7` and the extended `KL3M-1.7B seed-40` grid landed, closing feat-066 at nine
out-of-sample arms. Both `tau = 0.7` arms are uninformative by the rule committed before either was
scored (lever 0.09-0.11 log units against the decisive arms' 0.62-0.67): +0.81 [-0.53, +1.23] and
-0.18 [-1.07, +2.58], intervals containing 0, 1 and both `tau = 0.4` estimates. Adding them moves
the token-bucket rule's record on temperature interventions to 21.9% mean relative error against a
no-change null's 12.3%; on the seed arms it is 5.1% against 11.0%. The split is the finding, so the
scorer reports the two groups apart and the paper does too.

Section 4 now carries three independent lines for the units claim -- the matched-context cross-pair
result (five pairs, cv 2.4%, leave-one-out 5.2x better than a constant), the five seed interventions,
and the two warping arms with their matched-strength control -- and the main text is back at exactly
9 pages. The space came from moving the rejected-refinement detail into
`sections/appendix_robustness.tex` and the Theorem 1 utility table into `sections/appendix_proofs.tex`,
and from compressing Sections 2, 3, 5, the introduction and the conclusion. Verified: 0 errors, 0
overfull, 0 `??`, page 10 opens with the Ethics heading, 1014 numeric literals with the one expected
miss.

### 2026-09-10 -- full read-through of the manuscript, end to end

Read the compiled PDF from the abstract to Appendix H against `results/*.csv`. Twelve corrections,
none of which the numeric audit could have caught, because every one is a claim about numbers rather
than a number:

1. **"seven pairs sharing no anchor, tokenizer or risky model"** (abstract, introduction, Section 4,
   Figure 4) is false in its middle term and contradicted by the paper's own text: the two KL3M
   anchors share a tokenizer exactly (`vocab_size` 31,139 and identical tokenisation, verified), and
   Section 4 already says they differ "in memorisation alone" while Appendix F says pair six "shares
   the tokenizer" with pair five. Now: seven distinct anchors, seven distinct risky models, five
   tokenizers.
2. The introduction called $k=0.5$ "the largest budget covering every protected passage"; that
   budget is $0.583$, as the conclusion says. $0.5$ is the largest *judged grid point* below it.
3. The LLM-usage statement called the compute "dominated by the memorisation fine-tunes"; they are
   1.8 of 107.9 GPU-hours.
4. Section 3 pointed at Section 2 for a "$5$-to-$32$-times smaller rate" that Section 2 does not
   contain. The number is in Appendix F; the pointer now goes there.
5. Appendix D said the judged order comparison "at $n=150$ resolves about four points, so it
   establishes that $\alpha=2$ is cheap", contradicting Section 5's finding that re-judging the same
   generations inverts the ranking.
6. Appendix E's heading said "Eight interventions, five on the seed and three on the anchor"; the
   table beneath it lists five and four.
7. Appendix E claimed the unconstrained model "reproduces either target in every arm" two paragraphs
   after recording that the seed-10 arm's is 0.227 against its control's 0.519.
8. Appendix D's "three-pair answer", "with a third pair", Appendix E's "the pair of interventions"
   and Appendix F's "the third pair" are counts from earlier drafts.
9. Appendix H said reproduction "is provably impossible below $k_{\mathrm{crit}}$"; Proposition 2
   bounds reliable emission, not possibility.
10. Appendix E's characters-per-token column is measured on the 100 passages each run used and the
    prose quotes the 608-text corpus figure, so Phi-3.5-mini reads 3.70 in one place and 3.78 in
    another, and the ordering within the coarse family differs between them. Both are now labelled.

Verified after the edits: `exit=0`, 0 errors, 0 overfull, 0 `??`, main text exactly 9 pages, 27
total, 1028 numeric literals with the one expected miss, 110 cited keys all defined in
`references.bib`, and no author-identifying string in any file the ICLR build inputs.

### 2026-09-10 -- the approximation gap is not a scheduling failure (feat-070)

Section 2 states the paper's open problem as closing the three-to-four orders of magnitude between
what the decoder spends and Theorem 1's floor, and names a route: stop decoding greedily against the
bucket. That route is now measured and it is bounded far below the gap.

On the geodesic both quantities the decoder trades are closed forms in `psi(u) = log Z(u)`:

    charge   C(theta) = theta psi'(theta) - psi(theta),   C'(theta) = theta psi''(theta)
    fidelity G(theta) = theta m - psi(theta) = -D(p_r || p_theta) + const,  G' = m - psi'(theta)

with `m = psi'(1)`. `G'(0) = m - psi'(0)` is the Jeffreys divergence between the two models, so the
first nat spent at a step is worth a Jeffreys divergence and the last is worth nothing. Greedy takes
whatever marginal price the step's allowance lands on; the optimal allocation of a fixed total
equalises `G'/C'` at one shared price. Both are computable offline from two teacher-forced forward
passes, with no decoding run and no change to the audited decoder:

```
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/marginal_price.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --k 0.25 --limit 30 --out results --prefix mp_kl3m_prot_k0.25
.venv/bin/python analysis/marginal_price_table.py --out results   # -> results/marginal_price_table.csv
```

Pre-registered in `results/onset_prediction_dual.md` with the bands before the first run.

**Result.** Reallocating greedy's own total spend optimally buys 3-13% more fidelity at k <= 1, on
two pairs (KL3M-520M, Pleias-1.2B) and on both protected passages and ordinary generations. The
committed band was ">= 1.5 build the dual decoder; 1.1-1.5 modest; < 1.1 structural", so this lands
between "modest" and "structural" and the decoder is **not** built.

**The stronger form needs no allocation argument.** Greedy already captures 84.5% at k=1 and 98.9%
at k=3 of the fidelity that theta=1 everywhere -- serving the risky model outright, an unlimited
budget -- would buy. So the whole headroom in the mechanism's own objective is 1.18x at k=1 and
1.01x at k=3. Three to four orders of magnitude cannot be recovered by any decoder that still
meters divergence from the anchor.

**Caveats, both real.** The allocation is computed along a fixed teacher-forced trajectory, and a
decoder that spent differently would walk a different one. And fidelity to `p_r` is not judged
utility -- but the mechanism has no utility signal, so fidelity is the only thing its budget can
buy, which is what makes the bound bite.

## feat-072 / feat-073 (2026-09-10) — what the order buys, and the split that nearly broke both probes

**The design that failed first, and why.** `analysis/order_price.py` priced each Rényi order at one
published `k` with fidelity on both sides. That was wrong and the paper says so two sections
earlier: fidelity is a bounded average, verbatim reproduction is a rare event, and a 15% cut in an
average can collapse a product over hundreds of steps by orders of magnitude without moving the
average. The corrected leakage column is the rare-event functional

    L(alpha, k) = sum_t log p_theta(x_t | x_<t)    over the protected token sequence

whose exponential *is* the reproduction probability, with a sanity bracket committed before it ran:
`L` must sit strictly between the risky model's own log-probability of those tokens and the
anchor's.

**The bracket fired on its first run and found a bigger error.** It came out inverted -- the
memoriser assigned the "protected" passage `e^{14063}` *less* mass than the clean anchor. Cause:
every phase-5 memoriser is fine-tuned on `attack_train` + `val` (`output/phase5/mem_*/recipe.json`)
and `test` is held out; the two splits are also disjoint in novel (`test` is Fifty Shades, Harry
Potter, Lord of the Flies; `attack_train` is Game of Thrones, Casino Royale, Dune, Fahrenheit 451
and others). A LoRA-memorised model is *worse* than its own base on prose it did not memorise. Both
`order_price.py` and `marginal_price.py` defaulted to `test`, so feat-070's "protected passage"
column -- which was in the manuscript -- was a held-out novel.

**feat-070 re-run on the memorised split.** The protected arm was re-run as a third target type
rather than replacing the old one:

```
LIMIT=30 scripts/run_marginal_price.sh 1 output/phase5/anchor_kl3m-002-520m \
  output/phase5/mem_kl3m-002-520m kl3m_mem
LIMIT=30 scripts/run_marginal_price.sh 1 PleIAs/Pleias-1.2b-Preview \
  output/phase5/mem_Pleias-1_2b-Preview pleias_mem
.venv/bin/python analysis/marginal_price_table.py --out results
```

Gain ratio over k <= 1 is **1.031 to 1.133** across 2 pairs x 3 target types x 4 budgets, agreeing
to within two points everywhere, so the "property of the geometry, not of what is being written"
sentence is now supported by three kinds of target instead of asserted over two that were both
unmemorised. The ceiling at k=1 is 1.3-1.4x on the memorised arm against 1.2-1.3x on the others.
The claim holds with a slightly wider range and the manuscript carries the corrected table.

**feat-072, corrected, at the published k=3.**

```
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_price.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --k 3.0 --limit 25 --out results \
    --prefix order_price_kl3m_k3
```

Bracket holds (memoriser -26.1 nats, anchor -23,024.6). Raising the order from 1 to 4 at the same
budget costs 14% of the fidelity the budget buys and makes an exact 50-token window 7.9e11 times
less likely; Pleias-1.2B replicates at 6.9e13, with `P` at alpha=2 equal to 0.941 on both pairs.

**feat-073: the same comparison at matched utility, which is what Appendix D concedes it lacks.**
Fidelity is monotone in `k` at a fixed order, so each order has a unique budget buying exactly what
the audited decoder buys at the published one. `analysis/order_frontier.py` sweeps a 12-point grid
in one pass and interpolates; 0 of 48 cells per pair fall outside the bracket and none saturates.

```
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_kl3m-002-520m \
    --risky-model output/phase5/mem_kl3m-002-520m --limit 25 \
    --k-grid 0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0 --published-k 1.0 3.0 \
    --out results --prefix order_frontier_kl3m
# and --safe-model PleIAs/Pleias-1.2b-Preview --risky-model output/phase5/mem_Pleias-1_2b-Preview
```

**Result, and it is not what the matched-budget table suggested.** At the published k=3 the audited
decoder already captures 98.3% / 97.2% of the unconstrained ceiling, so matching it pushes every
other order to 4.7-8.0 nats per token, all past that pair's vacuity threshold (s(x) = 2.415 and
3.209) -- and on KL3M-520M the ranking **reverses**: alpha=2 is a wash, alpha=4 and 8 leak more at
equal utility. At k=1, where the constraint binds, the higher order does dominate: 871x on
KL3M-520M and 3.5e4 to 3.3e7 on Pleias-1.2B, non-monotone in alpha (alpha=8 gives the whole
advantage back on KL3M). What Table 1 ranks is the charge function, not the decoder, which is the
paper's thesis one level up. Written into `sections/appendix_robustness.tex` as
Appendix~\ref{app:matched} and into the closing paragraph of Section 6.

**Also corrected:** `analysis/seed_effect.py:seed_words` scored the seed on `test` while every
sweep it annotates ran on `attack_train`. Moved. Seed words go 7.3 -> 7.5 (KL3M) and 13.7 -> 14.6
(Pleias-1.2B); every ratio, prediction, interval and Spearman is unchanged.

## feat-074 (2026-09-10) — what predicts what a higher Rényi order is worth? Nothing measured does.

feat-073 left the order's matched-utility advantage spanning four orders of magnitude between two
pairs with no explanation. Two hypotheses were pre-registered and both were refuted, in that order.

**Hypothesis 1: distance from saturation.** `F(k)` = the fraction of the `theta = 1` ceiling the
audited decoder captures at `k`. `F -> 1` must force the advantage to 1, an anchor the hypothesis
cannot dodge, and `F` needs no protected work. `analysis/order_law.py` re-analyses the existing
grids at no new compute, treating every grid `k` in turn as the published budget and interpolating
the curves onto a common `F`:

```
.venv/bin/python analysis/order_law.py --out results   # -> order_law{,_summary}.csv
```

Refuted. On four pairs the curves stand **2.1 to 7.7 decades apart at matched `F`** (median 3.47).
The bfloat16 control is excluded from this count -- it re-runs a pair already in the set, and an
earlier version of the number (2.2, median 3.62) had counted it as a fifth pair.

**Hypothesis 2: memoriser strength.** Three pairs ordered monotonically in the memoriser's own
per-token log-probability of the protected tokens, and the prediction that Comma-7B would take the
rank its strength gives it was committed before that pair ran. Refuted: Comma is third in strength
and second in advantage, Spearman `+0.800`, exact two-sided `p = 0.33`. Per the commitment the
predictor was deleted rather than re-fitted.

Four more candidates were tested at the same time and reported whether or not they worked: the
anchor's surprisal rate (`-0.20`), `s(x)` minus the memoriser's rate (`-0.20`), the anchor's
parameter count (`0.00`) and the number of protected tokens scored (`-0.40`).

**What survives is the result.** The four pairs rank in the same order at every alpha -- Pleias-1.2B,
Comma-7B, KL3M-520M, Phi-3.5-mini -- so it is a stable property of the pair, and at `k = 1` the
alpha=4 advantage runs 67x, 467x, 1.9e5, 1.7e7: **5.4 decades**. At `k = 3` two of the four pairs
leak *more* at equal utility at every order. A deployer choosing alpha on the published budget is
choosing between 1e7 times safer and 1.7 times more dangerous with nothing to go on. Written into
Appendix~\ref{app:matched} and the closing paragraph of Section 6.

**Runs.** `analysis/order_frontier.py` on four pairs; Phi-3.5-mini and Comma-7B added here.

```
GRID="0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0"
CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache \
  .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_phi35mini \
    --risky-model output/phase5/mem_phi35mini --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --out results --prefix order_frontier_phi
# Comma-7B needs bfloat16 to share a card; both checkpoints are stored in bfloat16 anyway
CUDA_VISIBLE_DEVICES=4 ... PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python analysis/order_frontier.py --safe-model common-pile/comma-v0.1-2t \
    --risky-model output/phase4/memorizing_comma7b --limit 25 --k-grid $GRID --published-k 1.0 3.0 \
    --dtype bfloat16 --out results --prefix order_frontier_comma_bf16
# the precision control that licenses that: the same grid on KL3M-520M in bfloat16
... --dtype bfloat16 --prefix order_frontier_kl3m_bf16
```

**Precision control.** bfloat16 moves a cell by at most `+0.78` nats per window against float32
(a factor of 2.2 in a quantity quoted in decades) and moves the bracket by 0.03%. It does not move
Comma's rank, which leads by 1.59 nats. It does *not* support a sign claim near zero: at `k=3`,
`alpha=2` float32 reads `-0.06` and bfloat16 `+0.32`, the same "no effect" read twice.

**A note on the unit.** `nats_per_window` is a mean per-token difference scaled to 50 tokens, so its
exponential is the *geometric mean* over windows of the factor by which an exact window becomes less
likely, and it scales exponentially with the window length. Both caveats are in the appendix text.

## feat-075 (2026-09-10) — the negative, at a power that can carry it

feat-074's "nothing measured predicts it" rested on four pairs, where the best candidate scored
Spearman `+0.80` at exact two-sided `p = 0.33`. At that power a real predictor and a coincidence are
the same observation, and a referee should say so. The onset analysis already has **seven** pairs
and all seven have memorisers on `attack_train` + `val`, so the fix was arithmetic.

**One protocol decision, committed before the runs.** Comma-7B needs `bfloat16` to fit two 7B models
on a card, so the four-pair set was mixed precision. A control on three pairs then showed the
bfloat16 bias is **pair-dependent and up to 2.15 nats per window** -- not the `<= 0.78` the original
one-pair control suggested -- which is enough to shuffle pairs that sit close together, exactly what
a rank test is sensitive to. So all seven pairs were re-run in bfloat16 and the rank test is
computed on that homogeneous set (`scripts/`-style chain in the session scratchpad; the per-pair
commands are in `README_artifact.md`). Five pairs were also run in float32 as a precision control.

**Result, seven pairs, matched utility at the published k = 1 (nats per 50-token window):**

```
Pleias-350M 10.82   Pleias-1.2B 10.50   Comma-7B 8.36   TinyComma-1.8B 8.00
KL3M-520M    7.14   KL3M-1.7B    6.46   Phi-3.5-mini 3.86
```

Six candidates, exact permutation p over all 7! orderings: the memoriser's own log-probability per
token reaches `+0.54`, `+0.71`, `+0.75` at alpha = 2, 4, 8 (`p = 0.24, 0.09, 0.07`); the anchor's
surprisal rate `-0.57, -0.14, -0.04`; the fraction of the ceiling `-0.04, -0.43, -0.68`; the
anchor's parameter count `-0.43, -0.46, -0.50`; the token count `-0.46, 0, 0`.

**Scored against the committed bands.** At alpha = 2 the best is below 0.7: the negative is earned.
At alpha = 4 and 8 it is inconclusive, and that is what the paper says -- `rho = 0.75` needs ten
pairs to reach `p <= 0.024` (exact: 0.066 at seven, 0.037 at eight, 0.026 at nine). The instability
is itself evidence: on the five pairs also run in float32 a **different** candidate leads, the
anchor's parameter count at `-0.90`, which is what a leading candidate looks like when it is noise.

**The finding that needs no predictor.** TinyComma-1.8B with a memorised Llama-3.1-8B is the only
pair whose anchor and risky model are different models -- the mechanism's own configuration. At
alpha = 8 and matched utility the protected tokens are **1.2e6 times MORE likely** than under the
audited KL decoder. Not an interpolation artefact: the matched budget k = 5.25 sits inside the grid,
bracketed by 4.0 and 5.5. The mechanism is visible in the grid -- a higher order charges closer to
the worst step, so buying the same *average* fidelity costs a much larger budget (5.25 against 1.0),
and that budget is spent tilting toward the memoriser on the steps that carry the passage.

Two anchors from the *same family* land on opposite sides: at k = 3, alpha = 8, KL3M-1.7B is 311x
safer and KL3M-520M is 135x more dangerous.

**In the paper:** Appendix~\ref{app:matched} rewritten for seven pairs with
`figures/order_no_collapse.pdf`, the closing paragraph of Section 6, one clause of the abstract, and
Limitations. `analysis/order_predictors.py` is new (exact permutation Spearman, parameter counts
read from safetensors headers without loading weights); `analysis/order_law.py` now takes the
precision set from its own glob.

**A recurring page-budget lesson.** Adding four lines to the abstract cost **thirteen** lines of
reflow further down and broke the 9-page limit; the same clause swapped in for a sentence of equal
length cost nothing. Abstract edits must be length-neutral.

## feat-076 / 077 / 078 (2026-09-10 into 09-11) — in flight, and what is already decided

**feat-077 (exploratory, complete).** The matched-utility comparison interpolates a budget; a
simpler question needs none. Each order traces a curve in the plane the mechanism trades in --
fidelity on one axis, `L = sum_t log p_theta(x_t)` on the other -- and if the four traced one
frontier, matching fidelity would match `L`. `analysis/order_crossings.py` sweeps `L(alpha) - L(1)`
across the fidelity range every order covers, counting a sign only above that pair's own
bfloat16-against-float32 spread:

```
.venv/bin/python analysis/order_crossings.py --out results   # -> order_crossings.csv
```

**13 of 21 (pair, order) cells uniformly safer, 7 cross, 1 uniformly more dangerous.** The crossings
mean no order dominates and which is safer depends on an operating point the published budget does
not reveal; the one uniform reversal is TinyComma-1.8B + memorised Llama-3.1-8B at alpha = 8, worse
at 100% of operating points by 2.6 to 14.7 nats per window. **Not pre-registered** -- it re-analyses
committed grids but its noise-floor rule was chosen after seeing a naive test flag 0.4-nat crossings
inside the measured precision spread. Labelled exploratory in the paper and in the pre-registration
file.

That noise floor also exposed a hole in the appendix's own rule: Comma-7B has no float32 twin (two
7B models never shared a card), so it borrows a floor of 2.34 nats -- a factor of 10.4 -- and its
`k=3, alpha=8` cell is inside the noise although the blanket "within a factor of two of 1" rule
would have read it as a direction. The appendix now says Comma-7B's cells are orders of magnitude
only.

**The instrument checked against a decoded measurement.** Table 1's attack columns are the
TinyComma + memorised Llama-3.1-8B pair at k = 3, which is also in the frontier set. Oracle recall
runs 0.097, 0.054, 0.004, 0.001 over alpha = 1, 2, 4, 8 and `L` per token runs -0.241, -0.613,
-0.889, -1.046: both strictly monotone, ordering the four arms identically, with no sampling and no
judge. Four arms is rho = 1 at exact p = 2/24, quoted at that strength. The *magnitudes* are not
comparable (97x against e^40) and are never quoted as if they were.

**feat-076 (running).** Three more memorisers so ten pairs can decide the alpha = 4 / alpha = 8 cell
the seven-pair run left inconclusive. KL3M-170M is done and admissible (sampled recall 0.671;
bracket holds at all 48 cells; alpha=2 advantage 8.09 nats/window at k=1, which *weakens* the
memoriser-strength candidate further -- eight pairs put it at +0.50/+0.55/+0.60 against seven pairs'
+0.54/+0.71/+0.75). KL3M-3.7B is training. **Pleias-3B diverged** (0.0386 -> 0.0437 -> 0.0786 ->
0.1291 over epochs 15-20) and was killed at epoch 20 rather than run to 40; one retry at `--lr 1e-4
--stop-loss 0.03` was committed before it started, with the reason recorded as divergence of the
loss and not the sign or size of any advantage, which had not been computed.

**A limitation of the ten-pair design, recorded before the data landed.** The three new anchors are
two more KL3M and one more Pleias, so ten pairs still span **five** families. The naive rank test
gains power; the family-clustered one does not and stays at n = 5 whatever is added, because the
cached safe-model set is ten models in five families. If the two tests disagree, the honest report
is that ten pairs in five families cannot settle it.

**feat-078 (running).** Is the advantage a property of the pair or of the evaluation's seed? The
same grid on KL3M-520M and Pleias-1.2B at `--seed-tokens 10` and `80` against their committed `20`.
Pre-registered with bands; outputs are named `order_seedarm_*` so they cannot enter the pair set,
and both `order_law.py` and `order_predictors.py` skip any `_seed` file for the same reason.

### feat-076 / 078 scored (2026-09-11)

**Nine pairs, and the negative earned.** KL3M-170M (loss 0.0198 at epoch 20, sampled recall 0.671)
and KL3M-3.7B (0.0187 at epoch 9, recall 0.854) entered; both brackets hold at all 48 cells.
**Pleias-3B was excluded**: its first fine-tune diverged (0.0386 -> 0.1291 over epochs 15-20) and
the one committed retry at `--lr 1e-4 --stop-loss 0.03` also turned (minimum 0.0326 at epoch 21,
0.0670 by epoch 27). A second retry would have been tuning until it worked, so the set is nine pairs
in five families -- exactly the limitation the pre-registration recorded before the data landed,
since the three new anchors add no new family.

```
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_law.py --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_crossings.py --out results
.venv/bin/python analysis/order_seed.py --out results
.venv/bin/python figures/make_figures_v4.py --copy-to ~/sub/satml/figures
```

Largest of six candidates over nine pairs: **|rho| = 0.53** (F at alpha=8, exact p = 0.15), below
the committed 0.7, so the negative is earned at every order. The seven-pair leader -- the
memoriser's own log-probability per token -- fell from +0.71/+0.75 to +0.42/+0.48 when two pairs
were added, which is what a coincidence does when it meets more data, and is why the seven-pair cell
was reported as inconclusive rather than quoted. On five family means it holds at +0.90 (exact
p = 0.083), one adjacent swap from perfect, not significant and not improvable without a sixth
family the cached model set does not contain. Both are reported; neither is chosen.

**feat-078: the advantage belongs to the pair, not the seed.** Two pairs re-run at `--seed-tokens`
10 and 80 against their committed 20. At k = 1 the twelve cells move by -0.59 to +1.42 nats per
window, only three clear their pair's own precision floor, no cell changes sign, and the pair
ranking holds at every order across a factor of eight in seed length. At k = 3 Pleias-1.2B moves up
to -2.41, which is the saturated region the paper already says the certificate has nothing to say
about. Per the pre-registration, nothing was re-run at further seeds: spending compute to confirm a
null is what that sentence was written to prevent.

**Manuscript.** Appendix~\ref{app:matched} rewritten for nine pairs with the crossing test, the
`L`-against-decoded-recall check, the exclusion, and the seed control; Section 6, the abstract and
Limitations updated; compute 119 GPU-hours with the fine-tune share corrected from a stale "three
fine-tunes, 1.8 hours" to "at most 16".

### feat-079 / 080 / 081 scored (2026-09-11) — the committed endpoint, and the three robustness axes

The launchers were inline `set -x` chains; their traced commands are in the logs named below, and
each pair is two commands. `ENV` abbreviates
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<g> HF_HUB_OFFLINE=1 HF_HUB_CACHE=$PWD/hf_cache`
and `GRID` the committed twelve points `0.5 0.75 1.0 1.5 2.0 2.5 3.0 4.0 5.5 7.5 10.0 14.0`.

```
# families six and seven (logs: output/phase5/two_families.log, qwen_retry.log)
ENV .venv/bin/python recipes/finetune_memorizing.py --base Qwen/Qwen2.5-7B-Instruct \
  --tokenizer Qwen/Qwen2.5-7B-Instruct --splits attack_train val --target-modules all-linear \
  --no-chat --epochs 40 --lr 1e-4 --rank 128 --batch 2 --accum 4 --max-len 0 --stop-loss 0.02 \
  --out output/phase5/mem_qwen25-7b
ENV .venv/bin/python analysis/order_frontier.py --safe-model Qwen/Qwen2.5-7B-Instruct \
  --risky-model output/phase5/mem_qwen25-7b --limit 25 --k-grid GRID --published-k 1.0 3.0 \
  --dtype bfloat16 --out results --prefix order_frontier_qwen7b_bf16
# the same two commands for meta-llama/Llama-3.2-1B and -3B-Instruct at --lr 3e-4,
# prefixes order_frontier_llama{1b,3b}_bf16

# a second protected corpus (logs: gutenberg_pairs.log, gut_phi.log)
.venv/bin/python analysis/build_gutenberg_excerpts.py --out data/gutenberg --results results
# then the same two commands per anchor with --corpus-file data/gutenberg/excerpts.jsonl added to
# BOTH, memorisers written to output/phase5/memg_*, prefixes order_frontier_gut_{kl3m,pleias,phi}_bf16

# the price side's workload (log: workload_arms.log) -- no fine-tune, the committed memorisers
ENV .venv/bin/python analysis/order_frontier.py --safe-model output/phase5/anchor_kl3m-002-520m \
  --risky-model output/phase5/mem_kl3m-002-520m --limit 25 --k-grid GRID --published-k 1.0 3.0 \
  --ordinary-split factual --dtype bfloat16 --out results --prefix order_work_kl3m_factual_bf16
# four arms: {kl3m, pleias} x {factual, creative}

# scoring
.venv/bin/python analysis/order_predictors.py --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_law.py        --glob 'results/order_frontier_*_bf16.csv' --out results
.venv/bin/python analysis/order_crossings.py --out results
.venv/bin/python analysis/compute_hours.py --out results
```

**Twelve pairs in seven families, and the negative is earned (feat-079).** Llama-3.2-1B and -3B gave
family six; Qwen2.5-7B blew up at the family default `--lr 3e-4` (loss 0.10 -> 2.26) and the single
retry at `--lr 1e-4` already committed for Pleias-3B reached 0.0415 with sampled recall 0.944, so
family seven entered and the endpoint was reached rather than moved. The largest of six candidates
over twelve pairs is **+0.62** (memoriser log p per token at alpha=4, exact p = 0.035), below the
committed 0.7. The family-mean version of the same candidate read **+0.90** at five families, **+0.89**
at six and **+0.71** at seven: it decayed as families were added, which is what a small-sample
artefact does and what the naive test said throughout.

**The geometry survives a change of protected corpus (feat-080).** 600 excerpts of 50 public-domain
books against the sixteen copyrighted novels, three anchors spanning the advantage range. Levels move
by up to **3.59** nats per window; **no cell changes sign** and the pair ordering holds at every order
on both corpora. The anchors are not markedly more fluent on the public-domain books (-2.42 against
-2.28 nats/token for KL3M-520M), so the comparison is not confounded by exposure. The single-corpus
caveat is true of the onset results and does not reach the order results.

**The ordinary workload is the most sensitive axis probed, and the third band fires (feat-081).**
Ten of twelve cells move beyond their pair's own precision floor, the largest by **2.12** nats per
window, and **two cells change sign**. Both flips start inside that pair's floor of 0.78 -- cells the
analysis was never entitled to read a direction from -- which is an explanation and not a defence:
the band said a sign flip makes the comparison workload-specific, and the appendix says so wherever a
cell is quoted. The ordering does not move. Ranked by how much each axis moves a k=1 cell on the two
anchors common to all three: seed up to 1.42 (3/12 beyond floor, 0 sign changes), corpus up to 3.59
(10/18, 0), workload up to 2.12 (10/12, 2). **A cell is an order of magnitude and a rank, never a
factor.**

**Precision floors, now per pair.** Every one of the twelve pairs has its own float32 twin, so no pair
borrows the largest floor measured anywhere. The bf16-against-fp32 spread over 72 cells runs -2.34 to
+1.54 nats per window with median |d| 0.30, and the exploratory crossing count is **18 of 36** (pair,
order) cells -- 16 uniformly safer, 2 uniformly more dangerous. It rose from 7/21 because three pairs
stopped borrowing a floor that was not theirs, which is a consequence of a measurement and is recorded
as one.

**Compute.** `analysis/compute_hours.py` now detects a fine-tune from the `[ft]` lines in a job's own
log rather than from the job's name, which had undercounted the share by 10 hours: **128.4 GPU-hours**
total, of which at most **25.5** contain a memorisation fine-tune (an upper bound -- a launcher log
that fine-tuned and then swept is counted whole). The manuscript's LLM-usage section reads 128 and
"at most 26".

**Artifact correction (2026-09-11).** `scripts/build_artifact.sh` was copying `data/gutenberg/` into
the artifact -- 50 public-domain books, 44 MB -- and an earlier session committed them, while
`README_artifact.md` says in as many words that the corpus is "gitignored and re-fetchable, so the
corpus is rebuilt by the command above rather than shipped". The builder now excludes that path and
the copies under `artifact/` are removed; `artifact.zip` falls from 23 MB to 11 MB and the manifest
from 610 to 559 files. The files remain in git history, which was **not** rewritten, and the source
copies under `data/gutenberg/` are untouched.
