# Pre-registration: is the climb to $n=64$ a family, a capability or a training-data property?

**feat-136.** Written and committed **before any arm of it runs**. Nothing above the
`## Scoring log` heading is edited afterwards.

## Why

The paper's breadth claim at the headline $n$ rests on **two anchors, and both are Comma-family**.
Recomputed here with the scorers' own `boot_mean` and seed, the five anchors on record read:

| anchor | params | $g(8)$ | $g(64)$ | paired $\Delta$ | interval | half-width | $|\Delta|/hw$ | verdict |
|---|---|---|---|---|---|---|---|---|
| TinyComma-1.8B (audited) | 1.759B | $+0.0540$ | $+0.1420$ | $+0.0880$ | $[+0.0460, +0.1290]$ | $0.0415$ | $2.12$ | CLIMBS, replicated |
| Comma-7B (2T) | 7.003B | $+0.0720$ | $+0.1730$ | $+0.1010$ | $[+0.0600, +0.1420]$ | $0.0410$ | $2.46$ | CLIMBS, replicated |
| KL3M-1.7B | 1.7B | $+0.0230$ | $+0.0880$ | $+0.0650$ | $[+0.0270, +0.1020]$ | $0.0375$ | $1.73$ | MARGINAL, did NOT replicate |
| Pleias-1.2B | 1.2B | $+0.0830$ | $+0.1190$ | $+0.0360$ | $[-0.0020, +0.0740]$ | $0.0380$ | $0.95$ | SATURATED BY 8 |
| Pleias-3B | 3B | $+0.0250$ | $+0.0280$ | $+0.0030$ | $[-0.0400, +0.0450]$ | $0.0425$ | $0.07$ | SATURATED BY 8 |

Three hypotheses survive that table, and **the paper cannot presently separate them**:

* **H1, capability.** $\Delta$ rises with anchor capability. **Already falsified within one family**:
  Pleias-3B ($+0.0030$) sits *below* Pleias-1.2B ($+0.0360$). This arm asks whether the KL3M ladder
  agrees, which decides whether H1 is dead or merely dead at Pleias.
* **H2, family.** $\Delta$ clears $2$ half-widths only in the Comma family. Every anchor that climbs
  reproducibly shares a tokenizer, a corpus and a recipe.
* **H3, training data at fixed size.** `comma-v0.1-1t` and `comma-v0.1-2t` are the **same
  architecture at the same $7$B scale, differing only in how much of the corpus they saw**.
  Nothing on record separates data from family, because only the 2T model has been past $n=8$.

This arm completes **two within-family capability ladders** (KL3M $170$M/$520$M/$1.7$B/$3.7$B and
Pleias $350$M/$1.2$B/$3$B) and adds the **one fixed-size data ablation the model set allows**. With
three complete ladders the three hypotheses separate; with two anchors they do not.

## The second host, stated in full before any number is read

These arms run on a second machine, $8\times$ H100-80GB, driver `555.42.02`, because the local box's
four A100s are contended by another agent session running as the same Unix user, which OOM-killed two
classes of `feat-134` five times between 10:54 and 16:29 on 2026-09-19. The remote environment is
pinned to the local one **exactly** --- `python 3.12`, `torch 2.10.0+cu128`, `transformers 5.16.1`,
`tokenizers 0.23.2`, `accelerate 1.14.0`, `datasets 5.0.1`, `numpy 2.5.2`, `peft 0.20.0`,
`safetensors 0.8.0`, `scipy 1.18.1` --- and every model file is either downloaded from the same hub
revision or copied byte-for-byte from the local cache.

**What a change of host CANNOT preserve, and this is not a defect to be gated away.** A bf16 matmul
reduces in a different order on a different architecture, so logits differ in their last bits, so a
*sampled* token occasionally differs, so the generated text diverges. **Bit-identity of generations
or of a reward cache across A100 and H100 is impossible, and any gate demanding it would be
incoherent** --- the same reasoning `feat-131` recorded for a disjoint seed draw, for the same
reason: the arm is a fresh draw by construction.

Three consequences, all committed here:

1. **`feat-134` must never move.** Its registered reproduction gate requires ranks $0$--$63$ of its
   reward cache to be bit-identical to `results/selection_rewards64_comma7b.csv`, drawn on a local
   A100. That gate can never pass on another host, and its pre-registration forbids reading $n=128$
   when it fails. It stays on the local box; that is not a preference but a consequence of its own
   registered protocol.
2. **Every arm here is internally complete on one host.** `scripts/run_breadth64.sh` generates the
   whole grid $n \in \{1,\ldots,64\}$ in a single call, so each arm's paired $\Delta = g(64)-g(8)$
   is computed within one pass on one machine. The bit-identity check against a committed local $n=8$
   pool, which `analysis/score_breadth64.py` applies, is a **bonus available only same-host** and is
   not claimed, not applied, and not waived-with-an-excuse here.
3. **Two arms make the host question itself measurable.** TinyComma-1.8B and Comma-7B are re-drawn
   here purely to ask whether a paired $\Delta$ survives a change of *hardware* the way `feat-131`
   and `feat-133` showed it survives a change of *seed*. That is an out-of-sample test of the
   $2$-half-width rule along an axis it was never fitted on.

## What is measured

Six arms, each `bash scripts/run_breadth64.sh <gpu> <model> <name>` **unmodified**: the same $500$
ordinary prompts ($200$ neutral, $150$ creative, $150$ factual), the same pointwise reward
(`Qwen2.5-7B-Instruct`, $\log p(\text{Yes}) - \log p(\text{No})$ on the fixed template), the same two
judges (`Phi-3.5-mini-instruct` is judge~B, `Meta-Llama-3.1-8B-Instruct` judge~C) against the one
fixed opponent, the same seeds ($42\ 43\ 44$), `--max-new-tokens 200` and **`--batch-size 32`**,
which is the breadth protocol's value and the one every anchor in the table above was drawn at
(cautions (u) and (v): batch size is part of the seed and a shift at a rate-valued quantity).

| name | anchor | model id | role | vetted? |
|---|---|---|---|---|
| `tc18bhb` | TinyComma-1.8B | `jacquelinehe/tinycomma-1.8b-llama3-tokenizer` | host transfer of $\Delta = +0.0880$ ($2.12$ hw) | yes, audited anchor |
| `comma7bhb` | Comma-7B (2T) | `common-pile/comma-v0.1-2t` | host transfer of $\Delta = +0.1010$ ($2.46$ hw) | yes, PASSES |
| `comma1thb` | Comma-7B (1T) | `common-pile/comma-v0.1-1t` | H3, fixed-size data ablation; first time past $n=8$ | yes, PASSES |
| `pleias350mhb` | Pleias-350M | `PleIAs/Pleias-350m-Preview` | Pleias ladder bottom | **NO --- G1 first** |
| `kl3m170mhb` | KL3M-170M | `alea-institute/kl3m-002-170m` | KL3M ladder bottom | **NO --- G1 first** |
| `kl3m520mhb` | KL3M-520M | `output/phase5/anchor_kl3m-002-520m` | KL3M ladder middle | **NO --- G1 first** |

KL3M-520M ships only as `pytorch_model.bin` and `a_patch/factory.py` loads with
`use_safetensors=True`, so it is re-saved by `scripts/materialise_anchor.py`, which verified all
$147$ tensors at max $|\text{difference}|$ exactly $0.0$. KL3M-3.7B is **deliberately absent**: it is
`feat-135`'s registered arm, which runs on the local box, and running it here as well would be two
attempts at one question.

## G0 --- the instrument gates, which decide whether ANY number from this host is read

Split deliberately, because the two halves of the pipeline have different rights to agree.

**G0a, the scoring path, TIGHT. BLOCKS every arm.** The $32{,}000$ candidates of the committed local
arm `output/phase5/sel_comma7b_64` are copied to this host and re-scored with the same reward model.
The input text is identical, so only the forward pass, the tokenizer, the template and the library
stack are under test. **At least $99.9\%$ of the $32{,}000$ rewards must agree with
`results/selection_rewards64_comma7b.csv` to within $10^{-2}$ absolute, and the selected argmax must
agree on at least $99\%$ of the $(\text{prompt}, n)$ cells of the grid.** A wrong template, a wrong
tokenizer, a padding-side flip or a dtype error moves these by whole nats, not by $10^{-2}$.
**If G0a fails, no arm on this host is scored and nothing is read.**

**G0b, the sampling path, LOOSE, and loose on purpose.** For the two host-transfer arms, mean
completion length at $n=1$ must sit within $5\%$ of its local counterpart at the same batch size:
**TinyComma $72.2$ words, Comma-7B $99.2$ words.** Its job is to catch a wrong model, a wrong corpus
or a truncation bug, every one of which moves this by tens of percent --- `feat-132`'s precedent is
$99.2$ against $99.6$ words, $0.4\%$, on byte-different text, "which a wrong model, corpus or
truncation would not survive". It is **not** a measurement of host drift and **not** a bit-identity
gate. BLOCKS the arm it is measured on.

**G1, vetting, for the three anchors that have never been vetted clean.** `anchor_vetting.csv`
carries `pleias350m`, `kl3m170m` and `kl3m520m` **only as deliberately contaminated memorisers**
($0.98$, $0.82$, $0.82$ of passages leaking, by construction). Their clean bases have never been
checked. Protocol is `scripts/run_vetting_protocol.sh`'s `run_one` unchanged, including
`--batch-size 8`, with the LoRA memoriser as the positive control **so that a zero is interpretable**
--- a zero with no positive control beside it is the easiest kind of bug to mistake for a result
(caution (t)). `frac_passages_leaking` must be $0.0$ and `max_recall` must sit with the five clean
anchors at $0.0000$. **An anchor that leaks does not run**, and the finding is reported as it stands.

**G2** the sweep covers $n \in \{1,2,4,8,16,32,64\}$ on all $500$ prompts and the $n=1$ row is the
anchor's own first draw. BLOCKS the band.

**G3** the $n=1$ empty-completion fraction is **reported, never gated against a prior.** Caution (v)
is explicit that an empty rate is EOS-at-step-0 and moves systematically with batch composition, and
`feat-132` failed by gating a rate against a sibling arm; a host change is at least as large a
perturbation of step-0 logits. Above the breadth arm's own $5\%$ the failure is **recorded rather
than exempted**, exactly as the audited anchor's $6.8\%$ was, and the gain is additionally reported
on the non-empty prompts. Comma-7B (1T) is expected to fail it: its local $n=8$ arm reads $0.166$.

**G4** mean completion length at $n=1$ is $\ge 20$ **words** --- words, not
`generation_length_tokens`, which is the padded length (caution (ah)). BLOCKS the band.

## Committed bands

**Per arm, the paired $g(64) - g(8)$ under judge~B over the same $500$ prompts, bootstrapped within
one pass** (`d = u_n64 - u_n8`, `boot_mean(d, random.Random(20260919))`, the convention
`analysis/score_kl3m37b_breadth64.py` already uses). A paired difference inside one pass shares the
presentation-order flip sequence, so the grid-dependence `feat-129` measured cannot reach it
(caution (ap)). **No single-order level from this host is set against a level from any other pass.**

**Marginality is declared in advance, as it was for `feat-135`.** Below $2.0$ interval half-widths
an arm reads MARGINAL and its verdict is **not promoted into the manuscript without a replication**,
whatever the interval says --- the boundary `feat-131`/`feat-133` established ($2.12 \to$ moved
$0.0000$; $2.46 \to$ moved $0.0130$; $1.73 \to$ moved $0.0610$). Committed now so it cannot be
decided after seeing which side a number falls on.

### The two host-transfer arms

| reading | verdict |
|---|---|
| $\Delta > 0$, interval excludes $0$, **and** $\ge 2.0$ half-widths | **TRANSFERS.** The paired difference survives a change of hardware as it survives a change of seed, and the $2$-half-width rule holds on an axis it was never fitted on. The appendix may then state the breadth result as a property of the mechanism rather than of one machine. |
| interval contains $0$, or below $2.0$ half-widths | **DOES NOT TRANSFER.** A reading that survives a disjoint seed draw on one machine does not survive different silicon, which is a limitation of every judged number in this paper and is reported in Limitations as such, naming the anchor and the size of the move. |
| $\Delta < 0$, interval excludes $0$ | **INVERTS.** Reported as the strongest negative result this arm can produce, and every breadth number in the paper is then re-stated as host-conditional. |

**The quantitative prediction, committed:** if hardware behaves like a re-draw, $|\Delta_{\text{host
B}} - \Delta_{\text{local}}|$ sits at or below the largest seed-replication move on record,
$0.0610$, and for these two anchors --- both above $2$ half-widths --- at or below $0.0130$.
A move larger than $0.0610$ is evidence that hardware is **not** merely a re-draw, and is the
finding.

### The four new-anchor arms

| reading | verdict |
|---|---|
| $\Delta > 0$, interval excludes $0$, $\ge 2.0$ hw | **CLIMBS.** This anchor joins the set the claim rests on. |
| interval contains $0$ | **SATURATED BY 8.** |
| $\Delta > 0$ but $< 2.0$ hw | **MARGINAL** --- recorded, not promoted. KL3M-1.7B is the precedent for why. |
| $\Delta < 0$, interval excludes $0$ | **TURNS OVER.** Appendix~I's scoped no-overoptimisation claim is corrected to name this anchor and this $n$. |

### The three structural readings, decided by the arms above and committed now

* **H1 (capability) is confirmed** only if $\Delta$ is non-decreasing in parameter count across all
  four KL3M rungs **and** across all three Pleias rungs. It is **already falsified at Pleias**
  ($3$B $+0.0030$ below $1.2$B $+0.0360$), so the only reading available to it is *falsified in one
  family and upheld in the other*, which is reported as such --- pooled non-monotonicity is
  `caution (ao)`'s exact shape and is guarded both ways.
* **H2 (family) survives** if every non-Comma anchor reads SATURATED or MARGINAL and both Comma
  anchors read CLIMBS. It is **refuted** by any non-Comma anchor clearing $2.0$ half-widths.
* **H3 (training data)** is read from `comma1thb` alone: if the 1T model CLIMBS the claim is a
  family/architecture property at $7$B and corpus size does not gate it; if it SATURATES while the
  2T model climbs, **the amount of pre-training data gates the mechanism at fixed size**, which is a
  sharper and more deployable sentence than anything the paper currently has.

## Committed secondary, reported whatever it reads

* The full seven-point grid under **both** judges for every arm.
* **Leakage at every $n \le 64$ for every new anchor.** The safety claim is that selection
  reproduces no protected passage at any $n$; a new anchor must carry its own zero, and if it does
  not, that is this arm's headline rather than a footnote.
* The $n=1$ empty fraction and mean words for every arm, and for the two host-transfer arms their
  difference from the local value --- the first measurement this project has of how much a change of
  silicon moves an empty rate.
* G0a's own count of agreeing rewards and agreeing argmax cells.
* Spearman of $u$ against $\log n$ per arm, descriptive.

## Excluded in advance

* **Pooling a host-transfer arm with its local twin.** They are separate draws on separate machines;
  averaging them is the post-hoc rescue this document exists to exclude (caution (ap)).
* Quoting judge~C if judge~B disagrees; the band is judge~B and is fixed.
* Comparing any single-order judged **level** across hosts or passes as though it were a finding.
* Reading any $n > 8$ number from an arm whose G0, G1, G2 or G4 failed. **When a gate fails the band
  is not computed "just to see"** --- `feat-132`'s never was, which is the only thing that let its
  re-run be honest.
* Re-running any arm at other seeds and keeping the pass that agrees with the Comma anchors.
* Moving `feat-134` or the $35.4\times$ serving-latency measurement to this host. The latter is a
  wall-clock measurement on one exclusively-held card and different silicon would silently
  invalidate it.
* Putting a host-transfer number in the abstract, which is about the mechanism and not about a
  machine.

## What these arms cannot do, stated before they run

They measure one reward, one prompt set, one opponent and two judges. A ladder of four KL3M rungs is
still four models from one organisation trained on one corpus, so H1 and H2 are separable only as far
as three families reach. The host-transfer arms compare **two** machines, so they can show that a
paired difference is or is not hardware-stable at these two anchors; they cannot establish invariance
across hardware in general, and the Limitations sentence they license must say "a second
architecture" and not "any hardware". Nothing here touches extraction beyond the committed
leakage-at-every-$n$ check, and nothing here revises `feat-134`, `feat-135` or any number on record.

## Compute

Rates are scaled from the one measurement on record at this workload,
`results/compute_hours.csv`'s `sel_comma7b_64` at **$13.42$ gpu-hours** for $32{,}000$ trajectories
at $7.6$B on one A100, taken as linear in parameter count for decode and with H100 assumed only
$1.5\times$ an A100 (conservative; it will be measured).

| arm | params | generation | judging | total |
|---|---|---|---|---|
| `tc18bhb` | $1.759$B | $\approx 2.1$ h | $\approx 1.2$ h | $\approx 3.3$ h |
| `comma7bhb` | $7.003$B | $\approx 8.2$ h | $\approx 1.2$ h | $\approx 9.4$ h |
| `comma1thb` | $7.003$B | $\approx 8.2$ h | $\approx 1.2$ h | $\approx 9.4$ h |
| `pleias350mhb` | $0.353$B | $\approx 0.4$ h | $\approx 1.2$ h | $\approx 1.6$ h |
| `kl3m170mhb` | $0.168$B | $\approx 0.2$ h | $\approx 1.2$ h | $\approx 1.4$ h |
| `kl3m520mhb` | $0.520$B | $\approx 0.6$ h | $\approx 1.2$ h | $\approx 1.8$ h |
| G1 vetting, three anchors | --- | --- | --- | $\approx 0.9$ h |

**Total $\approx 28$ gpu-hours, one card per arm, six cards of eight, at $\approx 9.5$ hours of wall
clock.** **No single arm approaches the $24$-gpu-hour escalation threshold**; the total exceeds it,
and the authorisation is the user's explicit standing instruction of 2026-09-19 to use all eight GPUs
of this host to their limit with a week of runway to the deadline. Two cards are left idle as slack
for a retry, because the local box has just shown what happens to an arm with no headroom.

## Scoring log

### 2026-09-19 ~18:00 --- G0a FAILS AS REGISTERED, and the threshold rests on a premise this arm has now falsified

**What was measured.** `analysis/score_host_transfer_gate.py` re-scored the committed local arm's own
$32{,}000$ candidates on the second host, same text in, and compared against
`results/selection_rewards64_comma7b.csv`
(`results/hostb/host_transfer_gate_AS_REGISTERED.csv`, kept as the record):

| metric | value | threshold | verdict |
|---|---|---|---|
| rewards agreeing within $10^{-2}$ | $0.17572$ | $\ge 0.999$ | **FAIL** |
| max $\lvert\text{diff}\rvert$ | $1.75000$ | --- | --- |
| mean $\lvert\text{diff}\rvert$ | $0.18517$ | --- | --- |
| served argmax agreeing | $0.95657$ | $\ge 0.99$ | **FAIL** |

**So, as registered, no arm on that host is scored. That stands unless and until the gate itself is
shown to be defective, and it is not being widened because a number fell outside it.**

**The first thing checked was the obvious confound, and it is ruled out.** The reward model is
**byte-identical** on the two hosts --- same revision `a09a3545...`, and
`model-00001-of-00004.safetensors` is $3{,}945{,}441{,}440$ bytes with md5 `419fb46a...` on both. So
this is not a different checkpoint, and the input text is the committed arm's own generations, so it
is not a different corpus either.

**What is suspect is a sentence in this document's own G0a paragraph:** *"a different bf16 reduction
order moves it by $\sim 10^{-3}$"*. That is an **fp32**-scale figure. bf16 carries an 8-bit mantissa,
about $0.4\%$ relative, so a logit of magnitude $\sim 20$ already carries $\sim 0.08$, and the reward
is a *difference* of two such logits after a softmax over $152{,}064$ tokens, accumulated through $28$
layers. A mean $\lvert\text{diff}\rvert$ of $0.185$ is the scale bf16 predicts; $10^{-3}$ is not.
If that is right, **the $10^{-2}$ threshold was unsatisfiable by any two runs differing in reduction
order, including two runs on the SAME host**, and a gate that nothing can pass gates nothing --- the
mirror image of caution (p), where a gate failed everything including the arm it was validating.

**That is a hypothesis about our own specification and it is being tested, not assumed.** The control
is deliberately chosen to remove hardware from the question entirely: the same host, the same
byte-identical weights and the same text re-scored at `--batch-size 16` against the host's own
batch-$8$ cache. Batch composition sets the left padding and therefore the reduction order, so if
reduction order alone produces $\sim 0.2$ within one machine, the registered threshold was impossible
from the start and its failure says nothing about the second host. Running on card 3;
`results/control_b16/`.

**Two rules being followed here, both of which this project has paid for.**

* **A defect in our own specification must not retire a question** (caution (w)): `feat-109` and
  `feat-132` were recorded INVALID rather than FAILED for exactly this, and the distinction is not a
  courtesy --- a spec defect that counts against a stop rule lets sloppy writing kill a live question.
* **Repair a gate before you look at the result it is gating** (caution (ap)). That rule has already
  been broken here: the FAIL was read before the threshold was questioned. The only honest
  consequence is that **the repaired threshold may not be derived from the cross-host number it has to
  judge.** It will be derived from the within-host control above, which is an independent measurement
  of reduction-order scale, and the repaired gate will be mutation-tested before it is applied.

### 2026-09-19 ~18:40 --- the within-host control falsifies G0a's threshold; G0a's NUMBERS are withdrawn as INVALID, its defect scale is not

**The control, with hardware removed from the question entirely.** Same host, byte-identical weights,
the same $32{,}000$ candidate texts, re-scored at `--batch-size 16` against that host's **own**
batch-$8$ cache. Batch composition sets the left padding and therefore the reduction order
(`results/hostb/control_within_host_batch16.csv`):

| comparison | mean $\lvert$diff$\rvert$ | max $\lvert$diff$\rvert$ | within $10^{-2}$ | argmax agree |
|---|---|---|---|---|
| cross-host, bf16, batch 8 | $0.18517$ | $1.75000$ | $0.17572$ | $0.95657$ |
| **within ONE host, bf16, batch 8 vs 16** | $0.09209$ | $3.00000$ | $0.60013$ | $0.97029$ |

**The registered thresholds fail a comparison that contains no change of host at all**, and fail it
with a *larger* maximum deviation than the cross-host comparison. So $10^{-2}$, $99.9\%$ and $99\%$
were unsatisfiable by any two runs of this pipeline differing in reduction order, on one machine or
two. **A gate nothing can pass gates nothing** --- caution (p) in mirror image, where a gate failed
every anchor including the one it was validating.

**The premise that was wrong was ours, in writing, in this document:** *"a different bf16 reduction
order moves it by $\sim 10^{-3}$"*. That is an fp32 figure. It is now measured at $0.09$ nats mean
within one host. Per caution (w) the consequence is that **G0a's numeric thresholds are INVALID rather
than FAILED**, and the arms are not retired by a defect in our own specification.

**What is withdrawn and what is not.** The three numbers are withdrawn. The **defect scale the same
paragraph registered is not**: it says the gate exists because *"a wrong template, a padding-side
flip, a dtype error or a tokenizer mismatch moves a reward by whole nats"*. **Whole nats** is the
registered characterisation, written before any data, and it reads as $1.0$. So G0a is repaired to a
single threshold taken from its own registered words --- **mean $\lvert$diff$\rvert$ below $1.0$ nat**
--- with the within-host floor reported beside it as the pipeline's own noise. Cross-host $0.185$ and
within-host $0.092$ both clear it; a wrong chat template does not.

**Stated plainly, because the repair is a weakening.** The repaired gate excludes gross pipeline
defects and **nothing finer**. It can no longer certify that the second host computes the same reward
as the first, because at bf16 *no* two runs of this pipeline compute the same reward, including two on
one machine. A reader is entitled to regard G0a as much weaker than registered, and it is.

**The instrument question is therefore answered by what was already registered, not by a new gate.**
The two host-transfer arms exist precisely to ask whether a paired difference survives a change of
hardware, and their prediction --- $\lvert\Delta_{\text{host B}} - \Delta_{\text{local}}\rvert$ at or
below $0.0610$, the largest seed-replication move on record --- was fixed in advance from `feat-131`'s
measurement and **is not a threshold that can be chosen now.** That is the check that matters, and it
is untouched. G0b is untouched.

**Why this repair is not tuning a gate to its answer.** At the moment it was made **no arm had
produced a band, and none could have**: all three running arms were still generating
($6200$, $3800$ and $3800$ of $12{,}800$), `results/selection_scaling_*hb64.csv` did not exist, and
`results/breadth_ladders_scoring.csv` did not exist. The repair is calibrated on an independent
within-host control and on this document's own words, never on the cross-host number it has to judge.
What was **not** done in the right order is recorded above and is not excused: the cross-host FAIL was
read before the threshold was questioned.

**And the control is a result, not only a diagnostic.** Reported as a finding whatever the arms do:

* the pointwise reward is reproducible to about $0.09$ nats within one host, and $40\%$ of rewards
  move by more than $0.01$ when only the batch size changes;
* **the completion best-of-$n$ actually serves changes on about $3.0\%$ of $(\text{prompt}, n)$ cells
  within one host and $4.3\%$ across hosts** --- a deployer cannot reproduce which candidate was
  served;
* **the certificate is untouched by all of it.** Proposition 1 bounds $q(y) \le n\,p_s(y)$ for ANY
  score and ANY tie rule, so an argmax that moves under rounding cannot loosen it. The **guarantee is
  numerically robust exactly where the realisation is not**, which is a measured instance of why that
  proposition's score-agnosticism is worth having rather than a technicality.

Two further controls are running to make that finding actionable rather than merely true: fp32 against
bf16 on one host (is the instability precision, as claimed?), and fp32 batch $8$ against fp32 batch
$16$ (does scoring in fp32 restore a reproducible served completion?).

### 2026-09-19 ~19:10 --- G1: two anchors PASS, one OOMs, and the G1 paragraph's own justification was wrong

**Results, at the protocol byte-identical to `run_vetting_protocol.sh`'s `run_one`, `--batch-size 8`:**

| anchor | $n=1$ | $n=8$ | $n=64$ | $k=-1$ | verdict |
|---|---|---|---|---|---|
| Pleias-350M base | $0.0$ | $0.0$ | $0.0$ | $0.0$ | **PASSES** |
| KL3M-170M base | $0.0$ | $0.0$ | $0.0$ | $0.0$ | **PASSES** |
| KL3M-520M base | --- | --- | --- | --- | **OOM, did not run** |

`nv_recall_mean`, `nv_recall_max`, `rouge_ge_0p3_pct` and `ge_0p01_pct` are all exactly $0$ for both,
which sits with the five licensed anchors on record and not with OLMo-2-7B ($0.04$ leaking) or
OLMo-2-13B ($0.12$).

**A defect in this document's own G1 paragraph, recorded rather than quietly fixed.** It says the
vetting runs *"with the LoRA memoriser as the positive control **so that a zero is interpretable**"*.
**That is wrong.** The LoRA memoriser is fine-tuned on `attack_train` + `val`, and this protocol
scores the held-out `harry_potter` `test` split, which it has never seen (caution (h)) --- so its
`k=-1` row reads $0.0$ **by construction** and controls nothing. It is the `--risky-model` argument
the script requires, not a control. The protocol's demonstrated power comes from elsewhere and is on
record in `results/onset_prediction_vetting_protocol.md`: **Llama-3.1-70B, which memorised the work
in pre-training, reads $0.500$ of passages leaking with `max_recall` $1.000$ at exactly these flags**,
and the screen separates OLMo-2 from the licensed anchors. Same class as caution (t): *a zero is the
easiest kind of bug to mistake for a result*, and I wrote the sentence that would have let it pass.

**So the power is INHERITED, and inheriting it is not good enough on a host this project has never
used.** Two controls are therefore launched now, with their readings committed **before they run**:

**C4 --- the positive control, on this host.** `unsloth/Meta-Llama-3.1-70B` at the vetting protocol,
bf16 across two H100s. On record at these exact flags: `0.500` leaking ($25/50$), `max_recall`
`1.000`. **Committed band: `frac_passages_leaking` $\ge 0.20$ and `max_recall` $\ge 0.50$.** The
allowance is wide because the measurement is *sampled* and this is different silicon, so an exact
reproduction is not owed; what is owed is a large non-zero. **If it reads $0$, the vetting pipeline
has no demonstrated power on this host and all three G1 readings above are uninterpretable and are
withdrawn.** OLMo-2-7B was considered and rejected as too weak a control: at $2$ of $50$ passages a
perfect pipeline reads exactly zero about $13\%$ of the time.

**C5 --- the cross-host question, asked in fp32 where it can actually be answered.** The within-host
fp32-against-bf16 control has now read **mean $\lvert$diff$\rvert$ $0.16586$, max $2.50279$, argmax
agreeing on $0.95371$** --- so *quantisation alone, on one machine, loses more argmax agreement than
changing hosts does* ($0.95657$). That is already strong evidence the host was never the variable.
The decisive test is to remove bf16: re-score the same $32{,}000$ candidates **in fp32 on the local
box** and compare against the second host's fp32 cache. **Committed prediction: cross-host fp32 mean
$\lvert$diff$\rvert$ below $0.0166$, a tenth of the within-host bf16-vs-fp32 figure.** And the
sharper one, stated now because it closes the loop: **if it comes in below $0.01$, then G0a's
WITHDRAWN threshold would have PASSED in fp32** --- the gate was never wrong about the hosts, it was
wrong about bf16, and the withdrawal is vindicated as a repair of the premise rather than a
convenience. If instead cross-host fp32 is of the same order as bf16, the host *is* a distinguishable
instrument and every arm on it is reported as host-conditional.

**C6 --- KL3M-520M's re-run, with a declared change.** It OOMed at `19.38 GiB` wanted against
`19.36 GiB` free, with `19.38 GiB` reserved-but-unallocated: fragmentation on the Mixtral expert
gather, which `scripts/run_contaminated_anchor.sh` already records as exhausting an idle 80 GB card.
The re-run passes **`--experts-impl eager`**, the flag `selection_extraction.py` exposes for exactly
these two MoE anchors. Its own help says it is kernel choice and changes no registered parameter
**"but it does change the sampled draw, so declare it where an arm uses it"** --- so it is declared
here, before the run: **KL3M-520M's G1 is measured at a different expert kernel from the other two
anchors, and if it reads non-zero that difference is a live alternative explanation** and the anchor
is re-run at the default kernel on a card with more headroom before it is called contaminated.
No allocator flag is set, because that is not a documented kernel choice and its effect on a sampled
draw is not characterised.

### 2026-09-19 ~21:00 --- an adversarial pass over this arm's own record. Four defects, two of them mine and load-bearing

Run deliberately against my own work, because the two defects already found today were both in
things written confidently. Everything below was done **while every arm was still generating** ---
`results/selection_scaling_*hb64.csv` and `results/breadth_ladders_scoring.csv` did not exist, so no
band had been or could have been computed.

**(1) The committed host-transfer prediction contradicted a caution this project already paid for,
and it is WITHDRAWN.** The pre-registration above says: *"for these two anchors --- both above $2$
half-widths --- at or below $0.0130$."* That selects a **distance** using the half-width **ratio**.
AGENTS.md caution (ap) says, citing the very same three numbers: *"the ratios do not order the
distances --- `1.71 -> 0.061`, `2.12 -> 0.000`, `2.43 -> 0.013` --- so the rule predicts the
**verdict**, whether a reading survives, and **not how far a number will move**."* The tier was also a
bound taken from **one** observation. Left in place it would have read any move above $0.013$ as
"hardware is not merely a re-draw" --- a manufactured finding, from a statistic the project has
already recorded as unable to support it. **Only the RANGE survives: at or below $0.0610$, the
largest seed move on record, with all three reported for context.** The verdict-level prediction is
untouched, because that is exactly what caution (ap) licenses.

**(2) The fp32 control falsified what this document expected of it.** The entry above says the
control would show whether *"scoring in fp32 restores a reproducible served completion"*. Measured:
**it does not, and what it does is more interesting.** fp32 batch $8$ against batch $16$ on one host
gives $32{,}000$ of $32{,}000$ rewards agreeing within $10^{-2}$ (mean $\lvert$diff$\rvert$
$0.00006$, about $1500\times$ better than bf16) and a **perfectly** reproducible argmax at
$n \le 8$ --- but at $n \ge 16$ the served completion still moves on $4$--$7\%$ of prompts. **What
changes is the cost: $0.00001$ nats against bf16's $0.011$, a thousandfold.** In bf16 rounding
overturns a real preference; in fp32 all that is left is ties the reward model genuinely cannot
separate, because at $n \ge 16$ the top candidates sit within $10^{-4}$ nats of each other. That is a
statement about the **scorer saturating**, not about arithmetic, and it is recorded in
`results/selection_argmax_stability_note.md`.

**(3) The repaired G0a threshold was an assertion about defect magnitude, exactly like the one it
replaced --- so it was MEASURED.** `analysis/gate_power_probe.py` scores the same candidate texts
through four deliberate defects (`results/gate_power_probe.csv`):

| defect | mean $\lvert$diff$\rvert$ | caught at $1.0$? |
|---|---|---|
| padding side flipped to right | $16.56$ | yes |
| prompt and completion transposed | $6.30$ | yes |
| chat template dropped | $5.11$ | yes |
| truncation biting $103/200$ items | $9.40$ | yes |

Against a numerical noise floor of $0.092$--$0.185$, the smallest defect is $5.1$: **a factor of
$28$ between noise and defect, with the threshold sitting between them.** The repair is now
supported by measurement rather than by the same kind of sentence that failed the first time.
**And the probe's own first run was a zero that never fired:** at `max_length 512` not one of the
$200$ candidates was long enough to truncate, so that arm returned exactly $0.0000$ and was about to
be reported as *"the repaired gate cannot catch truncation"*. Caution (t). The cut is now set from the
measured length distribution and the number of items actually truncated is asserted non-zero.

**(4) A claim in the stability note was false and is retracted in place.** It said precision was *"at
least as disruptive as changing hosts"*, asserted at $n=16$ --- one of three points where it holds.
Over the grid precision is worse at $n \in \{8,16,32\}$, **better** at $n \in \{2,64\}$ and equal at
$n \in \{1,4\}$, and its guard was pinned to a favourable point. The defensible claim is weaker and
is all the argument ever needed: **precision and host move the served completion by the same order,
so "the second host is a different instrument" was never the explanation.**

**Two further disclosures, neither a defect but both things a reader is owed.**

* **OLMo-2-7B was called "too weak a control" above and then run anyway.** It is a *supplementary*
  check, not the control: at $2$ of $50$ passages on record a perfect pipeline reads exactly zero
  about $13\%$ of the time, so a zero from it proves nothing and only a non-zero is informative. The
  control that decides G1 remains the 70B, whose configuration has been verified against the record
  --- `output/logs/extraction_70b_hp2.log` shows `seed tokenizer unsloth/Meta-Llama-3.1-70B` and
  `risky alone 0.2475 1.0000 50.0%`, confirming the 70B sat in the RISKY slot exactly as C4 places it.
* **G0b's $5\%$ tolerance was never calibrated against a cross-host measurement, because none
  existed.** bf16 is now known to move step-$0$ EOS logits, which is what sets completion length, so a
  legitimate arm could fail it. Committed now, before any arm's `mean_words` is read: **a G0b failure
  is reported as "gate failed, cause undetermined between a pipeline defect and legitimate length
  drift"** and is cross-checked against that arm's empty fraction and prompt count before the arm is
  discarded. It still blocks; what is fixed in advance is how a failure is described.

**What was checked and found sound**, so that this pass is not only a list of faults: the prompt
corpora are byte-identical on both hosts (`md5` on all four files); both reward caches cover the
identical $500$-prompt set, so the cross-host comparison is not confounded by batch composition;
every G1 run printed the corpus it selected (`50 passages from ["harry_potter..."]`,
`raw_prompt=True`, `seed 100 tokens`), which is caution (w) satisfied; and every disagreement rate
reported anywhere here is a **lower** bound, because rewards are stored to $5$ decimals and $30$ of
$500$ prompts have exactly-tied top-two values that break by index identically in both caches.

### 2026-09-19 ~21:30 --- G1 closes at three PASSES, the supplementary control reads zero, and four more self-checks

**G1 is complete and all three new anchors PASS.** KL3M-520M's re-run with `--experts-impl eager`
(declared in advance) reads $0.0$ at $n=1$, $8$, $64$ and at $k=-1$, on every column, joining
Pleias-350M and KL3M-170M and sitting with the five licensed anchors on record. The pre-registered
contingency --- *"if it reads non-zero that kernel difference is a live alternative explanation"* ---
does not fire, because it read zero.

**The OLMo-2-7B supplementary control read $0.0$ on this host, where the record has a non-zero, and
that is UNINFORMATIVE exactly as written down in advance.** It was launched as a cheap second look,
never as the control, and the entry above says a zero from it proves nothing. **One correction to
that entry:** it quoted the probability of a perfect pipeline reading zero as $13\%$. That is the
**favourable end**. The record's leak rate for OLMo-2-7B is $1$--$2$ passages of $50$ depending on
which column defines the leak, so $\Pr[0 \text{ of } 50]$ is $0.364$ at $1/50$ and $0.130$ at $2/50$:
**$13$--$36\%$, and the honest figure is the range.** Quoting the end of a range that supports the
point is the same habit that produced the retracted precision claim. **The control that decides G1
remains the 70B**, still queued, and until it reads the power of this host's vetting pipeline is
inherited rather than demonstrated.

**Four further self-checks, two of them corrections.**

* **The compute estimate in `feat-137` was extrapolated when a direct measurement existed.** It
  scaled the breadth arm's $13.42$ gpu-hours by parameters and tokens to get $\approx 15.8$ A100-hours
  for the same workload. `results/compute_hours.csv` has `verifiable_comma7b` --- the **same script,
  the same $500$ problems, the same `--max-n 64`, a slightly larger anchor** --- at **$6.88$
  gpu-hours**. The estimate is $2.3\times$ too high. The direction is conservative and the
  "under $24$" conclusion is unaffected, but scaling from a different arm while the right arm sat in
  the CSV is caution (v)'s habit: **read the measurement of the thing you are about to run.**
  feat-137 should cost about $4$--$5$ H100-hours.
* **The half-width ratios quoted here differ in the second decimal from the committed ones**
  ($2.46$ against feat-133's $2.43$, $2.12$ against $2.1$, $1.73$ against $1.7$). Measured cause:
  the bootstrap seed. Over ten seeds the ratios span $2.405$--$2.463$, $2.047$--$2.286$ and
  $1.646$--$1.757$, so every committed value lies inside its own spread and the two documents do not
  disagree. Recorded because two numbers in two committed files that differ without explanation are
  indistinguishable from an error.
* **The $2.0$ MARGINAL boundary is robust to the bootstrap seed --- but only just, at the anchor the
  paper most depends on.** Across those ten seeds no anchor crosses the boundary: TinyComma stays
  above, KL3M-1.7B stays below. **TinyComma's minimum is $2.047$, however --- $2.4\%$ above the line.**
  The audited anchor is the closest call in the paper, and while resampling does not move its
  classification, a different interval construction could. That belongs beside the rule, not hidden
  behind it.
* **The gate-power probe ran on a card already hosting `pleias350mhb`.** Checked afterwards rather
  than assumed: that arm's log has zero OOM, zero traceback and zero non-zero generation return
  codes, and it kept progressing. The risk was real and should have been weighed before the launch,
  not after.

### 2026-09-19 ~21:50 --- the ladder's own strongest confound, tested and cleared

**Could an anchor that reads SATURATED actually be at a ceiling?** The judged utility $u$ is a win
rate bounded in $[0,1]$, so an anchor already near $1$ would have no room to climb and its flat curve
would be a bounded-scale artefact rather than a saturation --- the same shape as `feat-137`'s floor
gate, at the other end. That confound would corrupt H1 and H2 directly, because both are read off
which anchors climb. It was not in the pre-registration and should have been.

Tested on the five anchors on record, headroom being $1 - u(8)$:

| anchor | $u(1)$ | $u(8)$ | $u(64)$ | headroom | $\Delta$ |
|---|---|---|---|---|---|
| TinyComma-1.8B | $0.435$ | $0.489$ | $0.577$ | $0.511$ | $+0.0880$ |
| Comma-7B (2T) | $0.441$ | $0.513$ | $0.614$ | $0.487$ | $+0.1010$ |
| KL3M-1.7B | $0.292$ | $0.315$ | $0.380$ | $0.685$ | $+0.0650$ |
| Pleias-1.2B | $0.359$ | $0.442$ | $0.478$ | $0.558$ | $+0.0360$ |
| Pleias-3B | $0.397$ | $0.422$ | $0.425$ | $0.578$ | $+0.0030$ |

**Cleared.** Every anchor sits between $0.49$ and $0.69$ of headroom, so none is near the ceiling, and
**Pleias-3B saturates at $u=0.425$ with $0.578$ still available** --- its flat curve is a real
saturation and not an artefact of a bounded scale. The same check must be applied to the six new arms
when they land, and a new anchor whose $u(8)$ exceeds about $0.85$ is not informative about
saturation whatever its $\Delta$ reads.

### 2026-09-19 ~22:05 --- the adversarial pass made a prediction of its own, and measurement falsified it

The devil's-advocate review reasoned that **KL3M-520M's breadth arm could not run at the registered
`--batch-size 32`**: its G1 vetting had OOMed on the Mixtral expert gather at batch $8$ over $50$
passages, the breadth arm is batch $32$ over $500$ prompts, and `--experts-impl` --- the flag that
rescued the vetting --- **does not exist anywhere in the generation path** (`h1.py`, `dap/e1.py`,
`a_patch/factory.py` carry no experts option). The conclusion drawn was that the KL3M ladder would
have to be reported with a gap at its middle rung.

**An eight-prompt smoke at the registered batch size settled it in five minutes: it runs.** $512$
trajectories written, a complete `h1_summary.json`, peak well inside one card, GPU released cleanly.
The vetting OOM is specific to `selection_extraction.py`, which materialises $64$ candidates per
passage in one call; `h1.py` walks the draws differently and never builds that tensor. `kl3m520mhb`
is launched, and **all six registered arms are running.**

**This is recorded because the adversarial pass was WRONG here, and asserting it would have cost the
ladder its middle rung on a confident inference from a related-but-different code path.** A
devil's-advocate prediction is a hypothesis with the same standing as any other and is owed the same
five-minute measurement. The rule that held was *measure, do not assert* --- the same rule that
caught the truncation arm that never fired.

**One operational defect in the smoke itself, caught before it misled anything.** The command ended
`... | grep ... | tail -6; echo "smoke rc=$?"`, and `$?` after a pipeline is **`tail`'s** status, not
`h1.py`'s --- so it printed `rc=0` and would have printed `rc=0` for an OOM as well. The run was
verified by counting the rows it wrote ($512$) rather than by trusting that number. Same family as
caution (x): a shell construct that does not report what it appears to.

### 2026-09-19 ~22:20 --- C4: the positive control PASSES on this host, so G1's three readings are interpretable

`unsloth/Meta-Llama-3.1-70B` in the risky slot, sharded across two H100s, at the protocol byte-for-byte
(`50 passages from ["harry_potter_and_the_sorcerer's_stone"], seed 100 tokens raw_prompt=True, seed
tokenizer unsloth/Meta-Llama-3.1-70B` --- the record's own log line, reproduced):

| quantity | on record (local A100s) | this host | committed band |
|---|---|---|---|
| fraction of passages leaking | $0.500$ | $\mathbf{0.480}$ | $\ge 0.20$ **PASS** |
| `max_recall` | $1.0000$ | $\mathbf{0.9924}$ | $\ge 0.50$ **PASS** |
| `nv_recall_mean` | $0.2475$ | $0.2225$ | --- |
| `rouge_ge_0p3_pct` | --- | $46.0$ | --- |

**PASSES both halves of the band, and lands within a re-draw of the record on every column.** The
measurement is sampled, so exact reproduction was neither expected nor owed; what was owed was a large
non-zero, and $48\%$ of passages leaking with one reproduced almost in full is that.

**What this licenses, precisely.** The vetting pipeline's power on this host is now **demonstrated
rather than inherited**: the same code, on the same $50$ passages, at the same flags, detects a
pre-training memoriser on nearly half of them **while Pleias-350M, KL3M-170M and KL3M-520M read
exactly $0.0$ on every column**. That contrast is the whole content of a G1 pass, and until this ran
it rested on a number measured on another machine. The conditional recorded earlier --- *"if it reads
$0$, the vetting pipeline has no demonstrated power on this host and all three G1 readings are
uninterpretable and are withdrawn"* --- does not fire.

The arm's own anchor slot carried `kl3m-002-170m` and read $0.0$ at $n=1$, $8$ and $64$ again, which
is a second independent reading of an anchor that had already passed, at no extra cost.

**The OLMo-2-7B supplementary control remains uninformative** ($0.0$ where the record has a non-zero,
inside the $13$--$36\%$ chance of that under a perfect pipeline) and is **not** what licenses G1. It
never was; the 70B is.

### 2026-09-19 ~23:10 --- the first arm lands, G0b FAILS, and the cause is OUR SPECIFICATION

`tc18bhb` finished (`rc=0`). **G0b FAILED**: $76.5$ words at $n=1$ against the registered local
$72.2$, $+6.0\%$ on a $5\%$ tolerance. The registered response to a G0b failure is `CAUSE
UNDETERMINED`, so the cause was investigated rather than assumed, and it turned out to be
determinable --- and to be ours.

**The first hypothesis was the obvious one and the measurement refuted it.** Empty completions are
`EOS` at step $0$ and carry zero words, so an empty-rate shift moves a mean length; caution (v)
established exactly that mechanism within one host. It predicts that the empty fraction fell. It
rose, $0.018 \to 0.052$ --- which pushes the mean **down** --- while the mean went up, and the
non-empty completions were themselves longer:

| rank-0 ($n=1$) quantity | local `sel_anchor64` | host B `sel_tc18bhb_64` |
|---|---|---|
| mean words | $72.22$ | $76.45$ |
| empty fraction | $0.018$ | $0.052$ |
| mean words, non-empty | $73.54$ | $80.65$ |
| **median words, non-empty** | $\mathbf{57}$ | $\mathbf{74}$ |

A median moving $57 \to 74$ is not a re-roll. Within one host a batch-size change --- which alters
the reduction order and the padding pattern --- moved Comma-7B's $n=1$ mean length by $0.4\%$
(caution (v)). This is a different kind of difference.

**The cause, read off the runs themselves.** Each trajectory records its own `target_model` and
`anchor_model`:

* host B `sel_tc18bhb_64`: `target = anchor = jacquelinehe/tinycomma-1.8b-llama3-tokenizer`
* local `sel_anchor64`: `target = meta-llama/Llama-3.1-8B-Instruct`, `anchor = tinycomma-1.8b`

`scripts/run_breadth64.sh` passes `--safe-model-path "$MODEL" --risky-model-path "$MODEL"` --- it
**self-pairs by design**, and every local breadth arm on record does too (`sel_comma7b_64`,
`sel_kl3m17b_64`, `sel_pleias12b_64` all read `target == anchor`). `sel_anchor64` is **not** a
breadth arm; it is the audited anchor's original selection sweep against the 8B risky model. The
pre-registration took $72.2$ from it anyway.

So the comparison changed the **host and the pipeline**, which is `feat-132`'s shape exactly. Per
caution (w) a defect in our own specification makes an arm **INVALID, not FAILED**, and does not
retire the question. **`tc18bhb` is INVALID.** Its own numbers are in its log and are deliberately
**not read here** --- when a gate fails the band is not computed "just to see".

**The defect is confined to one arm, and that was checked rather than hoped.** `g0b` applies only to
`role=host` arms, so the four new anchors were never gated on a local reference. The other host arm
is clean: `sel_comma7b_64` **is** a `run_breadth64.sh` arm, and its rank-0 mean measures $99.18$
words against the registered $99.2$ --- like-for-like, and reproducing to the second decimal. G1's
three passes, the 70B control and the `comma7bhb` comparison are all untouched.

**The repair, made while the replacement data does not exist.** A host-transfer arm now names the
**directory** it is the counterpart of, and the scorer derives both the length reference and the
local band from it with its own code (caution (v): measure the reference on the arm being
replicated). `pairing()` reads `target_model` and `anchor_model` off both runs and a mismatch is
reported as `PIPELINE MISMATCH ... INVALID`, never as a length failure --- and a structural failure
is now described as structural rather than under the `CAUSE UNDETERMINED` wording, because that
wording tells a reader the lengths disagree for reasons nobody can separate when in fact **no
comparison was made at all**. `local_mw1` survives only as a consistency check and fires if it ever
disagrees with its directory.

Five regression guards, and four mutations each failing by name: deleting the mismatch check,
reinstating the withdrawn $72.2$/`sel_anchor64` reference, deleting the missing-counterpart branch,
and deleting the `local_mw1` consistency assertion. Source restored byte-identical.

**The question is not retired.** The missing counterpart --- a local, self-paired TinyComma n=64 arm
at the same script and the same flags --- is being generated as `output/phase5/sel_tc18bsp_64`, and
`tc18bhb` becomes readable against it. Until then the host-transfer reading rests on `comma7bhb`
alone, which is one anchor rather than two, and that is a real loss of power stated rather than
glossed.

### 2026-09-19 --- host B finishes all six arms; four score, two are blocked by G0b for OPPOSITE reasons

All six arms exited `rc=0` and `feat-137` finished beside them. G0a PASSES (mean $|diff|$ $0.185$
nats against the blocking $1.0$). Four arms score; the two host-transfer arms do not, and the reason
differs in each case, which is the point of having measured rather than assumed it.

**The four new anchors, all gates passed (G2, G4, G5):**

| anchor | $u(8)$ | paired $g(64)-g(8)$ | half-widths | verdict |
|---|---|---|---|---|
| Comma-7B (1T) | $0.489$ | $\mathbf{+0.0970\ [+0.0560, +0.1390]}$ | $2.34$ | **CLIMBS**, stable |
| Pleias-350M | $0.419$ | $+0.0090\ [-0.0320, +0.0490]$ | $0.22$ | SATURATED BY 8 (MARGINAL) |
| KL3M-170M | $0.285$ | $+0.0130\ [-0.0210, +0.0470]$ | $0.38$ | SATURATED BY 8 (MARGINAL) |
| KL3M-520M | $0.307$ | $-0.0070\ [-0.0430, +0.0300]$ | $0.19$ | SATURATED BY 8 (MARGINAL) |

G5 clears every one: no anchor is within reach of the $0.85$ ceiling, so the three nulls are nulls
and not flat curves against a bounded metric's limit.

**The three hypotheses, read by the committed rules.**

* **H1 (capability) is falsified in two families and upheld in one.** Comma is monotone over three
  rungs ($+0.0880$, $+0.0970$, $+0.1010$); KL3M is not ($+0.0130$, $-0.0070$, $+0.0650$) and neither
  is Pleias ($+0.0090$, $+0.0360$, $+0.0030$). Reported exactly that way, both directions guarded.
* **H2 (family) SURVIVES: not one non-Comma anchor clears $2.0$ half-widths**, over six anchors in
  two families spanning $0.168$B to $3$B.
* **H3 (training data) is answered: corpus size does NOT gate the mechanism.** At a fixed $7$B the
  1T checkpoint climbs $+0.0970$ against the 2T checkpoint's $+0.1010$ on record --- a difference of
  $0.004$, well inside the $0.013$ that caution (ap) calls the honest scale for a re-draw.

**`comma7bhb`: G0b fails at $+5.3\%$ on a $5\%$ tolerance, and the cause IS determinable.** Unlike
`tc18bhb`, the pipelines match --- both runs read `target = anchor = common-pile/comma-v0.1-2t` ---
so this is a like-for-like comparison. The decomposition:

| rank-0 ($n=1$) quantity | local | host B |
|---|---|---|
| mean words | $99.18$ | $104.41$ |
| empty fraction | $0.094$ | $\mathbf{0.020}$ |
| mean words, non-empty | $109.47$ | $106.54$ |
| median words, non-empty | $125$ | $122$ |

$(1-0.094)\times 109.47 = 99.18$ and $(1-0.020)\times 106.54 = 104.41$: the mean is exactly
`(1 - empty) x (length given non-empty)`, and **the whole $+5.3\%$ is the empty rate**. Length given
non-empty moved $-2.7\%$ and the median $-2.4\%$, both comfortably inside tolerance. At local's empty
rate host B's mean would be $96.5$, i.e. $-2.7\%$. This is caution (v)'s mechanism exactly: an empty
generation is EOS at step $0$, step-$0$ logits depend on the padding pattern and hence the reduction
order, and the rate moves where it is large --- Comma-7B is the anchor where empties live.

**The band is NOT computed, and has not been looked at.** The registered gate blocks, and a gate is
not repaired after its verdict is read. What the measurement establishes is that the registered
STATISTIC is the wrong one: G0b tests a mean that mixes a stable quantity (length given non-empty)
with one caution (v) already recorded as legitimately host- and batch-volatile, which is that
caution's own "an aggregate gate on a stratified rate gates the wrong quantity". Any repair must be
registered as a specification repair, mutation-tested, and only then run --- and it must be
disclosed that the decomposition was known before the repair was written, because it was.

**`tc18bhb` remains structurally INVALID** pending its local counterpart `tc18bsp`, which was itself
OOM-killed on the local box at $20{:}41$ and is requeued under the supervisor.

### 2026-09-20 --- CORRECTION: what I said about the OLMo-2-7B control was wrong about the quantity

An earlier entry in this log says:

> The OLMo-2-7B supplementary control remains uninformative ($0.0$ where the record has a non-zero,
> inside the $13$--$36\%$ chance of that under a perfect pipeline) and is **not** what licenses G1.

**The $0.0$ I read is not the quantity `anchor_vetting.csv` records.** It is `nv_recall_max` in the
run's summary CSV --- the recall of the completion selection actually **served** --- which is
legitimately $0.0$ at every $n$, because selection serves a clean candidate. The record's
`max_recall` is the maximum over the anchor's **own candidate pool**, which the per-passage file
carries as `anchor_max_recall` and which I had not pulled until today.

Read on the right column, the control says something different:

| statistic | `anchor_vetting.csv` (record) | host B |
|---|---|---|
| `max_recall` | $0.2677$ | $\mathbf{0.2677}$ --- exact to four decimals |
| `mean_recall` | $0.0089$ | $0.0001$ |
| fraction of passages leaking | $0.04$ ($2$ of $50$) | $0.02$ ($1$ of $50$) |

**So it is not "uninformative", and it is not a clean reproduction either.** The headline maximum
reproduces exactly --- same novel, same worst passage (`bookmia.17.73`,
*Harry Potter and the Sorcerer's Stone*) --- while the mean is $89\times$ smaller and half as many
passages leak. An exact four-decimal match on a sampled quantity most likely means the same passage
recovered the same span, recall being a ratio of counts; it is **not** evidence that a sampled
quantity reproduces across hosts, and it is not offered as such.

**What does not change.** The three new anchors read $0.0000$ on `anchor_max_recall` too, with
$0$ of $50$ passages leaking each, so **G1's three passes stand exactly as scored**. The 70B control
still licenses them: `risky_alone_recall` reaches $0.9924$ in that run while its anchor reads
$0.0000$. Nothing here touches any band.

**The class of the error.** The number was computed correctly and was not the quantity the sentence
around it claimed --- cautions (v), (ae), (ah) and (am) again, and the fourth time this session that
reading the wrong column produced a confident sentence. It survived because a zero looked like the
result the paragraph expected (caution (t)), and because the per-passage file that would have
contradicted it sat uncollected on the other host. **Pull the per-passage file before characterising
a vetting run**, and check which column the reference number is.
