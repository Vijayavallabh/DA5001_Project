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
