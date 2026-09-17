# Pre-registration: is $61.3\times$ the price a deployer actually pays?

Committed **2026-09-16 23:15**, before the arm ran. Nothing above `## Scoring log` is edited after.

## Why

The paper's cost headline --- selection at $n=64$ costs $61.3\times$ the metered decoder's
forward-pass cost --- is **not a measurement**. It is an analytical model in
`analysis/serving_cost.py`: parameter counts times token counts,
$(P_s + P_r)(L_p + T)$ with $P_s = 1.7586$ and $P_r = 7.6156$ billion. A reviewer's objection is
exact: *"forward-pass counts overstate wall-clock and understate memory. GPU-seconds per served
token would be the honest unit."* Batched $n$-sampling amortises heavily on a GPU, and $64$ short
completions generated in one batch do not cost $64$ times one completion in wall-clock. The paper
quotes $283$ GPU-hours in total and nothing per arm.

So the number that reaches the abstract is a FLOP proxy presented where a deployer will read a price.
This measures the price.

## What is measured

**Wall-clock GPU-seconds per served token**, for the two serving paths, on the same prompts, on the
same card, through the same entry points the project actually uses:

- **selection, $n=64$**: `h1.py --k-values 0.0 --trajectories-per-prompt 64` (the $64$ anchor draws)
  **plus** the pointwise reward pass that scores all $64$ and takes the argmax. Both stages count:
  the reward model is the price, and pretending otherwise is the thing being corrected.
- **metered, $k=10$**: `h1.py --k-values 10.0 --trajectories-per-prompt 1`, the anchored decoder.

A *served token* is a token actually delivered to the user: one completion per prompt in both paths.
Both run at `--max-new-tokens 200` on the same prompt set, so the denominators are comparable by
construction and the ratio of seconds-per-served-token is the ratio of wall-clock.

**Interleaved `SEL, MET, MET, SEL`** on one card, so that any thermal or co-tenancy drift over the
run cancels in the mean rather than loading onto whichever arm went first. Both repeats of each arm
are reported, not just their mean, so the drift is visible.

**The card is exclusively ours** (no second job of ours on it). Other users hold GPUs 0 and 4
throughout; that is recorded rather than controlled, and it is why the design interleaves.

## Committed bands

Let $R$ be the measured ratio, selection-$n{=}64$ seconds per served token divided by
metered-$k{=}10$ seconds per served token.

| $R$ | verdict | what the paper must do |
|---|---|---|
| $R < 30$ | **MODEL OVERSTATES** | the analytical $61.3\times$ is more than $2\times$ pessimistic as a price; the abstract must quote the measured ratio beside it, because the FLOP proxy is not what a deployer pays |
| $30 \le R \le 123$ | **MODEL HOLDS** (within $2\times$ either way) | $61.3\times$ survives as a price as well as a FLOP count; report the measurement as corroboration |
| $R > 123$ | **MODEL UNDERSTATES** | selection is *more* expensive in practice than the paper admits, and the cost concession must be strengthened |

Committed secondary, reported whatever $R$ is: the **split** of selection's cost between the $64$
anchor draws and the reward pass. The paper already argues the $61.3\times$ "is the price of the
reward model, not of the mechanism"; that claim is currently derived from parameter counts and this
measures it. If the reward pass is **less** than half the measured selection time, that argument is
weakened and the text must say so.

## Excluded in advance

We will not, after seeing the number: drop the reward pass from selection's cost to make the ratio
smaller; change `--max-new-tokens`, the prompt set, or the batch size between the two arms; report
only the faster of the two repeats of an arm; or re-run on a quieter box and report that instead
without reporting this one. Batch sizes are each path's own registered default and are held fixed
across repeats.

**This arm cannot refute the paper's scientific claims.** It prices them. A bad ratio costs the
cost-concession sentence, not the certificate.

## Scoring log

## Scoring, 2026-09-17 11:00 — MODEL HOLDS, but the reason the paper gives for the price is wrong

Ran 22:53--23:25 on GPU 2, exclusively ours, interleaved `SEL, MET, MET, SEL`. Other users held
GPUs 0 and 4 throughout and our own arm B held GPU 1; the box state is in the log at both ends.

**Denominators are identical by construction**, as the registration required: both arms serve one
completion per prompt over the same $40$ prompts, mean length $204.5$ tokens, **$8{,}179$ served
tokens each**. So the ratio of seconds-per-served-token *is* the wall-clock ratio.

| arm | rep 1 | rep 2 | mean | s / served token |
|---|---|---|---|---|
| selection $n=64$: anchor draws | $824.6$ | $822.6$ | $823.60$ | |
| selection $n=64$: reward pass | $85.8$ | $83.1$ | $84.44$ | |
| **selection, total** | $910.4$ | $905.7$ | **$908.03$** | $0.1110$ |
| **metered $k=10$** | $23.7$ | $27.6$ | **$25.64$** | $0.00314$ |

Repeat spread: $0.52\%$ on selection, $15.3\%$ on the metered arm --- $3.9$ seconds on a $25$-second
job, which is why both repeats are shown rather than a mean alone. Taking the extremes gives
$R \in [32.8, 38.5]$, inside the band at either end. (Corrected 2026-09-17 12:10: this line
first read $[32.9, 38.3]$, an arithmetic slip in the scoring pass; the band verdict is unchanged
and the manuscript carries the corrected pair.)

### Primary: $R = 35.4\times$ --- **MODEL HOLDS**

$30 \le 35.4 \le 123$, so the analytical $61.3\times$ survives as an order-of-magnitude price. But it
is **$1.73\times$ pessimistic**: a deployer pays about $35\times$, not $61\times$, because $64$ short
completions generated in one batch do not cost $64$ sequential ones. The paper should quote the
measured number beside the modelled one, and say which is which.

### Secondary: **WEAKENED**, and this is the finding that matters

The committed secondary was: *if the reward pass is less than half the measured selection time, the
"price of the reward model, not of the mechanism" argument is weakened and the text must say so.*

**The reward pass is $9.3\%$ of selection's wall-clock. The $64$ anchor draws are $90.7\%$.**

The parameter-count model has it backwards, and the mechanism is clear once measured: scoring $64$
candidates is **one batched forward pass**, compute-bound and parallel, while drawing them is
$64 \times 204$ sequential decode steps, memory-bandwidth-bound. Parameters are the wrong currency
for a cost that is dominated by how many times you must walk the weights.

The consequence is concrete and unfavourable to the paper's current text: **if the $7.6$B scorer were
free, selection would still cost $32.1\times$ the metered decoder.** So "a smaller scorer cuts the
price" is true of the FLOP count and close to false of the clock --- a $1.5$B scorer can cut at most
the $9.3\%$ it occupies. The real lever on serving cost is $n$, not the scorer. Section~5,
Appendix~I and the closing all assert the scorer version and all must be corrected.

That correction cuts both ways, and both halves go in the text: the price is *lower* than the paper
claims ($35\times$, not $61\times$), and the *reason* the paper gives for it is wrong.

### Band D honoured

The reward pass was not dropped from selection's cost; `--max-new-tokens`, the prompt set and both
batch sizes were held across repeats; both repeats of each arm are reported; and no quieter-box
re-run was substituted for this one.

**This arm prices the claims; it cannot and does not touch the certificate.**
