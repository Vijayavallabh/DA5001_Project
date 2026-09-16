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
