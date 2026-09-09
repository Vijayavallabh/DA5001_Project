# Pre-registration: where does the decoder become useful, relative to where the certificate fails?

Committed before the k in {0.6, 0.7, 0.8} generation was launched. Written 2026-09-10.

## What changed, and why this needs a finer grid

The manuscript's headline was that a judge cannot separate the decoder from serving its own safe
model at k=1 (-0.45 sigma) and can at k=3 (-2.35 sigma), so the useful region and the certified
region do not overlap. That rested on **180 judged pairs per arm**. At **600 pairs per arm**
(`results/utility_v5_summary.csv`, Qwen2.5-7B-Instruct) the same comparison gives **-3.66 sigma at
k=1**: the decoder is clearly better than the anchor there. The old number was under-powered, and
so was the null arm, which moves from 41.7% to 46.4% loss and now agrees with the independent
`feat-029` run at the same sample size (47.0%).

The claim therefore fails as stated and must be replaced. What the finer grid shows instead is a
near-coincidence of two boundaries measured in completely different ways:

    judged crossover (-2 sigma from the anchor), interpolated in [0.5, 1.0]   k = 0.68
    budget at which the FIRST protected work loses its certificate            k = 0.583
    budget at which 1% of protected works have lost it                        k = 0.682
    budget at which half have                                                 k = 1.023

The crossover is interpolated across a bracket half a nat wide, so 0.68 is not yet a measurement.
This experiment makes it one.

## The experiment

Generate the ordinary-prompt workload at **k in {0.6, 0.7, 0.8}** -- 500 prompts x 3 seeds x 3
classes = 1,500 trajectories per budget, the same prompts and seeds as every other judged arm --
and judge them at 200 pairs per cell with both judges.

## Prediction, committed blind

The crossover, re-interpolated on the finer grid, falls in **k in [0.58, 0.80]**, and the budget at
which the certificate first fails (0.583) lies **inside** the interval between the last budget where
the decoder is indistinguishable from the anchor and the first where it is not.

If instead the crossover falls **below 0.583**, the decoder becomes useful while the certificate
still covers every protected work, and the paper must report a genuine window in which the mechanism
is both useful and fully certified -- the opposite of its thesis. We commit to reporting that
outcome in those words if it occurs.

If the crossover falls **above 0.80**, the two boundaries are separated rather than coincident, and
the claim becomes the weaker "the decoder is useful only once the certificate has begun to fail",
without the coincidence.

Scored on the loss rate against the anchor-only arm, the statistic the published numbers use, with
the three-point utility score reported alongside.

## Scored (2026-09-10), both judges, 600 comparisons per arm

Generation: `output/phase5/util_cross` at k in {0.6, 0.7, 0.8}, 500 prompts x 3 seeds x 3 classes.
Judged by Qwen2.5-7B-Instruct (`results/utility_v6_summary.csv`) and Phi-3.5-mini-instruct
(`results/utility_v6_judge2_summary.csv`), scored by `analysis/judge_separation.py` into
`results/judge_separation_v6.csv` and `results/judge_separation_v6_judge2.csv`.

     k      works uncertified    Qwen z    Phi z
     0.5                 0.0%     -1.03    +1.52
     0.6                 0.1%     -1.21    +0.57
     0.7                 1.7%     -2.44    +0.57
     0.8                 9.1%     -2.79    -1.41
     1.0                43.9%     -3.66    +0.11
     10                100.0%     -6.97    -3.29

**The committed prediction holds on the primary judge.** Qwen's crossover, re-interpolated on the
fine grid, is **k = 0.66**, inside the committed [0.58, 0.80]. At that budget the certificate has
failed for 0.3% of the 758 protected works.

**The refutation condition did not occur on either judge.** A crossover below 0.583 would have
meant the decoder becomes useful while the certificate still covers every protected work, and we
committed to reporting that in those words. Qwen crosses at 0.66 and Phi at 5.39; both are above.

**The finer sub-claim fails and is withdrawn.** It said 0.583 would lie *inside* the interval
between the last budget where the decoder is indistinguishable from the anchor and the first where
it is not. On Qwen that interval is [0.6, 0.7] and 0.583 sits just below it. The two boundaries are
therefore **ordered, not coincident**: the certificate begins to fail first, the decoder is still
indistinguishable from its own anchor at k = 0.6, and only at 0.66 does it become useful. The paper
must say ordered, and must not claim a coincidence.

**The second judge takes the weaker branch the pre-registration also anticipated.** Phi does not
reach -2 sigma until **k = 5.39**, where the certificate covers **none** of the works, so on that
instrument the useful region opens an order of magnitude later and entirely inside the vacuous
region. The two judges agree on both endpoints -- neither separates at k <= 0.6, both do at
k >= 10 -- and disagree by 8x on where the useful region opens.

**Recorded against ourselves: the instrument moves between runs.** The null arm (the unconstrained
risky model judged against the anchor, n = 500, identical data both times) reads -5.47 sigma on
Qwen and -1.49 on Phi in the v5 run, and -6.09 and -2.49 in this one. A full sigma of run-to-run
drift on the same comparison is the resolution of the instrument, and any statement about a
separation smaller than that should be read with it in mind.
