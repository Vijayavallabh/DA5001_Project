# Pre-registration: does the certificate's paraphrase claim survive a non-literal event?

Committed **2026-09-17 14:25**, before the arm ran. Nothing above `## Scoring log` is edited after.

## Why

Section~2 carries one sentence answering the strongest standing objection to any verbatim-recall
copyright guarantee:

> *"Unlike an $n$-gram blocklist it names no work, so paraphrase and unlisted works are bounded on
> the same terms as exact quotation."*

A reviewer's objection is precise and, on inspection, correct:

> *"The bound $q(E) \le n\,p_s(E)$ is term-identical, but the event $E$ = 'output is a legal
> paraphrase of $x$' can have anchor base rate orders of magnitude above $p_s(\{x\})$, and the
> certificate then multiplies a base rate that is not small. Nothing in the paper measures
> paraphrase-class leakage at all; the entire extraction evaluation is near-verbatim recall and
> longest-common-substring."*

The second half is simply true of this repository. `nv_recall` and `lcs_word` are both **exact
substring** metrics (caution (j): `dap/stats.py:73` computes the longest common *substring*, not
subsequence), so every extraction number in the paper scores verbatim quotation and nothing else.
The sentence is an assertion with no experiment under it.

## What is measured

`dap/stats.py:rouge_l_score` is LCS-as-**subsequence**: it survives word insertions, deletions and
substitutions, so a reordered or partially reworded passage still scores. That is the standard
non-literal copying metric in the CopyBench line of work and it is a genuinely looser event than
anything this paper has scored. It already exists and `analysis/selection_extraction.py` has simply
never imported it.

feat-127 surfaces it on the **same arm, same 100 protected passages, same adversarial scorer** (the
memorising model's own likelihood, which searches the $n$ anchor draws for whichever is closest to
the protected text). Three quantities at each ROUGE-L threshold $\theta \in \{0.3, 0.5\}$:

- $p_s(E_\theta)$ --- the anchor's own base rate, the $n=1$ arm;
- $q(E_\theta)$ --- the served rate at $n=64$;
- the **measured amplification** $q/p_s$, against the permitted $64$.

## Committed bands --- read on $p_s(E)$ at ROUGE-L $\ge 0.5$

| anchor base rate | verdict |
|---|---|
| $0/100$ | **CLAIM HOLDS AT THIS LOOSENING.** At a genuinely non-literal event the anchor still never lands, so $64\times$ it is still nothing. The sentence stands, now with a measurement under it instead of an assertion. |
| $1$--$4$ of $100$ | **QUALIFIED.** The bound is term-identical; the base rate is not. The sentence must say the paraphrase guarantee is $64\times$ a rate that has been *measured*, and quote it, rather than implying it is $64\times$ zero. |
| $\ge 5$ of $100$ | **WITHDRAWN AS WRITTEN.** "Bounded on the same terms as exact quotation" misleads: at paraphrase level the certificate multiplies a base rate large enough that $n=64$ permits a rate a rights-holder would care about. The sentence is rewritten and the limitation stated in Section~7. |

## Committed secondary

The measured amplification $q(E_\theta)/p_s(E_\theta)$ at each $\theta$, against the permitted $64$,
reported whatever it reads. If $p_s(E_\theta) = 0$ the ratio is **undefined** and is reported as
undefined --- never as $1$, and never quietly dropped.

## Excluded in advance

We will not, after seeing results: move the ROUGE-L thresholds to make the base rate look smaller;
substitute a different non-literal metric; drop the $n=64$ column; report the verbatim numbers in
place of these; or change the scorer, the passages or the anchor.

## What this arm cannot do, stated before it runs

ROUGE-L is a **lexical** proxy for paraphrase, not a semantic or a legal one. It captures reordering,
insertion and substitution --- which is what "non-literal copying" means in this literature --- and it
does **not** capture a genuine rewrite that preserves meaning while sharing little word order. A null
result here therefore bounds *lexical* paraphrase only, and the manuscript must say exactly that
rather than claiming paraphrase-class safety in general. The honest ceiling on this arm is that it
converts an unsupported sentence into a supported narrower one.
