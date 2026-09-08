# Bibliography verification, plan v5 additions (2026-09-08)

Non-existent references are grounds for desk rejection, so every entry added today was checked
against a primary or authoritative record before it was cited. All fields below (volume, issue,
pages, year, publisher) match what is in `references.bib`.

| entry | claim in bib | verified against | result |
|---|---|---|---|
| `loynes1962stability` | Math. Proc. Camb. Phil. Soc. **58**(3), 497–520, 1962, CUP | Cambridge Core record | exact |
| `cruz1991calculus` | IEEE Trans. Inf. Theory **37**(1), 114–131, 1991 | IEEE Xplore / dblp vol. 37 | exact |
| `vanerven2014renyi` | IEEE Trans. Inf. Theory **60**(7), 3797–3820, 2014 | author's publication list + arXiv:1206.2459 | exact |
| `donsker1975asymptotic` | Comm. Pure Appl. Math. **28**(1), 1–47, 1975, Part I | Wiley, DOI 10.1002/cpa.3160280102 | exact |
| `leboudec2001network` | Springer LNCS **2050**, 2001 | Springer Link, DOI 10.1007/3-540-45318-0 | exact |
| `monteiropaes2026limits` | arXiv:2605.07105, 4 authors, 8 May 2026 | arXiv abstract page | exact |
| `dembo1998large` | Large Deviations Techniques and Applications, 2nd ed., Springer, 1998 | standard reference, not individually re-checked | assumed |
| `schaeffer2023mirage` | NeurIPS 2023, arXiv:2304.15004 | arXiv comment field ("NeurIPS 2023") | exact |

**One correction the verification forced.** The manuscript originally said the reward--KL frontier
of `monteiropaes2026limits` "is attained rather than merely approached". The abstract says something
weaker and more precise: the paper derives a *closed form* for the maximum expected reward at a
fixed KL budget, governed by a Jeffreys divergence rather than the sqrt(KL) of earlier analyses, and
shows empirically that best-of-N *approaches* the limit while PPO and GRPO remain substantially
suboptimal. The sentence in `sections/frontier.tex` was weakened to match the record.
