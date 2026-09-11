"""The one correctness-critical line of analysis/exact_rate.py, and the manifest it runs on.

The whole point of the arm is the difference between two quantities that coincide in the limit the
refuted refinement assumed: the exact D_KL(p_r||p_s) integrates over the vocabulary under p_r, the
plug-in reads the single column of the realised token. Getting that backwards, or silently
comparing densities on differently padded supports, would make the diagnostic answer its own
question."""
import csv
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

from analysis.exact_rate import rates, spearman  # noqa: E402

MANIFEST = "results/exact_rate_pairs.tsv"


def _lp(rows):
    return torch.log(torch.tensor(rows, dtype=torch.float64))


def test_the_exact_rate_and_the_plug_in_are_what_they_claim_to_be():
    p_s = [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]
    p_r = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]]
    tgt = torch.tensor([0, 1])
    kl, plug = rates(_lp(p_s), _lp(p_r), tgt)
    want_kl = [sum(r * math.log(r / s) for r, s in zip(pr, ps)) for pr, ps in zip(p_r, p_s)]
    want_plug = [math.log(p_r[0][0] / p_s[0][0]), math.log(p_r[1][1] / p_s[1][1])]
    assert kl == pytest.approx(want_kl, abs=1e-9), (kl, want_kl)
    assert plug == pytest.approx(want_plug, abs=1e-9), (plug, want_plug)
    # and they disagree here, which is the entire reason for the measurement
    assert abs(kl[1] - plug[1]) > 0.5, (kl[1], plug[1])


def test_they_coincide_when_the_risky_model_is_a_point_mass_on_the_target():
    """Eq. (req)'s derivation assumes exactly this. As p_r concentrates on x_t, the plug-in
    converges to the exact rate -- so where the memoriser is near-deterministic the refuted
    estimator is fine, and the diagnostic is only about the pairs where it is not."""
    prev = None
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        p_r = [[1 - 2 * eps, eps, eps]]
        p_s = [[0.5, 0.3, 0.2]]
        kl, plug = rates(_lp(p_s), _lp(p_r), torch.tensor([0]))
        gap = abs(kl[0] - plug[0])
        if prev is not None:
            assert gap < prev, (eps, gap, prev)
        prev = gap
    assert prev < 1e-2, prev


def test_a_padded_vocabulary_is_renormalised_rather_than_compared_across_supports():
    """Comma-7B carries 64256 embedding rows for a 64000-token vocabulary; a raw gather across two
    different widths would compare densities that do not sum to the same thing."""
    p_s = [[0.5, 0.3, 0.2]]
    p_r_wide = [[0.35, 0.35, 0.2, 0.1]]          # four columns, one of them padding
    kl, plug = rates(_lp(p_s), _lp(p_r_wide), torch.tensor([0]))
    ren = [x / sum(p_r_wide[0][:3]) for x in p_r_wide[0][:3]]
    want = sum(r * math.log(r / s) for r, s in zip(ren, p_s[0]))
    assert kl[0] == pytest.approx(want, abs=1e-9), (kl[0], want)
    assert plug[0] == pytest.approx(math.log(ren[0] / p_s[0][0]), abs=1e-9)


def test_spearman_matches_a_known_value():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_the_manifest_names_the_nine_onset_pairs_with_their_own_anchors():
    rows = [l.rstrip("\n").split("\t") for l in open(MANIFEST, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    assert len(rows) == 9, len(rows)
    table = {r["pair"]: r for r in csv.DictReader(open("results/onset_table.csv"))}
    for name, anchor, mem, bpf, onset in rows:
        assert name in table, name
        assert float(onset) == float(table[name]["onset"]), (name, onset)
        assert os.path.isdir(mem), mem
        base = json.load(open(os.path.join(mem, "recipe.json")))["base"]
        # self-paired everywhere but pair 1, where the LoRA sits on the RISKY model
        if name.startswith("TinyComma"):
            assert anchor != base and "tinycomma" in anchor, (anchor, base)
        else:
            assert anchor == base, (name, anchor, base)
