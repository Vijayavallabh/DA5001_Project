"""feat-165: TokenSwap as a decoder, checked against its authors' definition.

The arm exists because Appendix J said the method could not be measured without a vetted
auxiliary, and that is now false. What these guard is that what we run IS their method: their
110-word set taken verbatim rather than reconstructed, and their Algorithm 1's two invariants --
the mass on G is preserved and nothing off G moves.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.tokenswap_decode import g_token_ids, load_G, swap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_G_is_their_published_set_verbatim():
    """110 unique words, from their Appendix C.3. A reconstruction from the DESCRIPTION (top-500
    COCA, NLTK POS filter) would be our set, not theirs, and the paper would be measuring a method
    of ours under their name."""
    w = load_G()
    assert len(w) == 110 and len(set(w)) == 110
    # spot-check the two the paper itself names as examples, and the shape of the list
    assert w[0] == "the" and "in" in w
    assert all(x.isalpha() and x.islower() for x in w), "G is a word list, not token pieces"
    # their POS classes must actually be present: determiners, modals, wh-words, auxiliaries
    for w_ in ("the", "a", "an", "might", "must", "which", "whether", "being", "having"):
        assert w_ in w, f"{w_} is in their list and not in ours"


def test_the_swap_preserves_the_mass_on_G_and_moves_nothing_off_it():
    """Their alpha exists precisely to keep p_final a distribution. Both halves are invariants:
    sum_G p_final == sum_G p_main, and p_final == p_main everywhere else."""
    torch.manual_seed(0)
    V, G = 50, torch.tensor([3, 7, 11, 29])
    p_main = torch.softmax(torch.randn(V), 0)
    p_aux = torch.softmax(torch.randn(V), 0)
    out, m = swap(p_main, p_aux, G)
    assert abs(float(out[G].sum()) - float(p_main[G].sum())) < 1e-6, "mass on G moved"
    assert abs(m - float(p_main[G].sum())) < 1e-9
    off = torch.ones(V, dtype=torch.bool)
    off[G] = False
    assert torch.allclose(out[off], p_main[off]), "a probability off G changed"
    assert abs(float(out.sum()) - 1.0) < 1e-5, "p_final is not a distribution"
    # and it really is the auxiliary that decides WITHIN G: the ordering on G follows p_aux
    assert torch.argsort(out[G]).tolist() == torch.argsort(p_aux[G]).tolist()


def test_the_swap_is_a_no_op_when_the_auxiliary_puts_nothing_on_G():
    V, G = 20, torch.tensor([2, 5])
    p_main = torch.softmax(torch.randn(V), 0)
    p_aux = torch.zeros(V)
    p_aux[0] = 1.0
    out, _ = swap(p_main, p_aux, G)
    assert torch.allclose(out, p_main), "a degenerate auxiliary must leave the served law alone"


def test_G_maps_into_the_llama_vocabulary_without_losing_the_set():
    """The auxiliary shares Llama-3's tokenizer, which is why this arm can use the identity
    mapping their paper only approximates. If most of G stopped being single tokens the rule would
    be weaker than the one they specify, so the loss is asserted rather than logged."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    cache = os.path.join(ROOT, "hf_cache")
    if not os.path.isdir(cache):
        pytest.skip("no local model cache")
    os.environ.setdefault("HF_HUB_CACHE", cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = AutoTokenizer.from_pretrained("jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"tokenizer unavailable offline: {e}")
    ids, missing = g_token_ids(tok, load_G())
    assert len(missing) <= 5, f"{len(missing)} of 110 words are not single tokens: {missing}"
    assert len(ids) >= 200, f"G should reach both cased and spaced variants, got {len(ids)}"
    # the empty set would make the rule a no-op, which G1 exists to catch at run time too
    assert ids == sorted(set(ids))


def test_a_shared_vocabulary_pairing_reduces_to_the_identity():
    """g_pairs exists so an auxiliary with its own vocabulary can be run at all. When the two
    tokenizers ARE the same it must produce exactly what the shared-vocabulary path produces,
    paired with itself -- otherwise adding the ladder would silently change the arm already run."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from analysis.tokenswap_decode import g_pairs
    cache = os.path.join(ROOT, "hf_cache")
    if not os.path.isdir(cache):
        pytest.skip("no local model cache")
    os.environ.setdefault("HF_HUB_CACHE", cache)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = AutoTokenizer.from_pretrained("jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"tokenizer unavailable offline: {e}")
    words = load_G()
    ids, _ = g_token_ids(tok, words)
    mi, ai, missing = g_pairs(tok, tok, words)
    assert mi == ai, "a vocabulary paired with itself must map every token to itself"
    assert sorted(mi) == ids, "the paired path disagrees with the shared-vocabulary path"
    assert not missing


def test_the_swap_uses_the_auxiliary_s_OWN_ids_when_the_vocabularies_differ():
    """The whole point of the pairing: p_final[main_id] takes its value from p_aux[aux_id]. If the
    swap indexed p_aux with the MAIN ids it would read whatever token happens to sit at that index
    in the other vocabulary, which is noise that would still look like a working defence."""
    V = 40
    p_main = torch.softmax(torch.randn(V), 0)
    p_aux = torch.zeros(V)
    p_aux[31] = 0.75                     # the auxiliary's id for a word that is id 4 in the main
    p_aux[32] = 0.25
    G, GA = torch.tensor([4, 5]), torch.tensor([31, 32])
    out, _ = swap(p_main, p_aux, G, GA)
    m = float(p_main[G].sum())
    assert abs(float(out[4]) - 0.75 * m) < 1e-6 and abs(float(out[5]) - 0.25 * m) < 1e-6
    # and indexing p_aux with the main ids would have read zeros, i.e. left the law alone
    same, _ = swap(p_main, p_aux, G, None)
    assert torch.allclose(same, p_main), "the control for this test is not what it claims"


class _StubTok:
    """A vocabulary that spells some of G's words in two pieces, which is the case the pairing
    exists to handle: a word that is not a single token in the AUXILIARY cannot have its
    probability swapped, whatever it is in the main model."""

    def __init__(self, two_piece):
        self.two_piece = set(two_piece)

    def encode(self, s, add_special_tokens=False):
        return [1, 2] if s.strip().lower() in self.two_piece else [abs(hash(s)) % 9000]


def test_a_word_single_token_in_only_one_vocabulary_is_not_paired():
    from analysis.tokenswap_decode import g_pairs
    words = load_G()
    main = _StubTok([])                       # every word is one token in the main vocabulary
    aux = _StubTok(words[:30])                # thirty of them are two tokens in the auxiliary
    mi, ai, missing = g_pairs(main, aux, words)
    assert len(mi) == len(ai), "the two id lists must stay aligned"
    assert set(missing) == set(words[:30]), (
        "a word that is not a single token in BOTH vocabularies must be reported missing, not "
        "paired against whatever id the main model happens to use")
    # up to four surface forms survive per word (bare, spaced, capitalised, spaced-capitalised),
    # so the count is not len(words) - 30; what must hold is that NONE of the thirty appear
    assert 0 < len(mi) <= 4 * (len(words) - 30)


def test_the_gate_table_rounds_from_the_arms_own_trajectories():
    """feat-165's gate table quoted a run the arm itself had declared INVALID.

    `scripts/run_tokenswap.sh` appends to one log per half, so the leakage log holds three runs and
    the two gate values were read off the middle one -- the pipeline-mismatched 17:19 block -- while
    every band came from the 18:04 relaunch. Both gates passed either way, so the verdict could not
    catch it. This pins the printed cells to `results/tokenswap_gates.csv`, which
    `analysis/tokenswap_gates.py` recomputes from the trajectories, so a gate value has to round
    from the arm it is about (caution (j) applied to gates rather than to paper numbers).
    """
    import csv as _csv
    gates = os.path.join(ROOT, "results", "tokenswap_gates.csv")
    rows = {(r["arm_dir"], r["arm"]): r for r in _csv.DictReader(open(gates, encoding="utf-8"))}
    leak = rows[("leakage", "tokenswap")]

    txt = open(os.path.join(ROOT, "results", "onset_prediction_tokenswap.md"),
               encoding="utf-8").read()
    scoring = txt.partition("\n## Scoring log")[2]
    assert scoring, "the scoring section is missing"
    # Scope to the gate TABLE, not the scoring section. The first version of this test asserted
    # over the whole section and the paragraph that RECORDS the correction quotes both values, so
    # restoring the superseded `25.62%` into the table left it passing on the other occurrence --
    # caution (an), inside a guard written for caution (ag). Found by mutation-testing the guard.
    head, _, rest = scoring.partition("| gate | value | reading |")
    assert rest, "the gate table is missing"
    table = rest.split("\n\n")[0]

    # The swap arm's own mass and bind rate, to the precision the table prints them at.
    assert f"gamma = {float(leak['mass_on_g']):.4f}" in table, \
        f"G0 does not quote the swap arm's mass ({float(leak['mass_on_g']):.4f})"
    assert f"{100 * float(leak['bind_rate']):.2f}%" in table, \
        f"G1 does not quote the swap arm's bind rate ({100 * float(leak['bind_rate']):.2f}%)"
    assert f"`{leak['g_token_ids']}` token ids" in table, "G0 does not quote |G|"

    # The control's mass is close to the swap arm's, which is how the wrong row was quoted in the
    # first place. Assert the table does NOT carry it, so the same substitution fails by name.
    ctrl = float(rows[("leakage", "-1")]["mass_on_g"])
    assert f"gamma = {ctrl:.4f}" not in table, \
        "G0 quotes the rule-OFF arm's mass, which is the defect this test exists for"
