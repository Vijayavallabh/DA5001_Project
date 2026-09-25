"""feat-201: the pure helpers of analysis/blockwise_selection.py (no GPU, no download)."""
import math

from analysis.blockwise_selection import block_lengths, certificate, cut_at_eos, pick, reward_prompt


def test_blocks_partition_t_max():
    assert block_lengths(200, 10) == [10] * 20
    assert block_lengths(200, 67) == [67, 67, 66]
    assert block_lengths(200, 34) == [34] * 5 + [30]
    assert block_lengths(200, 200) == [200]
    for L in (10, 25, 34, 50, 67, 100, 200):
        assert sum(block_lengths(200, L)) == 200


def test_certificates_are_the_registered_ones():
    # (whole output, any 50-token window) in nats, as the pre-registration's table states them
    want = {(10, 64): (83.18, 24.95), (25, 64): (33.27, 12.48), (50, 64): (16.64, 8.32),
            (200, 64): (4.16, 4.16), (100, 8): (4.16, 4.16), (67, 4): (4.16, 2.77), (34, 2): (4.16, 2.08)}
    for (L, n), (whole, window) in want.items():
        got = certificate(200, L, n)
        assert round(got[0], 2) == whole and round(got[1], 2) == window, (L, n, got)
    # the matched-certificate arms spend exactly log 64 on the whole output
    for L, n in ((100, 8), (67, 4), (34, 2)):
        assert abs(certificate(200, L, n)[0] - math.log(64)) < 1e-12


def test_cut_keeps_the_first_end_of_text_and_drops_the_padding():
    assert cut_at_eos([5, 6, 1, 1, 1], {1}) == [5, 6, 1]
    assert cut_at_eos([1, 1], {1}) == [1]
    assert cut_at_eos([5, 6, 7], {1}) == [5, 6, 7]


def test_pick_is_argmax_with_ties_to_the_lowest_index_at_five_decimals():
    assert pick([0.1, 0.3, 0.2]) == 1
    assert pick([0.3, 0.1, 0.3]) == 0
    assert pick([0.300001, 0.3000049]) == 0      # equal once rounded as the cache stores them
    assert pick([-1.0, -0.5, -0.5]) == 1


def test_reward_prompt_drops_the_harness_header_only():
    assert reward_prompt("Complete the prefix:\nWrite a poem.") == "Write a poem."
    assert reward_prompt("Who was Ada Lovelace? ") == "Who was Ada Lovelace?"
