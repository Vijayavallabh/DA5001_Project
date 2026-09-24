"""feat-195: the harness can decode at He et al.'s book settings (temperature 0.7, repetition penalty
1.1), and the default leaves every run on record unchanged (no penalty processor at all)."""
import sys

import pytest

sys.argv = ["h1.py"]
from dap.e1 import AuditConfig, parse_args  # noqa: E402


def test_default_is_no_penalty():
    sys.argv = ["h1.py"]
    cfg = parse_args()
    assert cfg.repetition_penalty == 1.0 and AuditConfig.repetition_penalty == 1.0


def test_flag_reaches_the_config():
    sys.argv = ["h1.py", "--temperature", "0.7", "--repetition-penalty", "1.1"]
    cfg = parse_args()
    assert (cfg.temperature, cfg.repetition_penalty) == (0.7, 1.1)


def test_the_factory_adds_the_processor_only_when_asked():
    pytest.importorskip("transformers")
    from transformers import GenerationConfig, RepetitionPenaltyLogitsProcessor

    from a_patch.factory import AnchoredDecodingFactory
    f = AnchoredDecodingFactory.__new__(AnchoredDecodingFactory)
    on = f._prepare_logits_processor(None, GenerationConfig(repetition_penalty=1.1))
    off = f._prepare_logits_processor(None, GenerationConfig(repetition_penalty=1.0))
    assert [type(x) for x in on] == [RepetitionPenaltyLogitsProcessor] and len(off) == 0
