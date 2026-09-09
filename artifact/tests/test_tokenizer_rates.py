"""feat-061: the group boundaries this script reports are the ones the six measured pairs fall
into, so they must stay tied to those measurements rather than drifting to whatever is convenient."""
import csv, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.tokenizer_rates import COARSE_MIN, FINE_MAX


def test_boundaries_bracket_the_measured_gap():
    """The measured pairs sit at 1.98 and at 3.62-4.18 characters per token. The boundaries must
    leave both groups intact and call the space between them empty."""
    assert FINE_MAX < 2.4 or FINE_MAX == 2.4
    assert COARSE_MIN <= 3.62 and COARSE_MIN > FINE_MAX
    assert 1.98 <= FINE_MAX and 4.18 >= COARSE_MIN


def test_measured_pairs_are_grouped_as_the_paper_reports():
    p = os.path.join(REPO, "results", "tokenizer_rates.csv")
    if not os.path.exists(p):
        return
    r = {x["tokenizer"]: x for x in csv.DictReader(open(p))}
    for t in ("alea-institute/kl3m-003-1.7b", "alea-institute/kl3m-002-520m"):
        assert r[t]["group"] == "fine", r[t]
    for t in ("common-pile/comma-v0.1-2t", "PleIAs/Pleias-1.2b-Preview",
              "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"):
        assert r[t]["group"] == "coarse", r[t]


def test_vocabulary_size_and_granularity_are_not_the_same_variable():
    """The reason a third family was chosen on evidence: two ~32k vocabularies that cut this text
    at 1.98 and 3.78 characters per token. If this ever stopped holding, the Phi pair would no
    longer isolate anything."""
    p = os.path.join(REPO, "results", "tokenizer_rates.csv")
    if not os.path.exists(p):
        return
    small = [x for x in csv.DictReader(open(p)) if int(x["vocab"]) < 40000]
    groups = {x["group"] for x in small}
    assert groups == {"fine", "coarse"}, small
