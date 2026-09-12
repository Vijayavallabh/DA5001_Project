"""The Reproducibility Statement counts the pre-registration logs, in words.

audit_numbers.py cannot see a number spelled out, so this is the only check on it, and it was
wrong: the statement said nineteen when twenty-seven existed. Every arm in this paper commits its
bands, its grid, its entry gate and its excluded alternatives before the run that scores it, so the
count grows with the work and the sentence has to grow with the count.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.manuscript import tex  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = {12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
         17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
         **{20 + i: f"twenty-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w},
         30: "thirty", **{30 + i: f"thirty-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w}}


def test_the_statement_counts_the_logs_that_exist():
    n = len(glob.glob(os.path.join(ROOT, "results", "onset_prediction_*.md")))
    assert n in WORDS, f"extend WORDS past {n}"
    body = open(tex("iclr_2027.tex"), encoding="utf-8").read().replace("\n", " ")
    m = re.search(r"the\s+([a-z-]+)\s+pre-registration logs", body)
    assert m, "the Reproducibility Statement no longer counts the logs"
    assert m.group(1) == WORDS[n], (m.group(1), WORDS[n], n)


def test_every_unscored_log_is_accounted_for_in_the_handoff():
    """A pre-registration with no scoring section is an arm that was committed and never reported.
    That is fine while it runs and not otherwise, so every unscored log must be named in
    session-handoff.md -- which is where the next session looks for what is in flight."""
    handoff = open(os.path.join(ROOT, "session-handoff.md"), encoding="utf-8").read()
    unscored = []
    for p in glob.glob(os.path.join(ROOT, "results", "onset_prediction_*.md")):
        t = open(p, encoding="utf-8").read()
        if "## Scoring, " not in t and "## Scoring log" in t:
            unscored.append(os.path.basename(p))
    missing = [f for f in unscored if f not in handoff]
    assert not missing, f"unscored and unaccounted for: {missing}"


def test_the_unregistered_arm_is_labelled_as_one():
    """results/selection_alpaca_note.md is deliberately not an onset_prediction_*.md, because it
    had no committed bands, and it must keep saying so in its first lines."""
    p = os.path.join(ROOT, "results", "selection_alpaca_note.md")
    head = open(p, encoding="utf-8").read()[:1600]
    assert "no committed bands" in head
    assert not glob.glob(os.path.join(ROOT, "results", "onset_prediction_*alpaca_note*"))
