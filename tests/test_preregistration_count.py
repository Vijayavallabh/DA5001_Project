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
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w},
         40: "forty", **{40 + i: f"forty-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w},
         50: "fifty", **{50 + i: f"fifty-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w},
         60: "sixty", **{60 + i: f"sixty-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w},
         70: "seventy", **{70 + i: f"seventy-{w}" for i, w in enumerate(
             ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]) if w}}


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


# RETIRED 2026-09-19 with the de-jargoning pass: the Reproducibility Statement no longer
# counts the protocol logs, because naming an internal record count is bookkeeping a reader
# cannot use. The statement still says every interval was fixed before the run that produced
# it, and test_every_unscored_log_is_accounted_for_in_the_handoff below still keeps the repo's
# own records honest -- which is where that discipline belongs.
#   test_the_statement_counts_the_logs_that_exist
