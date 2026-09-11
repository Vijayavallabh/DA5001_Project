"""Where the manuscript lives.

`~` is /home/sports on this box and is NOT the project home, so `expanduser("~/sub/satml")` names a
directory that does not exist. Three tests guarded on that path and therefore **skipped silently**
for as long as they existed: the 72-cell appendix table check, the compute-hours check against the
LLM-usage sentence, and this session's Section 4 table check. Use the `$SATML_DIR` convention the
scripts already use, and fail loudly if the manuscript is genuinely absent rather than returning.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.environ.get("SATML_DIR") or os.path.normpath(os.path.join(ROOT, os.pardir, "sub", "satml"))


def tex(name):
    return os.path.join(DIR, name)
