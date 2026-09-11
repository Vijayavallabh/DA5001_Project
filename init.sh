#!/bin/bash
# Baseline verification for the DA5001 / SaTML-2027 audit repo. Fails fast.
set -e

cd "$(dirname "$0")"
echo "=== Harness Init: DA5001 Project (SaTML 2027 audit) ==="
echo "cwd: $(pwd)"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "ERROR: .venv missing. Run: uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt"
  exit 1
fi
echo "[OK] .venv present ($($PY --version 2>&1))"

# 1. Data files
for f in copybench_attack_train copybench_test copybench_val neutral creative factscore; do
  [ -s "data/$f.jsonl" ] || { echo "ERROR: data/$f.jsonl missing or empty"; exit 1; }
done
echo "[OK] 6 data/*.jsonl files present"

# 2. Static check (compile) and imports
$PY -m compileall -q a_patch dap analysis figures recipes >/dev/null && echo "[OK] compileall a_patch dap analysis figures recipes"
$PY -c "from a_patch import AnchoredDecodingFactory; from dap.shared import load_prompt_corpus, SOURCE_FILES; assert len(SOURCE_FILES)==6; from dap.stats import build_trajectory_seeds; print('[OK] a_patch / dap import')"

# 3. Harness state files
$PY - <<'EOF'
import json, sys
d = json.load(open("feature_list.json"))
ids = {f["id"] for f in d["features"]}
bad = [(f["id"], dep) for f in d["features"] for dep in f.get("dependencies", []) if dep not in ids]
assert not bad, f"unknown dependencies: {bad}"
active = [f["id"] for f in d["features"] if f["status"] == "in-progress"]
assert len(active) <= 1, f"more than one feature in-progress: {active}"
allowed = {"not-started", "in-progress", "blocked", "done"}
assert all(f["status"] in allowed for f in d["features"]), "bad status value"
print(f"[OK] feature_list.json valid: {len(ids)} features, in-progress={active or 'none'}")
EOF
for f in AGENTS.md progress.md session-handoff.md GOAL.md; do [ -s "$f" ] || { echo "ERROR: $f missing"; exit 1; }; done
echo "[OK] harness files present"

# 4. Plan and manuscript locations (outside the repo)
SUB=/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml
[ -s "$SUB/IMPROVEMENT_PLAN.md" ] || { echo "ERROR: $SUB/IMPROVEMENT_PLAN.md missing"; exit 1; }
[ -s "$SUB/satml_2027.tex" ] || { echo "ERROR: $SUB/satml_2027.tex missing"; exit 1; }
echo "[OK] plan + manuscript found in $SUB"

# 5. Tests (present after feat-003)
if [ -d tests ]; then
  $PY -m pytest -q tests
  echo "[OK] pytest"
else
  echo "[--] no tests/ yet (created by feat-003)"
fi

# 6. Entry points
$PY h1.py --help >/dev/null && echo "[OK] h1.py --help"
$PY h2.py --help >/dev/null && echo "[OK] h2.py --help"

# 7. GPU (informational; all runs were local: 4xA100 80GB at nvidia-smi indices 0,1,2,4 - index 3 is a 4 GB T400.
#    Always export CUDA_DEVICE_ORDER=PCI_BUS_ID with CUDA_VISIBLE_DEVICES, or index 4 lands on the T400.)
$PY -c "import torch; print(f'[info] local GPU available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')" || echo "[info] torch not importable locally"

# 8. Guard: files that belong to THIS repo must not end up in the manuscript tree, which lives
#    inside an unrelated git repo. A `cd` into ~/sub/satml that persists through a command block has
#    put progress.md there three times; each was caught by hand. This catches it automatically.
SAT="/mnt/md0/IITM/BackUp/Home/vijayavallabh/sub/satml"
for f in progress.md feature_list.json session-handoff.md AGENTS.md init.sh; do
  if [ -f "$SAT/$f" ] && [ "$(wc -l < "$SAT/$f")" -gt 1 ]; then
    echo "[FAIL] $SAT/$f looks like this repo's $f written into the manuscript tree." >&2
    echo "       Recover its content into ./$f, then: git -C $(dirname "$SAT") checkout -- sub/satml/$f" >&2
    exit 1
  fi
done
echo "[OK] no repo files stranded in the manuscript tree"

echo "=== Init Complete ==="
echo ""
echo "Plan v5 in progress (2026-09-08), branch iclr-2027, target ICLR 2027 (abstract Sep 18, paper Sep 25).
feat-035..087 are done except the optional ones; nothing is in progress. The last thread finished
was feat-087: the paper measures the audited decoder paying 165 nats for a gain the Cramer rate
function prices at 0.052, and then declines to make the constructive claim. SELECTION ANCHORING
makes it: draw n completions from the anchor, score them, serve the argmax, and the certificate is
Proposition 1 with K = log n -- vacuous only at n = e^S(x), about e^850, and it does not grow with
the work the way kT does. Ranked by the RISKY MODEL'S OWN LIKELIHOOD it buys nothing (its committed
band is refuted), because that likelihood predicts the judge at AUC 0.526, worse than the
completion's length. Ranked by a quality model and scored by a judge that did no ranking, n=8 gains
+0.081 [0.034, 0.130] for 1.204 nats where the metered decoder's best arm gains +0.072 for 171.3.
Recall is 0.0000 at every n up to 64. Before that, feat-086: the paper's largest stated limitation turned into an intervention. Hand every one of
the nine pairs' adversaries the SAME NUMBER OF WORDS instead of the same twenty tokens and the
spread in onset/s(x) goes 0.289 -> 0.113, S_match/S_20 = 0.392 against a band of <= 0.5 committed
before either new arm was swept -- 61% of what reads as a property of the anchors is the benchmark's
seed convention. Every pair that moved moved DOWN; the five that did not move were already at the
matched context, which makes them the control. Before that, feat-084/085: the onset ratio's two-value split was a GAP IN THE ANCHORS, not in the phenomenon.
A survey of 21 openly licensed models found exactly one in the 2.4-3.4 chars/token gap Limitations
called unfillable; built into a pair with everything committed beforehand it lands at 1.027, between
the clusters. Its one confound that fired -- the weakest memoriser in the set -- was answered by the
contingent control committed in the same file BEFORE that sweep: open-calm-3b, same tokenizer, s(x)
within 0.5%, a memoriser 5.1x stronger, lands at 0.993, also between the clusters. Nine pairs now,
and the context ranking strengthens to -0.958 at exact p = 0.0002.
Before that, feat-082: the onset re-measured on a second, disjoint protected corpus, where the
paper's central split reproduces -- the fine-tokenizer pair again the only one above 1 with its
interval excluding it, every ratio within 0.05 of its twin on the novels. Before that, the matched-utility
line (feat-072..081): twelve pairs in seven families, an earned negative on every predictor, and
three robustness axes that move levels by more than the precision floor and never move the ranking. feat-010/011 stay optional, feat-012 is superseded by feat-024, and
feat-016 is human-only - never start it. master holds the verified SaTML fallback at dd7e801.

The ICLR manuscript ~/sub/satml/iclr_2027.tex is structurally complete: main text exactly 9 of 9
pages - Ethics, Reproducibility and LLM Usage do not count, and pdftotext page 10 must carry NO
body prose at all (counting characters before 'Ethics' missed a two-line spill on 2026-09-11);
38 pages total, 0 overfull, 0 '??', 1807 numeric literals audited with one expected miss. It is NOT in this repo and must never be committed to the
stray git repo it sits inside.

If you are here to work:
1. Read AGENTS.md, then progress.md (bottom first) and session-handoff.md
2. No feature is open. The read-through is done end to end (2026-09-11, thirteen corrections):
   every table in the compiled document has been checked against its own CSV mechanically.
   Four live cautions: a paper number must round
   from its CSV ONCE -- double rounding put six of Appendix D's 72 cells one off, and reading
   a table against its CSV mechanically is what found that and five stale seed-words besides; a pair enters the onset analysis only if its
   SAMPLED k=-1 recall >= 0.10 (greedy recall lies); every arm is pre-registered with a
   refuting band in results/onset_prediction_*.md, so score against the band and do not
   refit; and pkill -f matches the shell that runs it -- kill by PID.
3. If you change something, rerun its evidence command, update feature_list.json and progress.md,
   and recompile the manuscript (0 '??', 0 overfull, no body prose on pdftotext page 10)
"
