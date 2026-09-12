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
# One feature in progress for work being EDITED; parallel GPU arms are allowed (AGENTS, Working
# Rules, amended 2026-09-12) provided each has bands committed before it started, its own
# results/onset_prediction_*.md, and a line in session-handoff.md. That last condition is the one
# that can rot, so it is checked here rather than taken on trust.
handoff = open("session-handoff.md", encoding="utf-8").read()
missing = [f for f in active if f not in handoff]
assert not missing, f"in-progress but not named in session-handoff.md: {missing}"
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
echo "Plan v5, the v6 restructure and the **v7 reframe**, branch iclr-2027, target ICLR 2027.
Everything through feat-095 is done; **feat-096 is in progress**.

The paper is v7 and argues ONE claim. One line of the chain rule leaves a per-token budget two
options: either K = kT grows with the work and the certificate is VACUOUS -- empty exactly at
K = S(x), identically for every Renyi order -- or the budget is bounded and the decoder is the
anchor at all but O(1) steps, which is TRIVIAL. Both horns are measured. The deployed class takes
the first by construction: its spend is the cost of IMITATING the risky model, Theta(T) nats
independent of the utility bought, stalling at 171.3 while its certificate is written against 4000.
Three repairs -- a sharper charge, a smaller cap, a better schedule -- are corollaries and fail
where the dichotomy says they must, and so does the ONE causal placement it allows: concentrating
the whole log-8 budget on the opening gains -0.008 [-0.056, +0.041] (feat-092, THE CAUSAL HORN IS
EMPTY). The escape is a different PLACE to spend. Serving the best of n anchor draws satisfies
q(y) <= n p_s(y) -- a PATHWISE certificate of exactly log n, under any score and any tie rule,
vacuous only at n = e^S(x).

What the last session added, and what it is allowed to claim:
 * feat-095, B1 GENERALISES. The constructive claim is no longer one setup: four anchors in three
   families, 1.2B to 7B, same prompts, judge and baseline. KL3M-1.7B +0.039 [+0.005, +0.076] and
   Comma-7B +0.111 [+0.072, +0.148] on the registered scorer, all three new anchors on the second.
   Comma-7B is now the paper's largest gain. Leakage is 0.0000 at n = 1, 8, 64 at every anchor.
 * feat-094, the domain split, is UNINFORMATIVE and is reported as such. The gain does not track
   the anchor control level across seven domain cells, but the -0.79/-0.70 it shows is inseparable
   from a no-effect null already giving -0.36 +/- 0.34. The support-ceiling EXPLANATION of the
   weaker AlpacaEval gain therefore has no evidence, results/selection_alpaca_note.md is corrected
   in place, and the paper says the gain is weaker off our prompts and that we cannot say why.
   The theorem q(y) <= n p_s(y) is untouched; what was refuted is using it to explain a number.
 * The vacuity statement now covers 9,870 BookMIA passages across 100 books (98.7% seen, 100%
   unseen and control) against 758 across 16. Its seen/unseen halves are CONFOUNDED under an anchor
   that saw neither and may never be treated as a matched pair.
 * feat-091: the judge is POSITION-DOMINATED. The same two texts win 261/500 shown second and 24
   shown first. Never quote an absolute judged level across passes; Table 1 is gains, not levels.

feat-010/011 stay optional, feat-012 is superseded by feat-024, and feat-016 is human-only - never
start it. master holds the verified SaTML fallback at dd7e801.

The ICLR manuscript ~/sub/satml/iclr_2027.tex is structurally complete: main text exactly 9 of 9
pages - Ethics, Reproducibility and LLM Usage do not count, and pdftotext page 10 must carry NO
body prose at all (counting characters before 'Ethics' missed a two-line spill on 2026-09-11);
0 overfull, 0 '??', 2080 numeric literals audited with one expected miss (64256). It is NOT in this
repo and must never be committed to the stray git repo it sits inside.

If you are here to work:
1. Read AGENTS.md, then progress.md (bottom first) and session-handoff.md
2. feat-096 is OPEN and may still be running: Comma-7B on AlpacaEval-805, the arm that decides
   whether the weaker benchmark gain is the anchor's support ceiling or an in-house prompt set that
   flatters the mechanism. Its bands AND the manuscript consequence of each reading were committed
   before it started, in results/onset_prediction_alpaca_comma7b.md -- score against them and do
   not refit. Sixteen live cautions are in AGENTS.md; the four that cost time most recently are
   (m) the judge is position-dominated, (n) the page budget moves with floats, headings and table
   rows and NOT with prose, (o) h1.py writes the CLI k string verbatim into filenames so 1e-9
   becomes trajectories_k1e-09_*.jsonl, and (p) a gate that fails everything is not a gate -- the
   breadth entry gate read a column its producer never writes and had therefore never run.
   And one habit rather than a caution: RENDER A MANUSCRIPT PAGE TO PNG AND LOOK AT IT. Figure 1
   had been drawn at 9.4in and printed at 0.88 textwidth for its whole life, putting every label
   at 3-4pt, and no compile-time check saw it.
3. If you change something, rerun its evidence command, update feature_list.json and progress.md,
   and recompile the manuscript (0 '??', 0 overfull, no body prose on pdftotext page 10)
"
