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
$PY -m compileall -q a_patch dap analysis figures >/dev/null && echo "[OK] compileall a_patch dap analysis figures"
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

# 7. GPU (informational; compute target is the DGX via the dgx-gpu skill)
$PY -c "import torch; print(f'[info] local GPU available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')" || echo "[info] torch not importable locally"

echo "=== Init Complete ==="
echo ""
echo "Next steps:"
echo "1. Read feature_list.json; pick ONE feature whose dependencies are done"
echo "2. Implement only that feature; run its evidence command"
echo "3. Paste command + output into feature_list.json evidence, update progress.md"
echo "4. Re-run ./init.sh before claiming done"
