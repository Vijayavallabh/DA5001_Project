#!/bin/bash
set -e

echo "=== Harness Init: DA5001 Project ==="

# 1. Environment check
if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run: uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt"
  exit 1
fi
echo "[OK] .venv exists"

# 2. Data files
for f in data/*.jsonl; do
  if [ ! -s "$f" ]; then
    echo "ERROR: $f is missing or empty"
    exit 1
  fi
done
echo "[OK] All 6 data/*.jsonl files present"

# 3. GPU check
python3 -c "import torch; print(f'  GPU available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')"

# 4. Module import check
echo ""
python3 -c "from a_patch import AnchoredDecodingFactory; print('[OK] a_patch imports')"
python3 -c "from dap.shared import load_prompt_corpus; print('[OK] dap.shared imports')"

# 5. Help smoke tests
echo ""
echo "=== h1.py --help ==="
python h1.py --help 2>&1 | head -5
echo ""
echo "=== h2.py --help ==="
python h2.py --help 2>&1 | head -5

echo ""
echo "=== Init Complete ==="
