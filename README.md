This is my DA5001 Project repository

Environment Setup
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

The output.zip file contains the compressed results

## Reproducing Results

### E1 — Anchored-Decoding Certification
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python h1.py \
  --data-dir data \
  --output-dir output/h1_outputs \
  --risky-model-path meta-llama/Llama-3.1-8B-Instruct \
  --safe-model-path jacquelinehe/tinycomma-1.8b-llama3-tokenizer \
  --trust-remote-code --parallelize > out.log 2>&1 & echo $! > h1.pid
```

### E2 — Adversarial Prompt Optimization (default K=3)
```bash
CUDA_VISIBLE_DEVICES=2,3 nohup python h2.py \
  --data-dir data \
  --output-dir output/h2_outputs \
  --trust-remote-code --parallelize > out_h2.log 2>&1 & echo $! > h2.pid
```

### E2 — Adversarial Prompt Optimization (K=5)
```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python h2.py \
  --k 5 \
  --data-dir data \
  --output-dir output/h2_k5_outputs \
  --trust-remote-code --parallelize > out_h2k.log 2>&1 & echo $! > h2_k5.pid
```