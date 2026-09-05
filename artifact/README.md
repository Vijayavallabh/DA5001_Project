# Artifact: What does a KL budget certify? An adversarial audit of inference-time near-access-freeness

Anonymised code, results, prompt sets, and recipes for the SaTML 2027 submission. Every number in the paper
traces to a file in `results/`, and every figure is rebuilt from those files by `figures/make_figures.py`.

## Layout

| Path | Content |
|---|---|
| `a_patch/` | The audited Anchored Decoding library (unchanged mechanism; `k_radius=-1` = risky only, `0` = anchor only). |
| `dap/` | Experiment code: `h1.py -> dap/e1.py` (fixed workload, baselines, copying metrics), `h2.py -> dap/e2/` (prompt search; Bernstein proxy retired), `dap/stats.py` (seeds, metrics, per-trajectory `budget_check`, anytime-valid confidence sequence). |
| `analysis/` | One script per audit: `reanalyze_logs.py` (C1/C2/C4 on released logs), `certificate_cap.py` (C3), `regime_sweep.py` (C2 sweep), `llr_tails.py` (C4), `composition_attack.py` (C5b), `bank_burst.py` (C5a; attempted, not evaluated in the paper: the fine-tuned memoriser ignores filler instructions, so nothing is banked; usable with a risky model that memorises and follows instructions). |
| `recipes/` | `finetune_memorizing.py` + `memorizing_model.md`: the memorising risky model (weights not redistributed: they reproduce copyrighted text). |
| `results/` | All CSV tables cited in the paper (see below). |
| `figures/` | Paper figures (PDF/PNG) and `make_figures.py`. |
| `data/` | Prompt sets (CopyBench book split, FactScore, WritingPrompts, neutral QA) exactly as sampled. |
| `tests/` | `pytest -q tests` (seeds, invariants on a log sample, metrics, confidence sequence, budget checks). |

## Results files

`regime_table.csv`, `llr_tails.csv`, `prefix_debt_forced_tokens.csv`, `surprisal.csv`, `seed_collisions.csv`,
`per_trajectory.csv` (released logs); `certificate_caps.csv`, `certificate_cap_summary.csv`,
`certificate_caps_memoriser.csv` (C3); `regime_sweep.csv` (C2 sweep); `llr_ratio_samples.csv` (C4);
`memorizing_model_recall.csv` (memoriser check); `composition.csv`, `composition_summary.csv` (C5b).

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install torch transformers peft accelerate numpy matplotlib pytest pypdf feedparser
.venv/bin/python -m pytest -q tests
# released logs (2.4 GB output.zip, shipped next to this artifact) -> results/regime_table.csv etc.
unzip -o output.zip 'output/h1_outputs/trajectories_k*.jsonl' && .venv/bin/python analysis/reanalyze_logs.py --logs output --out results
# certificate strength (one A100, ~2 min)
.venv/bin/python analysis/certificate_cap.py --data data --out results
# small-budget sweep (two A100s, ~10 h) and its summaries
scripts/run_regime_sweep.sh 0 1
.venv/bin/python analysis/regime_sweep.py --run plain=output/sweep_plain --run chat=output/sweep_chat --out results
.venv/bin/python analysis/llr_tails.py --sweep plain=output/sweep_plain --sweep chat=output/sweep_chat --out results
# memorising model (one A100, ~75 min) and its check
.venv/bin/python recipes/finetune_memorizing.py --out output/memorizing_llama8b && scripts/run_memorizing_check.sh 0
# composition attack (one A100, ~70 min)
.venv/bin/python analysis/composition_attack.py --risky-model output/memorizing_llama8b --limit 100 --k-values -1 0 0.15 0.5 1 3 5 10 20 --windows 20 50 --out results
# figures
.venv/bin/python figures/make_figures.py --copy-to ""
```

Models: anchor `jacquelinehe/tinycomma-1.8b-llama3-tokenizer`; risky `meta-llama/Llama-3.1-8B-Instruct` (gated; set `HF_TOKEN`);
optimiser `Qwen/Qwen2.5-7B-Instruct`. Set `CUDA_VISIBLE_DEVICES` for every job. The paper compiles with
`pdflatex + bibtex` or `tectonic -X compile satml_2027.tex`.

## Integrity

`MANIFEST.sha256` lists every file; verify with `sha256sum -c MANIFEST.sha256`.
