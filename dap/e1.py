import argparse
import gc
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GenerationConfig

from a_patch import AnchoredDecodingFactory
from dataclasses import replace

from .shared import CLASS_ORDER, PromptRecord, chat_eos_ids, load_prompt_corpus, true_gen_len, wrap_chat
from .stats import ebb_upper_bound_chapman, build_trajectory_seeds, copying_metrics
from .sampling import apply_e1_sampling, validate_sample_counts


@dataclass
class AuditConfig:
    data_dir: str = "data"
    output_dir: str = "output/h1_outputs"
    resume_from_trajectories: bool = False
    safe_model_path: str = "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"
    risky_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    k_values: Tuple[float, ...] = (1.0, 3.0, 5.0)
    trajectories_per_prompt: int = 30
    seeds: Tuple[int, ...] = (42, 43, 44)
    prefix_n: int = 5
    use_prefix_debt: bool = True
    temperature: float = 1.0
    max_new_tokens: int = 200
    delta: float = 0.05
    num_classes: int = 6
    batch_size: int = 8
    length_bucket_width: int = 32
    verbose: bool = False
    trust_remote_code: bool = True
    device: str = "cuda"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    parallelize: bool = True
    save_full_trajectories: bool = True
    factscore_field: str = "factscore_prompt"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    cap_neutral: int = 200
    cap_val: int = 150
    cap_test: int = 150
    cap_attack_train: int = 100
    cap_factual: int = 150
    cap_creative: int = 150
    use_chat_template: bool = False
    skip_existing: bool = False
    greedy: bool = False

    @property
    def num_hypotheses(self) -> int:
        return self.num_classes * len(self.k_values)

    @property
    def effective_num_classes(self) -> int:
        return len(CLASS_ORDER)

    @property
    def bonferroni_delta(self) -> float:
        return self.delta / self.num_hypotheses


class H1AuditRunner:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        dtype = getattr(torch, config.dtype)
        print("[stage] loading models", flush=True)
        self.factory = AnchoredDecodingFactory.from_pretrained(
            safe_model_path=config.safe_model_path,
            risky_model_path=config.risky_model_path,
            k_radius=max(0.0, config.k_values[0]),  # constructor rejects -1; generate(k_radius=k) sets the real k per run
            verbose=config.verbose,
            use_prefix_debt=config.use_prefix_debt,
            prefix_n=config.prefix_n,
            log_kl_stats=True,
            device=config.device,
            dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            token=os.getenv("HF_TOKEN"),
        )
        self.tokenizer = self.factory.tokenizer
        # Z is in [0, K] deterministically, so K is the only range used (feat-003; E1 used T*ln|V| before)
        self.eos_ids = chat_eos_ids(self.tokenizer) if config.use_chat_template else self.tokenizer.eos_token_id
        print(f"[stage] models ready; chat_template={config.use_chat_template} eos={self.eos_ids}", flush=True)

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            do_sample=not self.config.greedy,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_ids,
        )

    def _estimate_prefix_debt(self, final_budget: float, gen_len: int, k: float) -> float:
        init_budget = final_budget - (gen_len * k)
        return max(0.0, -init_budget)

    def _prompt_token_length(self, text: str) -> int:
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        return int(ids.shape[0])

    def _build_jobs(self, prompts: List[PromptRecord]) -> List[Dict[str, Any]]:
        jobs = []
        for prompt in prompts:
            traj_seeds = build_trajectory_seeds(prompt.prompt_id, self.config.seeds, self.config.trajectories_per_prompt)
            prompt_len = self._prompt_token_length(prompt.prompt_text)
            for traj_idx, seed in enumerate(traj_seeds):
                jobs.append({"prompt": prompt, "seed": int(seed), "trajectory_id": int(traj_idx), "prompt_len": prompt_len})
        return jobs

    def _group_jobs_by_seed(self, jobs: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for job in jobs:
            grouped[job["seed"]].append(job)
        return grouped

    def _make_length_buckets(self, jobs: List[Dict[str, Any]], bucket_width: int = 32) -> List[List[Dict[str, Any]]]:
        by_bucket = defaultdict(list)
        for job in jobs:
            bucket_id = job["prompt_len"] // bucket_width
            by_bucket[bucket_id].append(job)
        ordered = []
        for bucket_id in sorted(by_bucket.keys()):
            bucket_jobs = sorted(by_bucket[bucket_id], key=lambda x: (x["prompt_len"], x["prompt"].prompt_id, x["trajectory_id"]))
            ordered.append(bucket_jobs)
        return ordered

    def _slice_batches(self, jobs: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        return [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

    def _records_from_batch(self, batch_jobs, output, stats, k):
        records = []
        per_step_stats = stats.get("per_step") or []
        final_cum_raw = stats.get("final_cum_kl_spent_per_seq") or [0.0] * len(batch_jobs)
        final_budget_raw = stats.get("final_budget_per_seq") or [0.0] * len(batch_jobs)
        budget_util_raw = stats.get("budget_utilization_per_seq") or [0.0] * len(batch_jobs)

        batch_texts = [job["prompt"].prompt_text for job in batch_jobs]
        enc = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
        prompt_lens = enc.attention_mask.sum(dim=1).tolist()
        seqs = output.sequences.detach().cpu()

        for i, job in enumerate(batch_jobs):
            prompt = job["prompt"]
            seed = job["seed"]
            trajectory_id = job["trajectory_id"]
            prompt_len = int(prompt_lens[i])
            full_ids = seqs[i].tolist()
            gen_ids = full_ids[prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

            final_cum_spend = float(final_cum_raw[i])
            final_budget = float(final_budget_raw[i])
            budget_utilization = float(budget_util_raw[i])
            gen_len = len(gen_ids)

            true_prefix_debt = None
            init_budget_tensor = None
            if per_step_stats:
                step0 = per_step_stats[0]
                prefix_debt_arr = step0.get("prefix_debt")
                if prefix_debt_arr is not None and i < len(prefix_debt_arr):
                    true_prefix_debt = float(prefix_debt_arr[i])
                init_budget_arr = step0.get("init_budget_tensor")
                if init_budget_arr is not None and i < len(init_budget_arr):
                    init_budget_tensor = float(init_budget_arr[i])

            delta_init = true_prefix_debt if true_prefix_debt is not None else self._estimate_prefix_debt(final_budget, gen_len, k)

            # feat-003: per-trajectory utilisation Z / max(0, B) and solver activity counts
            own_len = true_gen_len(gen_ids, [self.tokenizer.pad_token_id, *([self.eos_ids] if isinstance(self.eos_ids, int) else self.eos_ids)])
            bd_i = [float(step["bd"][i]) for step in per_step_stats[:own_len] if "bd" in step and i < len(step["bd"])]
            steps_forced = sum(b <= 1e-6 for b in bd_i)
            steps_free = sum(b >= 1 - 1e-6 for b in bd_i)
            utilisation = final_cum_spend / final_budget if final_budget > 0 and math.isfinite(final_budget) else None

            per_step_log = []
            if self.config.save_full_trajectories:
                for t, step in enumerate(per_step_stats):
                    rec = {
                        "t": t,
                        "k_t": float(step["k_t"][i]) if "k_t" in step and i < len(step["k_t"]) else 0.0,
                        "a_t": float(step["kl_to_safe"][i]) if "kl_to_safe" in step and i < len(step["kl_to_safe"]) else 0.0,
                        "a_t_recomputed": float(step["kl_to_safe"][i]) if "kl_to_safe" in step and i < len(step["kl_to_safe"]) else 0.0,
                        "lambda": step.get("lambda"),
                        "budget_remaining": float(step["budget_remaining"][i]) if "budget_remaining" in step and i < len(step["budget_remaining"]) else 0.0,
                        "budget_so_far": float(step["budget_so_far"][i]) if "budget_so_far" in step and i < len(step["budget_so_far"]) else 0.0,
                        "cum_kl_spent": float(step["cum_kl_spent"][i]) if "cum_kl_spent" in step and i < len(step["cum_kl_spent"]) else 0.0,
                        "sampled_token": step["sampled_token"][i] if "sampled_token" in step and i < len(step["sampled_token"]) else None,
                        "sampled_token_id": step["sampled_token_id"][i] if "sampled_token_id" in step and i < len(step["sampled_token_id"]) else None,
                        "p_star_prob": float(step["p_star_prob"][i]) if "p_star_prob" in step and i < len(step["p_star_prob"]) else None,
                        "p_s_prob": float(step["p_s_prob"][i]) if "p_s_prob" in step and i < len(step["p_s_prob"]) else None,
                        "p_risky_prob": float(step["p_risky_prob"][i]) if "p_risky_prob" in step and i < len(step["p_risky_prob"]) else None,
                        "bc": float(step["bc"][i]) if "bc" in step and i < len(step["bc"]) else None,
                        "bd": float(step["bd"][i]) if "bd" in step and i < len(step["bd"]) else None,
                    }
                    per_step_log.append(rec)

            record = {
                "metadata": {
                    "prompt_id": prompt.prompt_id,
                    "domain": prompt.domain,
                    "split": prompt.split,
                    "novel_source": prompt.novel_source,
                    "model_pair": "tinycomma_llama31_8b",
                    "target_model": self.config.risky_model_path,
                    "anchor_model": self.config.safe_model_path,
                    "level": "token",
                    "k": k,
                    "K": budget_K(k, self.config.max_new_tokens),
                    "T_max": self.config.max_new_tokens,
                    "B_max": None,
                    "n": self.config.prefix_n,
                    "seed": seed,
                    "trajectory_id": trajectory_id,
                    "chat_template": self.config.use_chat_template,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "prefix_analysis": {
                    "delta_init": delta_init,
                    "true_prefix_debt": true_prefix_debt,
                    "init_budget_tensor": init_budget_tensor,
                    "prefix_text": prompt.prompt_text,
                    "prefix_length_tokens": prompt_len,
                },
                "per_step_log": per_step_log,
                "aggregate": {
                    "total_spend": final_cum_spend,
                    "generation": gen_text,
                    "full_text": full_text,
                    "generation_length_tokens": gen_len,
                    "generation_length_bytes": len(gen_text.encode("utf-8")),
                    **copying_metrics(gen_text, prompt.reference),
                    "fluency_score": None,
                    "final_budget": final_budget,
                    "budget_utilization": budget_utilization,
                    "utilisation": utilisation,
                    "invariant_ok": bool(final_cum_spend <= max(0.0, final_budget) + 1e-3),
                    "steps_forced_safe": steps_forced,
                    "steps_active": len(bd_i) - steps_forced - steps_free,
                    "steps_risky_unchanged": steps_free,
                },
                "source_record": prompt.raw,
            }
            records.append(record)
        return records

    def _run_seed_group(self, jobs_for_seed, k, batch_size, bucket_width):
        records = []
        length_buckets = self._make_length_buckets(jobs_for_seed, bucket_width=bucket_width)
        for bucket_jobs in length_buckets:
            batches = self._slice_batches(bucket_jobs, batch_size=batch_size)
            for batch_jobs in batches:
                batch_seed = batch_jobs[0]["seed"]
                batch_texts = [job["prompt"].prompt_text for job in batch_jobs]
                gen_cfg = self.generation_config()
                output = self.factory.generate(text=batch_texts, generation_config=gen_cfg, k_radius=k, seed=batch_seed, parallelize=self.config.parallelize, show_progress=False)
                stats = self.factory.get_kl_stats_summary()
                batch_records = self._records_from_batch(batch_jobs, output, stats, k)
                records.extend(batch_records)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        return records

    def run_split_batched(self, split_prompts, k, batch_size, bucket_width=32):
        jobs = self._build_jobs(split_prompts)
        jobs_by_seed = self._group_jobs_by_seed(jobs)
        all_records = []
        done = 0
        total = len(jobs)
        for seed in sorted(jobs_by_seed.keys()):
            seed_jobs = jobs_by_seed[seed]
            seed_records = self._run_seed_group(seed_jobs, k=k, batch_size=batch_size, bucket_width=bucket_width)
            all_records.extend(seed_records)
            done += len(seed_jobs)
            print(f"[stage] k={k} seed={seed} processed {done}/{total}", flush=True)
        return all_records

    def run(self, prompts: List[PromptRecord]) -> Dict[str, Any]:
        summary_rows = []
        grouped = defaultdict(list)
        for p in prompts:
            grouped[p.split].append(p)
        print(f"[stage] starting E1 with {len(prompts)} prompts", flush=True)

        for k in self.config.k_values:
            K = budget_K(k, self.config.max_new_tokens)
            class_spends = defaultdict(list)
            class_records = defaultdict(list)
            print(f"[stage] running k={k}", flush=True)

            for split_name in CLASS_ORDER:
                split_prompts = grouped.get(split_name, [])
                print(f"[stage] split={split_name} prompts={len(split_prompts)}", flush=True)
                path = self.output_dir / f"trajectories_k{k:g}_{split_name}.jsonl"
                if self.config.skip_existing and path.exists():  # resume a killed sweep (feat-005)
                    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
                    print(f"[stage] reusing {path} ({len(records)} records)", flush=True)
                else:
                    records = self.run_split_batched(split_prompts=split_prompts, k=k, batch_size=self.config.batch_size, bucket_width=self.config.length_bucket_width)

                if self.config.save_full_trajectories and not (self.config.skip_existing and path.exists()):
                    with open(path, "w", encoding="utf-8") as fout:
                        for record in records:
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                for record in records:
                    class_spends[split_name].append(float(record["aggregate"]["total_spend"]))
                    class_records[split_name].append(record["aggregate"])

            for split_name in CLASS_ORDER:
                summary_rows.append(summary_row(split_name, k, K, class_spends[split_name], self.config.bonferroni_delta, class_records[split_name]))

        summary_json = self.output_dir / "h1_summary.json"
        summary_csv = self.output_dir / "h1_summary.csv"

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(",".join(SUMMARY_HEADERS) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[h]) for h in SUMMARY_HEADERS) + "\n")

        return {"summary_json": str(summary_json), "summary_csv": str(summary_csv), "rows": summary_rows}


SUMMARY_HEADERS = ["class", "k", "K", "M", "mean_Z", "var_Z", "R", "delta", "U_EBB", "certified",
                   "util_max", "util_gt_0p9", "invariant_violations", "active_step_pct", "forced_safe_step_pct",
                   "rouge_l_mean", "lcs_word_mean", "lcs_char_mean", "acs_word_mean", "nv_recall_mean", "gen_len_mean"]
METRIC_KEYS = ("rouge_l", "lcs_word", "lcs_char", "acs_word", "nv_recall")


def budget_K(k: float, t_max: int) -> float:
    """Sequence budget. k = -1 is the risky-only baseline (no budget, K = inf); k = 0 is safe-only (K = 0)."""
    return float("inf") if k == -1.0 else k * t_max


def summary_row(split_name: str, k: float, K: float, spends: List[float], delta: float, aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-class summary. R = K because Z is in [0, K] by construction (feat-003).
    Baselines (k in {-1, 0}) carry no certificate, so no Bernstein arithmetic is done for them (feat-004)."""
    mean_z = float(np.mean(spends)) if spends else 0.0
    var_z = float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0
    baseline = k in (-1.0, 0.0)
    u_ebb = None if baseline else ebb_upper_bound_chapman(spends, K, delta)
    utils = [a["utilisation"] for a in aggregates if a.get("utilisation") is not None]
    steps = sum(a.get("steps_forced_safe", 0) + a.get("steps_active", 0) + a.get("steps_risky_unchanged", 0) for a in aggregates)
    means = {f"{m}_mean": float(np.mean([a[m] for a in aggregates])) if aggregates else 0.0 for m in METRIC_KEYS}
    means["gen_len_mean"] = float(np.mean([a["generation_length_tokens"] for a in aggregates])) if aggregates else 0.0
    return {
        "class": split_name, "k": k, "K": K, "M": len(spends),
        "mean_Z": mean_z, "var_Z": var_z, "R": None if baseline else K, "delta": delta, "U_EBB": u_ebb,
        "certified": None if baseline else bool(u_ebb <= K), **means,
        "util_max": max(utils) if utils else 0.0, "util_gt_0p9": sum(u > 0.9 for u in utils),
        "invariant_violations": sum(1 for a in aggregates if a.get("invariant_ok") is False),
        "active_step_pct": 100.0 * sum(a.get("steps_active", 0) for a in aggregates) / steps if steps else 0.0,
        "forced_safe_step_pct": 100.0 * sum(a.get("steps_forced_safe", 0) for a in aggregates) / steps if steps else 0.0,
    }


def rebuild_summary_from_saved_trajectories(config: AuditConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    summary_rows = []

    for k in config.k_values:
        K = budget_K(k, config.max_new_tokens)
        for split_name in CLASS_ORDER:
            path = output_dir / f"trajectories_k{k:g}_{split_name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing trajectory file: {path}")
            aggregates = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        aggregates.append(json.loads(line)["aggregate"])
            spends = [float(a["total_spend"]) for a in aggregates]
            summary_rows.append(summary_row(split_name, k, K, spends, config.bonferroni_delta, aggregates))

    summary_json = output_dir / "h1_summary.json"
    summary_csv = output_dir / "h1_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(SUMMARY_HEADERS) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[h]) for h in SUMMARY_HEADERS) + "\n")

    return {"summary_json": str(summary_json), "summary_csv": str(summary_csv), "rows": summary_rows}


def parse_args() -> AuditConfig:
    p = argparse.ArgumentParser(description="Experiment E1: multi-domain K-NAF certification")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="output/h1_outputs")
    p.add_argument("--safe-model-path", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    p.add_argument("--risky-model-path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--k-values", nargs="+", type=float, default=[3.0, 5.0])
    p.add_argument("--trajectories-per-prompt", type=int, default=10)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--prefix-n", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--factscore-field", default="factscore_prompt", choices=["factscore_prompt", "hundredw_prompt", "around_100", "one_fact_prompt", "prompt_text"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--parallelize", action="store_true")
    p.add_argument("--no-prefix-debt", action="store_true")
    p.add_argument("--no-save-full-trajectories", action="store_true")
    p.add_argument("--num-classes", type=int, default=len(CLASS_ORDER))
    p.add_argument("--cap-neutral", type=int, default=200)
    p.add_argument("--cap-val", type=int, default=150)
    p.add_argument("--cap-test", type=int, default=150)
    p.add_argument("--cap-attack-train", type=int, default=100)
    p.add_argument("--cap-factual", type=int, default=150)
    p.add_argument("--cap-creative", type=int, default=150)
    p.add_argument("--resume-from-trajectories", action="store_true")
    p.add_argument("--use-chat-template", action="store_true", help="wrap each prompt as one user turn of the Llama-3.1 chat template and stop on <|eot_id|>")
    p.add_argument("--skip-existing", action="store_true", help="reuse trajectory files already in --output-dir (resume a killed run)")
    p.add_argument("--greedy", action="store_true", help="argmax decoding; only valid for the baselines k in {-1, 0} (feat-008 extraction check)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--length-bucket-width", type=int, default=32)
    args = p.parse_args()

    if args.num_classes != len(CLASS_ORDER):
        raise ValueError(f"--num-classes={args.num_classes} does not match len(CLASS_ORDER)={len(CLASS_ORDER)}. Keep them aligned for correct Bonferroni correction.")
    if not args.k_values:
        raise ValueError("--k-values must contain at least one value.")
    if any(k < 0 and k != -1.0 for k in args.k_values):
        raise ValueError("--k-values must be -1 (risky only), 0 (safe only), or positive.")
    if args.greedy and any(k not in (-1.0, 0.0) for k in args.k_values):
        raise ValueError("--greedy is only meaningful for the baselines k in {-1, 0}; anchored decoding requires sampling.")

    return AuditConfig(
        data_dir=args.data_dir, output_dir=args.output_dir,
        safe_model_path=args.safe_model_path, risky_model_path=args.risky_model_path,
        k_values=tuple(args.k_values), trajectories_per_prompt=args.trajectories_per_prompt,
        seeds=tuple(args.seeds), prefix_n=args.prefix_n, use_prefix_debt=not args.no_prefix_debt,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens, delta=args.delta,
        num_classes=args.num_classes, verbose=args.verbose, trust_remote_code=args.trust_remote_code,
        device=args.device, batch_size=args.batch_size, length_bucket_width=args.length_bucket_width,
        device_map=args.device_map, dtype=args.dtype, parallelize=args.parallelize,
        save_full_trajectories=True, factscore_field=args.factscore_field,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        cap_neutral=args.cap_neutral, cap_val=args.cap_val, cap_test=args.cap_test,
        cap_attack_train=args.cap_attack_train, cap_factual=args.cap_factual, cap_creative=args.cap_creative,
        resume_from_trajectories=args.resume_from_trajectories,
        use_chat_template=args.use_chat_template, skip_existing=args.skip_existing, greedy=args.greedy,
    )


def main():
    print("[stage] parsing config", flush=True)
    cfg = parse_args()

    if getattr(cfg, "resume_from_trajectories", False):
        result = rebuild_summary_from_saved_trajectories(cfg)
        print(json.dumps({
            "summary_json": result["summary_json"], "summary_csv": result["summary_csv"],
            "bonferroni_delta": cfg.bonferroni_delta, "num_classes": cfg.num_classes,
            "num_k_values": len(cfg.k_values), "num_summary_rows": len(result["rows"]),
            "mode": "resumed_from_saved_trajectories",
        }, indent=2))
        return

    prompts = load_prompt_corpus(cfg.data_dir, cfg.factscore_field)
    prompts = apply_e1_sampling(prompts, cfg.cap_neutral, cfg.cap_val, cfg.cap_test, cfg.cap_attack_train, cfg.cap_factual, cfg.cap_creative)

    counts = Counter(p.split for p in prompts)
    print(f"[stage] sampled counts={dict(counts)}", flush=True)
    caps = {"neutral": cfg.cap_neutral, "val": cfg.cap_val, "test": cfg.cap_test, "attack_train": cfg.cap_attack_train, "factual": cfg.cap_factual, "creative": cfg.cap_creative}
    validate_sample_counts(counts, caps)

    runner = H1AuditRunner(cfg)
    if cfg.use_chat_template:
        prompts = [replace(p, prompt_text=wrap_chat(p.prompt_text, runner.tokenizer)) for p in prompts]
    result = runner.run(prompts)
    print(json.dumps({
        "sampled_counts": dict(counts),
        "summary_json": result["summary_json"], "summary_csv": result["summary_csv"],
        "bonferroni_delta": cfg.bonferroni_delta, "num_classes": cfg.num_classes,
        "num_k_values": len(cfg.k_values), "num_summary_rows": len(result["rows"]),
    }, indent=2))
