import argparse
import json
import math
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch
from transformers import GenerationConfig
from a_patch import AnchoredDecodingFactory
from dotenv import load_dotenv
load_dotenv()
import gc

@dataclass
class PromptRecord:
    prompt_id: str
    domain: str
    split: str
    prompt_text: str
    novel_source: Optional[str] = None
    reference: Optional[str] = None
    expected_answer: Optional[str] = None
    question_type: Optional[str] = None
    truncation_type: Optional[str] = None
    debt_init_estimated: Optional[float] = None
    atomic_fact_source: Optional[str] = None
    reddit_id: Optional[str] = None
    score: Optional[int] = None
    source_file: Optional[str] = None
    cleaning_passed: Optional[bool] = None
    raw: Optional[Dict[str, Any]] = None

CLASS_ORDER = ["neutral", "val", "test", "attack_train", "factual", "creative"]
@dataclass
class AuditConfig:
    data_dir: str = "data"
    output_dir: str = "output/h1_outputs"
    resume_from_trajectories: bool = False
    # Experiment E1 defaults
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

    # E1 prompt counts
    cap_neutral: int = 200
    cap_val: int = 150
    cap_test: int = 150
    cap_attack_train: int = 100
    cap_factual: int = 150
    cap_creative: int = 150

    @property
    def num_hypotheses(self) -> int:
        return self.num_classes * len(self.k_values)

    @property
    def effective_num_classes(self) -> int:
        return len(CLASS_ORDER)
    @property
    def bonferroni_delta(self) -> float:
        return self.delta / self.num_hypotheses
    


SOURCE_FILES = {
    "copybench_attack_train.jsonl": ("copyright", "attack_train"),
    "copybench_test.jsonl": ("copyright", "test"),
    "copybench_val.jsonl": ("copyright", "val"),
    "neutral.jsonl": ("copyright", "neutral"),
    "creative.jsonl": ("creative", "creative"),
    "factscore.jsonl": ("factual", "factual"),
}


def stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


def build_trajectory_seeds(prompt_id: str, base_seeds: Tuple[int, ...], n: int) -> List[int]:
    offset = stable_hash(prompt_id) % 100000
    out = []
    for i in range(n):
        out.append(base_seeds[i % len(base_seeds)] + offset + i)
    return out


def rouge_l_score(hypothesis: str, reference: Optional[str]) -> float:
    if not reference:
        return 0.0
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec = lcs / len(hyp)
    rec = lcs / len(ref)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def minhash_5gram_score(hypothesis: str, reference: Optional[str]) -> float:
    if not reference:
        return 0.0
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if len(h) < 5 or len(r) < 5:
        return 0.0
    h5 = set(tuple(h[i:i + 5]) for i in range(len(h) - 4))
    r5 = set(tuple(r[i:i + 5]) for i in range(len(r) - 4))
    if not h5 or not r5:
        return 0.0
    return len(h5 & r5) / len(h5 | r5)



def ebb_upper_bound_chapman(samples: List[float], R: float, delta: float) -> float:
    if not samples:
        return float("inf")
    arr = np.asarray(samples, dtype=np.float64)
    M = len(arr)
    mean_z = float(arr.mean())
    var_z = float(arr.var(ddof=1)) if M > 1 else 0.0
    log_term = math.log(2.0 / delta)
    width = math.sqrt((2.0 * var_z * log_term) / M) + (3.0 * R * log_term) / M
    return mean_z + width

def first_of(step: Dict[str, Any], *keys, default=None):
    for key in keys:
        val = step.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            return val[0] if val else default
        return val
    return default


class PromptNormalizer:
    def __init__(self, factscore_field: str = "factscore_prompt"):
        self.factscore_field = factscore_field

    def normalize(self, raw: Dict[str, Any], filename: str, idx: int) -> PromptRecord:
        if filename not in SOURCE_FILES:
            raise ValueError(f"Unsupported file: {filename}")
        domain, split = SOURCE_FILES[filename]
        if filename.startswith("copybench_") or filename == "neutral.jsonl":
            return self._normalize_copybench_like(raw, filename, idx, domain, split)
        if filename == "creative.jsonl":
            return self._normalize_creative(raw, filename, idx, domain, split)
        if filename == "factscore.jsonl":
            return self._normalize_factscore(raw, filename, idx, domain, split)
        raise ValueError(f"No normalizer registered for {filename}")

    def _normalize_copybench_like(self, raw, filename, idx, domain, split):
        prompt_id = str(raw.get("prompt_id") or raw.get("source_excerpt_id") or f"{split}_{idx:05d}")
        prompt_text = raw.get("prefix") or raw.get("raw_text") or raw.get("prompt_text") or raw.get("prompt")
        if prompt_text is None:
            raise ValueError(f"Missing prefix/raw_text/prompt for {filename} line {idx + 1}")
        prompt_text = "Complete the prefix:\n" + str(prompt_text)
        return PromptRecord(
            prompt_id=prompt_id,
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=prompt_text,
            novel_source=raw.get("novel_source") or raw.get("source_novel"),
            reference=raw.get("reference") or raw.get("reference_text") or "",
            expected_answer=raw.get("reference") or raw.get("reference_text") or "",
            question_type=raw.get("question_type"),
            truncation_type=raw.get("truncation_type") or ("cliffhanger" if split == "attack_train" else None),
            debt_init_estimated=raw.get("debt_init_estimated"),
            source_file=filename,
            raw=raw,
        )

    def _normalize_creative(self, raw, filename, idx, domain, split):
        meta = raw.get("metadata") or {}
        prompt_text = raw.get("prompt_text") or raw.get("input") or meta.get("title")
        if prompt_text is None:
            raise ValueError(f"Missing input/prompt_text for {filename} line {idx + 1}")
        prompt_id = str(raw.get("prompt_id") or meta.get("submission_id") or f"creative_{idx:05d}")
        prompt_text = "Complete the prefix:\n" + str(prompt_text)
        return PromptRecord(
            prompt_id=prompt_id,
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=prompt_text,
            novel_source=raw.get("novel_source"),
            reference=raw.get("reference"),
            expected_answer=None,
            reddit_id=raw.get("reddit_id") or meta.get("submission_id"),
            score=raw.get("score") or meta.get("score"),
            source_file=filename,
            cleaning_passed=raw.get("cleaning_passed", True),
            raw=raw,
        )

    def _normalize_factscore(self, raw, filename, idx, domain, split):
        prompt_text = (
            raw.get("prompt_text")
            or raw.get(self.factscore_field)
            or raw.get("factscore_prompt")
            or raw.get("hundredw_prompt")
            or raw.get("around_100")
            or raw.get("one_fact_prompt")
        )
        if prompt_text is None:
            raise ValueError(f"Missing factual prompt field for {filename} line {idx + 1}")
        entity = raw.get("entity") or f"entity_{idx:05d}"
        return PromptRecord(
            prompt_id=str(raw.get("prompt_id") or f"fact_{idx:05d}_{entity}"),
            domain=str(raw.get("domain") or domain),
            split=str(raw.get("split") or split),
            prompt_text=str(prompt_text).strip(),
            novel_source=raw.get("novel_source") or raw.get("source_novel") or str(entity),
            reference=raw.get("expected_answer") or raw.get("wikipedia_text"),
            expected_answer=raw.get("expected_answer") or raw.get("wikipedia_text"),
            question_type=raw.get("question_type") or "plot_event",
            atomic_fact_source=raw.get("atomic_fact_source") or "factscore_books",
            source_file=filename,
            raw=raw,
        )


def load_prompt_corpus(data_dir: str, factscore_field: str) -> List[PromptRecord]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    normalizer = PromptNormalizer(factscore_field=factscore_field)
    prompts: List[PromptRecord] = []
    for filename in SOURCE_FILES:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"[stage] reading {path}", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                prompts.append(normalizer.normalize(raw, filename, idx))
    return prompts


def _stable_sort_key(p: PromptRecord):
    return p.prompt_id


def _take_first_n(prompts: List[PromptRecord], n: int) -> List[PromptRecord]:
    return sorted(prompts, key=_stable_sort_key)[:n]


def _quartile_bucket(value: float, cuts: List[float]) -> int:
    if value <= cuts[0]:
        return 0
    if value <= cuts[1]:
        return 1
    if value <= cuts[2]:
        return 2
    return 3


def stratified_attack_sample(prompts: List[PromptRecord], target_n: int) -> List[PromptRecord]:
    with_debt = [p for p in prompts if p.debt_init_estimated is not None]
    without_debt = [p for p in prompts if p.debt_init_estimated is None]

    if len(with_debt) < target_n:
        return sorted(with_debt + without_debt, key=_stable_sort_key)[:target_n]

    debt_vals = sorted(float(p.debt_init_estimated) for p in with_debt)
    q1 = debt_vals[len(debt_vals) // 4]
    q2 = debt_vals[len(debt_vals) // 2]
    q3 = debt_vals[(3 * len(debt_vals)) // 4]
    cuts = [q1, q2, q3]

    buckets = {0: [], 1: [], 2: [], 3: []}
    for p in with_debt:
        b = _quartile_bucket(float(p.debt_init_estimated), cuts)
        buckets[b].append(p)

    per_bucket = target_n // 4
    sampled = []
    for b in range(4):
        sampled.extend(sorted(buckets[b], key=_stable_sort_key)[:per_bucket])

    if len(sampled) < target_n:
        used = {p.prompt_id for p in sampled}
        remainder = [p for p in sorted(with_debt, key=_stable_sort_key) if p.prompt_id not in used]
        sampled.extend(remainder[: target_n - len(sampled)])

    return sampled[:target_n]


def stratified_factual_sample(prompts: List[PromptRecord], target_n: int) -> List[PromptRecord]:
    by_source = defaultdict(list)
    for p in prompts:
        by_source[p.novel_source or "unknown"].append(p)

    groups = sorted(by_source.items(), key=lambda x: x[0])
    if not groups:
        return []

    base = target_n // len(groups)
    extra = target_n % len(groups)
    sampled = []

    for i, (_, items) in enumerate(groups):
        quota = base + (1 if i < extra else 0)
        sampled.extend(sorted(items, key=_stable_sort_key)[:quota])

    if len(sampled) < target_n:
        used = {p.prompt_id for p in sampled}
        flat = []
        for _, items in groups:
            flat.extend(sorted(items, key=_stable_sort_key))
        sampled.extend([p for p in flat if p.prompt_id not in used][: target_n - len(sampled)])

    return sampled[:target_n]


def apply_e1_sampling(prompts: List[PromptRecord], cfg: AuditConfig) -> List[PromptRecord]:
    grouped = defaultdict(list)
    for p in prompts:
        grouped[p.split].append(p)

    sampled = []
    sampled.extend(_take_first_n(grouped["neutral"], cfg.cap_neutral))
    sampled.extend(_take_first_n(grouped["val"], cfg.cap_val))
    sampled.extend(_take_first_n(grouped["test"], cfg.cap_test))
    sampled.extend(stratified_attack_sample(grouped["attack_train"], cfg.cap_attack_train))
    sampled.extend(stratified_factual_sample(grouped["factual"], cfg.cap_factual))
    creative_pool = [p for p in grouped["creative"] if p.cleaning_passed is not False and (p.score is None or float(p.score) >= 10)]
    sampled.extend(_take_first_n(creative_pool, cfg.cap_creative))
    return sampled


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
            k_radius=config.k_values[0],
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
        self.R_token = config.max_new_tokens * math.log(len(self.tokenizer))
        print(f"[stage] models ready; R_token={self.R_token:.4f}", flush=True)

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            do_sample=True,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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
            traj_seeds = build_trajectory_seeds(
                prompt.prompt_id,
                self.config.seeds,
                self.config.trajectories_per_prompt,
            )
            prompt_len = self._prompt_token_length(prompt.prompt_text)
            for traj_idx, seed in enumerate(traj_seeds):
                jobs.append(
                    {
                        "prompt": prompt,
                        "seed": int(seed),
                        "trajectory_id": int(traj_idx),
                        "prompt_len": prompt_len,
                    }
                )
        return jobs

    def _group_jobs_by_seed(self, jobs: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for job in jobs:
            grouped[job["seed"]].append(job)
        return grouped

    def _make_length_buckets(
        self,
        jobs: List[Dict[str, Any]],
        bucket_width: int = 32,
    ) -> List[List[Dict[str, Any]]]:
        by_bucket = defaultdict(list)
        for job in jobs:
            bucket_id = job["prompt_len"] // bucket_width
            by_bucket[bucket_id].append(job)

        ordered = []
        for bucket_id in sorted(by_bucket.keys()):
            bucket_jobs = sorted(
                by_bucket[bucket_id],
                key=lambda x: (x["prompt_len"], x["prompt"].prompt_id, x["trajectory_id"]),
            )
            ordered.append(bucket_jobs)
        return ordered

    def _slice_batches(
        self,
        jobs: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[List[Dict[str, Any]]]:
        return [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]

    def _records_from_batch(
        self,
        batch_jobs: List[Dict[str, Any]],
        output,
        stats,
        k: float,
    ) -> List[Dict[str, Any]]:
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

            delta_init = (
                true_prefix_debt
                if true_prefix_debt is not None
                else self._estimate_prefix_debt(final_budget, gen_len, k)
            )

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
                    "K": k * self.config.max_new_tokens,
                    "T_max": self.config.max_new_tokens,
                    "B_max": None,
                    "n": self.config.prefix_n,
                    "seed": seed,
                    "trajectory_id": trajectory_id,
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
                    "rouge_l": rouge_l_score(gen_text, prompt.reference),
                    "minhash_5gram": minhash_5gram_score(gen_text, prompt.reference),
                    "fluency_score": None,
                    "final_budget": final_budget,
                    "budget_utilization": budget_utilization,
                },
                "source_record": prompt.raw,
            }
            records.append(record)

        return records

    def _run_seed_group(
        self,
        jobs_for_seed: List[Dict[str, Any]],
        k: float,
        batch_size: int,
        bucket_width: int,
    ) -> List[Dict[str, Any]]:
        records = []

        length_buckets = self._make_length_buckets(jobs_for_seed, bucket_width=bucket_width)

        for bucket_jobs in length_buckets:
            batches = self._slice_batches(bucket_jobs, batch_size=batch_size)

            for batch_jobs in batches:
                batch_seed = batch_jobs[0]["seed"]
                batch_texts = [job["prompt"].prompt_text for job in batch_jobs]
                gen_cfg = self.generation_config()

                output = self.factory.generate(
                    text=batch_texts,
                    generation_config=gen_cfg,
                    k_radius=k,
                    seed=batch_seed,
                    parallelize=self.config.parallelize,
                    show_progress=False,
                )
                stats = self.factory.get_kl_stats_summary()

                batch_records = self._records_from_batch(batch_jobs, output, stats, k)
                records.extend(batch_records)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return records

    def run_split_batched(
        self,
        split_prompts: List[PromptRecord],
        k: float,
        batch_size: int,
        bucket_width: int = 32,
    ) -> List[Dict[str, Any]]:
        jobs = self._build_jobs(split_prompts)
        jobs_by_seed = self._group_jobs_by_seed(jobs)

        all_records = []
        done = 0
        total = len(jobs)

        for seed in sorted(jobs_by_seed.keys()):
            seed_jobs = jobs_by_seed[seed]
            seed_records = self._run_seed_group(
                jobs_for_seed=seed_jobs,
                k=k,
                batch_size=batch_size,
                bucket_width=bucket_width,
            )
            all_records.extend(seed_records)
            done += len(seed_jobs)
            print(
                f"[stage] k={k} seed={seed} processed {done}/{total}",
                flush=True,
            )

        return all_records

    def run(self, prompts: List[PromptRecord]) -> Dict[str, Any]:
        summary_rows = []
        grouped = defaultdict(list)
        for p in prompts:
            grouped[p.split].append(p)

        print(f"[stage] starting E1 with {len(prompts)} prompts", flush=True)

        batch_size = getattr(self.config, "batch_size", 8)
        bucket_width = getattr(self.config, "length_bucket_width", 32)

        for k in self.config.k_values:
            K = k * self.config.max_new_tokens
            class_spends = defaultdict(list)
            print(f"[stage] running k={k}", flush=True)

            for split_name in CLASS_ORDER:
                split_prompts = grouped.get(split_name, [])
                print(f"[stage] split={split_name} prompts={len(split_prompts)}", flush=True)

                records = self.run_split_batched(
                    split_prompts=split_prompts,
                    k=k,
                    batch_size=batch_size,
                    bucket_width=bucket_width,
                )

                if self.config.save_full_trajectories:
                    path = self.output_dir / f"trajectories_k{k:g}_{split_name}.jsonl"
                    with open(path, "w", encoding="utf-8") as fout:
                        for record in records:
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                for record in records:
                    class_spends[split_name].append(float(record["aggregate"]["total_spend"]))

            for split_name in CLASS_ORDER:
                spends = class_spends[split_name]
                mean_z = float(np.mean(spends)) if spends else 0.0
                var_z = float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0
                u_ebb = ebb_upper_bound_chapman(
                    spends,
                    self.R_token,
                    self.config.bonferroni_delta,
                )
                summary_rows.append(
                    {
                        "class": split_name,
                        "k": k,
                        "K": K,
                        "M": len(spends),
                        "mean_Z": mean_z,
                        "var_Z": var_z,
                        "R": self.R_token,
                        "delta": self.config.bonferroni_delta,
                        "U_EBB": u_ebb,
                        "certified": bool(u_ebb <= K),
                    }
                )

        summary_json = self.output_dir / "h1_summary.json"
        summary_csv = self.output_dir / "h1_summary.csv"

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)

        headers = ["class", "k", "K", "M", "mean_Z", "var_Z", "R", "delta", "U_EBB", "certified"]
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

        return {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "rows": summary_rows,
        }

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
    p.add_argument(
        "--factscore-field",
        default="factscore_prompt",
        choices=["factscore_prompt", "hundredw_prompt", "around_100", "one_fact_prompt", "prompt_text"],
    )
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--length-bucket-width", type=int, default=32)
    args = p.parse_args()

    
    if args.num_classes != len(CLASS_ORDER):
        raise ValueError(
            f"--num-classes={args.num_classes} does not match len(CLASS_ORDER)={len(CLASS_ORDER)}. "
            "Keep them aligned for correct Bonferroni correction."
        )

    if not args.k_values:
        raise ValueError("--k-values must contain at least one value.")

    return AuditConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        safe_model_path=args.safe_model_path,
        risky_model_path=args.risky_model_path,
        k_values=tuple(args.k_values),
        trajectories_per_prompt=args.trajectories_per_prompt,
        seeds=tuple(args.seeds),
        prefix_n=args.prefix_n,
        use_prefix_debt=not args.no_prefix_debt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        delta=args.delta,
        num_classes=args.num_classes,
        verbose=args.verbose,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        batch_size=args.batch_size,
        length_bucket_width=args.length_bucket_width,
        device_map=args.device_map,
        dtype=args.dtype,
        parallelize=args.parallelize,
        save_full_trajectories=True,
        factscore_field=args.factscore_field,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        cap_neutral=args.cap_neutral,
        cap_val=args.cap_val,
        cap_test=args.cap_test,
        cap_attack_train=args.cap_attack_train,
        cap_factual=args.cap_factual,
        cap_creative=args.cap_creative,
        resume_from_trajectories=args.resume_from_trajectories,
    )

def validate_sample_counts(counts: Counter, cfg: AuditConfig) -> None:
    expected = {
        "neutral": cfg.cap_neutral,
        "val": cfg.cap_val,
        "test": cfg.cap_test,
        "attack_train": cfg.cap_attack_train,
        "factual": cfg.cap_factual,
        "creative": cfg.cap_creative,
    }
    for split, exp in expected.items():
        got = counts.get(split, 0)
        if got != exp:
            raise ValueError(
                f"Sample count mismatch for split='{split}': expected {exp}, got {got}. "
                "This usually means the filtered source pool is smaller than the configured cap."
            )
    total_expected = sum(expected.values())
    total_got = sum(counts.values())
    if total_got != total_expected:
        raise ValueError(
            f"Total sampled prompt count mismatch: expected {total_expected}, got {total_got}."
        )
def rebuild_summary_from_saved_trajectories(config: AuditConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir)
    summary_rows = []

    tokenizer = None
    R_token = None

    for k in config.k_values:
        K = k * config.max_new_tokens

        for split_name in CLASS_ORDER:
            path = output_dir / f"trajectories_k{k:g}_{split_name}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing trajectory file: {path}")

            spends = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    spends.append(float(record["aggregate"]["total_spend"]))

                    if R_token is None:
                        meta = record.get("metadata", {})
                        t_max = meta.get("T_max", config.max_new_tokens)
                        if tokenizer is None:
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(
                                config.safe_model_path,
                                trust_remote_code=config.trust_remote_code,
                            )
                        R_token = t_max * math.log(len(tokenizer))

            mean_z = float(np.mean(spends)) if spends else 0.0
            var_z = float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0
            u_ebb = ebb_upper_bound_chapman(spends, R_token, config.bonferroni_delta)

            summary_rows.append(
                {
                    "class": split_name,
                    "k": k,
                    "K": K,
                    "M": len(spends),
                    "mean_Z": mean_z,
                    "var_Z": var_z,
                    "R": R_token,
                    "delta": config.bonferroni_delta,
                    "U_EBB": u_ebb,
                    "certified": bool(u_ebb <= K),
                }
            )

    summary_json = output_dir / "h1_summary.json"
    summary_csv = output_dir / "h1_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    headers = ["class", "k", "K", "M", "mean_Z", "var_Z", "R", "delta", "U_EBB", "certified"]
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in summary_rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "rows": summary_rows,
    }


def main():
    print("[stage] parsing config", flush=True)
    cfg = parse_args()

    if getattr(cfg, "resume_from_trajectories", False):
        result = rebuild_summary_from_saved_trajectories(cfg)
        print(json.dumps({
            "summary_json": result["summary_json"],
            "summary_csv": result["summary_csv"],
            "bonferroni_delta": cfg.bonferroni_delta,
            "num_classes": cfg.num_classes,
            "num_k_values": len(cfg.k_values),
            "num_summary_rows": len(result["rows"]),
            "mode": "resumed_from_saved_trajectories",
        }, indent=2))
        return

    prompts = load_prompt_corpus(cfg.data_dir, cfg.factscore_field)
    prompts = apply_e1_sampling(prompts, cfg)

    counts = Counter(p.split for p in prompts)
    print(f"[stage] sampled counts={dict(counts)}", flush=True)
    validate_sample_counts(counts, cfg)

    runner = H1AuditRunner(cfg)
    result = runner.run(prompts)
    print(json.dumps({
        "sampled_counts": dict(counts),
        "summary_json": result["summary_json"],
        "summary_csv": result["summary_csv"],
        "bonferroni_delta": cfg.bonferroni_delta,
        "num_classes": cfg.num_classes,
        "num_k_values": len(cfg.k_values),
        "num_summary_rows": len(result["rows"]),
    }, indent=2))
if __name__ == "__main__":
    main()
