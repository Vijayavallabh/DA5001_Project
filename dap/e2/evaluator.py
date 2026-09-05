import gc
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GenerationConfig

from a_patch import AnchoredDecodingFactory
from ..shared import chat_eos_ids, true_gen_len, wrap_chat
from ..stats import ebb_upper_bound_chapman, build_trajectory_seeds
from .types import E2Config, EvalResult


def utilisation(spend: float, budget: float):
    """Z / max(0, B) per trajectory; None when no positive budget was accrued (feat-003)."""
    return spend / budget if budget > 0 and np.isfinite(budget) else None


def activity_counts(per_step, j: int, gen_len: int) -> List[int]:
    """[theta==0 (safe forced), 0<theta<1 (solver active), theta==1 (risky unchanged)] over the first gen_len steps."""
    bd = [float(s["bd"][j]) for s in per_step[:gen_len] if "bd" in s and j < len(s["bd"])]
    forced = sum(b <= 1e-6 for b in bd)
    free = sum(b >= 1 - 1e-6 for b in bd)
    return [forced, len(bd) - forced - free, free]


def safe_rho(u_ebb: float, effective_budget_min: float):
    if not np.isfinite(u_ebb):
        return None, "nonfinite_u_ebb"
    if not np.isfinite(effective_budget_min):
        return None, "nonfinite_effective_budget_min"
    if effective_budget_min <= 0.0:
        return None, "nonpositive_effective_budget_min"
    rho = u_ebb / effective_budget_min
    if not np.isfinite(rho):
        return None, "nonfinite_rho"
    return float(rho), None


class AnchoredEvaluator:
    def __init__(self, cfg: E2Config):
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        self.factory = AnchoredDecodingFactory.from_pretrained(
            safe_model_path=cfg.safe_model_path,
            risky_model_path=cfg.risky_model_path,
            k_radius=cfg.k,
            verbose=cfg.verbose,
            use_prefix_debt=cfg.use_prefix_debt,
            prefix_n=cfg.prefix_n,
            log_kl_stats=True,
            device=cfg.device,
            dtype=dtype,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            token=os.getenv("HF_TOKEN"),
        )
        self.tokenizer = self.factory.tokenizer
        self.R_token = self.cfg.K  # Z is in [0, K] deterministically; the only range used anywhere (feat-003)
        self.gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=chat_eos_ids(self.tokenizer) if cfg.use_chat_template else self.tokenizer.eos_token_id,
        )

    def _fmt(self, text: str) -> str:
        return wrap_chat(text, self.tokenizer) if self.cfg.use_chat_template else text

    def _estimate_prefix_debt(self, final_budget: float, gen_len: int) -> float:
        init_budget = final_budget - (gen_len * self.cfg.k)
        return max(0.0, -init_budget)

    def _prompt_token_length(self, text: str) -> int:
        ids = self.tokenizer(self._fmt(text), return_tensors="pt").input_ids[0]
        return int(ids.shape[0])

    def _slice_batches(self, items, batch_size):
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _bucket_specs_by_length(self, specs, length_bucket_width=32):
        buckets = defaultdict(list)
        for idx, spec in enumerate(specs):
            prompt_len = self._prompt_token_length(spec["prompt_text"])
            spec = dict(spec)
            spec["_prompt_len"] = prompt_len
            spec["_original_index"] = idx
            bucket_id = prompt_len // max(1, length_bucket_width)
            buckets[bucket_id].append(spec)
        out = []
        for bucket_id in sorted(buckets.keys()):
            bucket_specs = sorted(buckets[bucket_id], key=lambda x: (x["_prompt_len"], x["_original_index"]))
            out.append(bucket_specs)
        return out

    def _init_accumulator(self, spec):
        return {
            "candidate_id": spec["candidate_id"],
            "lineage_id": spec["lineage_id"],
            "generation": spec["generation"],
            "source": spec["source"],
            "domain": spec["domain"],
            "split": spec["split"],
            "prompt_text": spec["prompt_text"],
            "parent_ids": spec.get("parent_ids") or [],
            "parent_lineage_ids": spec.get("parent_lineage_ids") or [],
            "spends": [],
            "final_budgets": [],
            "delta_inits": [],
            "utilisations": [],
            "activity": [],
        }

    def _finalize_eval_result(self, acc, n, delta):
        spends = acc["spends"]
        final_budgets = acc["final_budgets"]
        delta_inits = acc["delta_inits"]

        mean_spend = float(np.mean(spends)) if spends else 0.0
        var_spend = float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0
        u_ebb = ebb_upper_bound_chapman(spends, self.R_token, delta)
        effective_budget_min = max(0.0, min(final_budgets)) if final_budgets else self.cfg.K

        rho, invalid_reason = safe_rho(u_ebb, effective_budget_min)
        candidate_valid = invalid_reason is None
        certified = bool(candidate_valid and u_ebb <= effective_budget_min)

        return EvalResult(
            candidate_id=acc["candidate_id"],
            lineage_id=acc["lineage_id"],
            generation=acc["generation"],
            source=acc["source"],
            domain=acc["domain"],
            split=acc["split"],
            prompt_text=acc["prompt_text"],
            N=n,
            spends=spends,
            final_budgets=final_budgets,
            delta_inits=delta_inits,
            mean_spend=mean_spend,
            var_spend=var_spend,
            U_EBB=u_ebb,
            rho=float(rho) if rho is not None else 0.0,
            certified=certified,
            effective_budget_min=float(effective_budget_min),
            delta_init_mean=float(np.mean(delta_inits)) if delta_inits else 0.0,
            final_budget_mean=float(np.mean(final_budgets)) if final_budgets else 0.0,
            parent_ids=acc["parent_ids"],
            parent_lineage_ids=acc["parent_lineage_ids"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            utilisations=acc["utilisations"],
            activity=acc["activity"],
        )

    def evaluate_text_batch(self, specs, n, delta, seed_offset=0, batch_size=8, length_bucket_width=32):
        if not specs:
            return []

        accumulators = [self._init_accumulator(spec) for spec in specs]
        buckets = self._bucket_specs_by_length(specs, length_bucket_width=length_bucket_width)

        for bucket_specs in buckets:
            bucket_batches = self._slice_batches(bucket_specs, batch_size=batch_size)

            for batch_specs in bucket_batches:
                batch_candidate_ids = [spec["candidate_id"] for spec in batch_specs]

                # seed_offset is a trajectory-index offset (stage-2 / top-up start at n0), never added to seed values
                seeds_per_example = [build_trajectory_seeds(cid, self.cfg.seeds, n, start=seed_offset) for cid in batch_candidate_ids]

                for _ in range(n):
                    shared_seed_groups = defaultdict(list)
                    for local_idx, spec in enumerate(batch_specs):
                        shared_seed = int(seeds_per_example[local_idx][_])
                        shared_seed_groups[shared_seed].append((local_idx, spec))

                    for shared_seed, grouped_items in shared_seed_groups.items():
                        grouped_specs = [x[1] for x in grouped_items]
                        grouped_texts = [self._fmt(spec["prompt_text"]) for spec in grouped_specs]

                        output = self.factory.generate(text=grouped_texts, generation_config=self.gen_cfg, k_radius=self.cfg.k, seed=shared_seed, parallelize=self.cfg.parallelize, show_progress=False)
                        stats = self.factory.get_kl_stats_summary()

                        final_cum_spend = stats.get("final_cum_kl_spent_per_seq") or stats.get("finalcumklspentperseq") or [0.0] * len(grouped_specs)
                        final_budget = stats.get("final_budget_per_seq") or stats.get("finalbudgetperseq") or [0.0] * len(grouped_specs)
                        per_step = stats.get("per_step") or []

                        enc = self.tokenizer(grouped_texts, return_tensors="pt", padding=True)
                        prompt_lens = enc.attention_mask.sum(dim=1).tolist()
                        seqs = output.sequences.detach().cpu()

                        for j, spec in enumerate(grouped_specs):
                            orig_idx = spec["_original_index"]

                            prefix_debt_val = None
                            if per_step and "prefix_debt" in per_step[0]:
                                prefix_arr = per_step[0].get("prefix_debt")
                                if prefix_arr is not None and j < len(prefix_arr):
                                    prefix_debt_val = float(prefix_arr[j])

                            gen_ids = seqs[j].tolist()[int(prompt_lens[j]):]
                            eos_ids = self.gen_cfg.eos_token_id
                            gen_len = true_gen_len(gen_ids, [self.tokenizer.pad_token_id, *([eos_ids] if isinstance(eos_ids, int) else eos_ids)])
                            if prefix_debt_val is None:
                                prefix_debt_val = self._estimate_prefix_debt(float(final_budget[j]), gen_len)

                            spend_j, budget_j = float(final_cum_spend[j]), float(final_budget[j])
                            accumulators[orig_idx]["spends"].append(spend_j)
                            accumulators[orig_idx]["final_budgets"].append(budget_j)
                            accumulators[orig_idx]["delta_inits"].append(float(prefix_debt_val))
                            accumulators[orig_idx]["utilisations"].append(utilisation(spend_j, budget_j))
                            accumulators[orig_idx]["activity"].append(activity_counts(per_step, j, gen_len))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return [self._finalize_eval_result(acc, n=n, delta=delta) for acc in accumulators]

    def evaluate_text(self, prompt_text, candidate_id, generation, source, lineage_id, domain, split, n, delta, parent_ids=None, parent_lineage_ids=None, seed_offset=0):
        spec = {
            "prompt_text": prompt_text,
            "candidate_id": candidate_id,
            "generation": generation,
            "source": source,
            "lineage_id": lineage_id,
            "domain": domain,
            "split": split,
            "parent_ids": parent_ids or [],
            "parent_lineage_ids": parent_lineage_ids or [],
        }
        return self.evaluate_text_batch(specs=[spec], n=n, delta=delta, seed_offset=seed_offset, batch_size=1, length_bucket_width=32)[0]
