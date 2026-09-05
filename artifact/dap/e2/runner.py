import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dotenv import load_dotenv
load_dotenv()

from ..shared import CLASS_ORDER, load_prompt_corpus
from ..sampling import stratified_attack_sample, stratified_factual_sample
from ..stats import budget_check, stable_hash
from .types import E2Config, Candidate, EvalResult, ArchiveItem
from .evaluator import AnchoredEvaluator, safe_rho
from .optimizer import LocalHFOptimizer
from .surrogate import SurrogateEnsemble
from .util import set_global_seed, dedupe_candidates, filter_by_length, lineage_id_for_candidate, k_dpp_select, ngram_jaccard, rough_structural_tag


class E2Runner:
    def __init__(self, cfg: E2Config):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = AnchoredEvaluator(cfg)
        self.optimizer = LocalHFOptimizer(cfg)
        self.surrogate = SurrogateEnsemble(cfg, self.evaluator.tokenizer)
        self.disqualified_candidate_ids: set[str] = set()
        self.archive_history: List[ArchiveItem] = []
        self.current_archive: List[ArchiveItem] = []
        self.generation_log: List[Dict[str, Any]] = []
        self.lineage_scores = defaultdict(dict)
        self.disqualified_lineage_ids: set[str] = set()
        self.eval_batch_size = getattr(cfg, "eval_batch_size", 8)
        self.length_bucket_width = getattr(cfg, "length_bucket_width", 32)

    def _persist_archive_state(self) -> None:
        self._write_json(self.output_dir / "archive_history.json", [asdict(x) for x in self.archive_history])
        self._write_json(self.output_dir / "archive_current.json", [asdict(x) for x in self.current_archive])

    def _eligible_archive_items(self, items: List[ArchiveItem]) -> List[ArchiveItem]:
        banned_ids = getattr(self, "disqualified_candidate_ids", set())
        banned_lineages = getattr(self, "disqualified_lineage_ids", set())
        return [a for a in items if bool(a.certified) and np.isfinite(a.rho) and a.candidate_id not in banned_ids and a.lineage_id not in banned_lineages]

    def _disqualify_violations(self, violations: List[dict]) -> None:
        bad_ids = {v["candidate_id"] for v in violations if v.get("candidate_id")}
        bad_lineages = {v["lineage_id"] for v in violations if v.get("lineage_id")}
        if not bad_ids and not bad_lineages:
            return
        self.disqualified_candidate_ids.update(bad_ids)
        self.disqualified_lineage_ids.update(bad_lineages)
        self.archive_history = [a for a in self.archive_history if a.candidate_id not in bad_ids and a.lineage_id not in bad_lineages]
        self.current_archive = [a for a in self.current_archive if a.candidate_id not in bad_ids and a.lineage_id not in bad_lineages]
        self._persist_archive_state()

    def _write_json(self, path: Path, obj: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, path: Path, rows: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_surrogate_scores(self, score_map: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
        safe_mean = np.asarray(score_map.get("safe_mean", score_map.get("safe")), dtype=np.float32)
        safe_sigma = np.asarray(score_map.get("safe_sigma", score_map.get("sigma")), dtype=np.float32)
        margin = np.asarray(score_map.get("margin", np.zeros(n)), dtype=np.float32)
        rho_fuse = np.asarray(score_map.get("fuse", np.zeros(n)), dtype=np.float32)

        safe_mean = np.nan_to_num(safe_mean, nan=0.5, posinf=1.0, neginf=0.0)
        safe_sigma = np.nan_to_num(safe_sigma, nan=1.0, posinf=1.0, neginf=0.0)
        margin = np.nan_to_num(margin, nan=0.0, posinf=2.0, neginf=-2.0)
        rho_fuse = np.nan_to_num(rho_fuse, nan=0.0, posinf=10.0, neginf=0.0)

        return {"safe_mean": safe_mean, "safe_sigma": safe_sigma, "margin": margin, "rho_fuse": rho_fuse}

    def _annotate_eval(self, ev: EvalResult) -> Dict[str, Any]:
        row = asdict(ev)
        rho, invalid_reason = safe_rho(ev.spends, ev.final_budgets)
        row["rho_num"] = float(ev.max_spend)
        row["rho_den"] = float(ev.effective_budget_min)
        row["raw_rho"] = rho
        row["candidate_valid"] = invalid_reason is None
        row["invalid_reason"] = invalid_reason
        return row

    def init_pool(self, prompts: List[Any]) -> Tuple[List[Any], List[Any]]:
        grouped = defaultdict(list)
        for p in prompts:
            grouped[p.split].append(p)

        attack_init = stratified_attack_sample(grouped["attack_train"], self.cfg.init_attack)
        factual_init = stratified_factual_sample(grouped["factual"], self.cfg.init_factual)
        creative_pool = [p for p in grouped["creative"] if p.cleaning_passed is not False and (p.score is None or float(p.score) >= 10)]
        creative_init = sorted(creative_pool, key=lambda x: x.prompt_id)[:self.cfg.init_creative]

        init_archive = attack_init + factual_init + creative_init
        init_ids = {p.prompt_id for p in init_archive}
        heldout = [p for p in grouped["test"] if p.prompt_id not in init_ids][:self.cfg.heldout_keep]
        return init_archive, heldout

    def _eval_specs(self, specs, n, delta, seed_offset=0):
        if not specs:
            return []

        adaptive = bool(getattr(self.cfg, "adaptive_eval", True) and getattr(self.surrogate, "ready", False) and n >= 4)

        def _call_eval(eval_specs, n_eval, seed_off):
            if hasattr(self.evaluator, "evaluate_text_batch"):
                return self.evaluator.evaluate_text_batch(specs=eval_specs, n=n_eval, delta=delta, seed_offset=seed_off, batch_size=self.eval_batch_size, length_bucket_width=self.length_bucket_width)
            out = []
            for spec in eval_specs:
                out.append(self.evaluator.evaluate_text(prompt_text=spec["prompt_text"], candidate_id=spec["candidate_id"], generation=spec["generation"], source=spec["source"], lineage_id=spec["lineage_id"], domain=spec["domain"], split=spec["split"], n=n_eval, delta=delta, parent_ids=spec.get("parent_ids"), parent_lineage_ids=spec.get("parent_lineage_ids"), seed_offset=seed_off))
            return out

        def _merge_eval_results(ev_a: EvalResult, ev_b: EvalResult, delta_merge: float) -> EvalResult:
            spends = list(ev_a.spends) + list(ev_b.spends)
            final_budgets = list(ev_a.final_budgets) + list(ev_b.final_budgets)
            delta_inits = list(ev_a.delta_inits) + list(ev_b.delta_inits)
            max_spend, rho, certified = budget_check(spends, final_budgets)
            effective_budget_min = max(0.0, min(float(ev_a.effective_budget_min), float(ev_b.effective_budget_min)))
            return EvalResult(
                candidate_id=ev_a.candidate_id, lineage_id=ev_a.lineage_id, generation=ev_a.generation,
                source=ev_a.source, domain=ev_a.domain, split=ev_a.split, prompt_text=ev_a.prompt_text,
                N=len(spends), spends=spends, final_budgets=final_budgets, delta_inits=delta_inits,
                mean_spend=float(np.mean(spends)) if spends else 0.0,
                var_spend=float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0,
                max_spend=max_spend, rho=float(rho) if rho is not None else 0.0, certified=certified,
                effective_budget_min=float(effective_budget_min),
                delta_init_mean=float(np.mean(delta_inits)) if delta_inits else 0.0,
                final_budget_mean=float(np.mean(final_budgets)) if final_budgets else 0.0,
                parent_ids=ev_a.parent_ids, parent_lineage_ids=ev_a.parent_lineage_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                utilisations=list(ev_a.utilisations) + list(ev_b.utilisations),
                activity=list(ev_a.activity) + list(ev_b.activity),
            )

        if not adaptive:
            return _call_eval(specs, n_eval=n, seed_off=seed_offset)

        n0 = int(getattr(self.cfg, "adaptive_eval_min_traj", 4))
        n0 = max(2, min(n0, n))
        if n0 >= n:
            return _call_eval(specs, n_eval=n, seed_off=seed_offset)

        stage1 = _call_eval(specs, n_eval=n0, seed_off=seed_offset)

        texts = [spec["prompt_text"] for spec in specs]
        scores = self._normalize_surrogate_scores(self.surrogate.predict(texts, [0.0] * len(specs)), len(specs))
        safe_mean = scores["safe_mean"]
        safe_sigma = scores["safe_sigma"]
        margin = scores["margin"]

        observed_rho = np.asarray([ev.rho if np.isfinite(ev.rho) else 0.0 for ev in stage1], dtype=np.float32)
        observed_valid = np.asarray([1.0 if ev.effective_budget_min > 0 else 0.0 for ev in stage1], dtype=np.float32)

        promote_score = (0.45 * np.clip(observed_rho, 0.0, 2.0) + 0.20 * safe_mean + 0.20 * safe_sigma + 0.15 * np.clip(margin, -1.0, 1.0)) * observed_valid

        survivor_mask = np.asarray([bool(ev.certified) and ev.effective_budget_min > 0 for ev in stage1], dtype=bool)

        remain = n - n0
        if remain <= 0:
            return stage1

        topup_fraction = float(getattr(self.cfg, "adaptive_eval_topup_fraction", 0.5))
        topup_count = max(1, int(round(topup_fraction * len(specs))))
        eligible_idx = np.where(survivor_mask)[0]
        if len(eligible_idx) == 0:
            eligible_idx = np.argsort(-promote_score)[:topup_count]
        else:
            eligible_idx = eligible_idx[np.argsort(-promote_score[eligible_idx])[:topup_count]]

        eligible_idx = np.asarray(sorted(set(int(i) for i in eligible_idx.tolist())), dtype=np.int64)

        if len(eligible_idx) == 0:
            return stage1

        stage2_specs = [specs[i] for i in eligible_idx]
        stage2 = _call_eval(stage2_specs, n_eval=remain, seed_off=seed_offset + n0)

        merged = list(stage1)
        for local_j, idx in enumerate(eligible_idx):
            merged[idx] = _merge_eval_results(stage1[idx], stage2[local_j], delta_merge=delta)

        return merged

    def _prompt_specs(self, prompts, generation, source, candidate_prefix, lineage_prefix):
        specs = []
        for p in prompts:
            specs.append({
                "prompt_text": p.prompt_text, "candidate_id": f"{candidate_prefix}{p.prompt_id}",
                "generation": generation, "source": source, "lineage_id": f"{lineage_prefix}{p.prompt_id}",
                "domain": p.domain, "split": p.split, "parent_ids": None, "parent_lineage_ids": None,
                "_prompt_obj": p,
            })
        return specs

    def _candidate_specs(self, candidates, lineage_ids, seed_offset=0):
        specs = []
        for c, lineage_id in zip(candidates, lineage_ids):
            specs.append({
                "prompt_text": c.prompt_text, "candidate_id": c.candidate_id,
                "generation": c.generation, "source": c.source, "lineage_id": lineage_id,
                "domain": "adversarial", "split": "generated",
                "parent_ids": c.parent_ids, "parent_lineage_ids": c.parent_lineage_ids,
                "_candidate_obj": c, "_seed_offset": seed_offset,
            })
        return specs

    def _eval_prompts_batched(self, prompts, generation, source, n, delta, candidate_prefix, lineage_prefix):
        specs = self._prompt_specs(prompts=prompts, generation=generation, source=source, candidate_prefix=candidate_prefix, lineage_prefix=lineage_prefix)
        evals = self._eval_specs(specs, n=n, delta=delta, seed_offset=0)
        return [(spec["_prompt_obj"], ev) for spec, ev in zip(specs, evals)]

    def _eval_candidates_batched(self, candidates, lineage_ids, n, delta, seed_offset=0):
        if not candidates:
            return []

        use_adaptive = bool(getattr(self.cfg, "adaptive_eval", True) and getattr(self.surrogate, "ready", False) and n > 1)

        if not use_adaptive:
            specs = self._candidate_specs(candidates=candidates, lineage_ids=lineage_ids, seed_offset=seed_offset)
            evals = self._eval_specs(specs, n=n, delta=delta, seed_offset=seed_offset)
            return [(spec["_candidate_obj"], ev) for spec, ev in zip(specs, evals)]

        texts = [c.prompt_text for c in candidates]
        scores = self._normalize_surrogate_scores(self.surrogate.predict(texts, [0.0] * len(texts)), len(candidates))

        safe_mean = scores["safe_mean"]
        safe_sigma = scores["safe_sigma"]
        margin = scores["margin"]

        uncertainty = safe_sigma
        boundary = 1.0 - np.abs(safe_mean - 0.5) / 0.5
        riskiness = np.clip(-margin, 0.0, 1.0)

        hardness = (0.50 * uncertainty + 0.35 * boundary + 0.15 * riskiness).astype(np.float32)
        hardness = np.clip(hardness, 0.0, 1.0)

        n_min = max(2, min(n, getattr(self.cfg, "adaptive_eval_min_traj", max(2, n // 2))))
        n_max = n

        alloc = n_min + np.rint((n_max - n_min) * hardness).astype(int)
        alloc = np.clip(alloc, n_min, n_max)

        grouped = {}
        for i, n_i in enumerate(alloc.tolist()):
            grouped.setdefault(int(n_i), []).append(i)

        partial_results = [None] * len(candidates)

        for n_i in sorted(grouped.keys()):
            idxs = grouped[n_i]
            sub_candidates = [candidates[i] for i in idxs]
            sub_lineages = [lineage_ids[i] for i in idxs]
            specs = self._candidate_specs(candidates=sub_candidates, lineage_ids=sub_lineages, seed_offset=seed_offset)
            evals = self._eval_specs(specs, n=n_i, delta=delta, seed_offset=seed_offset)
            for local_j, ev in enumerate(evals):
                global_i = idxs[local_j]
                partial_results[global_i] = (sub_candidates[local_j], ev)

        return [x for x in partial_results if x is not None]

    def to_archive_item(self, ev: EvalResult, c: Optional[Candidate] = None) -> ArchiveItem:
        return ArchiveItem(
            candidate_id=ev.candidate_id, lineage_id=ev.lineage_id, generation=ev.generation,
            source=ev.source, domain=ev.domain, split=ev.split, prompt_text=ev.prompt_text,
            rho=ev.rho, max_spend=ev.max_spend, certified=ev.certified,
            effective_budget_min=ev.effective_budget_min, final_budget_mean=ev.final_budget_mean,
            delta_init_mean=ev.delta_init_mean, N=ev.N,
            rationale=c.rationale if c else "", novelty_tag=c.novelty_tag if c else "",
            expected_rho=c.expected_rho if c else 0.0,
            parent_ids=ev.parent_ids, parent_lineage_ids=ev.parent_lineage_ids,
        )

    def initialize(self, prompts: List[Any]) -> Tuple[List[Any], List[Any]]:
        init_archive, heldout = self.init_pool(prompts)

        init_pairs = self._eval_prompts_batched(
            prompts=init_archive, generation=0, source="init", n=self.cfg.init_traj,
            delta=self.cfg.delta_screen, candidate_prefix="init_", lineage_prefix="seed_",
        )

        init_rows = []
        eval_rows = []

        for p, ev in init_pairs:
            eval_rows.append(self._annotate_eval(ev))
            if ev.certified:
                init_rows.append(self.to_archive_item(ev))
                self.lineage_scores[f"seed_{p.prompt_id}"][0] = ev.rho

        self.archive_history.extend(init_rows)
        self.current_archive = list(init_rows)

        self._write_jsonl(self.output_dir / "init_eval.jsonl", eval_rows)
        self._write_json(self.output_dir / "archive_after_init.json", [asdict(x) for x in self.current_archive])
        return init_archive, heldout

    def load_resume_state(self) -> bool:
        init_path = self.output_dir / "archive_after_init.json"
        current_path = self.output_dir / "archive_current.json"
        history_path = self.output_dir / "archive_history.json"
        genlog_path = self.output_dir / "generation_log.json"

        if history_path.exists():
            hist_rows = self._read_json(history_path)
        elif init_path.exists():
            hist_rows = self._read_json(init_path)
        else:
            return False

        curr_rows = self._read_json(current_path) if current_path.exists() else hist_rows

        self.archive_history = [ArchiveItem(**row) for row in hist_rows]
        self.current_archive = [ArchiveItem(**row) for row in curr_rows]
        self.generation_log = self._read_json(genlog_path) if genlog_path.exists() else []

        self.lineage_scores = defaultdict(dict)
        for a in self.archive_history:
            prev = self.lineage_scores[a.lineage_id].get(a.generation, 0.0)
            self.lineage_scores[a.lineage_id][a.generation] = max(prev, a.rho)

        return True

    def run_resume(self, prompts: List[Any]):
        init_pool, heldout_pool = self.init_pool(prompts)

        resumed = self.load_resume_state()
        if not resumed:
            init_pool, heldout_pool = self.initialize(prompts)

        start_gen = max([g["generation"] for g in self.generation_log], default=0) + 1

        for g in range(start_gen, self.cfg.generations + 1):
            self.run_generation(g, init_pool)
        assert all(bool(a.certified) for a in self.current_archive)
        assert all(np.isfinite(a.rho) for a in self.current_archive)
        assert all(bool(a.certified) for a in self.archive_history)
        report = self.final_validation(heldout_pool)
        print(json.dumps(report, indent=2))

    def lineage_context(self) -> List[ArchiveItem]:
        if not self.archive_history:
            return []
        rho_all = np.asarray([a.rho for a in self.archive_history], dtype=np.float32)
        p40 = float(np.quantile(rho_all, 0.4)) if len(rho_all) else 0.0
        out = []
        for item in sorted(self.current_archive, key=lambda x: x.rho, reverse=True):
            hist = self.lineage_scores[item.lineage_id]
            recent_gens = sorted(hist.keys())[-3:]
            recent_best = max([hist[g] for g in recent_gens], default=0.0)
            if recent_best >= p40:
                out.append(item)
        return out

    def generation_candidates(self, g: int) -> List[Candidate]:
        context = self.lineage_context() or sorted(self.current_archive, key=lambda x: x.rho, reverse=True)
        if not context:
            return []
        ranked = sorted(context, key=lambda x: x.rho, reverse=True)
        parent_pool_k = min(len(ranked), max(6, self.cfg.archive_keep))
        parent_pool = ranked[:parent_pool_k]
        out = []

        for call_i in range(self.cfg.calls_per_generation):
            if len(parent_pool) <= 3:
                local_context = list(parent_pool)
            else:
                sample_k = min(len(parent_pool), 4 + (call_i % 3))
                local_context = random.sample(parent_pool, sample_k)
            random.shuffle(local_context)
            out.extend(self.optimizer.generate(g, local_context, self.cfg.candidates_per_call))

        if len(parent_pool) >= 3:
            for call_i in range(self.cfg.crossover_calls_per_generation):
                parents = random.sample(parent_pool, 3)
                out.extend(self.optimizer.crossover(g, parents, self.cfg.crossover_candidates_per_call))

        random.shuffle(out)
        return out

    def run_generation(self, g: int, init_pool: List[Any]):
        surrogate_info = self.surrogate.fit(self.archive_history, self.cfg.K)
        raw_candidates = self.generation_candidates(g)

        history_texts = [a.prompt_text for a in self.archive_history]
        deduped = dedupe_candidates(raw_candidates, history_texts)
        length_ok = filter_by_length(deduped, self.evaluator.tokenizer, self.cfg.min_prompt_tokens, self.cfg.max_prompt_tokens)

        screened = []
        no_surrogate_pool = []
        screen_debug = {"mode": "fallback", "ready": bool(self.surrogate.ready), "length_ok": len(length_ok)}

        if self.surrogate.ready and length_ok:
            texts = [c.prompt_text for c in length_ok]
            scores = self._normalize_surrogate_scores(self.surrogate.predict(texts, [0.0] * len(length_ok)), len(length_ok))

            safe_mean = scores["safe_mean"]
            safe_sigma = scores["safe_sigma"]
            margin = scores["margin"]

            exploit_score = (safe_mean - 0.50 * safe_sigma + 0.15 * np.clip(margin, -1.0, 1.0)).astype(np.float32)
            boundary_score = (-np.abs(safe_mean - 0.5) + 0.50 * safe_sigma).astype(np.float32)
            explore_score = (safe_sigma - 0.15 * np.abs(safe_mean - 0.5) + 0.10 * np.clip(margin, -1.0, 1.0)).astype(np.float32)

            exploit_order = np.argsort(-exploit_score)
            boundary_order = np.argsort(-boundary_score)
            explore_order = np.argsort(-explore_score)

            keep_n = min(self.cfg.prescreen_keep, len(length_ok))
            exploit_k = min(keep_n, max(1, int(round(0.60 * keep_n))))
            boundary_k = min(keep_n - exploit_k, max(0, int(round(0.20 * keep_n))))
            explore_k = max(0, keep_n - exploit_k - boundary_k)

            chosen = []
            seen = set()

            def _take(order, k):
                for idx in order:
                    if idx in seen:
                        continue
                    seen.add(int(idx))
                    chosen.append(length_ok[int(idx)])
                    if len(chosen) >= k:
                        break

            _take(exploit_order, exploit_k)
            _take(boundary_order, exploit_k + boundary_k)
            _take(explore_order, keep_n)

            if len(chosen) < keep_n:
                fallback_order = np.argsort(-(safe_mean - 0.25 * safe_sigma))
                _take(fallback_order, keep_n)

            chosen_idx = np.asarray(list(seen), dtype=np.int64)
            chosen_rank_score = exploit_score[chosen_idx]
            rerank_local = np.argsort(-chosen_rank_score)
            screened = [length_ok[int(chosen_idx[i])] for i in rerank_local]

            no_surrogate_pool = [c for i, c in enumerate(length_ok) if i not in seen]

            screen_debug = {
                "mode": "surrogate", "ready": True, "length_ok": len(length_ok), "keep_n": keep_n,
                "exploit_k": exploit_k, "boundary_k": boundary_k, "explore_k": explore_k,
                "safe_mean_max": float(np.max(safe_mean)) if len(safe_mean) else 0.0,
                "safe_mean_min": float(np.min(safe_mean)) if len(safe_mean) else 0.0,
                "safe_sigma_mean": float(np.mean(safe_sigma)) if len(safe_sigma) else 0.0,
                "margin_mean": float(np.mean(margin)) if len(margin) else 0.0,
            }
        else:
            screened = length_ok[:self.cfg.prescreen_keep]
            no_surrogate_pool = length_ok[self.cfg.prescreen_keep:]

        med_pool = screened[:self.cfg.med_fid_keep]
        med_lineages = [lineage_id_for_candidate(c, self.current_archive) for c in med_pool]
        med_pairs = self._eval_candidates_batched(candidates=med_pool, lineage_ids=med_lineages, n=self.cfg.med_fid_traj, delta=self.cfg.delta_screen, seed_offset=0)

        med_results = []
        med_rows = []
        for c, ev12 in med_pairs:
            med_results.append((c, ev12))
            med_rows.append(self._annotate_eval(ev12))

        self._write_jsonl(self.output_dir / f"gen_{g:02d}_medfid.jsonl", med_rows)

        survivors = [(c, ev) for c, ev in med_results if ev.certified and ev.effective_budget_min > 0]
        survivors.sort(key=lambda x: (np.isfinite(x[1].rho), x[1].rho if np.isfinite(x[1].rho) else -float("inf")), reverse=True)

        topup_candidates = [c for c, _ in survivors[:self.cfg.topup_keep]]
        topup_lineages = [ev.lineage_id for _, ev in survivors[:self.cfg.topup_keep]]
        ev12_by_id = {ev.candidate_id: ev for _, ev in survivors[:self.cfg.topup_keep]}

        ev8_pairs = self._eval_candidates_batched(candidates=topup_candidates, lineage_ids=topup_lineages, n=self.cfg.topup_traj, delta=self.cfg.delta_screen, seed_offset=self.cfg.med_fid_traj)

        updated_items = []
        topup_rows = []
        invalid_topup_rows = []

        for c, ev8 in ev8_pairs:
            ev12 = ev12_by_id[c.candidate_id]
            spends = ev12.spends + ev8.spends
            final_budgets = ev12.final_budgets + ev8.final_budgets
            delta_inits = ev12.delta_inits + ev8.delta_inits

            max_spend, rho, certified = budget_check(spends, final_budgets)
            effective_budget_min = max(0.0, min(ev12.effective_budget_min, ev8.effective_budget_min))

            ev20 = EvalResult(
                candidate_id=ev12.candidate_id, lineage_id=ev12.lineage_id, generation=ev12.generation,
                source=ev12.source, domain=ev12.domain, split=ev12.split, prompt_text=ev12.prompt_text,
                N=len(spends), spends=spends, final_budgets=final_budgets, delta_inits=delta_inits,
                mean_spend=float(np.mean(spends)), var_spend=float(np.var(spends, ddof=1)) if len(spends) > 1 else 0.0,
                max_spend=max_spend, rho=float(rho) if rho is not None else 0.0, certified=certified,
                effective_budget_min=float(effective_budget_min),
                delta_init_mean=float(np.mean(delta_inits)) if delta_inits else 0.0,
                final_budget_mean=float(np.mean(final_budgets)) if final_budgets else 0.0,
                parent_ids=ev12.parent_ids, parent_lineage_ids=ev12.parent_lineage_ids,
                timestamp=datetime.now(timezone.utc).isoformat(),
                utilisations=list(ev12.utilisations) + list(ev8.utilisations),
                activity=list(ev12.activity) + list(ev8.activity),
            )

            row = self._annotate_eval(ev20)

            if certified:
                topup_rows.append(row)
                item = self.to_archive_item(ev20, c)
                updated_items.append(item)
                self.lineage_scores[item.lineage_id][g] = max(self.lineage_scores[item.lineage_id].get(g, 0.0), item.rho)
            else:
                row["rejected_reason"] = "not_certified"
                invalid_topup_rows.append(row)

        self._write_jsonl(self.output_dir / f"gen_{g:02d}_topup.jsonl", topup_rows)
        self._write_jsonl(self.output_dir / f"gen_{g:02d}_topup_invalid.jsonl", invalid_topup_rows)

        ablation_rows = []
        if g % 2 == 0:
            rand_pool = random.sample(init_pool, min(self.cfg.ablation_random, len(init_pool)))
            rand_pairs = self._eval_prompts_batched(prompts=rand_pool, generation=g, source="ablation_random", n=8, delta=self.cfg.delta_screen, candidate_prefix=f"abl_rand_{g}_", lineage_prefix="seed_")
            for _, ev in rand_pairs:
                ablation_rows.append(self._annotate_eval(ev))

            no_surrogate_candidates = no_surrogate_pool[:self.cfg.ablation_no_surrogate]
            no_surrogate_lineages = [lineage_id_for_candidate(c, self.current_archive) for c in no_surrogate_candidates]
            no_surrogate_pairs = self._eval_candidates_batched(candidates=no_surrogate_candidates, lineage_ids=no_surrogate_lineages, n=8, delta=self.cfg.delta_screen, seed_offset=0)
            for _, ev in no_surrogate_pairs:
                row = self._annotate_eval(ev)
                row["ablation"] = "no_surrogate"
                ablation_rows.append(row)

            self._write_jsonl(self.output_dir / f"gen_{g:02d}_ablations.jsonl", ablation_rows)

        self.archive_history.extend(updated_items)

        certified_items = [a for a in self.archive_history if a.certified]
        certified_items = [a for a in certified_items if np.isfinite(a.rho)]
        certified_items.sort(key=lambda x: x.rho, reverse=True)

        embeds = self.surrogate.sentence_embed([a.prompt_text for a in certified_items]) if certified_items else np.zeros((0, 768), dtype=np.float32)
        quality = np.asarray([max(1e-4, float(a.rho)) for a in certified_items], dtype=np.float64) if certified_items else np.zeros(0, dtype=np.float64)
        selected_idx, dpp_info = k_dpp_select(embeds, quality, self.cfg.archive_keep) if len(certified_items) else ([], {"mode": "empty"})

        self.current_archive = [certified_items[i] for i in selected_idx] if selected_idx else certified_items[:self.cfg.archive_keep]
        self.current_archive.sort(key=lambda x: x.rho, reverse=True)

        gen_log = {
            "generation": g, "raw_candidates": len(raw_candidates), "after_dedup": len(deduped),
            "after_length": len(length_ok), "screened": len(screened), "med_fid": len(med_pool),
            "survivors_under_0.9K": len(survivors), "topup_promoted": len(updated_items),
            "candidate_validity_rate": len(updated_items) / max(1, len(raw_candidates)),
            "invalid_topup_count": len(invalid_topup_rows), "surrogate": surrogate_info,
            "surrogate_screen": screen_debug, "dpp": dpp_info,
            "best_rho": max([a.rho for a in self.current_archive if np.isfinite(a.rho)], default=0.0),
        }
        self.generation_log.append(gen_log)

        self._write_json(self.output_dir / "generation_log.json", self.generation_log)
        self._write_json(self.output_dir / "archive_current.json", [asdict(x) for x in self.current_archive])
        self._write_json(self.output_dir / "archive_history.json", [asdict(x) for x in self.archive_history])

    def pareto_front(self) -> List[ArchiveItem]:
        pool = self._eligible_archive_items(self.current_archive)
        if not pool:
            return []
        if len(pool) == 1:
            return list(pool)

        embeds = self.surrogate.sentence_embed([a.prompt_text for a in pool])
        dists = np.sqrt(((embeds[:, None, :] - embeds[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        diversity = dists.min(axis=1)

        keep = []
        for i, a in enumerate(pool):
            dominated = False
            for j, b in enumerate(pool):
                if i == j:
                    continue
                if b.rho >= a.rho and diversity[j] >= diversity[i] and (b.rho > a.rho or diversity[j] > diversity[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(a)

        keep.sort(key=lambda x: x.rho, reverse=True)
        return keep

    def final_validation(self, heldout_pool: List[Any]):
        def build_pools():
            certified_history = self._eligible_archive_items(self.archive_history)
            certified_history.sort(key=lambda x: x.rho, reverse=True)
            front = self.pareto_front()
            final_pool = front[:self.cfg.final_keep]
            stress_archive = certified_history[:self.cfg.stress_keep]
            return certified_history, final_pool, stress_archive

        def eval_candidate_pool(pool, n, delta, generation, pool_name):
            candidates = [Candidate(candidate_id=a.candidate_id, generation=generation, prompt_text=a.prompt_text, rationale=a.rationale, novelty_tag=a.novelty_tag, expected_rho=a.expected_rho, source=a.source, parent_ids=a.parent_ids, parent_lineage_ids=a.parent_lineage_ids) for a in pool]
            lineages = [a.lineage_id for a in pool]
            pairs = self._eval_candidates_batched(candidates=candidates, lineage_ids=lineages, n=n, delta=delta, seed_offset=0)
            rows = []
            rows_pass = []
            violations = []
            for _, ev in pairs:
                row = self._annotate_eval(ev)
                rows.append(row)
                if ev.certified:
                    rows_pass.append(row)
                else:
                    violations.append({"pool": pool_name, "candidate_id": ev.candidate_id, "lineage_id": ev.lineage_id, "max_spend": ev.max_spend, "delta_init_mean": ev.delta_init_mean})
            return rows, rows_pass, violations

        max_cleanup_rounds = 5
        seen_bad_pairs = set()

        final_rows = []
        final_rows_pass = []
        stress_rows = []
        stress_rows_pass = []
        violations = []
        certified_history = []

        for _ in range(max_cleanup_rounds):
            certified_history, final_pool, stress_archive = build_pools()

            for a in final_pool:
                assert a.candidate_id not in self.disqualified_candidate_ids
                assert a.lineage_id not in self.disqualified_lineage_ids

            for a in stress_archive:
                assert a.candidate_id not in self.disqualified_candidate_ids
                assert a.lineage_id not in self.disqualified_lineage_ids

            final_rows, final_rows_pass, final_viol = eval_candidate_pool(final_pool, self.cfg.final_traj, self.cfg.delta_final, 999, "final")
            stress_rows, stress_rows_pass, stress_viol = eval_candidate_pool(stress_archive, self.cfg.stress_traj, self.cfg.delta_stress, 1000, "stress")

            violations = final_viol + stress_viol
            if not violations:
                break

            new_bad_pairs = {(v.get("candidate_id"), v.get("lineage_id")) for v in violations} - seen_bad_pairs

            if not new_bad_pairs:
                break

            seen_bad_pairs |= new_bad_pairs
            self._disqualify_violations(violations)

        heldout_pairs = self._eval_prompts_batched(prompts=heldout_pool[:self.cfg.heldout_keep], generation=999, source="heldout", n=self.cfg.heldout_traj, delta=self.cfg.delta_heldout, candidate_prefix="heldout_", lineage_prefix="heldout_")
        heldout_rows = [self._annotate_eval(ev) for _, ev in heldout_pairs]

        self._write_jsonl(self.output_dir / "final_validation.jsonl", final_rows)
        self._write_jsonl(self.output_dir / "final_validation_pass.jsonl", final_rows_pass)
        self._write_jsonl(self.output_dir / "heldout_validation.jsonl", heldout_rows)
        self._write_jsonl(self.output_dir / "stress_validation.jsonl", stress_rows)
        self._write_jsonl(self.output_dir / "stress_validation_pass.jsonl", stress_rows_pass)

        final_rhos = [r["rho"] for r in final_rows_pass]
        heldout_rhos = [r["rho"] for r in heldout_rows]

        report = {
            "K": self.cfg.K,
            "max_rho_archive": max([a.rho for a in certified_history], default=0.0),
            "final_candidates_evaluated": len(final_rows),
            "final_candidates_passed": len(final_rows_pass),
            "final_pass_rate": float(len(final_rows_pass) / max(1, len(final_rows))),
            "stress_candidates_evaluated": len(stress_rows),
            "stress_candidates_passed": len(stress_rows_pass),
            "heldout_generalization_gap": (float(np.mean(final_rhos)) - float(np.mean(heldout_rhos))) if final_rhos and heldout_rhos else None,
            "candidate_validity_rate_mean": float(np.mean([g["candidate_validity_rate"] for g in self.generation_log])) if self.generation_log else None,
            "violations": violations,
        }

        self._write_json(self.output_dir / "final_report.json", report)
        return report

    def run(self, prompts: List[Any]):
        init_pool, heldout_pool = self.initialize(prompts)
        for g in range(1, self.cfg.generations + 1):
            self.run_generation(g, init_pool)
        assert all(bool(a.certified) for a in self.current_archive)
        assert all(bool(a.certified) for a in self.archive_history)
        assert all(np.isfinite(a.rho) for a in self.current_archive)
        assert all(np.isfinite(a.rho) for a in self.archive_history)
        report = self.final_validation(heldout_pool)
        print(json.dumps(report, indent=2))


def parse_args() -> E2Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="output/e2_outputs")
    p.add_argument("--safe-model-path", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    p.add_argument("--risky-model-path", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--length-bucket-width", type=int, default=32)
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.set_defaults(trust_remote_code=True)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--parallelize", dest="parallelize", action="store_true")
    p.add_argument("--no-parallelize", dest="parallelize", action="store_false")
    p.set_defaults(parallelize=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prefix-n", type=int, default=5)
    p.add_argument("--use-chat-template", action="store_true", help="wrap prompts as one user turn of the risky model's chat template")
    p.add_argument("--delta-final", type=float, default=0.0033)
    p.add_argument("--delta-heldout", type=float, default=0.0033)
    p.add_argument("--delta-stress", type=float, default=0.0033)
    p.add_argument("--delta-screen", type=float, default=0.0033)
    p.add_argument("--factscore-field", default="factscore_prompt")

    args = p.parse_args()
    return E2Config(
        data_dir=args.data_dir, output_dir=args.output_dir,
        safe_model_path=args.safe_model_path, risky_model_path=args.risky_model_path,
        device=args.device, device_map=args.device_map, dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        parallelize=args.parallelize, verbose=args.verbose,
        k=args.k, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        prefix_n=args.prefix_n, use_chat_template=args.use_chat_template, delta_screen=args.delta_screen, delta_final=args.delta_final,
        delta_heldout=args.delta_heldout, delta_stress=args.delta_stress,
        factscore_field=args.factscore_field,
        eval_batch_size=args.eval_batch_size, length_bucket_width=args.length_bucket_width,
    )


def main():
    cfg = parse_args()
    set_global_seed(cfg.seeds[0] if cfg.seeds else 42)
    prompts = load_prompt_corpus(cfg.data_dir, cfg.factscore_field)
    runner = E2Runner(cfg)
    runner.run(prompts)
