import gc
import os
import time
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from .tokenizer import init_tokenizer
from .loader import _is_bitsandbytes_available, _build_quantization_config


class AnchoredDecodingFactory:

    @classmethod
    def from_pretrained(
        cls,
        safe_model_path: Optional[str] = None,
        risky_model_path: Optional[str] = None,
        safe_model: Optional[torch.nn.Module] = None,
        risky_model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        k_radius: float = 0.15,
        verbose: bool = False,
        use_prefix_debt: bool = True,
        prefix_n: int = 5,
        log_kl_stats: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        max_memory: Optional[dict] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs,
    ):
        if use_prefix_debt:
            assert (
                prefix_n is not None
            ), "prefix_n must be set when use_prefix_debt is True"

        if tokenizer is None:
            if safe_model_path is None:
                raise ValueError("tokenizer or safe_model_path must be provided")
            tokenizer = init_tokenizer(
                safe_model_path,
                padding_side="left",
                trust_remote_code=trust_remote_code,
            )

        num_gpus = torch.cuda.device_count()
        if verbose:
            print(f"[INFO] AnchoredDecoding: Detected {num_gpus} GPUs")

        if num_gpus >= 2 and safe_model is None and risky_model is None:
            if verbose:
                print(f"[INFO] Loading risky model on cuda:0, safe model on cuda:1")
            quantization_config = _build_quantization_config(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                dtype=dtype,
            )
            risky_load = dict(
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
                device_map={"": 0},
                **kwargs,
            )
            safe_load = dict(
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
                device_map={"": 1},
                **kwargs,
            )
            if quantization_config is not None:
                risky_load["quantization_config"] = quantization_config
                safe_load["quantization_config"] = quantization_config
        else:
            if verbose:
                print(f"[INFO] Using device_map={device_map}")
            quantization_config = _build_quantization_config(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                dtype=dtype,
            )
            common_load = dict(
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                **kwargs,
            )
            if quantization_config is not None:
                common_load["quantization_config"] = quantization_config
            risky_load = safe_load = common_load

        if safe_model is None:
            if safe_model_path is None:
                raise ValueError("safe_model or safe_model_path must be provided")
            safe_model = AutoModelForCausalLM.from_pretrained(
                safe_model_path, **safe_load
            )

        if risky_model is None:
            if risky_model_path is None:
                raise ValueError("risky_model or risky_model_path must be provided")
            risky_model = AutoModelForCausalLM.from_pretrained(
                risky_model_path, **risky_load
            )

        target_vocab = len(tokenizer)
        if safe_model.get_input_embeddings().weight.shape[0] != target_vocab:
            raise ValueError(
                f"Safe model vocab size ({safe_model.get_input_embeddings().weight.shape[0]}) "
                f"does not match tokenizer vocab size ({target_vocab}). "
                "Please use byte-level decoding (Coming soon...)"
            )
        if risky_model.get_input_embeddings().weight.shape[0] != target_vocab:
            raise ValueError(
                f"Risky model vocab size ({risky_model.get_input_embeddings().weight.shape[0]}) "
                f"does not match tokenizer vocab size ({target_vocab}). "
                "Please use byte-level decoding (Coming soon...)"
            )

        for mdl in (safe_model, risky_model):
            mdl.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            mdl.config.eos_token_id = tokenizer.eos_token_id
            if not use_prefix_debt:
                mdl.config.num_logits_to_keep = 1
            mdl.config.output_attentions = False
            mdl.config.output_hidden_states = False

        return cls(
            safe_model=safe_model,
            risky_model=risky_model,
            tokenizer=tokenizer,
            k_radius=k_radius,
            verbose=verbose,
            use_prefix_debt=use_prefix_debt,
            prefix_n=prefix_n,
            log_kl_stats=log_kl_stats,
            device=device,
        )

    def __init__(
        self,
        safe_model: torch.nn.Module,
        risky_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        k_radius: float = 0.15,
        use_prefix_debt: bool = True,
        prefix_n: int = 5,
        log_kl_stats: bool = False,
        verbose: bool = False,
        device: Optional[torch.device] = None,
        eps_kl: float = 1e-4,
        solver_max_iter: int = 20,
    ) -> None:
        self.config = safe_model.config
        self.safe_model = safe_model
        self.risky_model = risky_model
        self.tokenizer = tokenizer
        self.k_radius = k_radius
        self.prefix_n = prefix_n
        self.eps_kl = eps_kl
        self.solver_max_iter = solver_max_iter

        assert self.k_radius >= 0.0, "k_radius must be positive"

        self.verbose = verbose
        self.use_prefix_debt = use_prefix_debt

        self.device = device or next(self.safe_model.parameters()).device

        self.safe_device = next(self.safe_model.parameters()).device
        self.risky_device = next(self.risky_model.parameters()).device

        self.safe_model.eval()
        self.risky_model.eval()

        self.log_kl_stats = log_kl_stats
        self.kl_stats_history = []

    def get_kl_stats_summary(self) -> dict:
        if not hasattr(self, "kl_stats_history") or not self.kl_stats_history:
            return {
                "per_step": [],
                "final_cum_kl_spent_per_seq": [0.0],
                "final_budget_per_seq": [0.0],
                "budget_utilization_per_seq": [0.0],
            }

        per_step = []
        for step in self.kl_stats_history:
            row = dict(step)
            for key, value in list(row.items()):
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                    if value.ndim == 0:
                        row[key] = value.item()
                    else:
                        row[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    row[key] = value.tolist()
                elif isinstance(value, tuple):
                    row[key] = list(value)
            per_step.append(row)

        last = per_step[-1]

        def _as_list(x, default=0.0):
            if x is None:
                return [default]
            if isinstance(x, list):
                return x
            return [x]

        final_cum = _as_list(last.get("cum_kl_spent", last.get("cumklspent", None)), default=0.0)
        final_budget = _as_list(last.get("budget_so_far", last.get("budgetsofar", None)), default=0.0)

        n = max(len(final_cum), len(final_budget))
        if len(final_cum) < n:
            final_cum = final_cum + [final_cum[-1] if final_cum else 0.0] * (n - len(final_cum))
        if len(final_budget) < n:
            final_budget = final_budget + [final_budget[-1] if final_budget else 0.0] * (n - len(final_budget))

        budget_util = []
        for spend, budget in zip(final_cum, final_budget):
            spend = float(spend)
            budget = float(budget)
            if np.isfinite(budget) and abs(budget) > 1e-12:
                budget_util.append(spend / budget)
            else:
                budget_util.append(0.0)

        return {
            "per_step": per_step,
            "final_cum_kl_spent_per_seq": [float(x) for x in final_cum],
            "final_budget_per_seq": [float(x) for x in final_budget],
            "budget_utilization_per_seq": [float(x) for x in budget_util],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        text: Optional[str | List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        parallelize: bool = False,
        k_radius: Optional[float] = None,
        seed: Optional[int] = None,
        **model_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        if seed is not None:
            from transformers import set_seed
            set_seed(seed)

        k_radius = k_radius if k_radius is not None else self.k_radius

        if generation_config is None:
            generation_config = GenerationConfig(**model_kwargs)

        if text is not None:
            if input_ids is not None:
                raise ValueError("Only one of `text` or `input_ids` should be provided.")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        if input_ids is None:
            raise ValueError("Either `text` or `input_ids` must be provided.")

        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id

        attention_mask = self._prepare_attention_mask(input_ids, attention_mask, generation_config)

        self._validate_generate_inputs(input_ids, generation_config, attention_mask=attention_mask)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        pad_token_id = self._prepare_pad_token_id(generation_config)
        eos_token_id = generation_config.eos_token_id

        stopping_criteria = self._prepare_stopping_criteria(stopping_criteria, generation_config)

        logits_warper = self._prepare_logits_warper(logits_warper, generation_config)
        logits_processor = self._prepare_logits_processor(logits_processor, generation_config)

        output = self._decode(
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            logits_processor=logits_processor,
            k_radius=k_radius,
            do_sample=bool(generation_config.do_sample),
            min_new_tokens=getattr(generation_config, "min_new_tokens", 0),
            **model_kwargs,
        )

        return output

    def _validate_generate_inputs(self, input_ids, generation_config, attention_mask=None):
        if hasattr(generation_config, "max_new_tokens") and generation_config.max_new_tokens is not None:
            if not (isinstance(generation_config.max_new_tokens, int) and generation_config.max_new_tokens > 0):
                raise ValueError("`max_new_tokens` should be a strictly positive integer.")
            if attention_mask is not None:
                input_len = attention_mask.sum(dim=-1).max().item()
            else:
                input_len = input_ids.shape[1]
            generation_config.max_length = int(input_len + generation_config.max_new_tokens)
        elif generation_config.max_length is not None:
            if not (isinstance(generation_config.max_length, int) and generation_config.max_length > 0):
                raise ValueError("`max_length` should be a strictly positive integer.")
        else:
            raise ValueError("Either `max_length` or `max_new_tokens` must be provided.")

        if generation_config.do_sample:
            if generation_config.temperature <= 0:
                raise ValueError("`temperature` should be positive for sampling decoding.")

        if generation_config.num_return_sequences not in (None, 1):
            raise ValueError("Only one generation is supported.")
        if generation_config.num_beams not in (None, 1):
            raise ValueError("Beam search is not supported.")

        if generation_config.pad_token_id is not None:
            if self.safe_model.config.pad_token_id is not None and generation_config.pad_token_id != self.safe_model.config.pad_token_id:
                warnings.warn(f"Pad token mismatch with safe model: {generation_config.pad_token_id} vs {self.safe_model.config.pad_token_id}")
            if self.risky_model.config.pad_token_id is not None and generation_config.pad_token_id != self.risky_model.config.pad_token_id:
                warnings.warn(f"Pad token mismatch with risky model: {generation_config.pad_token_id} vs {self.risky_model.config.pad_token_id}")

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if input_ids.dim() != 2:
            raise ValueError("Input prompt should be of shape (batch_size, sequence length).")
        if self.safe_model.config.vocab_size != self.risky_model.config.vocab_size:
            raise ValueError("Models must have the same vocabulary.")

    def _prepare_attention_mask(self, input_ids, attention_mask, generation_config):
        if attention_mask is None:
            if generation_config.pad_token_id is not None and (input_ids == generation_config.pad_token_id).any():
                attention_mask = input_ids.ne(generation_config.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, device=self.device)
        if generation_config.pad_token_id is not None and (input_ids[:, -1] == generation_config.pad_token_id).sum() > 0 and self.verbose:
            print("A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.")
        return attention_mask

    def _prepare_pad_token_id(self, generation_config):
        if generation_config.eos_token_id is not None:
            if self.verbose and generation_config.pad_token_id != generation_config.eos_token_id:
                print(f"Overriding `pad_token_id` to {generation_config.eos_token_id} (`eos_token_id`) to generate safe sequences.")
            pad_token_id = generation_config.eos_token_id
        elif generation_config.pad_token_id is not None:
            pad_token_id = generation_config.pad_token_id
        else:
            raise ValueError("Neither `pad_token_id` nor `eos_token_id` is defined.")
        return pad_token_id

    def _prepare_stopping_criteria(self, stopping_criteria, generation_config):
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        stopping_criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        return stopping_criteria

    def _prepare_logits_processor(self, logits_processor, generation_config):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        rp = getattr(generation_config, "repetition_penalty", None)
        if rp is not None and rp != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=rp))
        nrng = getattr(generation_config, "no_repeat_ngram_size", None)
        if nrng is not None and nrng > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(nrng))
        return logits_processor

    def _prepare_logits_warper(self, logits_warper, generation_config):
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        if generation_config.temperature is not None and generation_config.temperature not in (0.0, 1.0):
            logits_warper.append(TemperatureLogitsWarper(generation_config.temperature))
        return logits_warper

    def _safe_kl_terms(self, log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        return (
            torch.nan_to_num(log_p.exp() * (log_p - log_q), nan=0.0, posinf=float("inf"), neginf=0.0)
            .sum(dim=-1)
            .clamp(min=0.0)
        )

    @torch.no_grad()
    def forward_direct(self, model, input_ids, attention_mask, past_key_values, move_to_device=None):
        dev = next(model.parameters()).device
        ids = input_ids.to(dev, non_blocking=True)
        mask = attention_mask.to(dev, non_blocking=True) if attention_mask is not None else None

        if past_key_values is None:
            out = model(input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True)
        else:
            out = model(input_ids=ids[:, -1:], attention_mask=mask, past_key_values=past_key_values, use_cache=True, return_dict=True)

        logits = out.logits[:, -1, :]
        if move_to_device is not None:
            logits = logits.to(move_to_device)
        pkv = out.past_key_values
        del out
        return logits, pkv

    def _compute_prefix_debt_fast(self, safe_logp_target, risky_logp_target, input_ids, attention_mask, k):
        llr = risky_logp_target - safe_logp_target
        valid = attention_mask[:, 1:].bool() if attention_mask is not None else torch.ones_like(llr, dtype=torch.bool)
        next_tok = input_ids[:, 1:]
        if len(self.tokenizer.all_special_ids) > 0:
            special_ids = torch.tensor(list(self.tokenizer.all_special_ids), device=next_tok.device, dtype=next_tok.dtype)
            is_special = (next_tok.unsqueeze(-1) == special_ids).any(dim=-1)
            valid = valid & (~is_special)
        positive = valid & (llr > 0)
        masked = llr.masked_fill(~positive, float("-inf"))
        k_eff = min(int(k), llr.size(1))
        if k_eff <= 0:
            return torch.zeros(llr.size(0), device=llr.device, dtype=torch.float32)
        vals, _ = masked.topk(k_eff, dim=-1, largest=True)
        chosen = torch.isfinite(vals)
        vals = torch.where(chosen, vals, torch.zeros_like(vals))
        denom = chosen.sum(dim=-1).clamp(min=1).to(torch.float32)
        debt = vals.sum(dim=-1) / denom
        debt = torch.where(chosen.any(dim=-1), debt, torch.zeros_like(debt))
        return debt.to(torch.float32)

    def solve_optimization_newton(
        self,
        safe_logits: torch.Tensor,
        risky_logits: torch.Tensor,
        k_radius,  # float or Tensor [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bc, bd, log_pc, log_pd = self._solve_theta_newton(
            safe_logits, risky_logits, k_radius, max_iter=self.solver_max_iter
        )
        assert torch.allclose(
            (bc + bd).float(), torch.ones_like((bc + bd).float()), atol=1e-5, rtol=0.0
        )
        return bc, bd, log_pc, log_pd

    def _get_logp_from_weights(
        self,
        bc: torch.Tensor,
        bd: torch.Tensor,
        log_pc: torch.Tensor,
        log_pd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bc.dim() == 2 and bc.size(1) == 1:
            bc = bc.squeeze(1)
        if bd.dim() == 2 and bd.size(1) == 1:
            bd = bd.squeeze(1)

        term_d = bd.unsqueeze(-1) * log_pd
        term_c = bc.unsqueeze(-1) * log_pc

        term_d = torch.nan_to_num(term_d, nan=0.0)
        term_c = torch.nan_to_num(term_c, nan=0.0)

        next_token_logits = term_d + term_c

        log_p = F.log_softmax(next_token_logits, dim=-1)

        return log_p, log_pc, next_token_logits

    def _solve_theta_newton(
        self,
        safe_logits: torch.Tensor,
        risky_logits: torch.Tensor,
        k_radius,
        max_iter: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = safe_logits.device
        B, V = safe_logits.shape

        log_pd = F.log_softmax(risky_logits.float(), dim=-1)
        log_pc = F.log_softmax(safe_logits.float(), dim=-1)

        k_t = torch.as_tensor(k_radius, device=device, dtype=torch.float32)
        if k_t.ndim == 0:
            k_t = k_t.expand(B)
        else:
            k_t = k_t.view(-1)
            assert k_t.numel() == B, f"k_t must be scalar or shape [B], got {k_t.shape}"

        mask_force_pc = k_t <= 0.0

        KL_pd_pc = self._safe_kl_terms(log_pd, log_pc)

        mask_use_pd = (KL_pd_pc <= k_t) & (~mask_force_pc)

        active = ~(mask_force_pc | mask_use_pd)

        w_c = torch.empty((B, 1), device=device, dtype=torch.float32)
        w_d = torch.empty((B, 1), device=device, dtype=torch.float32)

        w_c[mask_force_pc] = 1.0
        w_d[mask_force_pc] = 0.0
        w_c[mask_use_pd] = 0.0
        w_d[mask_use_pd] = 1.0

        if not active.any():
            return w_c, w_d, log_pc, log_pd

        log_pc_a = log_pc[active]
        log_pd_a = log_pd[active]
        k_a = k_t[active]
        Ba = log_pc_a.size(0)

        a = log_pd_a - log_pc_a
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        def kl_theta(th: torch.Tensor) -> torch.Tensor:
            q_log_unnorm = log_pc_a + th[:, None] * a
            logZ = torch.logsumexp(q_log_unnorm, dim=-1)
            log_q = q_log_unnorm - logZ[:, None]
            return self._safe_kl_terms(log_q, log_pc_a)

        lo = torch.zeros(Ba, device=device, dtype=torch.float32)
        hi = torch.ones(Ba, device=device, dtype=torch.float32)

        theta = torch.clamp(k_a / (k_a + 1.0), 1e-4, 1.0 - 1e-4)

        eps = 1e-9

        for _ in range(max_iter):
            q = log_pc_a + theta[:, None] * a
            logZ = torch.logsumexp(q, dim=-1)
            q.sub_(logZ[:, None])
            q.exp_()

            mean_a = (q * a).sum(dim=-1)
            mean_a2 = (q * (a * a)).sum(dim=-1)
            var_a = (mean_a2 - mean_a * mean_a).clamp_min(0.0)

            KL = theta * mean_a - logZ
            KL = torch.nan_to_num(KL, nan=float("inf"), posinf=float("inf"), neginf=0.0)

            f = KL - k_a

            hi = torch.where(f > 0, theta, hi)
            lo = torch.where(f <= 0, theta, lo)

            fp = (theta * var_a).clamp_min(eps)
            theta_new = theta - f / fp

            bad = (theta_new <= lo) | (theta_new >= hi) | ~torch.isfinite(theta_new)
            theta = torch.where(bad, 0.5 * (lo + hi), theta_new)

            if (hi - lo).max() < 1e-6:
                break

        for _ in range(12):
            mid = 0.5 * (lo + hi)
            KL_mid = kl_theta(mid)
            feas = KL_mid <= k_a
            lo = torch.where(feas, mid, lo)
            hi = torch.where(feas, hi, mid)

        theta = lo

        wd = theta[:, None]
        wc = 1.0 - wd

        w_c[active] = wc
        w_d[active] = wd

        return w_c, w_d, log_pc, log_pd

    def _model_forward_all_logits(self, model, input_ids, attention_mask):
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = self.device

        if input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        all_logits = outputs.logits
        past_key_values = outputs.past_key_values
        del outputs
        return all_logits, past_key_values

    @torch.no_grad()
    def _decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stopping_criteria: StoppingCriteriaList,
        logits_warper: LogitsProcessorList,
        logits_processor: LogitsProcessorList,
        pad_token_id: int,
        eos_token_id: int,
        k_radius: float,
        output_logits: bool = False,
        return_dict_in_generate: bool = True,
        do_sample: bool = False,
        parallelize: bool = False,
        no_kv_cache: bool = False,
        post_hoc_logits_warper: bool = False,
        min_new_tokens: int = 0,
        show_progress: bool = False,
        **model_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        if k_radius not in (0.0, -1.0):
            if logits_processor is not None and len(logits_processor) > 0:
                raise ValueError("Anchored raw-anchor guarantee does not allow logits processors before the KL solve.")
            if logits_warper is not None and len(logits_warper) > 0:
                raise ValueError("Anchored raw-anchor guarantee does not allow logits warpers before the KL solve.")

        if k_radius not in (0.0, -1.0) and not do_sample:
            raise ValueError("Anchored Decoding guarantees apply to sampling from the fused distribution, not greedy argmax decoding. Set do_sample=True.")
        if post_hoc_logits_warper:
            raise ValueError("post_hoc_logits_warper=True breaks the per-step KL constraint on the decoding distribution. Keep post_hoc_logits_warper=False for global K-NAF guarantees.")

        logits_list: Optional[List[torch.Tensor]] = [] if (return_dict_in_generate and output_logits) else None

        if self.log_kl_stats:
            self.kl_stats_history = []

        if isinstance(eos_token_id, int):
            eos_token_id_list = [eos_token_id]
        else:
            eos_token_id_list = list(eos_token_id)
        eos_token_id_tensor = torch.tensor(eos_token_id_list, device=self.device)

        batch_size, prompt_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = input_ids.new(batch_size).fill_(1)

        model_kwargs["use_cache"] = True
        safe_past_key_values = None
        risky_past_key_values = None

        if parallelize:
            stream1 = torch.cuda.Stream(device=self.safe_device)
            stream2 = torch.cuda.Stream(device=self.risky_device)

        step_count = 0
        start_time = time.time()
        if self.verbose:
            print(f"Starting generation with prompt length {prompt_len} tokens.")

        cum_kl_spent = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        eps_kl = self.eps_kl

        if self.use_prefix_debt:
            assert self.prefix_n is not None, "prefix_n must be set when use_prefix_debt is True"

            labels = input_ids[:, 1:].unsqueeze(-1)

            if parallelize and self.safe_device != self.risky_device:
                with torch.cuda.stream(stream1):
                    c_all_out = self.safe_model(input_ids=input_ids.to(self.safe_device), attention_mask=attention_mask.to(self.safe_device), use_cache=True)
                with torch.cuda.stream(stream2):
                    d_all_out = self.risky_model(input_ids=input_ids.to(self.risky_device), attention_mask=attention_mask.to(self.risky_device), use_cache=True)
                torch.cuda.synchronize(self.safe_device)
                torch.cuda.synchronize(self.risky_device)

                safe_past_key_values = c_all_out.past_key_values
                safe_logits = c_all_out.logits[:, -1, :].to(self.device)
                c_logits_prefix = c_all_out.logits[:, :-1, :]
                c_lp = (c_logits_prefix.gather(-1, labels.to(c_logits_prefix.device)).squeeze(-1).float() - c_logits_prefix.logsumexp(dim=-1).float()).to(self.device)
                del c_all_out, c_logits_prefix

                risky_past_key_values = d_all_out.past_key_values
                risky_logits = d_all_out.logits[:, -1, :].to(self.device)
                d_logits_prefix = d_all_out.logits[:, :-1, :]
                d_lp = (d_logits_prefix.gather(-1, labels.to(d_logits_prefix.device)).squeeze(-1).float() - d_logits_prefix.logsumexp(dim=-1).float()).to(self.device)
                del d_all_out, d_logits_prefix
            else:
                c_all_out = self.safe_model(input_ids=input_ids.to(self.safe_device), attention_mask=attention_mask.to(self.safe_device), use_cache=True)
                safe_past_key_values = c_all_out.past_key_values
                safe_logits = c_all_out.logits[:, -1, :].to(self.device)
                c_logits_prefix = c_all_out.logits[:, :-1, :]
                c_lp = (c_logits_prefix.gather(-1, labels.to(c_logits_prefix.device)).squeeze(-1).float() - c_logits_prefix.logsumexp(dim=-1).float()).to(self.device)
                del c_all_out, c_logits_prefix

                d_all_out = self.risky_model(input_ids=input_ids.to(self.risky_device), attention_mask=attention_mask.to(self.risky_device), use_cache=True)
                risky_past_key_values = d_all_out.past_key_values
                risky_logits = d_all_out.logits[:, -1, :].to(self.device)
                d_logits_prefix = d_all_out.logits[:, :-1, :]
                d_lp = (d_logits_prefix.gather(-1, labels.to(d_logits_prefix.device)).squeeze(-1).float() - d_logits_prefix.logsumexp(dim=-1).float()).to(self.device)
                del d_all_out, d_logits_prefix

            prefix_debt = self._compute_prefix_debt_fast(c_lp, d_lp, input_ids, attention_mask, self.prefix_n)
            init_budget_tensor = -prefix_debt.to(torch.float32)
            if self.verbose:
                print(f"[INFO] Using prefix debt True with prefix_n={self.prefix_n}")
                print(f"[INFO] Prefix debt: {prefix_debt.tolist()}")
        else:
            if parallelize and self.safe_device != self.risky_device:
                with torch.cuda.stream(stream1):
                    safe_logits, safe_past_key_values = self.forward_direct(self.safe_model, input_ids, attention_mask, None)
                with torch.cuda.stream(stream2):
                    risky_logits, risky_past_key_values = self.forward_direct(self.risky_model, input_ids, attention_mask, None)
                torch.cuda.synchronize(self.safe_device)
                torch.cuda.synchronize(self.risky_device)
            else:
                safe_logits, safe_past_key_values = self.forward_direct(self.safe_model, input_ids, attention_mask, None)
                risky_logits, risky_past_key_values = self.forward_direct(self.risky_model, input_ids, attention_mask, None)

            init_budget_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        use_precomputed_logits = True

        max_new_tokens = getattr(stopping_criteria[0], "max_length", prompt_len + 100) - prompt_len

        if show_progress:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=int(max_new_tokens), desc="Generating", leave=False)
            except ImportError:
                pbar = None
        else:
            pbar = None

        while not this_peer_finished:
            if not use_precomputed_logits:
                if no_kv_cache:
                    safe_pkv_in = None
                    risky_pkv_in = None
                else:
                    safe_pkv_in = safe_past_key_values
                    risky_pkv_in = risky_past_key_values

                if parallelize and self.safe_device != self.risky_device:
                    with torch.cuda.stream(stream1):
                        safe_logits, safe_past_key_values = self.forward_direct(self.safe_model, input_ids, attention_mask, safe_pkv_in)
                    with torch.cuda.stream(stream2):
                        risky_logits, risky_past_key_values = self.forward_direct(self.risky_model, input_ids, attention_mask, risky_pkv_in)
                    torch.cuda.synchronize(self.safe_device)
                    torch.cuda.synchronize(self.risky_device)
                else:
                    safe_logits, safe_past_key_values = self.forward_direct(self.safe_model, input_ids, attention_mask, safe_pkv_in)
                    risky_logits, risky_past_key_values = self.forward_direct(self.risky_model, input_ids, attention_mask, risky_pkv_in)

                if no_kv_cache:
                    safe_past_key_values = None
                    risky_past_key_values = None
            else:
                use_precomputed_logits = False

            if safe_logits.device != risky_logits.device:
                safe_logits = safe_logits.to(risky_logits.device)

            safe_logits = logits_processor(input_ids, safe_logits)
            risky_logits = logits_processor(input_ids, risky_logits)

            if logits_warper is not None and len(logits_warper) > 0:
                safe_logits = logits_warper(input_ids, safe_logits)
                risky_logits = logits_warper(input_ids, risky_logits)

            generated_tokens = input_ids.shape[1] - prompt_len
            apply_min_tokens = (min_new_tokens is not None and min_new_tokens > 0 and generated_tokens < min_new_tokens and eos_token_id is not None)
            if apply_min_tokens:
                for eid in eos_token_id_list:
                    safe_logits[:, eid] = -float("inf")
                    risky_logits[:, eid] = -float("inf")

            B = safe_logits.size(0)
            device = safe_logits.device
            dtype = safe_logits.dtype
            t_gen = int(input_ids.shape[1] - prompt_len)

            if k_radius == 0.0:
                bc = torch.ones((B, 1), device=device, dtype=dtype)
                bd = torch.zeros((B, 1), device=device, dtype=dtype)
                k_t = torch.zeros((B,), device=device, dtype=torch.float32)
                budget_so_far = torch.zeros((B,), device=device, dtype=torch.float32)
                log_pc = F.log_softmax(safe_logits.float(), dim=-1)
                log_pd = F.log_softmax(risky_logits.float(), dim=-1)
            elif k_radius == -1.0:
                bc = torch.zeros((B, 1), device=device, dtype=dtype)
                bd = torch.ones((B, 1), device=device, dtype=dtype)
                k_t = torch.full((B,), float("inf"), device=device, dtype=torch.float32)
                budget_so_far = torch.full((B,), float("inf"), device=device, dtype=torch.float32)
                log_pc = F.log_softmax(safe_logits.float(), dim=-1)
                log_pd = F.log_softmax(risky_logits.float(), dim=-1)
            else:
                budget_so_far = (float(t_gen + 1) * float(k_radius)) + init_budget_tensor
                remaining = (budget_so_far - cum_kl_spent).clamp(min=0.0)
                k_t = remaining * unfinished_sequences.float()
                bc, bd, log_pc, log_pd = self.solve_optimization_newton(safe_logits, risky_logits, k_t)

            log_p, log_pc, next_token_logits = self._get_logp_from_weights(bc, bd, log_pc, log_pd)
            log_pc_realized = log_pc

            kl_step = torch.zeros((B,), device=device, dtype=torch.float32)
            if k_radius not in (0.0, -1.0):
                kl_step = self._safe_kl_terms(log_p, log_pc_realized).float()
                mask = unfinished_sequences.bool()
                violation = kl_step[mask] - k_t[mask]
                max_violation = violation.max().item() if mask.any() else 0.0
                if max_violation > eps_kl:
                    warnings.warn(f"KL constraint exceeded by {max_violation:.6f} (eps={eps_kl}). max(KL)={kl_step[mask].max().item():.6f}, max(k_t)={k_t[mask].max().item():.6f}", RuntimeWarning)
                cum_kl_spent = cum_kl_spent + kl_step * unfinished_sequences.float()

            if self.log_kl_stats:
                kl_to_safe = self._safe_kl_terms(log_p, log_pc).float()
                kl_to_risky = self._safe_kl_terms(log_p, log_pd).float()

                probs = log_p.exp()
                if k_radius not in (0.0, -1.0):
                    next_tokens = torch.multinomial(probs, 1).squeeze(1)
                elif do_sample:
                    next_tokens = torch.multinomial(probs, 1).squeeze(1)
                else:
                    next_tokens = torch.argmax(log_p, dim=-1)

                sampled_token_ids = next_tokens.detach().cpu().tolist()
                sampled_tokens = [self.tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in sampled_token_ids]

                p_star_prob = probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                p_s_prob = log_pc.exp().gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                p_risky_prob = log_pd.exp().gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()

                self.kl_stats_history.append({
                    "step": step_count,
                    "kl_to_safe": kl_to_safe.detach().cpu().tolist(),
                    "kl_to_risky": kl_to_risky.detach().cpu().tolist(),
                    "bc": bc.squeeze(-1).detach().cpu().tolist() if bc.dim() > 1 else bc.detach().cpu().tolist(),
                    "bd": bd.squeeze(-1).detach().cpu().tolist() if bd.dim() > 1 else bd.detach().cpu().tolist(),
                    "k_t": k_t.detach().cpu().tolist(),
                    "cum_kl_spent": cum_kl_spent.detach().cpu().tolist(),
                    "budget_so_far": budget_so_far.detach().cpu().tolist() if isinstance(budget_so_far, torch.Tensor) else [budget_so_far] * B,
                    "budget_remaining": (budget_so_far - cum_kl_spent).detach().cpu().tolist() if isinstance(budget_so_far, torch.Tensor) else [budget_so_far - x for x in cum_kl_spent.detach().cpu().tolist()],
                    "sampled_token_id": sampled_token_ids,
                    "sampled_token": sampled_tokens,
                    "p_star_prob": p_star_prob,
                    "p_s_prob": p_s_prob,
                    "p_risky_prob": p_risky_prob,
                    "lambda": None,
                    "prefix_debt": prefix_debt.detach().cpu().tolist() if self.use_prefix_debt else [0.0] * B,
                    "init_budget_tensor": init_budget_tensor.detach().cpu().tolist() if isinstance(init_budget_tensor, torch.Tensor) else [init_budget_tensor] * B,
                })

            if self.verbose:
                if k_radius not in (0.0, -1.0):
                    print(f"[DEBUG] bc: {bc.squeeze(-1).tolist()}, bd: {bd.squeeze(-1).tolist()}")
                    print(f"[DEBUG] step {step_count}: KL(fused || safe) = {kl_step.detach().cpu().tolist()} (mean={kl_step.mean().item():.4f})")
                    print(f"[DEBUG] step {step_count}: cum_kl_spent={cum_kl_spent.detach().cpu().tolist()}")
                    print(f"[DEBUG] step {step_count}: k_t[0:6]={k_t[:6].detach().cpu().tolist()}")

            if eos_token_id is not None:
                is_eos_token = (next_tokens.unsqueeze(-1) == eos_token_id_tensor).any(dim=-1)
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences * (~is_eos_token).long()

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            if logits_list is not None:
                logits_list.append(next_token_logits.detach().cpu())

            stop = stopping_criteria(input_ids, None)
            unfinished_sequences = unfinished_sequences & (~stop).long()
            this_peer_finished = unfinished_sequences.max() == 0

            if self.verbose:
                elapsed = time.time() - start_time
                total_gen = input_ids.shape[1] - prompt_len
                print(f"Step {step_count + 1}: Generated {total_gen} tokens in {elapsed:.2f} seconds.")

            if pbar is not None:
                pbar.update(1)

            step_count += 1

        if pbar is not None:
            pbar.close()

        if self.verbose:
            total_elapsed = time.time() - start_time
            total_gen = input_ids.shape[1] - prompt_len
            print(f"Generation completed: {total_gen} tokens generated in {total_elapsed:.2f} seconds.")

        del safe_past_key_values, risky_past_key_values
        gc.collect()

        if logits_list is not None:
            logits = tuple([logit.to(input_ids.device) for logit in logits_list])
        else:
            logits = None

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=input_ids, logits=logits)
        return input_ids
