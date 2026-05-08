import gc
import importlib.util
import os
import time
import warnings
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
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
from dotenv import load_dotenv
load_dotenv()


def init_tokenizer(
    model_checkpoint: str, padding_side: str = "left", **kwargs
) -> AutoTokenizer:
    """
    Initializes the tokenizer with special tokens added.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        dtype=kwargs.get("dtype", "auto"),
        trust_remote_code=kwargs.get("trust_remote_code", True),
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _is_bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def _build_quantization_config(
    load_in_4bit: bool,
    load_in_8bit: bool,
    dtype: torch.dtype,
):
    if not (load_in_4bit or load_in_8bit):
        return None
    if not _is_bitsandbytes_available():
        warnings.warn(
            "bitsandbytes is not installed. Ignoring load_in_4bit/load_in_8bit and loading full precision. "
            "Install bitsandbytes to enable quantized model loading."
        )
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        warnings.warn(
            "Transformers does not expose BitsAndBytesConfig. Ignoring quantized loading "
            "and loading full precision."
        )
        return None

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    return None


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
        """
        Class method to initialize the factory directly from model paths or existing models.
        """
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

        # Detect GPU configuration for optimal model placement
        num_gpus = torch.cuda.device_count()
        if verbose:
            print(f"[INFO] AnchoredDecoding: Detected {num_gpus} GPUs")

        if num_gpus >= 2 and safe_model is None and risky_model is None:
            # Best config: put each model on its own GPU for parallel inference
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
                device_map={"": 0},  # Entire model on GPU 0
                **kwargs,
            )
            safe_load = dict(
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=trust_remote_code,
                device_map={"": 1},  # Entire model on GPU 1
                **kwargs,
            )
            if quantization_config is not None:
                risky_load["quantization_config"] = quantization_config
                safe_load["quantization_config"] = quantization_config
        else:
            # Single GPU or pre-loaded models: use provided device_map
            if verbose:
                print(f"[INFO] Using device_map={device_map}")
            quantization_config = _build_quantization_config(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
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

        # Only resize if vocab actually differs
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

        # Pad/EOS ids
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
        """Initialize the AnchoredDecodingFactory with two language models.

        Args:
            safe_model (torch.nn.Module): The safe language model.
            risky_model (torch.nn.Module): The risky language model.
            k_radius (float, optional): The radius for k-NAF guarantee. Defaults to 0.15.
            verbose (bool, optional): If True, prints detailed logs. Defaults to False.
            eps_kl (float, optional): Numerical slack for KL constraint checks. Defaults to 1e-4.
            solver_max_iter (int, optional): Max iterations for the Newton solver. Defaults to 20.
        """
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

        # Device management
        self.device = device or next(self.safe_model.parameters()).device

        # Detect model devices
        self.safe_device = next(self.safe_model.parameters()).device
        self.risky_device = next(self.risky_model.parameters()).device

        # Ensure models are in eval mode
        self.safe_model.eval()
        self.risky_model.eval()

        # KL statistics logging
        self.log_kl_stats = log_kl_stats
        self.kl_stats_history = []

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
        """Generate text sequences using the combined models."""
        if seed is not None:
            from transformers import set_seed

            set_seed(seed)

        k_radius = k_radius if k_radius is not None else self.k_radius

        if generation_config is None:
            generation_config = GenerationConfig(**model_kwargs)

        # Handle text input
        if text is not None:
            if input_ids is not None:
                raise ValueError(
                    "Only one of `text` or `input_ids` should be provided."
                )

            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(
                self.device
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        if input_ids is None:
            raise ValueError("Either `text` or `input_ids` must be provided.")

        # Set defaults from tokenizer if not provided
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Prepare attention masks and special tokens
        attention_mask = self._prepare_attention_mask(
            input_ids, attention_mask, generation_config
        )

        # Input validation
        self._validate_generate_inputs(
            input_ids, generation_config, attention_mask=attention_mask
        )

        # Move input_ids and attention_mask to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        pad_token_id = self._prepare_pad_token_id(generation_config)
        eos_token_id = generation_config.eos_token_id

        # Stopping criteria
        stopping_criteria = self._prepare_stopping_criteria(
            stopping_criteria, generation_config
        )

        # Logits warper
        logits_warper = self._prepare_logits_warper(logits_warper, generation_config)
        logits_processor = self._prepare_logits_processor(
            logits_processor, generation_config
        )

        # Generate outputs
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

    def _validate_generate_inputs(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Validate inputs for the generate method.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            generation_config (GenerationConfig): Configuration for generation.
            attention_mask (Optional[torch.Tensor]): Attention mask.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        # Handle max_new_tokens or max_length
        if (
            hasattr(generation_config, "max_new_tokens")
            and generation_config.max_new_tokens is not None
        ):
            if not (
                isinstance(generation_config.max_new_tokens, int)
                and generation_config.max_new_tokens > 0
            ):
                raise ValueError(
                    "`max_new_tokens` should be a strictly positive integer."
                )
            # Calculate max_length from max_new_tokens and input length
            if attention_mask is not None:
                input_len = attention_mask.sum(dim=-1).max().item()
            else:
                input_len = input_ids.shape[1]
            generation_config.max_length = int(
                input_len + generation_config.max_new_tokens
            )
        elif generation_config.max_length is not None:
            if not (
                isinstance(generation_config.max_length, int)
                and generation_config.max_length > 0
            ):
                raise ValueError("`max_length` should be a strictly positive integer.")
        else:
            raise ValueError(
                "Either `max_length` or `max_new_tokens` must be provided."
            )

        if generation_config.do_sample:
            if generation_config.temperature <= 0:
                raise ValueError(
                    "`temperature` should be positive for sampling decoding."
                )

        if generation_config.num_return_sequences not in (None, 1):
            raise ValueError("Only one generation is supported.")

        if generation_config.num_beams not in (None, 1):
            raise ValueError("Beam search is not supported.")

        # Token ID consistency checks (warnings only - models may have different configs)
        if generation_config.pad_token_id is not None:
            if (
                self.safe_model.config.pad_token_id is not None
                and generation_config.pad_token_id
                != self.safe_model.config.pad_token_id
            ):
                warnings.warn(
                    f"Pad token mismatch with safe model: {generation_config.pad_token_id} vs {self.safe_model.config.pad_token_id}"
                )
            if (
                self.risky_model.config.pad_token_id is not None
                and generation_config.pad_token_id
                != self.risky_model.config.pad_token_id
            ):
                warnings.warn(
                    f"Pad token mismatch with risky model: {generation_config.pad_token_id} vs {self.risky_model.config.pad_token_id}"
                )

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        if input_ids.dim() != 2:
            raise ValueError(
                "Input prompt should be of shape (batch_size, sequence length)."
            )

        if self.safe_model.config.vocab_size != self.risky_model.config.vocab_size:
            raise ValueError("Models must have the same vocabulary.")

    def _prepare_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        """Prepare the attention mask for generation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Existing attention mask.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            torch.Tensor: Prepared attention mask.
        """
        if attention_mask is None:
            if (
                generation_config.pad_token_id is not None
                and (input_ids == generation_config.pad_token_id).any()
            ):
                attention_mask = input_ids.ne(generation_config.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, device=self.device)

        if (
            generation_config.pad_token_id is not None
            and (input_ids[:, -1] == generation_config.pad_token_id).sum() > 0
            and self.verbose
        ):
            print(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )
        return attention_mask

    def _prepare_pad_token_id(
        self,
        generation_config: GenerationConfig,
    ) -> int:
        """Prepare the pad token ID.

        Args:
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            int: Pad token ID.
        """
        # Always prefer EOS token for padding in generation to ensure it is skipped during decoding
        if generation_config.eos_token_id is not None:
            if (
                self.verbose
                and generation_config.pad_token_id != generation_config.eos_token_id
            ):
                print(
                    f"Overriding `pad_token_id` to {generation_config.eos_token_id} "
                    "(`eos_token_id`) to generate safe sequences."
                )
            pad_token_id = generation_config.eos_token_id
        elif generation_config.pad_token_id is not None:
            pad_token_id = generation_config.pad_token_id
        else:
            raise ValueError("Neither `pad_token_id` nor `eos_token_id` is defined.")

        return pad_token_id

    def _prepare_stopping_criteria(
        self,
        stopping_criteria: Optional[StoppingCriteriaList],
        generation_config: GenerationConfig,
    ) -> StoppingCriteriaList:
        """Prepare the stopping criteria for generation.

        Args:
            stopping_criteria (Optional[StoppingCriteriaList]): Existing stopping criteria.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            StoppingCriteriaList: Prepared stopping criteria.
        """
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(
            MaxLengthCriteria(max_length=generation_config.max_length)
        )
        stopping_criteria.append(
            EosTokenCriteria(eos_token_id=generation_config.eos_token_id)
        )
        return stopping_criteria

    def _prepare_logits_processor(
        self,
        logits_processor: Optional[LogitsProcessorList],
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """Prepare the logits processor for generation."""
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        rp = getattr(generation_config, "repetition_penalty", None)
        if rp is not None and rp != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=rp))

        nrng = getattr(generation_config, "no_repeat_ngram_size", None)
        if nrng is not None and nrng > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(nrng))
        return logits_processor

    def _prepare_logits_warper(
        self,
        logits_warper: Optional[LogitsProcessorList],
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """Prepare the logits warper for generation.

        Args:
            logits_warper (Optional[LogitsProcessorList]): Existing logits warper.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            LogitsProcessorList: Prepared logits warper.
        """
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        if generation_config.temperature > 0.0:
            logits_warper.append(TemperatureLogitsWarper(generation_config.temperature))
        return logits_warper

    def _safe_kl_terms(self, log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        # log_p, log_q: [..., V] fp32
        # KL(P || Q) = sum_i p_i * (log p_i - log q_i) >= 0 always
        # Use memory-efficient implementation: p * (log_p - log_q) and handle p=0 via nan_to_num
        return (
            torch.nan_to_num(
                log_p.exp() * (log_p - log_q), nan=0.0, posinf=float("inf"), neginf=0.0
            )
            .sum(dim=-1)
            .clamp(min=0.0)
        )

    @torch.no_grad()
    def forward_direct(
        self, model, input_ids, attention_mask, past_key_values, move_to_device=None
    ):
        """Direct forward pass, bypassing prepare_inputs_for_generation complexity."""
        dev = next(model.parameters()).device
        ids = input_ids.to(dev, non_blocking=True)
        mask = (
            attention_mask.to(dev, non_blocking=True)
            if attention_mask is not None
            else None
        )

        if past_key_values is None:
            # Prefill: process all tokens
            out = model(
                input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True
            )
        else:
            # Incremental: only the new token, but full attention mask
            out = model(
                input_ids=ids[:, -1:],
                attention_mask=mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = out.logits[:, -1, :]
        if move_to_device is not None:
            logits = logits.to(move_to_device)
        pkv = out.past_key_values
        del out
        return logits, pkv

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
        post_hoc_logits_warper: bool = False,  # MUST remain False for KL guarantee
        min_new_tokens: int = 0,
        show_progress: bool = False,
        **model_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        """
        Option B: bank ACTUAL KL spend.
        - Budget grows as (t_gen+1) * k_radius.
        - Spend is cum_kl_spent += KL(q_t || p_c_t) each step (distribution-level).
        - Per-step cap is NOT enforced (no max clamp); a step may spend > k_radius if bank allows.
        """

        if post_hoc_logits_warper:
            raise ValueError(
                "post_hoc_logits_warper=True breaks the per-step KL constraint on the decoding distribution. "
                "Keep post_hoc_logits_warper=False for global K-NAF guarantees."
            )

        logits_list: Optional[List[torch.Tensor]] = (
            [] if (return_dict_in_generate and output_logits) else None
        )

        # Clear KL stats for this generation
        if self.log_kl_stats:
            self.kl_stats_history = []

        # EOS handling - cache tensor for loop efficiency
        if isinstance(eos_token_id, int):
            eos_token_id_list = [eos_token_id]
        else:
            eos_token_id_list = list(eos_token_id)
        eos_token_id_tensor = torch.tensor(eos_token_id_list, device=self.device)

        batch_size, prompt_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = input_ids.new(batch_size).fill_(1)

        # KV cache config - let HF manage cache_position internally
        model_kwargs["use_cache"] = True
        safe_past_key_values = None
        risky_past_key_values = None

        # optional parallel CUDA streams
        if parallelize:
            # Use separate streams on the respective devices for true parallelism
            stream1 = torch.cuda.Stream(device=self.safe_device)
            stream2 = torch.cuda.Stream(device=self.risky_device)

        step_count = 0
        start_time = time.time()
        if self.verbose:
            print(f"Starting generation with prompt length {prompt_len} tokens.")

        # Banked KL spend (fp32)
        cum_kl_spent = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        # Numerical slack
        eps_kl = self.eps_kl

        if self.use_prefix_debt:
            assert (
                self.prefix_n is not None
            ), "prefix_n must be set when use_prefix_debt is True"

            # 1. Run Forward Passes
            labels = input_ids[:, 1:].unsqueeze(-1)  # [B, L-1, 1]

            if parallelize and self.safe_device != self.risky_device:
                # Parallel prefill for better TTFT since they are on different GPUs
                with torch.cuda.stream(stream1):
                    c_all_out = self.safe_model(
                        input_ids=input_ids.to(self.safe_device),
                        attention_mask=attention_mask.to(self.safe_device),
                        use_cache=True,
                    )
                with torch.cuda.stream(stream2):
                    d_all_out = self.risky_model(
                        input_ids=input_ids.to(self.risky_device),
                        attention_mask=attention_mask.to(self.risky_device),
                        use_cache=True,
                    )
                torch.cuda.synchronize(self.safe_device)
                torch.cuda.synchronize(self.risky_device)

                # Extract safe Components
                safe_past_key_values = c_all_out.past_key_values
                safe_logits = c_all_out.logits[:, -1, :].to(self.device)
                c_logits_prefix = c_all_out.logits[:, :-1, :]
                c_lp = (
                    c_logits_prefix.gather(-1, labels.to(c_logits_prefix.device))
                    .squeeze(-1)
                    .float()
                    - c_logits_prefix.logsumexp(dim=-1).float()
                ).to(self.device)
                del c_all_out, c_logits_prefix

                # Extract risky Components
                risky_past_key_values = d_all_out.past_key_values
                risky_logits = d_all_out.logits[:, -1, :].to(self.device)
                d_logits_prefix = d_all_out.logits[:, :-1, :]
                d_lp = (
                    d_logits_prefix.gather(-1, labels.to(d_logits_prefix.device))
                    .squeeze(-1)
                    .float()
                    - d_logits_prefix.logsumexp(dim=-1).float()
                ).to(self.device)
                del d_all_out, d_logits_prefix

            else:
                # Sequential prefill - maximize memory efficiency by clearing safe logits before risky prefill
                c_all_out = self.safe_model(
                    input_ids=input_ids.to(self.safe_device),
                    attention_mask=attention_mask.to(self.safe_device),
                    use_cache=True,
                )

                safe_past_key_values = c_all_out.past_key_values
                safe_logits = c_all_out.logits[:, -1, :].to(self.device)
                c_logits_prefix = c_all_out.logits[:, :-1, :]
                c_lp = (
                    c_logits_prefix.gather(-1, labels.to(c_logits_prefix.device))
                    .squeeze(-1)
                    .float()
                    - c_logits_prefix.logsumexp(dim=-1).float()
                ).to(self.device)

                del c_all_out, c_logits_prefix

                d_all_out = self.risky_model(
                    input_ids=input_ids.to(self.risky_device),
                    attention_mask=attention_mask.to(self.risky_device),
                    use_cache=True,
                )

                risky_past_key_values = d_all_out.past_key_values
                risky_logits = d_all_out.logits[:, -1, :].to(self.device)
                d_logits_prefix = d_all_out.logits[:, :-1, :]
                d_lp = (
                    d_logits_prefix.gather(-1, labels.to(d_logits_prefix.device))
                    .squeeze(-1)
                    .float()
                    - d_logits_prefix.logsumexp(dim=-1).float()
                ).to(self.device)

                del d_all_out, d_logits_prefix

            # 4. Compute Debt
            prefix_debt = self._compute_prefix_debt_fast(
                c_lp, d_lp, input_ids, attention_mask, self.prefix_n
            )

            init_budget_tensor = -prefix_debt.to(torch.float32)
            if self.verbose:
                print(f"[INFO] Using prefix debt True with prefix_n={self.prefix_n}")
                print(f"[INFO] Prefix debt: {prefix_debt.tolist()}")
        else:
            # Standard path - direct prefill (no cache yet)
            if parallelize and self.safe_device != self.risky_device:
                with torch.cuda.stream(stream1):
                    safe_logits, safe_past_key_values = self.forward_direct(
                        self.safe_model, input_ids, attention_mask, None
                    )
                with torch.cuda.stream(stream2):
                    risky_logits, risky_past_key_values = self.forward_direct(
                        self.risky_model, input_ids, attention_mask, None
                    )
                torch.cuda.synchronize(self.safe_device)
                torch.cuda.synchronize(self.risky_device)
            else:
                safe_logits, safe_past_key_values = self.forward_direct(
                    self.safe_model, input_ids, attention_mask, None
                )
                risky_logits, risky_past_key_values = self.forward_direct(
                    self.risky_model, input_ids, attention_mask, None
                )

            # No prefix debt - initialize budget tensor to zero
            init_budget_tensor = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )

        use_precomputed_logits = True

        # Progress bar setup
        max_new_tokens = (
            getattr(stopping_criteria[0], "max_length", prompt_len + 100) - prompt_len
        )

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
                # Direct forward pass - no prepare_inputs_for_generation complexity
                if no_kv_cache:
                    if self.verbose:
                        print("Using no KV cache")
                    safe_pkv_in = None
                    risky_pkv_in = None
                else:
                    safe_pkv_in = safe_past_key_values
                    risky_pkv_in = risky_past_key_values

                # Forward passes
                if parallelize and self.safe_device != self.risky_device:
                    # Launch both forwards in parallel streams (only if on different devices)
                    with torch.cuda.stream(stream1):
                        safe_logits, safe_past_key_values = self.forward_direct(
                            self.safe_model, input_ids, attention_mask, safe_pkv_in
                        )
                    with torch.cuda.stream(stream2):
                        risky_logits, risky_past_key_values = self.forward_direct(
                            self.risky_model, input_ids, attention_mask, risky_pkv_in
                        )
                    torch.cuda.synchronize(self.safe_device)
                    torch.cuda.synchronize(self.risky_device)
                else:
                    safe_logits, safe_past_key_values = self.forward_direct(
                        self.safe_model, input_ids, attention_mask, safe_pkv_in
                    )
                    risky_logits, risky_past_key_values = self.forward_direct(
                        self.risky_model, input_ids, attention_mask, risky_pkv_in
                    )

                # If no_kv_cache, don't keep the cache
                if no_kv_cache:
                    safe_past_key_values = None
                    risky_past_key_values = None
            else:
                # Reuse precomputed logits from beginning, set to false
                use_precomputed_logits = False

            # Move logits to same device for fusion (risky model is typically larger/primary)
            if safe_logits.device != risky_logits.device:
                safe_logits = safe_logits.to(risky_logits.device)

            # Apply logits processors BEFORE solve (no clone needed - processors return new tensors)
            safe_logits = logits_processor(input_ids, safe_logits)
            risky_logits = logits_processor(input_ids, risky_logits)

            # Apply warpers (temperature is OK) BEFORE solve
            if logits_warper is not None and len(logits_warper) > 0:
                safe_logits = logits_warper(input_ids, safe_logits)
                risky_logits = logits_warper(input_ids, risky_logits)

            # Enforce min_new_tokens by banning EOS BEFORE solve
            generated_tokens = input_ids.shape[1] - prompt_len
            apply_min_tokens = (
                min_new_tokens is not None
                and min_new_tokens > 0
                and generated_tokens < min_new_tokens
                and eos_token_id is not None
            )
            if apply_min_tokens:
                for eid in eos_token_id_list:
                    safe_logits[:, eid] = -float("inf")
                    risky_logits[:, eid] = -float("inf")

            B = safe_logits.size(0)
            device = safe_logits.device
            dtype = safe_logits.dtype

            # t_gen = number of new tokens already generated so far (before selecting next)
            t_gen = int(input_ids.shape[1] - prompt_len)

            # Determine per-example k_t (banked KL)
            if k_radius == 0.0:
                bc = torch.ones((B, 1), device=device, dtype=dtype)
                bd = torch.zeros((B, 1), device=device, dtype=dtype)
                k_t = torch.zeros((B,), device=device, dtype=torch.float32)
                budget_so_far = torch.zeros(
                    (B,), device=device, dtype=torch.float32
                )  # no budget in safe-only mode
                # Compute log probs once for edge case
                log_pc = F.log_softmax(safe_logits.float(), dim=-1)
                log_pd = F.log_softmax(risky_logits.float(), dim=-1)

            elif k_radius == -1.0:
                # risky-only (NO guarantee)
                bc = torch.zeros((B, 1), device=device, dtype=dtype)
                bd = torch.ones((B, 1), device=device, dtype=dtype)
                k_t = torch.full((B,), float("inf"), device=device, dtype=torch.float32)
                budget_so_far = torch.full(
                    (B,), float("inf"), device=device, dtype=torch.float32
                )  # infinite budget in risky-only mode
                # Compute log probs once for edge case
                log_pc = F.log_softmax(safe_logits.float(), dim=-1)
                log_pd = F.log_softmax(risky_logits.float(), dim=-1)

            else:
                # Budget accrued so far: (t_gen+1)*k + init_budget
                # budget_so_far = (float(t_gen + 1) * float(k_radius)) + init_budget_tensor  # scalar float
                # budget_so_far = torch.full((B,), budget_so_far, device=device, dtype=torch.float32)
                budget_so_far = (
                    float(t_gen + 1) * float(k_radius)
                ) + init_budget_tensor

                remaining = (budget_so_far - cum_kl_spent).clamp(min=0.0)  # [B] fp32

                # Only apply to unfinished sequences
                k_t = remaining * unfinished_sequences.float()  # [B] fp32

                # Solver returns bc, bd AND cached log_pc, log_pd (avoids redundant log_softmax)
                bc, bd, log_pc, log_pd = self.solve_optimization_newton(
                    safe_logits, risky_logits, k_t
                )

            # Fused distribution in decoding space (reuses cached log_pc, log_pd)
            log_p, log_pc, next_token_logits = self._get_logp_from_weights(
                bc, bd, log_pc, log_pd
            )
            log_pc_realized = log_pc

            # Local KL check + BANK the spend (do this BEFORE logging so cum_kl_spent is up-to-date)
            kl_step = torch.zeros(
                (B,), device=device, dtype=torch.float32
            )  # default for edge cases
            if k_radius not in (0.0, -1.0):
                kl_step = self._safe_kl_terms(log_p, log_pc_realized).float()  # [B]

                mask = unfinished_sequences.bool()
                # Solver should enforce KL <= k_t; allow tiny numerical slack.
                # Note: Due to numerical precision in Newton solver, small violations can occur.
                violation = kl_step[mask] - k_t[mask]
                max_violation = violation.max().item() if mask.any() else 0.0
                if max_violation > eps_kl:
                    warnings.warn(
                        f"KL constraint exceeded by {max_violation:.6f} (eps={eps_kl}). "
                        f"max(KL)={kl_step[mask].max().item():.6f}, max(k_t)={k_t[mask].max().item():.6f}",
                        RuntimeWarning,
                    )

                # Bank the KL spend (distribution-level)
                cum_kl_spent = cum_kl_spent + kl_step * unfinished_sequences.float()

            # Log KL statistics if enabled (AFTER updating cum_kl_spent so it includes current step)
            if self.log_kl_stats:
                # log_pd is already computed above - no redundant log_softmax needed
                kl_to_safe = self._safe_kl_terms(
                    log_p, log_pc
                ).float()  # KL(p* || p_c) [B]
                kl_to_risky = self._safe_kl_terms(
                    log_p, log_pd
                ).float()  # KL(p* || p_d) [B]


                if do_sample:
                    probs = log_p.exp()
                    next_tokens = torch.multinomial(probs, 1).squeeze(1)
                else:
                    probs = log_p.exp()
                    next_tokens = torch.argmax(log_p, dim=-1)

                sampled_token_ids = next_tokens.detach().cpu().tolist()
                sampled_tokens = [
                    self.tokenizer.decode([tok_id], skip_special_tokens=False)
                    for tok_id in sampled_token_ids
                ]

                p_star_prob = probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                p_s_prob = log_pc.exp().gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                p_risky_prob = log_pd.exp().gather(1, next_tokens.unsqueeze(1)).squeeze(1).detach().cpu().tolist()

                self.kl_stats_history.append({
                                                    "step": step_count,
                                                    "kl_to_safe": kl_to_safe.detach().cpu().tolist(),
                                                    "kl_to_risky": kl_to_risky.detach().cpu().tolist(),
                                                    "bc": (
                                                        bc.squeeze(-1).detach().cpu().tolist()
                                                        if bc.dim() > 1
                                                        else bc.detach().cpu().tolist()
                                                    ),
                                                    "bd": (
                                                        bd.squeeze(-1).detach().cpu().tolist()
                                                        if bd.dim() > 1
                                                        else bd.detach().cpu().tolist()
                                                    ),
                                                    "k_t": k_t.detach().cpu().tolist(),
                                                    "cum_kl_spent": cum_kl_spent.detach().cpu().tolist(),
                                                    "budget_so_far": (
                                                        budget_so_far.detach().cpu().tolist()
                                                        if isinstance(budget_so_far, torch.Tensor)
                                                        else [budget_so_far] * B
                                                    ),
                                                    "budget_remaining": (
                                                        (budget_so_far - cum_kl_spent).detach().cpu().tolist()
                                                        if isinstance(budget_so_far, torch.Tensor)
                                                        else [budget_so_far - x for x in cum_kl_spent.detach().cpu().tolist()]
                                                    ),
                                                    "sampled_token_id": sampled_token_ids,
                                                    "sampled_token": sampled_tokens,
                                                    "p_star_prob": p_star_prob,
                                                    "p_s_prob": p_s_prob,
                                                    "p_risky_prob": p_risky_prob,
                                                    "lambda": None,
                                                }
                                            )

            if self.verbose:
                if k_radius not in (0.0, -1.0):
                    print(
                        f"[DEBUG] bc: {bc.squeeze(-1).tolist()}, bd: {bd.squeeze(-1).tolist()}"
                    )
                    print(
                        f"[DEBUG] step {step_count}: KL(fused || safe) = {kl_step.detach().cpu().tolist()} "
                        f"(mean={kl_step.mean().item():.4f})"
                    )
                    print(
                        f"[DEBUG] step {step_count}: cum_kl_spent={cum_kl_spent.detach().cpu().tolist()}"
                    )
                    print(
                        f"[DEBUG] step {step_count}: k_t[0:6]={k_t[:6].detach().cpu().tolist()}"
                    )

            # EOS handling + unfinished mask update
            if eos_token_id is not None:
                is_eos_token = (next_tokens.unsqueeze(-1) == eos_token_id_tensor).any(
                    dim=-1
                )

                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
                unfinished_sequences = unfinished_sequences * (~is_eos_token).long()

            # Append token
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

            if logits_list is not None:
                logits_list.append(next_token_logits.detach().cpu())

            # Stopping
            stop = stopping_criteria(input_ids, None)
            unfinished_sequences = unfinished_sequences & (~stop).long()
            this_peer_finished = unfinished_sequences.max() == 0

            if self.verbose:
                elapsed = time.time() - start_time
                total_gen = input_ids.shape[1] - prompt_len
                print(
                    f"Step {step_count + 1}: Generated {total_gen} tokens in {elapsed:.2f} seconds."
                )

            if pbar is not None:
                pbar.update(1)

            step_count += 1

        if pbar is not None:
            pbar.close()

        if self.verbose:
            total_elapsed = time.time() - start_time
            total_gen = input_ids.shape[1] - prompt_len
            print(
                f"Generation completed: {total_gen} tokens generated in {total_elapsed:.2f} seconds."
            )

        del safe_past_key_values, risky_past_key_values
        gc.collect()

        if logits_list is not None:
            logits = tuple([logit.to(input_ids.device) for logit in logits_list])
        else:
            logits = None

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=input_ids, logits=logits)
        return input_ids

    @torch.no_grad()
    def _model_forward_all_logits(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass that returns ALL logits (not just last token) plus KV cache.
        Used for initial prefill when we need logits for prefix debt calculation.

        NOTE: Calls model DIRECTLY (like get_prefix_debt True) without prepare_inputs_for_generation
        to ensure prefix debt computation matches exactly.
        """
        # Handle device placement (same as get_prefix_debt True)
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = self.device

        if input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        # Call model directly - MUST match get_prefix_debt True for identical prefix debt
        # Key difference: use_cache=True to get KV cache for subsequent generation
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,  # Need KV cache for generation
            return_dict=True,
        )

        # Return ALL logits [B, L, V] and KV cache
        all_logits = outputs.logits  # [B, L, V]
        past_key_values = outputs.past_key_values
        del outputs
        return all_logits, past_key_values

    @torch.no_grad()
    def _compute_prefix_debt_fast(
        self,
        safe_logp_target: torch.Tensor,  # [B, L-1] - pre-gathered log-probs
        risky_logp_target: torch.Tensor,  # [B, L-1] - pre-gathered log-probs
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Lightweight prefix debt calculation using pre-gathered log-probabilities.
        Peak memory complexity: O(B * L) instead of O(B * L * V).
        """
        # 1. Compute Log-Likelihood Ratio directly
        llr = risky_logp_target - safe_logp_target  # [B, L-1]

        # 2. Setup mask for valid positions
        m = (
            attention_mask[:, 1:].bool()
            if attention_mask is not None
            else torch.ones_like(llr, dtype=torch.bool)
        )

        # 3. Mask special tokens (EOS, BOS, etc.)
        next_tok = input_ids[:, 1:]
        special_ids = torch.tensor(
            list(self.tokenizer.all_special_ids), device=self.device
        )
        # Vectorized special token check: [B, L-1, 1] == [S] -> [B, L-1, S] -> any(dim=-1)
        is_special = (next_tok.unsqueeze(-1) == special_ids).any(dim=-1)

        valid = m & (~is_special)

        # Fill invalid positions with a very low value so they aren't picked by top-k
        llr = llr.masked_fill(~valid, -1e9)

        # 4. Top-k LLR values
        k_eff = min(int(k), llr.size(1))
        vals, _ = llr.topk(k_eff, dim=-1, largest=True)

        # 5. Return mean of positive LLRs
        return vals.clamp(min=0.0).mean(dim=-1)

    @torch.no_grad()
    def _compute_prefix_debt_from_logits(
        self,
        safe_logits: torch.Tensor,  # [B, L-1, V] - predictions for tokens at positions 1 to L-1
        risky_logits: torch.Tensor,  # [B, L-1, V]
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Compute prefix debt from pre-computed logits (avoids extra forward pass).

        Args:
            safe_logits: Logits from safe model for positions 0 to L-2 (predicting tokens 1 to L-1)
            risky_logits: Logits from risky model for positions 0 to L-2
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            k: Number of top tokens to consider for debt

        Returns:
            Tensor [B] with prefix debt for each sequence
        """
        # Convert to log probabilities
        logp_c = F.log_softmax(safe_logits.float(), dim=-1).to(
            self.device
        )  # [B, L-1, V]
        logp_d = F.log_softmax(risky_logits.float(), dim=-1).to(
            self.device
        )  # [B, L-1, V]

        # Target tokens: positions 1 to L-1 (the tokens being predicted)
        next_tok = input_ids[:, 1:].unsqueeze(-1).to(self.device)  # [B, L-1, 1]

        if next_tok.max() >= logp_c.size(-1):
            raise ValueError(
                f"Token ID {next_tok.max()} exceeds vocab size {logp_c.size(-1)}"
            )

        # Log-likelihood ratio: log p_d(token) - log p_c(token)
        llr = (logp_d.gather(-1, next_tok) - logp_c.gather(-1, next_tok)).squeeze(
            -1
        )  # [B, L-1]

        # Mask for valid positions (attended and non-special tokens)
        m = (
            attention_mask[:, 1:].bool().to(self.device)
            if attention_mask is not None
            else torch.ones_like(llr, dtype=torch.bool)
        )

        special_ids = set(self.tokenizer.all_special_ids)
        is_special = torch.zeros_like(m)
        for tid in special_ids:
            is_special |= next_tok.squeeze(-1) == tid

        valid = m & (~is_special)
        llr = llr.masked_fill(~valid, -1e9)

        # Top-k LLR values (positions where risky is most preferred over safe)
        k_eff = min(int(k), llr.size(1))
        vals, _ = llr.topk(k_eff, dim=-1, largest=True)

        # Clamp to 0 so "good" tokens (where safe is better) don't reduce debt
        return vals.clamp(min=0.0).mean(dim=-1)

    def solve_optimization_newton(
        self,
        safe_logits: torch.Tensor,
        risky_logits: torch.Tensor,
        k_radius,  # float or Tensor [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (bc, bd, log_pc, log_pd) - reuse log_pc/log_pd to avoid redundant computation."""
        bc, bd, log_pc, log_pd = self._solve_theta_newton(
            safe_logits, risky_logits, k_radius, max_iter=self.solver_max_iter
        )
        # If bc/bd are bf16, abs(...) < 1e-6 can fail due to precision; use fp32 in the check.
        assert torch.allclose(
            (bc + bd).float(), torch.ones_like((bc + bd).float()), atol=1e-5, rtol=0.0
        )
        return bc, bd, log_pc, log_pd

    def _get_logp_from_weights(
        self,
        bc: torch.Tensor,  # [B,1] or [B]
        bd: torch.Tensor,  # [B,1] or [B]
        log_pc: torch.Tensor,  # [B,V] pre-computed log_softmax of safe logits
        log_pd: torch.Tensor,  # [B,V] pre-computed log_softmax of risky logits
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused distribution from pre-computed log probabilities."""
        # 3) Geometric blend in log-prob space, then renormalize
        if bc.dim() == 2 and bc.size(1) == 1:
            bc = bc.squeeze(1)
        if bd.dim() == 2 and bd.size(1) == 1:
            bd = bd.squeeze(1)

        # Compute un-normalized log distribution / next token logits
        # next_token_logits = bd.unsqueeze(-1) * log_pd + bc.unsqueeze(-1) * log_pc   # [B,V]

        term_d = bd.unsqueeze(-1) * log_pd
        term_c = bc.unsqueeze(-1) * log_pc

        # Handle 0 * -inf = NaN
        term_d = torch.nan_to_num(term_d, nan=0.0)
        term_c = torch.nan_to_num(term_c, nan=0.0)

        next_token_logits = term_d + term_c

        # Normalize to a proper log prob distribution (fused)
        log_p = F.log_softmax(next_token_logits, dim=-1)  # [B,V]

        return log_p, log_pc, next_token_logits

    @torch.no_grad()
    def _solve_theta_newton(
        self,
        safe_logits: torch.Tensor,  # [B,V]
        risky_logits: torch.Tensor,  # [B,V]
        k_radius,  # float or Tensor [B]
        max_iter: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve for theta = w_d in q_theta ∝ p_c^{1-theta} p_d^{theta} s.t. KL(q_theta || p_c) <= k_t.

        Returns:
            (w_c, w_d, log_pc, log_pd) - weights and cached log-probs to avoid redundant computation.

        Guarantee-critical properties:
        - All computations in fp32.
        - Robust to masked tokens (-inf) and -inf - -inf -> NaN.
        - Returns a theta that is *numerically feasible* via a final bisection projection.
        """
        device = safe_logits.device
        B, V = safe_logits.shape

        # fp32 log-probs
        log_pd = F.log_softmax(risky_logits.float(), dim=-1)  # [B,V] fp32
        log_pc = F.log_softmax(safe_logits.float(), dim=-1)  # [B,V] fp32

        # k_t as fp32 [B]
        k_t = torch.as_tensor(k_radius, device=device, dtype=torch.float32)
        if k_t.ndim == 0:
            k_t = k_t.expand(B)
        else:
            k_t = k_t.view(-1)
            assert k_t.numel() == B, f"k_t must be scalar or shape [B], got {k_t.shape}"

        # Corners
        mask_force_pc = k_t <= 0.0

        # KL(p_d || p_c) robust (fp32)
        KL_pd_pc = self._safe_kl_terms(log_pd, log_pc)  # [B]

        mask_use_pd = (KL_pd_pc <= k_t) & (~mask_force_pc)

        active = ~(mask_force_pc | mask_use_pd)

        # Output weights (fp32)
        w_c = torch.empty((B, 1), device=device, dtype=torch.float32)
        w_d = torch.empty((B, 1), device=device, dtype=torch.float32)

        w_c[mask_force_pc] = 1.0
        w_d[mask_force_pc] = 0.0
        w_c[mask_use_pd] = 0.0
        w_d[mask_use_pd] = 1.0

        if not active.any():
            return w_c, w_d, log_pc, log_pd  # fp32, plus cached log probs

        # Active subset
        log_pc_a = log_pc[active]  # [Ba,V] fp32
        log_pd_a = log_pd[active]  # [Ba,V] fp32
        k_a = k_t[active]  # [Ba] fp32
        Ba = log_pc_a.size(0)

        # a = log p_d - log p_c ; sanitize to avoid NaNs/Infs from masking
        a = log_pd_a - log_pc_a
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)  # [Ba,V]

        # Helper: robust KL(q_theta || p_c)
        def kl_theta(th: torch.Tensor) -> torch.Tensor:
            q_log_unnorm = log_pc_a + th[:, None] * a
            logZ = torch.logsumexp(q_log_unnorm, dim=-1)  # [Ba]
            log_q = q_log_unnorm - logZ[:, None]  # [Ba,V]
            return self._safe_kl_terms(log_q, log_pc_a)  # [Ba]

        # Bracket in theta-space
        lo = torch.zeros(Ba, device=device, dtype=torch.float32)
        hi = torch.ones(Ba, device=device, dtype=torch.float32)

        # Initialize theta away from endpoints
        theta = torch.clamp(k_a / (k_a + 1.0), 1e-4, 1.0 - 1e-4)

        eps = 1e-9

        for _ in range(max_iter):
            # log q_theta ∝ log_pc + theta*a
            # Reuse tensor memory to reduce peak allocation
            q = log_pc_a + theta[:, None] * a
            logZ = torch.logsumexp(q, dim=-1)  # [Ba]
            q.sub_(logZ[:, None])  # [Ba,V] in-place
            q.exp_()  # [Ba,V] in-place

            # Moments under q (fp32)
            mean_a = (q * a).sum(dim=-1)  # [Ba]
            mean_a2 = (q * (a * a)).sum(dim=-1)  # [Ba]
            var_a = (mean_a2 - mean_a * mean_a).clamp_min(0.0)  # [Ba]

            # KL(q||pc) = theta*E[a] - logZ
            KL = theta * mean_a - logZ  # [Ba]
            # If KL has NaNs (shouldn't, but just in case), treat as infeasible
            KL = torch.nan_to_num(KL, nan=float("inf"), posinf=float("inf"), neginf=0.0)

            f = KL - k_a  # [Ba]

            # Update bracket: f<=0 is feasible
            hi = torch.where(f > 0, theta, hi)
            lo = torch.where(f <= 0, theta, lo)

            # Newton step: f' = theta * Var_q[a]
            fp = (theta * var_a).clamp_min(eps)
            theta_new = theta - f / fp

            # Safeguard: stay inside (lo, hi), else bisect
            bad = (theta_new <= lo) | (theta_new >= hi) | ~torch.isfinite(theta_new)
            theta = torch.where(bad, 0.5 * (lo + hi), theta_new)

            if (hi - lo).max() < 1e-6:
                break

        # --- Final feasibility projection (guarantee-critical) ---
        # Ensure returned theta is numerically feasible under kl_theta
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            KL_mid = kl_theta(mid)
            feas = KL_mid <= k_a
            lo = torch.where(feas, mid, lo)
            hi = torch.where(feas, hi, mid)

        theta = lo  # feasible by construction (KL <= k_a)

        wd = theta[:, None]  # [Ba,1]
        wc = 1.0 - wd

        w_c[active] = wc
        w_d[active] = wd

        return w_c, w_d, log_pc, log_pd  # fp32, plus cached log probs

    def get_kl_stats_summary(self) -> dict:
        """Get a summary of KL statistics from the last generation.

        Returns:
            dict with keys:
                - 'per_step': list of per-step stats (includes k_t, cum_kl_spent, budget_so_far)
                - 'mean_kl_to_safe': mean KL(p* || p_c) across all steps
                - 'mean_kl_to_risky': mean KL(p* || p_d) across all steps
                - 'total_kl_to_safe_per_seq': sum of KL(p* || p_c) across all steps per sequence (realized k)
                - 'total_kl_to_risky_per_seq': sum of KL(p* || p_d) across all steps per sequence
                - 'final_cum_kl_spent_per_seq': final cumulative KL spent per sequence
                - 'final_budget_per_seq': final total budget per sequence
                - 'budget_utilization_per_seq': percentage of budget used per sequence
        """
        if not self.kl_stats_history:
            return {
                "per_step": [],
                "mean_kl_to_safe": 0.0,
                "mean_kl_to_risky": 0.0,
                "total_kl_to_safe": 0.0,
                "total_kl_to_risky": 0.0,
            }

        # Aggregate across steps (take mean across batch for each step, then aggregate)
        all_kl_safe = []
        all_kl_risky = []
        for step_data in self.kl_stats_history:
            all_kl_safe.extend(step_data["kl_to_safe"])
            all_kl_risky.extend(step_data["kl_to_risky"])

        # Per-sequence totals (sum across steps for each batch element)
        n_steps = len(self.kl_stats_history)
        batch_size = len(self.kl_stats_history[0]["kl_to_safe"]) if n_steps > 0 else 0

        total_kl_safe_per_seq = [0.0] * batch_size
        total_kl_risky_per_seq = [0.0] * batch_size
        for step_data in self.kl_stats_history:
            for i, (kc, kd) in enumerate(
                zip(step_data["kl_to_safe"], step_data["kl_to_risky"])
            ):
                total_kl_safe_per_seq[i] += kc
                total_kl_risky_per_seq[i] += kd

        # Get final budget tracking stats from the last step
        last_step = self.kl_stats_history[-1]
        final_cum_kl_spent = last_step.get("cum_kl_spent", [0.0] * batch_size)
        final_budget = last_step.get("budget_so_far", [0.0] * batch_size)

        # Compute budget utilization (handle inf budget case)
        budget_utilization = []
        for spent, budget in zip(final_cum_kl_spent, final_budget):
            if budget == float("inf") or budget == 0:
                budget_utilization.append(0.0)  # risky-only or safe-only mode
            else:
                budget_utilization.append((spent / budget) * 100.0)

        return {
            "per_step": self.kl_stats_history,
            "mean_kl_to_safe": np.mean(all_kl_safe) if all_kl_safe else 0.0,
            "mean_kl_to_risky": np.mean(all_kl_risky) if all_kl_risky else 0.0,
            "total_kl_to_safe_per_seq": total_kl_safe_per_seq,  # realized k per sequence
            "total_kl_to_risky_per_seq": total_kl_risky_per_seq,
            "mean_total_kl_to_safe": (
                np.mean(total_kl_safe_per_seq) if total_kl_safe_per_seq else 0.0
            ),
            "mean_total_kl_to_risky": (
                np.mean(total_kl_risky_per_seq) if total_kl_risky_per_seq else 0.0
            ),
            "final_cum_kl_spent_per_seq": final_cum_kl_spent,
            "final_budget_per_seq": final_budget,
            "budget_utilization_per_seq": budget_utilization,  # percentage of budget used
            "mean_budget_utilization": (
                np.mean(budget_utilization) if budget_utilization else 0.0
            ),
        }


def generate(
    input_ids: Optional[torch.Tensor] = None,
    safe_model: Optional[torch.nn.Module] = None,
    risky_model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    k_radius: float = 0.15,
    generation_config: Optional[GenerationConfig] = None,
    text: Optional[str | List[str]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> GenerateDecoderOnlyOutput:
    """
    Convenience function to generate text using AnchoredDecodingFactory.
    """
    # Separate factory-specific arguments from generation-specific ones
    factory_args = {
        "use_prefix_debt",
        "prefix_n",
        "log_kl_stats",
        "verbose",
        "device",
        "eps_kl",
        "solver_max_iter",
        "dtype",
        "device_map",
        "trust_remote_code",
        "max_memory",
        "seed",
    }

    factory_kwargs = {k: v for k, v in kwargs.items() if k in factory_args}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in factory_args}

    factory = AnchoredDecodingFactory(
        safe_model=safe_model,
        risky_model=risky_model,
        tokenizer=tokenizer,
        k_radius=k_radius,
        **factory_kwargs,
    )

    if generation_config is None:
        generation_config = GenerationConfig(**gen_kwargs)

    return factory.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        text=text,
        seed=seed,
        **gen_kwargs,
    )



