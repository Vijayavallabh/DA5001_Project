from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class E2Config:
    data_dir: str = "data"
    output_dir: str = "output/e2_outputs"
    adaptive_eval: bool = True
    adaptive_eval_min_traj: int = 4
    adaptive_eval_topup_fraction: float = 0.5
    safe_model_path: str = "jacquelinehe/tinycomma-1.8b-llama3-tokenizer"
    risky_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    eval_batch_size: int = 8
    length_bucket_width: int = 32
    device: str = "cuda"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    parallelize: bool = True
    verbose: bool = False

    k: float = 3.0
    max_new_tokens: int = 200
    use_chat_template: bool = False
    temperature: float = 1.0
    prefix_n: int = 5
    use_prefix_debt: bool = True

    delta_screen: float = 0.0033
    delta_final: float = 0.0033
    delta_heldout: float = 0.0033
    delta_stress: float = 0.0033

    init_attack: int = 48
    init_factual: int = 24
    init_creative: int = 24

    init_traj: int = 12
    med_fid_traj: int = 10
    topup_traj: int = 16
    final_traj: int = 20
    heldout_traj: int = 20
    stress_traj: int = 30

    generations: int = 4
    calls_per_generation: int = 8
    candidates_per_call: int = 8
    crossover_calls_per_generation: int = 3
    crossover_candidates_per_call: int = 4

    prescreen_keep: int = 48
    med_fid_keep: int = 24
    topup_keep: int = 6
    archive_keep: int = 24

    ablation_random: int = 0
    ablation_no_surrogate: int = 4

    final_keep: int = 6
    heldout_keep: int = 8
    stress_keep: int = 4

    min_prompt_tokens: int = 20
    max_prompt_tokens: int = 250

    seeds: Tuple[int, ...] = (42, 43, 44)
    factscore_field: str = "factscore_prompt"

    sentence_model_name: str = "sentence-transformers/sentence-t5-base"
    tfidf_features: int = 3000
    surrogate_lr: float = 3e-4
    surrogate_epochs: int = 80
    surrogate_patience: int = 10
    surrogate_batch_size: int = 32
    replay_fraction: float = 0.4
    violator_weight: float = 4.0
    surrogate_device: str = "cuda:0"

    optimizer_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    optimizer_device: str = "cuda:1"
    optimizer_dtype: str = "bfloat16"
    optimizer_temperature: float = 1.0
    optimizer_top_p: float = 0.98
    optimizer_max_tokens: int = 768
    optimizer_retries: int = 2
    optimizer_max_input_tokens: int = 3072

    @property
    def K(self) -> float:
        return self.k * self.max_new_tokens


@dataclass
class Candidate:
    candidate_id: str
    generation: int
    prompt_text: str
    rationale: str
    novelty_tag: str
    expected_rho: float
    source: str
    parent_ids: List[str] = field(default_factory=list)
    parent_lineage_ids: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    candidate_id: str
    lineage_id: str
    generation: int
    source: str
    domain: str
    split: str
    prompt_text: str
    N: int
    spends: List[float]
    final_budgets: List[float]
    delta_inits: List[float]
    mean_spend: float
    var_spend: float
    max_spend: float  # largest per-trajectory spend Z_j (the Bernstein proxy U_EBB is retired, feat-007)
    rho: float  # largest per-trajectory utilisation Z_j / B_j (0 if no trajectory accrued budget)
    certified: bool  # every trajectory satisfies Z_j <= max(0, B_j) + 1e-3
    delta_init_mean: float
    effective_budget_min: float
    final_budget_mean: float
    parent_ids: List[str]
    parent_lineage_ids: List[str]
    timestamp: str
    utilisations: List[float] = field(default_factory=list)  # Z / max(0, B) per trajectory (None if B <= 0)
    activity: List[List[int]] = field(default_factory=list)  # [steps theta==0, 0<theta<1, theta==1] per trajectory

    @property
    def U_EBB(self) -> float:  # backwards-compatible name used by the surrogate; now the max spend
        return self.max_spend


@dataclass
class ArchiveItem:
    candidate_id: str
    lineage_id: str
    generation: int
    source: str
    domain: str
    split: str
    prompt_text: str
    rho: float  # max per-trajectory utilisation
    max_spend: float
    certified: bool
    delta_init_mean: float
    effective_budget_min: float
    N: int
    final_budget_mean: float = 0.0
    rationale: str = ""
    novelty_tag: str = ""
    expected_rho: float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    parent_lineage_ids: List[str] = field(default_factory=list)

    @property
    def U_EBB(self) -> float:
        return self.max_spend
