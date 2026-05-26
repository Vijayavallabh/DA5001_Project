import importlib
import warnings

import torch


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
