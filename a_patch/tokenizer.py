from transformers import AutoTokenizer


def init_tokenizer(
    model_checkpoint: str, padding_side: str = "left", **kwargs
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        dtype=kwargs.get("dtype", "auto"),
        trust_remote_code=kwargs.get("trust_remote_code", True),
    )
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
