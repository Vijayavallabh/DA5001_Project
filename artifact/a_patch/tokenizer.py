from transformers import AutoTokenizer


def ensure_pad_token(tokenizer):
    """Give a tokenizer a usable pad token, without growing the vocabulary.

    `pad = eos` is the usual fallback and it is not enough: some permissively licensed families
    (Pleias 1.2b/3b) register pad, eos, unk AND bos all as None, so that assignment stores None and
    the first padded batch dies inside `_get_padding_truncation_strategies`. We fall through the
    special tokens to a literal that is already in the vocabulary -- `<|end_of_text|>` is present in
    those checkpoints at id 2, merely unregistered -- and finally to token 0.

    Nothing is ever added. Adding a pad token would push len(tokenizer) past the model's embedding
    rows and trip the padded-vocabulary check in a_patch/factory.py, which is what makes the
    self-paired experiments possible in the first place. Padding is attention-masked, so any
    existing id is sound.
    """
    if tokenizer.pad_token_id is not None:
        return tokenizer
    vocab = tokenizer.get_vocab()
    for cand in (tokenizer.eos_token, tokenizer.unk_token, tokenizer.bos_token,
                 "<|end_of_text|>", "<|endoftext|>", "</s>", "<pad>"):
        if cand and cand in vocab:
            tokenizer.pad_token = cand
            return tokenizer
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    return tokenizer


def init_tokenizer(
    model_checkpoint: str, padding_side: str = "left", **kwargs
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        dtype=kwargs.get("dtype", "auto"),
        trust_remote_code=kwargs.get("trust_remote_code", True),
    )
    tokenizer.padding_side = padding_side
    return ensure_pad_token(tokenizer)
