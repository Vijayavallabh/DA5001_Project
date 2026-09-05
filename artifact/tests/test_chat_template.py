"""feat-003: --use-chat-template wraps a prompt as one user turn and adds <|eot_id|> as a stop token."""
from dap.shared import LLAMA3_CHAT, chat_eos_ids, wrap_chat


class _Tok:  # shared Llama-3 tokenizer shape, no template (as shipped with TinyComma)
    chat_template = None
    bos_token = "<|begin_of_text|>"
    eos_token_id = 128001
    unk_token_id = None

    def convert_tokens_to_ids(self, t):
        return 128009 if t == "<|eot_id|>" else None


def test_wrap_and_eos():
    s = wrap_chat("Complete the prefix:\nabc", _Tok())
    assert s.startswith("<|start_header_id|>system") and s.endswith("assistant<|end_header_id|>\n\n")
    assert "user<|end_header_id|>\n\nComplete the prefix:\nabc<|eot_id|>" in s
    assert not s.startswith("<|begin_of_text|>")  # tokenizer adds BOS
    assert chat_eos_ids(_Tok()) == [128001, 128009]
    assert LLAMA3_CHAT.count("{content}") == 1


def test_true_gen_len_stops_at_first_eos():
    from dap.shared import true_gen_len
    assert true_gen_len([5, 6, 128009, 128001, 128001], [128001, 128009]) == 3
    assert true_gen_len([5, 6, 7], 128001) == 3
