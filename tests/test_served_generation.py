"""caution (bc): judged text is the generation alone, recovered as full_text minus the prompt.

The helper must strip exactly the prompt tail a left-padded row carried into `generation`, handle
a chat-template prompt whose special tokens full_text never contains, leave an un-echoed record
alone, and fall back rather than guess when a record does not decompose."""
from dap.shared import served_generation

PLAIN = "Complete the prefix:\nDescribe the technique for making homemade yogurt."
CHAT = ("<|start_header_id|>system<|end_header_id|>\n\nsys<|eot_id|><|start_header_id|>user"
        "<|end_header_id|>\n\nQ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def test_the_echoed_prompt_tail_is_removed():
    a = {"generation": " homemade yogurt. Heat the milk.",
         "full_text": PLAIN + " Heat the milk."}
    assert served_generation(a, PLAIN) == " Heat the milk."


def test_a_chat_prompt_loses_its_special_tokens_the_way_full_text_did():
    a = {"generation": "assistant\n\nTo make it,",
         "full_text": "system\n\nsysuser\n\nQ?assistant\n\nTo make it,"}
    assert served_generation(a, CHAT) == "To make it,"


def test_an_unpadded_row_is_left_alone():
    a = {"generation": " Heat the milk.", "full_text": PLAIN + " Heat the milk."}
    assert served_generation(a, PLAIN) == " Heat the milk."


def test_an_empty_generation_stays_empty_and_is_not_the_prompt_tail():
    a = {"generation": " making homemade yogurt.", "full_text": PLAIN}
    assert served_generation(a, PLAIN) == ""


def test_a_record_that_does_not_decompose_falls_back_to_generation():
    a = {"generation": "xyz", "full_text": "something else entirely"}
    assert served_generation(a, PLAIN) == "xyz"
