from cot_eval.scoring import (
    answers_equal,
    extract_gsm8k_gold,
    normalize_numeric,
    parse_answer,
    parse_last_letter_answer,
    parse_numeric_answer,
    parse_yes_no_answer,
)


def test_normalize_numeric_strips_commas_currency_and_trailing_zeroes() -> None:
    assert normalize_numeric("$1,234.00") == "1234"
    assert normalize_numeric("8.50") == "8.5"


def test_extract_gsm8k_gold_uses_hash_marker() -> None:
    assert extract_gsm8k_gold("Some reasoning.\n#### 1,234") == "1234"


def test_parse_numeric_prefers_answer_phrase() -> None:
    assert parse_numeric_answer("We compute 10 + 5 = 15. The answer is $8.") == "8"
    assert parse_numeric_answer("Reasoning has 15 and 20.\nFinal answer: 8") == "8"
    assert parse_numeric_answer("Reasoning has 15 and 20.\nFinal answer is 8") == "8"


def test_parse_numeric_prefers_first_number_after_prose_answer_marker() -> None:
    text = "Answer: Jason would need to make 750 telephone calls to sell 15 cars."
    assert parse_numeric_answer(text) == "750"


def test_parse_numeric_handles_equation_after_answer_marker() -> None:
    assert parse_numeric_answer("The answer is 99 + 10 = 109.") == "109"


def test_parse_numeric_prefers_boxed_or_bold_final_answer() -> None:
    assert parse_numeric_answer(r"Rounded to the nearest integer: \boxed{$139}") == "139"
    assert parse_numeric_answer("So she spent **70 minutes** (1 hour 10 minutes).") == "70"
    assert parse_numeric_answer("Indras has **6** letters. So together they have 6 + 7 = 13 letters.") == "13"


def test_parse_numeric_ignores_explanation_after_answer_marker() -> None:
    text = "**Answer: $4,400**\n\n**Explanation**\n1. Series movies cost $6."
    assert parse_numeric_answer(text) == "4400"


def test_parse_numeric_falls_back_to_last_number() -> None:
    assert parse_numeric_answer("First 3, then 9, finally 12.") == "12"


def test_parse_yes_no_answer() -> None:
    assert parse_yes_no_answer("After one flip it is tails. So the answer is no.") == "no"
    assert parse_yes_no_answer("The answer is no, not yes.") == "no"
    assert parse_yes_no_answer("The reasoning mentions no as a possibility.\nFinal answer: yes") == "yes"
    assert parse_yes_no_answer("The coin is still **heads up**.") == "yes"
    assert parse_yes_no_answer("The coin is tails up.") == "no"
    assert parse_yes_no_answer("The coin is not heads up.") == "no"


def test_parse_last_letter_answer() -> None:
    assert parse_last_letter_answer('Concatenating them is "nk". The answer is nk.') == "nk"


def test_parse_answer_dispatch_and_equality() -> None:
    assert parse_answer("coin_flip", "The answer is yes.") == "yes"
    assert answers_equal("gsm8k", "8.0", "8")
    assert not answers_equal("last_letter", None, "nk")
