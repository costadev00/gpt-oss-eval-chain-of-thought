from cot_eval.math_tasks import (
    build_math_items_from_rows,
    extract_boxed_contents,
    extract_math_gold,
    normalize_math_numeric,
    parse_generated_math_answer,
)


def test_extract_boxed_contents_handles_nested_latex() -> None:
    text = r"Then the answer is \boxed{\frac{3}{4}} and not \boxed{2}."
    assert extract_boxed_contents(text) == [r"\frac{3}{4}", "2"]


def test_normalize_math_numeric_accepts_numbers_and_simple_fractions() -> None:
    assert normalize_math_numeric(r"1,234") == "1234"
    assert normalize_math_numeric(r"\frac{1}{2}") == "0.5"
    assert normalize_math_numeric("6.00") == "6"


def test_extract_math_gold_uses_last_parseable_boxed_answer() -> None:
    assert extract_math_gold(r"Some work. \boxed{x=7}") == "7"
    assert extract_math_gold(r"Some work. \boxed{\text{meters}}") is None


def test_build_math_items_from_rows_filters_non_numeric_gold() -> None:
    rows = [
        {"problem": "p1", "solution": r"work \boxed{5}", "level": "Level 1"},
        {"problem": "p2", "solution": r"work \boxed{x+y}", "level": "Level 2"},
    ]

    items, skipped, total_seen = build_math_items_from_rows(rows, config="algebra")

    assert total_seen == 2
    assert skipped == 1
    assert len(items) == 1
    assert items[0].gold == "5"


def test_parse_generated_math_answer_requires_explicit_final_answer_or_boxed() -> None:
    assert parse_generated_math_answer("1. Incomplete step mentions 4") is None
    assert parse_generated_math_answer("1/83") == "0.01204819277108433734939759036"
    assert parse_generated_math_answer("1. Work.\nFinal answer: 4") == "4"
    assert parse_generated_math_answer("1. Work.\nFinal answer: 1/3") == "0.3333333333333333333333333333"
    assert parse_generated_math_answer(r"1. Work.\nFinal answer: \(\frac{5}{9}\)") == "0.5555555555555555555555555556"
    assert parse_generated_math_answer(r"1. Work.\nFinal answer: $\displaystyle 240/13$") == "18.46153846153846153846153846"
    assert parse_generated_math_answer(r"Some work. \boxed{9}") == "9"
