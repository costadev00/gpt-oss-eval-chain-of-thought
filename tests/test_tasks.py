import pytest

from cot_eval.tasks import generate_coin_flip_items, generate_last_letter_items


def test_last_letter_generation_is_deterministic() -> None:
    first = generate_last_letter_items(limit=5, seed=123)
    second = generate_last_letter_items(limit=5, seed=123)
    assert first == second
    for item in first:
        quoted = item.question.split('"')[1]
        words = quoted.split()
        assert item.gold == "".join(word[-1] for word in words).lower()


def test_last_letter_generation_rejects_impossible_limit() -> None:
    with pytest.raises(ValueError):
        generate_last_letter_items(limit=10_000, seed=123)


def test_coin_flip_generation_is_deterministic_and_scored_by_parity() -> None:
    items = generate_coin_flip_items(limit=10, seed=99)
    assert items == generate_coin_flip_items(limit=10, seed=99)
    for item in items:
        flip_count = item.question.count(" flips the coin.")
        expected = "yes" if flip_count % 2 == 0 else "no"
        assert item.gold == expected
