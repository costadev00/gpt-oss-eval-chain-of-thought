import math

from cot_eval.prm_scoring import (
    label_probabilities_from_logprobs,
    majority_vote_answer,
    reduce_step_scores,
    split_solution_steps,
    step_score_from_probabilities,
)
from cot_eval.prm_scoring import StepScore


def test_split_solution_steps_removes_final_answer_and_numbering() -> None:
    solution = "1. Add 2 and 3.\n2) This gives 5.\nFinal answer: 5\nExtra text"
    assert split_solution_steps(solution) == ["Add 2 and 3.", "This gives 5."]


def test_label_probabilities_from_first_token_top_logprobs() -> None:
    probabilities = label_probabilities_from_logprobs(
        [
            {
                " positive": math.log(0.55),
                " neutral": math.log(0.25),
                " negative": math.log(0.20),
            }
        ]
    )

    assert math.isclose(probabilities["positive"], 0.55)
    assert math.isclose(probabilities["neutral"], 0.25)
    assert math.isclose(probabilities["negative"], 0.20)


def test_step_score_treats_neutral_as_positive() -> None:
    label, score, log_score = step_score_from_probabilities({"positive": 0.4, "neutral": 0.3, "negative": 0.3})
    assert label == "positive"
    assert score == 0.7
    assert math.isclose(log_score, math.log(0.7))


def test_reduce_step_scores_uses_product_and_min() -> None:
    scores = [
        StepScore("math", "i", "c", 0, "p", [], "s1", "positive", 0.8, 0.1, 0.1, 0.9, math.log(0.9), 0.1, 10, 1),
        StepScore("math", "i", "c", 1, "p", ["s1"], "s2", "positive", 0.5, 0.1, 0.4, 0.6, math.log(0.6), 0.1, 10, 1),
    ]

    reduced = reduce_step_scores(scores)

    assert math.isclose(reduced.solution_score, 0.54)
    assert math.isclose(reduced.min_step_score, 0.6)
    assert reduced.step_count == 2


def test_majority_vote_tie_breaks_by_first_occurrence() -> None:
    assert majority_vote_answer(["7", "8", "8", "7"]) == "7"
    assert majority_vote_answer([None, "5", None]) == "5"
    assert majority_vote_answer([None, None]) is None
