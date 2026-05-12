from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from cot_eval.types import TextCompletionLogprobsResult

PRM_LABELS = ("positive", "neutral", "negative")
MIN_SCORE = 1e-12


@dataclass(frozen=True)
class StepScore:
    task: str
    item_id: str
    candidate_id: str
    step_index: int
    problem: str
    previous_steps: list[str]
    step: str
    label_predicted: str
    p_positive: float
    p_neutral: float
    p_negative: float
    step_score: float
    step_log_score: float
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class SolutionScore:
    solution_score: float
    solution_log_score: float
    min_step_score: float
    step_count: int


def strip_step_prefix(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^\s*(?:step\s*)?\d+[\).\:-]\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"^\s*[-*]\s*", "", stripped)
    return stripped.strip()


def split_solution_steps(solution: str) -> list[str]:
    steps: list[str] = []
    for raw_line in solution.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"\bfinal\s+answer\s*(?:is|[:=])", line, flags=re.IGNORECASE):
            break
        cleaned = strip_step_prefix(line)
        if cleaned:
            steps.append(cleaned)
    return steps


def normalize_label_token(token: str) -> str | None:
    cleaned = token.strip().lower()
    cleaned = cleaned.strip("\"'`:.()[]{}")
    cleaned = re.split(r"\s+", cleaned)[0] if cleaned else ""
    cleaned = cleaned.strip("\"'`:.()[]{}")
    return cleaned if cleaned in PRM_LABELS else None


def label_probabilities_from_logprobs(
    top_logprobs: list[dict[str, float]],
    generated_content: str = "",
) -> dict[str, float]:
    probabilities = {label: 0.0 for label in PRM_LABELS}
    if top_logprobs:
        for token, logprob in top_logprobs[0].items():
            label = normalize_label_token(token)
            if label is not None:
                probabilities[label] += math.exp(logprob)

    if not any(probabilities.values()):
        generated_label = normalize_label_token(generated_content)
        if generated_label is not None:
            probabilities[generated_label] = 1.0
    return probabilities


def predicted_label(probabilities: dict[str, float]) -> str:
    return max(PRM_LABELS, key=lambda label: probabilities.get(label, 0.0))


def step_score_from_probabilities(probabilities: dict[str, float]) -> tuple[str, float, float]:
    label = predicted_label(probabilities)
    score = max(MIN_SCORE, probabilities.get("positive", 0.0) + probabilities.get("neutral", 0.0))
    return label, score, math.log(score)


def make_step_score(
    task: str,
    item_id: str,
    candidate_id: str,
    step_index: int,
    problem: str,
    previous_steps: list[str],
    step: str,
    result: TextCompletionLogprobsResult,
) -> StepScore:
    probabilities = label_probabilities_from_logprobs(result.top_logprobs, generated_content=result.content)
    label, step_score, step_log_score = step_score_from_probabilities(probabilities)
    return StepScore(
        task=task,
        item_id=item_id,
        candidate_id=candidate_id,
        step_index=step_index,
        problem=problem,
        previous_steps=previous_steps,
        step=step,
        label_predicted=label,
        p_positive=probabilities["positive"],
        p_neutral=probabilities["neutral"],
        p_negative=probabilities["negative"],
        step_score=step_score,
        step_log_score=step_log_score,
        latency_s=result.latency_s,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def reduce_step_scores(step_scores: list[StepScore]) -> SolutionScore:
    if not step_scores:
        return SolutionScore(
            solution_score=MIN_SCORE,
            solution_log_score=math.log(MIN_SCORE),
            min_step_score=MIN_SCORE,
            step_count=0,
        )
    solution_log_score = sum(score.step_log_score for score in step_scores)
    min_step_score = min(score.step_score for score in step_scores)
    return SolutionScore(
        solution_score=math.exp(solution_log_score),
        solution_log_score=solution_log_score,
        min_step_score=min_step_score,
        step_count=len(step_scores),
    )


def majority_vote_answer(parsed_answers: list[str | None]) -> str | None:
    present = [answer for answer in parsed_answers if answer is not None]
    if not present:
        return None
    counts = Counter(present)
    best_count = max(counts.values())
    for answer in present:
        if counts[answer] == best_count:
            return answer
    return None
