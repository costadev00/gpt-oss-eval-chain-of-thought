from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from cot_eval.types import Condition, Prediction, TaskName


@dataclass(frozen=True)
class MetricRow:
    task: TaskName
    condition: Condition
    n: int
    correct: int
    accuracy: float
    ci_low: float
    ci_high: float
    accuracy_delta_vs_standard: float | None
    avg_latency_s: float | None
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_total_tokens: float | None
    parse_failures: int


def bootstrap_accuracy_ci(values: list[bool], seed: int, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        accuracy = 1.0 if values[0] else 0.0
        return accuracy, accuracy
    rng = random.Random(seed)
    accuracies = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        accuracies.append(sum(sample) / len(sample))
    accuracies.sort()
    low_index = int(0.025 * (len(accuracies) - 1))
    high_index = int(0.975 * (len(accuracies) - 1))
    return accuracies[low_index], accuracies[high_index]


def optional_mean(values: list[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def aggregate_metrics(predictions: list[Prediction], seed: int) -> list[MetricRow]:
    grouped: dict[tuple[TaskName, Condition], list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction.task, prediction.condition)].append(prediction)

    accuracies: dict[tuple[TaskName, Condition], float] = {}
    rows: list[MetricRow] = []
    task_order = {"gsm8k": 0, "last_letter": 1, "coin_flip": 2}
    condition_order = {"standard": 0, "cot": 1}
    ordered_groups = sorted(grouped.items(), key=lambda item: (task_order[item[0][0]], condition_order[item[0][1]]))
    for key, group in ordered_groups:
        task, condition = key
        correct_values = [prediction.correct for prediction in group]
        correct = sum(correct_values)
        accuracy = correct / len(group) if group else 0.0
        accuracies[key] = accuracy
        ci_low, ci_high = bootstrap_accuracy_ci(correct_values, seed=seed)
        rows.append(
            MetricRow(
                task=task,
                condition=condition,
                n=len(group),
                correct=correct,
                accuracy=accuracy,
                ci_low=ci_low,
                ci_high=ci_high,
                accuracy_delta_vs_standard=None,
                avg_latency_s=optional_mean([prediction.latency_s for prediction in group]),
                avg_prompt_tokens=optional_mean([prediction.prompt_tokens for prediction in group]),
                avg_completion_tokens=optional_mean([prediction.completion_tokens for prediction in group]),
                avg_total_tokens=optional_mean([prediction.total_tokens for prediction in group]),
                parse_failures=sum(1 for prediction in group if prediction.parse_failed),
            )
        )

    final_rows: list[MetricRow] = []
    for row in rows:
        delta = None
        if row.condition == "cot":
            standard = accuracies.get((row.task, "standard"))
            if standard is not None:
                delta = row.accuracy - standard
        final_rows.append(
            MetricRow(
                task=row.task,
                condition=row.condition,
                n=row.n,
                correct=row.correct,
                accuracy=row.accuracy,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                accuracy_delta_vs_standard=delta,
                avg_latency_s=row.avg_latency_s,
                avg_prompt_tokens=row.avg_prompt_tokens,
                avg_completion_tokens=row.avg_completion_tokens,
                avg_total_tokens=row.avg_total_tokens,
                parse_failures=row.parse_failures,
            )
        )
    return final_rows


def write_metrics_csv(path: Path, rows: list[MetricRow]) -> None:
    fieldnames = [
        "task",
        "condition",
        "n",
        "correct",
        "accuracy",
        "ci_low",
        "ci_high",
        "accuracy_delta_vs_standard",
        "avg_latency_s",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "parse_failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row.task,
                    "condition": row.condition,
                    "n": row.n,
                    "correct": row.correct,
                    "accuracy": f"{row.accuracy:.6f}",
                    "ci_low": f"{row.ci_low:.6f}",
                    "ci_high": f"{row.ci_high:.6f}",
                    "accuracy_delta_vs_standard": ""
                    if row.accuracy_delta_vs_standard is None
                    else f"{row.accuracy_delta_vs_standard:.6f}",
                    "avg_latency_s": "" if row.avg_latency_s is None else f"{row.avg_latency_s:.6f}",
                    "avg_prompt_tokens": ""
                    if row.avg_prompt_tokens is None
                    else f"{row.avg_prompt_tokens:.2f}",
                    "avg_completion_tokens": ""
                    if row.avg_completion_tokens is None
                    else f"{row.avg_completion_tokens:.2f}",
                    "avg_total_tokens": "" if row.avg_total_tokens is None else f"{row.avg_total_tokens:.2f}",
                    "parse_failures": row.parse_failures,
                }
            )
