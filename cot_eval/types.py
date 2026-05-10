from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Condition = Literal["standard", "cot"]
TaskName = Literal["gsm8k", "last_letter", "coin_flip"]


@dataclass(frozen=True)
class EvalItem:
    task: TaskName
    item_id: str
    question: str
    gold: str


@dataclass(frozen=True)
class CompletionResult:
    content: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    retried_without_extra_body: bool = False
    discarded_reasoning: bool = False


@dataclass(frozen=True)
class Prediction:
    task: TaskName
    condition: Condition
    item_id: str
    question: str
    gold: str
    parsed_answer: str | None
    correct: bool
    parse_failed: bool
    response: str
    prompt_hash: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    metadata: dict[str, Any]
