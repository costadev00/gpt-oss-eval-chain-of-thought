from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from cot_eval.types import TaskName

ANSWER_IS_RE = re.compile(r"(?:the\s+answer\s+is|answer\s*[:=])\s*([^\n]+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LAST_LETTER_RE = re.compile(r"\b([a-zA-Z]{1,12})\b")


def normalize_numeric(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.replace("$", "").replace("%", "")
    cleaned = cleaned.strip().rstrip(".")
    if not cleaned:
        return None
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation:
        return None
    if decimal == decimal.to_integral_value():
        return str(int(decimal))
    normalized = format(decimal.normalize(), "f")
    return normalized.rstrip("0").rstrip(".")


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    parsed = parse_numeric_answer(answer)
    if parsed is None:
        raise ValueError(f"Could not extract GSM8K gold answer from: {answer!r}")
    return parsed


def parse_numeric_answer(text: str) -> str | None:
    answer_match = ANSWER_IS_RE.search(text)
    search_space = answer_match.group(1) if answer_match else text
    candidates = [candidate.group(0) for candidate in NUMBER_RE.finditer(search_space)]
    candidates = [candidate for candidate in candidates if candidate not in {"", "+", "-", "."}]
    if not candidates and answer_match:
        candidates = [candidate.group(0) for candidate in NUMBER_RE.finditer(text)]
    if not candidates:
        return None
    return normalize_numeric(candidates[-1])


def parse_yes_no_answer(text: str) -> str | None:
    answer_match = ANSWER_IS_RE.search(text)
    search_space = answer_match.group(1) if answer_match else text
    matches = YES_NO_RE.findall(search_space)
    if not matches and answer_match:
        matches = YES_NO_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()


def parse_last_letter_answer(text: str) -> str | None:
    answer_match = ANSWER_IS_RE.search(text)
    search_space = answer_match.group(1) if answer_match else text
    quoted = re.findall(r'"([a-zA-Z]+)"', search_space)
    if quoted:
        return quoted[-1].lower()
    words = LAST_LETTER_RE.findall(search_space)
    words = [word for word in words if word.lower() not in {"the", "answer", "is"}]
    if not words and answer_match:
        words = LAST_LETTER_RE.findall(text)
    if not words:
        return None
    return words[-1].lower()


def parse_answer(task: TaskName, text: str) -> str | None:
    if task == "gsm8k":
        return parse_numeric_answer(text)
    if task == "coin_flip":
        return parse_yes_no_answer(text)
    if task == "last_letter":
        return parse_last_letter_answer(text)
    raise ValueError(f"Unknown task: {task}")


def answers_equal(task: TaskName, parsed: str | None, gold: str) -> bool:
    if parsed is None:
        return False
    if task == "gsm8k":
        parsed_numeric = normalize_numeric(parsed)
        gold_numeric = normalize_numeric(gold)
        if parsed_numeric is None or gold_numeric is None:
            return False
        try:
            return math.isclose(float(parsed_numeric), float(gold_numeric), rel_tol=0.0, abs_tol=1e-9)
        except ValueError:
            return parsed_numeric == gold_numeric
    return parsed.strip().lower() == gold.strip().lower()
