from __future__ import annotations

import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

BOXED_PREFIX = r"\boxed"
FRAC_RE = re.compile(r"^(-?)\\frac\{([-+]?\d+(?:\.\d+)?)\}\{([-+]?\d+(?:\.\d+)?)\}$")
SLASH_FRAC_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class MathItem:
    item_id: str
    problem: str
    gold: str
    config: str
    level: str
    solution: str


@dataclass(frozen=True)
class MathLoadResult:
    items: list[MathItem]
    skipped: int
    total_seen: int


def normalize_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".")


def strip_latex_number(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\,", "").replace("\\!", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("\\textstyle", "")
    cleaned = cleaned.replace("\\scriptstyle", "")
    cleaned = cleaned.replace("\\scriptscriptstyle", "")
    cleaned = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    while cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def normalize_math_numeric(value: str) -> str | None:
    cleaned = strip_latex_number(value)
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1]
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]

    frac_match = FRAC_RE.fullmatch(cleaned)
    if frac_match:
        sign, numerator_raw, denominator_raw = frac_match.groups()
        try:
            numerator = Decimal(numerator_raw)
            denominator = Decimal(denominator_raw)
        except InvalidOperation:
            return None
        if denominator == 0:
            return None
        value_decimal = numerator / denominator
        if sign:
            value_decimal = -value_decimal
        return normalize_decimal(value_decimal)

    slash_match = SLASH_FRAC_RE.fullmatch(cleaned)
    if slash_match:
        try:
            numerator = Decimal(slash_match.group(1))
            denominator = Decimal(slash_match.group(2))
        except InvalidOperation:
            return None
        if denominator == 0:
            return None
        return normalize_decimal(numerator / denominator)

    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", cleaned):
        return None
    try:
        return normalize_decimal(Decimal(cleaned))
    except InvalidOperation:
        return None


def extract_boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    start = 0
    while True:
        prefix_index = text.find(BOXED_PREFIX, start)
        if prefix_index == -1:
            break
        brace_index = text.find("{", prefix_index + len(BOXED_PREFIX))
        if brace_index == -1:
            break
        depth = 0
        for index in range(brace_index, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[brace_index + 1 : index])
                    start = index + 1
                    break
        else:
            break
    return contents


def extract_math_gold(solution: str) -> str | None:
    for boxed in reversed(extract_boxed_contents(solution)):
        normalized = normalize_math_numeric(boxed)
        if normalized is not None:
            return normalized
    return None


def build_math_items_from_rows(rows: Iterable[dict[str, object]], config: str) -> tuple[list[MathItem], int, int]:
    items: list[MathItem] = []
    skipped = 0
    total_seen = 0
    for index, row in enumerate(rows):
        total_seen += 1
        solution = str(row.get("solution", ""))
        gold = extract_math_gold(solution)
        if gold is None:
            skipped += 1
            continue
        items.append(
            MathItem(
                item_id=f"math-{config}-{index}",
                problem=str(row.get("problem", "")),
                gold=gold,
                config=config,
                level=str(row.get("level", "")),
                solution=solution,
            )
        )
    return items, skipped, total_seen


def load_math_items(limit: int | None, seed: int, configs: list[str] | None = None) -> MathLoadResult:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run MATH evaluations.") from exc

    selected_configs = configs or MATH_CONFIGS
    items: list[MathItem] = []
    skipped = 0
    total_seen = 0
    for config in selected_configs:
        dataset = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        config_items, config_skipped, config_total = build_math_items_from_rows(dataset, config=config)
        items.extend(config_items)
        skipped += config_skipped
        total_seen += config_total

    rng = random.Random(seed)
    rng.shuffle(items)
    if limit is not None:
        items = items[: min(limit, len(items))]
    return MathLoadResult(items=items, skipped=skipped, total_seen=total_seen)


def parse_generated_math_answer(text: str) -> str | None:
    from cot_eval.scoring import answer_segments, parse_numeric_answer

    bare_numeric = normalize_math_numeric(text)
    if bare_numeric is not None:
        return bare_numeric

    segments = answer_segments(text)
    for segment in reversed(segments):
        normalized = normalize_math_numeric(segment)
        if normalized is not None:
            return normalized
        for boxed in reversed(extract_boxed_contents(segment)):
            normalized = normalize_math_numeric(boxed)
            if normalized is not None:
                return normalized

    if segments:
        parsed = parse_numeric_answer(text)
        if parsed is not None:
            return parsed

    boxed_values = extract_boxed_contents(text)
    for value in reversed(boxed_values):
        normalized = normalize_math_numeric(value)
        if normalized is not None:
            return normalized
    return None
