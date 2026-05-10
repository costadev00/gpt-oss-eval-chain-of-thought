from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from cot_eval.types import TaskName

ANSWER_MARKER_RE = re.compile(
    r"(?:\bfinal\s+answer\s*(?:is|[:=])|\b(?:so\s+)?the\s+answer\s+is\b|\banswer\s*[:=])",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")
YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LAST_LETTER_RE = re.compile(r"\b([a-zA-Z]{1,12})\b")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
NEGATED_HEADS_RE = re.compile(r"\b(?:not|no\s+longer|is\s+not|isn't|isnt)\s+(?:still\s+)?heads(?:\s+up)?\b", re.IGNORECASE)
HEADS_UP_RE = re.compile(r"\b(?:still\s+)?heads\s+up\b|\b(?:is|remains|stays)\s+(?:still\s+)?heads\b", re.IGNORECASE)
TAILS_UP_RE = re.compile(r"\btails\s+up\b|\b(?:is|remains|stays|lands|landed)\s+(?:on\s+)?tails\b", re.IGNORECASE)
ANSWER_BOUNDARY_RE = re.compile(
    r"\n\s*(?:\*\*)?(?:explanation|how\s+we\s+got|step[-\s]*by[-\s]*step|steps?|calculation)\b",
    re.IGNORECASE,
)


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


def normalize_text(text: str) -> str:
    return (
        text.replace("\u202f", " ")
        .replace("\xa0", " ")
        .replace("\u2212", "-")
        .replace("−", "-")
        .strip()
    )


def numeric_candidates(text: str) -> list[str]:
    candidates = [candidate.group(0) for candidate in NUMBER_RE.finditer(normalize_text(text))]
    return [candidate for candidate in candidates if candidate not in {"", "+", "-", "."}]


def boxed_numeric_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in BOXED_RE.finditer(text):
        candidates.extend(numeric_candidates(match.group(1)))
    return candidates


def bold_numeric_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in BOLD_RE.finditer(text):
        candidates.extend(numeric_candidates(match.group(1)))
    return candidates


def marked_numeric_candidates(text: str) -> list[str]:
    return boxed_numeric_candidates(text) + bold_numeric_candidates(text)


def answer_segments(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches = list(ANSWER_MARKER_RE.finditer(normalized))
    segments: list[str] = []
    for match in matches:
        segment = normalized[match.end() :].strip()
        boundary = ANSWER_BOUNDARY_RE.search(segment)
        if boundary:
            segment = segment[: boundary.start()].strip()
        segments.append(segment)
    return segments


def terminal_numeric_segment(text: str) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"\n+|(?<=[.!?])\s+", normalize_text(text)) if chunk.strip()]
    for chunk in reversed(chunks):
        if numeric_candidates(chunk):
            return chunk
    return normalize_text(text)


def terminal_text_segment(text: str) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"\n+|(?<=[.!?])\s+", normalize_text(text)) if chunk.strip()]
    return chunks[-1] if chunks else normalize_text(text)


def parse_coin_state_answer(text: str) -> str | None:
    normalized = normalize_text(text)
    if NEGATED_HEADS_RE.search(normalized):
        return "no"
    if HEADS_UP_RE.search(normalized):
        return "yes"
    if TAILS_UP_RE.search(normalized):
        return "no"
    return None


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    parsed = parse_numeric_answer(answer)
    if parsed is None:
        raise ValueError(f"Could not extract GSM8K gold answer from: {answer!r}")
    return parsed


def parse_numeric_answer(text: str) -> str | None:
    normalized = normalize_text(text)
    for segment in reversed(answer_segments(normalized)):
        marked = marked_numeric_candidates(segment)
        if marked:
            return normalize_numeric(marked[0])

        candidates = numeric_candidates(segment)
        if not candidates:
            continue

        prefix = segment.lstrip(" :*$`([{")
        if re.match(NUMBER_RE, prefix) and "=" in segment:
            return normalize_numeric(candidates[-1])
        return normalize_numeric(candidates[0])

    boxed = boxed_numeric_candidates(normalized)
    if boxed:
        return normalize_numeric(boxed[-1])

    terminal_segment = terminal_numeric_segment(normalized)
    terminal_bold = bold_numeric_candidates(terminal_segment)
    if terminal_bold:
        return normalize_numeric(terminal_bold[0])

    candidates = numeric_candidates(terminal_segment)
    if not candidates:
        return None
    return normalize_numeric(candidates[-1])


def parse_yes_no_answer(text: str) -> str | None:
    normalized = normalize_text(text)
    matches: list[str] = []
    for segment in reversed(answer_segments(normalized)):
        matches = YES_NO_RE.findall(segment)
        if matches:
            return matches[0].lower()
        state = parse_coin_state_answer(segment)
        if state:
            return state
    matches = YES_NO_RE.findall(normalized)
    if not matches:
        return parse_coin_state_answer(terminal_text_segment(normalized))
    return matches[-1].lower()


def parse_last_letter_answer(text: str) -> str | None:
    normalized = normalize_text(text)
    segments = answer_segments(normalized)
    search_space = segments[-1] if segments else normalized
    quoted = re.findall(r'"([a-zA-Z]+)"', search_space)
    if quoted:
        return quoted[-1].lower()
    words = LAST_LETTER_RE.findall(search_space)
    words = [word for word in words if word.lower() not in {"the", "answer", "is"}]
    if not words and segments:
        words = LAST_LETTER_RE.findall(normalized)
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
