from __future__ import annotations

import random

from cot_eval.scoring import extract_gsm8k_gold
from cot_eval.types import EvalItem, TaskName

FIRST_NAMES = [
    "Ada",
    "Grace",
    "Linus",
    "Katherine",
    "Alan",
    "Edsger",
    "Barbara",
    "Donald",
    "Frances",
    "Margaret",
    "Tim",
    "Radia",
]

LAST_NAMES = [
    "Lovelace",
    "Hopper",
    "Torvalds",
    "Johnson",
    "Turing",
    "Dijkstra",
    "Liskov",
    "Knuth",
    "Allen",
    "Hamilton",
    "Berners",
    "Perlman",
]

COIN_NAMES = [
    "Alex",
    "Brianna",
    "Casey",
    "Devon",
    "Elliot",
    "Fatima",
    "Gabriel",
    "Harper",
    "Indira",
    "Jules",
    "Kai",
    "Lena",
    "Mina",
    "Noah",
    "Omar",
    "Priya",
]


def load_gsm8k_items(limit: int | None, seed: int) -> list[EvalItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run GSM8K evaluations.") from exc

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    items: list[EvalItem] = []
    for index, row in enumerate(dataset):
        items.append(
            EvalItem(
                task="gsm8k",
                item_id=f"gsm8k-{index}",
                question=row["question"],
                gold=extract_gsm8k_gold(row["answer"]),
            )
        )
    return items


def generate_last_letter_items(limit: int, seed: int) -> list[EvalItem]:
    max_items = len(FIRST_NAMES) * len(LAST_NAMES)
    if limit > max_items:
        raise ValueError(f"last_letter limit cannot exceed {max_items} unique generated examples")
    rng = random.Random(seed)
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < limit:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        if (first, last) not in pairs:
            pairs.add((first, last))
    items: list[EvalItem] = []
    for index, (first, last) in enumerate(sorted(pairs)):
        phrase = f"{first} {last}"
        gold = f"{first[-1]}{last[-1]}".lower()
        items.append(
            EvalItem(
                task="last_letter",
                item_id=f"last_letter-{index}",
                question=f'Take the last letters of the words in "{phrase}" and concatenate them.',
                gold=gold,
            )
        )
    return items


def generate_coin_flip_items(limit: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    items: list[EvalItem] = []
    seen: set[tuple[tuple[str, bool], ...]] = set()
    while len(items) < limit:
        step_count = rng.randint(2, 5)
        people = rng.sample(COIN_NAMES, step_count)
        actions = tuple((person, rng.choice([True, False])) for person in people)
        if actions in seen:
            continue
        seen.add(actions)
        clauses = [f"{person} {'flips' if flips else 'does not flip'} the coin." for person, flips in actions]
        flip_count = sum(1 for _, flips in actions if flips)
        gold = "yes" if flip_count % 2 == 0 else "no"
        question = "A coin is heads up. " + " ".join(clauses) + " Is the coin still heads up?"
        items.append(EvalItem(task="coin_flip", item_id=f"coin_flip-{len(items)}", question=question, gold=gold))
    return items


def load_items(tasks: list[TaskName], gsm8k_limit: int | None, symbolic_limit: int, seed: int) -> list[EvalItem]:
    items: list[EvalItem] = []
    for task in tasks:
        if task == "gsm8k":
            items.extend(load_gsm8k_items(gsm8k_limit, seed))
        elif task == "last_letter":
            items.extend(generate_last_letter_items(symbolic_limit, seed))
        elif task == "coin_flip":
            items.extend(generate_coin_flip_items(symbolic_limit, seed))
        else:
            raise ValueError(f"Unknown task: {task}")
    return items
