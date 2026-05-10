from __future__ import annotations

import hashlib

from cot_eval.types import Condition, TaskName

MATH_COT_EXEMPLARS = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]

LAST_LETTER_COT_EXEMPLARS = [
    (
        'Take the last letters of the words in "Elon Musk" and concatenate them.',
        'The last letter of "Elon" is "n". The last letter of "Musk" is "k". '
        'Concatenating them is "nk". The answer is nk.',
    ),
    (
        'Take the last letters of the words in "Larry Page" and concatenate them.',
        'The last letter of "Larry" is "y". The last letter of "Page" is "e". '
        'Concatenating them is "ye". The answer is ye.',
    ),
    (
        'Take the last letters of the words in "Sergey Brin" and concatenate them.',
        'The last letter of "Sergey" is "y". The last letter of "Brin" is "n". '
        'Concatenating them is "yn". The answer is yn.',
    ),
    (
        'Take the last letters of the words in "Bill Gates" and concatenate them.',
        'The last letter of "Bill" is "l". The last letter of "Gates" is "s". '
        'Concatenating them is "ls". The answer is ls.',
    ),
]

COIN_FLIP_COT_EXEMPLARS = [
    (
        "A coin is heads up. Ka flips the coin. Sherrie flips the coin. Is the coin still heads up?",
        "The coin was flipped by Ka and Sherrie. So the coin was flipped 2 times, which is an even number. "
        "The coin started heads up, so after an even number of flips, it will still be heads up. "
        "So the answer is yes.",
    ),
    (
        "A coin is heads up. Jamey flips the coin. Teressa flips the coin. Is the coin still heads up?",
        "The coin was flipped by Jamey and Teressa. So the coin was flipped 2 times, which is an even number. "
        "The coin started heads up, so after an even number of flips, it will still be heads up. "
        "So the answer is yes.",
    ),
    (
        "A coin is heads up. Maybelle flips the coin. Shalonda does not flip the coin. Is the coin still heads up?",
        "The coin was flipped by Maybelle. So the coin was flipped 1 time, which is an odd number. "
        "The coin started heads up, so after an odd number of flips, it will be tails up. "
        "So the answer is no.",
    ),
    (
        "A coin is heads up. Millicent does not flip the coin. Conception flips the coin. Is the coin still heads up?",
        "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. "
        "The coin started heads up, so after an odd number of flips, it will be tails up. "
        "So the answer is no.",
    ),
    (
        "A coin is heads up. Sal flips the coin. Raymond does not flip the coin. Is the coin still heads up?",
        "The coin was flipped by Sal. So the coin was flipped 1 time, which is an odd number. "
        "The coin started heads up, so after an odd number of flips, it will be tails up. "
        "So the answer is no.",
    ),
    (
        "A coin is heads up. Conception flips the coin. Kristian does not flip the coin. Is the coin still heads up?",
        "The coin was flipped by Conception. So the coin was flipped 1 time, which is an odd number. "
        "The coin started heads up, so after an odd number of flips, it will be tails up. "
        "So the answer is no.",
    ),
    (
        "A coin is heads up. Inga does not flip the coin. Elanor does not flip the coin. Is the coin still heads up?",
        "The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, "
        "and it was not flipped, so it is still heads up. So the answer is yes.",
    ),
    (
        "A coin is heads up. Ryan flips the coin. Shaunda flips the coin. Is the coin still heads up?",
        "The coin was flipped by Ryan and Shaunda. So the coin was flipped 2 times, which is an even number. "
        "The coin started heads up, so after an even number of flips, it will still be heads up. "
        "So the answer is yes.",
    ),
]

EXEMPLARS: dict[TaskName, list[tuple[str, str]]] = {
    "gsm8k": MATH_COT_EXEMPLARS,
    "last_letter": LAST_LETTER_COT_EXEMPLARS,
    "coin_flip": COIN_FLIP_COT_EXEMPLARS,
}


def direct_answer_from_cot(answer: str) -> str:
    marker = "The answer is "
    index = answer.rfind(marker)
    if index == -1:
        return answer
    return marker + answer[index + len(marker) :]


def build_prompt(task: TaskName, condition: Condition, question: str) -> str:
    lines: list[str] = []
    for exemplar_question, exemplar_answer in EXEMPLARS[task]:
        answer = exemplar_answer if condition == "cot" else direct_answer_from_cot(exemplar_answer)
        lines.append(f"Q: {exemplar_question}\nA: {answer}")
    lines.append(f"Q: {question}\nA:")
    return "\n\n".join(lines)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
