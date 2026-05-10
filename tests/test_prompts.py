from cot_eval.prompts import DEFAULT_SYSTEM_PROMPT, build_prompt, direct_answer_from_cot, ensure_final_answer_line


def test_direct_answer_from_cot_is_case_insensitive() -> None:
    answer = "The coin was flipped once. So the answer is no."
    assert direct_answer_from_cot(answer) == "Final answer: no"


def test_cot_exemplar_keeps_reasoning_and_adds_final_answer_line() -> None:
    answer = "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    formatted = ensure_final_answer_line(answer)
    assert "There are originally 3 cars" in formatted
    assert formatted.endswith("Final answer: 5")


def test_coin_flip_standard_prompt_differs_from_cot_prompt() -> None:
    question = "A coin is heads up. Alex flips the coin. Is the coin still heads up?"
    standard = build_prompt("coin_flip", "standard", question)
    cot = build_prompt("coin_flip", "cot", question)
    assert standard != cot
    assert "The coin was flipped by Ka and Sherrie" not in standard
    assert "The coin was flipped by Ka and Sherrie" in cot


def test_standard_prompts_strip_reasoning_for_all_tasks() -> None:
    questions = {
        "gsm8k": "Darrell and Allen ages are in the ratio of 7:11. Their total age is 162. How old is Allen in 10 years?",
        "last_letter": 'Take the last letters of the words in "Tim Hopper" and concatenate them.',
        "coin_flip": "A coin is heads up. Alex does not flip the coin. Lena flips the coin. Is the coin still heads up?",
    }
    forbidden_reasoning = {
        "gsm8k": ["There are 15 trees originally", "3 + 2 = 5", "Olivia had 23 dollars"],
        "last_letter": ['The last letter of "Elon"', "Concatenating them"],
        "coin_flip": ["The coin was flipped by", "odd number", "even number"],
    }
    for task, question in questions.items():
        standard = build_prompt(task, "standard", question)
        cot = build_prompt(task, "cot", question)
        assert standard != cot
        for phrase in forbidden_reasoning[task]:
            assert phrase not in standard
            assert phrase in cot


def test_prompts_use_stable_final_answer_contract() -> None:
    question = "If there are 2 cars and 3 arrive, how many cars are there?"
    standard = build_prompt("gsm8k", "standard", question)
    cot = build_prompt("gsm8k", "cot", question)
    assert "Final answer:" in DEFAULT_SYSTEM_PROMPT
    assert "Final answer:" in standard
    assert "Final answer:" in cot
