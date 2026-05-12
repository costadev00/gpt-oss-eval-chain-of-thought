from __future__ import annotations

PRM_GENERATION_SYSTEM_PROMPT = (
    "You solve competition math problems. Write a concise step-by-step solution with one step per line. "
    "End with exactly one final line formatted as `Final answer: <answer>`, with no text after it. "
    "For this evaluation, the final answer must be numeric only, without units."
)


def build_generation_prompt(problem: str) -> str:
    return (
        "Solve the math problem. Use numbered steps, one step per line. "
        "Do not skip algebraic reasoning. End with `Final answer: <answer>`.\n\n"
        f"Problem:\n{problem}\n\nSolution:"
    )


def format_steps(steps: list[str]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps))


def build_step_judge_prompt(problem: str, previous_steps: list[str], step: str) -> str:
    return (
        "You are a process-supervision reward model for math solutions.\n"
        "Label the CURRENT STEP only.\n\n"
        "Labels:\n"
        "- positive: the current step is correct, reasonable, and makes progress.\n"
        "- neutral: the current step is correct and reasonable, but does not clearly make progress.\n"
        "- negative: the current step is incorrect, unreasonable, or invalid in context.\n\n"
        "Return exactly one label: positive, neutral, or negative.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Previous steps:\n{format_steps(previous_steps)}\n\n"
        f"Current step:\n{step}\n\n"
        "Label:"
    )
