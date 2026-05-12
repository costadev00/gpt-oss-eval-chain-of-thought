import math
from argparse import Namespace
from pathlib import Path

from cot_eval.math_tasks import MathItem, MathLoadResult
from cot_eval.prm_run import PRMCandidate, run_prm_evaluation, select_for_problem
from cot_eval.types import CompletionResult, TextCompletionLogprobsResult


class FakePRMClient:
    def __init__(self) -> None:
        self.generation_calls = 0

    def chat_complete(self, prompt: str, temperature: float, max_tokens: int) -> CompletionResult:
        self.generation_calls += 1
        if self.generation_calls % 2 == 1:
            content = "1. Compute incorrectly.\nFinal answer: 4"
        else:
            content = "1. Compute 2 + 3 = 5.\nFinal answer: 5"
        return CompletionResult(content=content, latency_s=0.01, prompt_tokens=20, completion_tokens=10, total_tokens=30)

    def completion_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> TextCompletionLogprobsResult:
        if "incorrectly" in prompt:
            top = {" negative": math.log(0.80), " positive": math.log(0.10), " neutral": math.log(0.05)}
        else:
            top = {" positive": math.log(0.80), " neutral": math.log(0.10), " negative": math.log(0.05)}
        return TextCompletionLogprobsResult(
            content=" positive",
            latency_s=0.01,
            tokens=[" positive"],
            token_logprobs=[math.log(0.8)],
            top_logprobs=[top],
            prompt_tokens=30,
            completion_tokens=1,
            total_tokens=31,
        )


def test_prm_run_writes_artifacts_and_selects_prm_best(tmp_path: Path) -> None:
    item = MathItem(
        item_id="math-test-0",
        problem="What is 2 + 3?",
        gold="5",
        config="algebra",
        level="Level 1",
        solution=r"\boxed{5}",
    )
    args = Namespace(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="openai/gpt-oss-20b",
        math_limit=1,
        samples_per_problem=2,
        math_configs=["algebra"],
        seed=42,
        max_tokens=128,
        generation_temperature=0.7,
        reasoning_effort="medium",
        top_logprobs=20,
        timeout=1,
        preflight_timeout=1,
        output_dir=tmp_path,
        write_incremental=False,
        skip_preflight=True,
    )

    output_dir = run_prm_evaluation(
        args,
        client=FakePRMClient(),
        math_load_result=MathLoadResult(items=[item], skipped=0, total_seen=1),
    )

    assert (output_dir / "config.json").exists()
    assert (output_dir / "candidates.jsonl").exists()
    assert (output_dir / "step_scores.jsonl").exists()
    assert (output_dir / "selected.jsonl").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "analysis.md").exists()
    assert (output_dir / "analysis_section.tex").exists()
    assert (output_dir / "final_report.tex").exists()

    selected = (output_dir / "selected.jsonl").read_text(encoding="utf-8")
    assert '"method": "prm_best_of_n"' in selected
    assert '"parsed_answer": "5"' in selected


def test_prm_selection_prefers_parseable_candidates() -> None:
    bad_unparseable = PRMCandidate(
        task="math",
        item_id="math-test",
        candidate_id="bad",
        sample_index=0,
        config="algebra",
        level="Level 1",
        problem="What is 2 + 3?",
        gold="5",
        response="partial",
        parsed_answer=None,
        correct=False,
        parse_failed=True,
        steps=["partial"],
        step_count=1,
        solution_score=0.9,
        solution_log_score=0.0,
        min_step_score=0.9,
        generation_latency_s=0.1,
        prm_latency_s=0.1,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prm_prompt_tokens=1,
        prm_completion_tokens=1,
    )
    good_parseable = PRMCandidate(
        task="math",
        item_id="math-test",
        candidate_id="good",
        sample_index=1,
        config="algebra",
        level="Level 1",
        problem="What is 2 + 3?",
        gold="5",
        response="Final answer: 5",
        parsed_answer="5",
        correct=True,
        parse_failed=False,
        steps=[],
        step_count=0,
        solution_score=0.1,
        solution_log_score=-10.0,
        min_step_score=0.1,
        generation_latency_s=0.1,
        prm_latency_s=0.1,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        prm_prompt_tokens=1,
        prm_completion_tokens=1,
    )

    selections = {selection.method: selection for selection in select_for_problem([bad_unparseable, good_parseable])}

    assert selections["prm_best_of_n"].candidate_id == "good"
