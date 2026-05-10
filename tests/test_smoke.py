from argparse import Namespace
from pathlib import Path

from cot_eval.run import run_evaluation
from cot_eval.types import CompletionResult, EvalItem


class FakeClient:
    def complete(self, prompt: str) -> CompletionResult:
        if "coin still heads up" in prompt:
            content = "The answer is yes."
        elif "last letters" in prompt:
            content = "The answer is aa."
        else:
            content = "The answer is 5."
        return CompletionResult(
            content=content,
            latency_s=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def test_run_evaluation_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fake_load_items(tasks, gsm8k_limit, symbolic_limit, seed):
        return [
            EvalItem(task="gsm8k", item_id="gsm8k-0", question="What is 2 + 3?", gold="5"),
            EvalItem(task="coin_flip", item_id="coin_flip-0", question="A coin is heads up. Is the coin still heads up?", gold="yes"),
            EvalItem(task="last_letter", item_id="last_letter-0", question='Take the last letters of the words in "Ada Lovelace".', gold="ae"),
        ]

    monkeypatch.setattr("cot_eval.run.load_items", fake_load_items)
    args = Namespace(
        tasks=["gsm8k", "coin_flip", "last_letter"],
        gsm8k_limit=1,
        symbolic_limit=1,
        seed=42,
        output_dir=tmp_path,
        model="openai/gpt-oss-20b",
        base_url="http://localhost:8000/v1",
        conditions=["standard", "cot"],
        reasoning_effort="medium",
        max_tokens=128,
        system_prompt="Always end with Final answer: <answer>.",
        api_key="EMPTY",
        timeout=1,
        preflight_timeout=1,
        write_incremental=False,
        skip_preflight=False,
    )

    output_dir = run_evaluation(args, client=FakeClient())

    assert (output_dir / "config.json").exists()
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "analysis.md").exists()
    assert (output_dir / "analysis_section.tex").exists()
