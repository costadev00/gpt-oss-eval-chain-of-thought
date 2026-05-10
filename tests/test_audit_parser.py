import json
from dataclasses import asdict
from pathlib import Path

from cot_eval.audit_parser import audit_predictions, read_predictions, rescore_predictions
from cot_eval.types import Prediction


def make_prediction(response: str, parsed_answer: str | None, correct: bool) -> Prediction:
    return Prediction(
        task="gsm8k",
        condition="cot",
        item_id="gsm8k-test",
        question="How many calls are needed?",
        gold="750",
        parsed_answer=parsed_answer,
        correct=correct,
        parse_failed=parsed_answer is None,
        response=response,
        prompt_hash="abc123",
        latency_s=0.1,
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
        metadata={},
    )


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")


def test_rescore_predictions_updates_parser_sensitive_answer() -> None:
    prediction = make_prediction(
        "Answer: Jason would need to make 750 telephone calls to sell 15 cars.",
        parsed_answer="15",
        correct=False,
    )

    rescored = rescore_predictions([prediction])[0]

    assert rescored.parsed_answer == "750"
    assert rescored.correct is True
    assert rescored.parse_failed is False


def test_audit_predictions_writes_artifacts_without_mutating_original(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    prediction = make_prediction(
        "Answer: Jason would need to make 750 telephone calls to sell 15 cars.",
        parsed_answer="15",
        correct=False,
    )
    write_predictions(predictions_path, [prediction])

    changed, correctness_changed = audit_predictions(
        predictions_path=predictions_path,
        output_dir=tmp_path,
        seed=42,
        sample_limit=10,
        write_rescored=True,
    )

    assert changed == 1
    assert correctness_changed == 1
    assert (tmp_path / "parser_audit.md").exists()
    assert (tmp_path / "parser_audit_changes.csv").exists()
    assert (tmp_path / "metrics_rescored.csv").exists()
    assert (tmp_path / "predictions_rescored.jsonl").exists()

    original = read_predictions(predictions_path)[0]
    rescored = read_predictions(tmp_path / "predictions_rescored.jsonl")[0]
    assert original.parsed_answer == "15"
    assert rescored.parsed_answer == "750"
