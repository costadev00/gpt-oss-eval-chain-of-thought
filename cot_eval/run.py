from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from cot_eval.client import OpenAIChatClient
from cot_eval.metrics import aggregate_metrics, write_metrics_csv
from cot_eval.prompts import build_prompt, prompt_hash
from cot_eval.reporting import write_latex_section, write_markdown_analysis
from cot_eval.scoring import answers_equal, parse_answer
from cot_eval.tasks import load_items
from cot_eval.types import CompletionResult, Condition, EvalItem, Prediction, TaskName


class CompletionClient(Protocol):
    def complete(self, prompt: str) -> CompletionResult:
        ...


def parse_tasks(value: str) -> list[TaskName]:
    tasks = [task.strip() for task in value.split(",") if task.strip()]
    allowed = {"gsm8k", "last_letter", "coin_flip"}
    unknown = sorted(set(tasks) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown tasks: {', '.join(unknown)}")
    return [task for task in tasks]  # type: ignore[return-value]


def make_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def evaluate_item(client: CompletionClient, item: EvalItem, condition: Condition) -> Prediction:
    prompt = build_prompt(item.task, condition, item.question)
    result = client.complete(prompt)
    parsed = parse_answer(item.task, result.content)
    correct = answers_equal(item.task, parsed, item.gold)
    return Prediction(
        task=item.task,
        condition=condition,
        item_id=item.item_id,
        question=item.question,
        gold=item.gold,
        parsed_answer=parsed,
        correct=correct,
        parse_failed=parsed is None,
        response=result.content,
        prompt_hash=prompt_hash(prompt),
        latency_s=result.latency_s,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        metadata={
            "retried_without_extra_body": result.retried_without_extra_body,
            "discarded_reasoning": result.discarded_reasoning,
        },
    )


def write_predictions(path: Path, predictions: list[Prediction]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")


def run_evaluation(args: argparse.Namespace, client: CompletionClient | None = None) -> Path:
    items = load_items(args.tasks, args.gsm8k_limit, args.symbolic_limit, args.seed)
    output_dir = make_output_dir(args.output_dir)
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "tasks": args.tasks,
        "gsm8k_limit": args.gsm8k_limit,
        "symbolic_limit": args.symbolic_limit,
        "seed": args.seed,
        "conditions": args.conditions,
        "temperature": 0,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "backend": "vllm-openai-compatible",
        "tensor_parallel_size": 4,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if client is None:
        client = OpenAIChatClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )

    predictions: list[Prediction] = []
    total = len(items) * len(args.conditions)
    progress = tqdm(total=total, desc="Evaluating", unit="prompt")
    try:
        for item in items:
            for condition in args.conditions:
                predictions.append(evaluate_item(client, item, condition))
                progress.update(1)
                if args.write_incremental:
                    write_predictions(output_dir / "predictions.jsonl", predictions)
    finally:
        progress.close()

    metrics = aggregate_metrics(predictions, seed=args.seed)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    write_markdown_analysis(output_dir / "analysis.md", metrics, predictions)
    write_latex_section(output_dir / "analysis_section.tex", metrics)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate gpt-oss-20b with and without chain-of-thought prompts.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key for the endpoint.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name served by vLLM.")
    parser.add_argument("--tasks", type=parse_tasks, default=parse_tasks("gsm8k,last_letter,coin_flip"))
    parser.add_argument("--conditions", type=lambda value: [item.strip() for item in value.split(",")], default=["standard", "cot"])
    parser.add_argument("--gsm8k-limit", type=int, default=200, help="Number of GSM8K test examples to sample.")
    parser.add_argument("--symbolic-limit", type=int, default=100, help="Number of generated examples per symbolic task.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--write-incremental", action="store_true", help="Rewrite predictions.jsonl after every prompt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    invalid_conditions = sorted(set(args.conditions) - {"standard", "cot"})
    if invalid_conditions:
        parser.error(f"Unknown conditions: {', '.join(invalid_conditions)}")
    args.conditions = [condition for condition in args.conditions]  # argparse stores strings; values validated above.
    output_dir = run_evaluation(args)
    print(f"Wrote evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
