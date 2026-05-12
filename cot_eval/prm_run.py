from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Protocol

from tqdm import tqdm

from cot_eval.client import EndpointUnavailableError, OpenAIChatClient
from cot_eval.math_tasks import MATH_CONFIGS, MathItem, MathLoadResult, load_math_items, parse_generated_math_answer
from cot_eval.prm_prompts import PRM_GENERATION_SYSTEM_PROMPT, build_generation_prompt, build_step_judge_prompt
from cot_eval.prm_reporting import write_prm_final_report, write_prm_latex_section, write_prm_markdown_analysis
from cot_eval.prm_scoring import StepScore, majority_vote_answer, make_step_score, reduce_step_scores, split_solution_steps
from cot_eval.scoring import answers_equal
from cot_eval.types import CompletionResult, TextCompletionLogprobsResult


class PRMClient(Protocol):
    def chat_complete(self, prompt: str, temperature: float, max_tokens: int) -> CompletionResult:
        ...

    def completion_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 0,
        stop: list[str] | None = None,
    ) -> TextCompletionLogprobsResult:
        ...


@dataclass(frozen=True)
class PRMCandidate:
    task: str
    item_id: str
    candidate_id: str
    sample_index: int
    config: str
    level: str
    problem: str
    gold: str
    response: str
    parsed_answer: str | None
    correct: bool
    parse_failed: bool
    steps: list[str]
    step_count: int
    solution_score: float
    solution_log_score: float
    min_step_score: float
    generation_latency_s: float
    prm_latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prm_prompt_tokens: int
    prm_completion_tokens: int


@dataclass(frozen=True)
class PRMSelection:
    method: str
    item_id: str
    candidate_id: str | None
    problem: str
    gold: str
    parsed_answer: str | None
    correct: bool
    parse_failed: bool
    response: str
    solution_score: float | None
    solution_log_score: float | None
    min_step_score: float | None
    step_count: int | None
    generation_latency_s: float
    prm_latency_s: float
    generation_tokens: int
    prm_tokens: int


def parse_math_configs(value: str) -> list[str]:
    configs = [config.strip() for config in value.split(",") if config.strip()]
    unknown = sorted(set(configs) - set(MATH_CONFIGS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown MATH configs: {', '.join(unknown)}")
    return configs


def make_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def optional_sum(values: list[int | None]) -> int:
    return sum(value for value in values if value is not None)


def optional_mean(values: list[int | float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def is_math_correct(parsed_answer: str | None, gold: str) -> bool:
    return answers_equal("gsm8k", parsed_answer, gold)


def generate_candidate(
    client: PRMClient,
    item: MathItem,
    sample_index: int,
    temperature: float,
    max_tokens: int,
) -> PRMCandidate:
    result = client.chat_complete(build_generation_prompt(item.problem), temperature=temperature, max_tokens=max_tokens)
    parsed = parse_generated_math_answer(result.content)
    correct = is_math_correct(parsed, item.gold)
    steps = split_solution_steps(result.content)
    return PRMCandidate(
        task="math",
        item_id=item.item_id,
        candidate_id=f"{item.item_id}-candidate-{sample_index}",
        sample_index=sample_index,
        config=item.config,
        level=item.level,
        problem=item.problem,
        gold=item.gold,
        response=result.content,
        parsed_answer=parsed,
        correct=correct,
        parse_failed=parsed is None,
        steps=steps,
        step_count=len(steps),
        solution_score=0.0,
        solution_log_score=-math.inf,
        min_step_score=0.0,
        generation_latency_s=result.latency_s,
        prm_latency_s=0.0,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        prm_prompt_tokens=0,
        prm_completion_tokens=0,
    )


def score_candidate_steps(client: PRMClient, candidate: PRMCandidate, top_logprobs: int) -> tuple[PRMCandidate, list[StepScore]]:
    step_scores: list[StepScore] = []
    for step_index, step in enumerate(candidate.steps):
        previous_steps = candidate.steps[:step_index]
        prompt = build_step_judge_prompt(candidate.problem, previous_steps, step)
        result = client.completion_with_logprobs(
            prompt,
            max_tokens=1,
            top_logprobs=top_logprobs,
            temperature=0,
            stop=["\n"],
        )
        step_scores.append(
            make_step_score(
                task=candidate.task,
                item_id=candidate.item_id,
                candidate_id=candidate.candidate_id,
                step_index=step_index,
                problem=candidate.problem,
                previous_steps=previous_steps,
                step=step,
                result=result,
            )
        )

    solution_score = reduce_step_scores(step_scores)
    scored_candidate = replace(
        candidate,
        solution_score=solution_score.solution_score,
        solution_log_score=solution_score.solution_log_score,
        min_step_score=solution_score.min_step_score,
        step_count=solution_score.step_count,
        prm_latency_s=sum(score.latency_s for score in step_scores),
        prm_prompt_tokens=optional_sum([score.prompt_tokens for score in step_scores]),
        prm_completion_tokens=optional_sum([score.completion_tokens for score in step_scores]),
    )
    return scored_candidate, step_scores


def candidate_costs(candidates: list[PRMCandidate]) -> tuple[float, int]:
    latency = sum(candidate.generation_latency_s for candidate in candidates)
    tokens = optional_sum([candidate.total_tokens for candidate in candidates])
    return latency, tokens


def prm_costs(candidates: list[PRMCandidate]) -> tuple[float, int]:
    latency = sum(candidate.prm_latency_s for candidate in candidates)
    tokens = sum(candidate.prm_prompt_tokens + candidate.prm_completion_tokens for candidate in candidates)
    return latency, tokens


def select_for_problem(candidates: list[PRMCandidate]) -> list[PRMSelection]:
    if not candidates:
        return []
    total_generation_latency, total_generation_tokens = candidate_costs(candidates)
    total_prm_latency, total_prm_tokens = prm_costs(candidates)
    first = candidates[0]
    majority_answer = majority_vote_answer([candidate.parsed_answer for candidate in candidates])
    majority_candidate = next((candidate for candidate in candidates if candidate.parsed_answer == majority_answer), first)
    rankable_candidates = [candidate for candidate in candidates if not candidate.parse_failed] or candidates
    prm_best = max(rankable_candidates, key=lambda candidate: candidate.solution_log_score)
    oracle = next((candidate for candidate in candidates if candidate.correct), first)

    selections = [
        make_selection("first", first, first.generation_latency_s, 0.0, first.total_tokens or 0, 0, parsed_override=None),
        make_selection(
            "majority_vote",
            majority_candidate,
            total_generation_latency,
            0.0,
            total_generation_tokens,
            0,
            parsed_override=majority_answer,
        ),
        make_selection(
            "prm_best_of_n",
            prm_best,
            total_generation_latency,
            total_prm_latency,
            total_generation_tokens,
            total_prm_tokens,
            parsed_override=None,
        ),
        make_selection(
            "oracle_best_of_n",
            oracle,
            total_generation_latency,
            0.0,
            total_generation_tokens,
            0,
            parsed_override=None,
        ),
    ]
    return selections


def make_selection(
    method: str,
    candidate: PRMCandidate,
    generation_latency_s: float,
    prm_latency_s: float,
    generation_tokens: int,
    prm_tokens: int,
    parsed_override: str | None,
) -> PRMSelection:
    parsed = candidate.parsed_answer if parsed_override is None else parsed_override
    return PRMSelection(
        method=method,
        item_id=candidate.item_id,
        candidate_id=candidate.candidate_id,
        problem=candidate.problem,
        gold=candidate.gold,
        parsed_answer=parsed,
        correct=is_math_correct(parsed, candidate.gold),
        parse_failed=parsed is None,
        response=candidate.response,
        solution_score=candidate.solution_score,
        solution_log_score=candidate.solution_log_score,
        min_step_score=candidate.min_step_score,
        step_count=candidate.step_count,
        generation_latency_s=generation_latency_s,
        prm_latency_s=prm_latency_s,
        generation_tokens=generation_tokens,
        prm_tokens=prm_tokens,
    )


def aggregate_selection_metrics(selections: list[PRMSelection]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    method_order = ["first", "majority_vote", "prm_best_of_n", "oracle_best_of_n"]
    for method in method_order:
        group = [selection for selection in selections if selection.method == method]
        if not group:
            continue
        n = len(group)
        correct = sum(1 for selection in group if selection.correct)
        parse_failures = sum(1 for selection in group if selection.parse_failed)
        rows.append(
            {
                "method": method,
                "n": n,
                "correct": correct,
                "accuracy": correct / n,
                "parse_failures": parse_failures,
                "parse_coverage": (n - parse_failures) / n,
                "avg_step_count": optional_mean([selection.step_count for selection in group]),
                "avg_solution_score": optional_mean([selection.solution_score for selection in group]),
                "avg_min_step_score": optional_mean([selection.min_step_score for selection in group]),
                "avg_generation_latency_s": optional_mean([selection.generation_latency_s for selection in group]),
                "avg_prm_latency_s": optional_mean([selection.prm_latency_s for selection in group]),
                "avg_generation_tokens": optional_mean([selection.generation_tokens for selection in group]),
                "avg_prm_tokens": optional_mean([selection.prm_tokens for selection in group]),
            }
        )
    return rows


def candidate_stats(candidates: list[PRMCandidate]) -> dict[str, object]:
    correct_scores = [candidate.solution_score for candidate in candidates if candidate.correct]
    incorrect_scores = [candidate.solution_score for candidate in candidates if not candidate.correct]
    return {
        "candidate_count": len(candidates),
        "correct_candidates": len(correct_scores),
        "incorrect_candidates": len(incorrect_scores),
        "avg_score_correct": optional_mean(correct_scores),
        "avg_score_incorrect": optional_mean(incorrect_scores),
        "avg_steps": optional_mean([candidate.step_count for candidate in candidates]),
    }


def write_jsonl(path: Path, records: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "n",
        "correct",
        "accuracy",
        "parse_failures",
        "parse_coverage",
        "avg_step_count",
        "avg_solution_score",
        "avg_min_step_score",
        "avg_generation_latency_s",
        "avg_prm_latency_s",
        "avg_generation_tokens",
        "avg_prm_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(output_dir: Path, config: dict[str, object], candidates: list[PRMCandidate], step_scores: list[StepScore], selections: list[PRMSelection]) -> None:
    metrics = aggregate_selection_metrics(selections)
    stats = candidate_stats(candidates)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_jsonl(output_dir / "step_scores.jsonl", step_scores)
    write_jsonl(output_dir / "selected.jsonl", selections)
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    write_prm_markdown_analysis(output_dir / "analysis.md", metrics, stats, config)
    write_prm_latex_section(output_dir / "analysis_section.tex", metrics, stats)
    write_prm_final_report(output_dir / "final_report.tex", metrics, stats, config)


def run_prm_evaluation(args: argparse.Namespace, client: PRMClient | None = None, math_load_result: MathLoadResult | None = None) -> Path:
    if client is None:
        client = OpenAIChatClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            system_prompt=PRM_GENERATION_SYSTEM_PROMPT,
        )
        if not args.skip_preflight:
            client.check_connection(timeout=args.preflight_timeout)

    if math_load_result is None:
        math_load_result = load_math_items(args.math_limit, args.seed, configs=args.math_configs)

    output_dir = make_output_dir(args.output_dir)
    config: dict[str, object] = {
        "timestamp": output_dir.name,
        "model": args.model,
        "base_url": args.base_url,
        "benchmark": "EleutherAI/hendrycks_math",
        "math_configs": args.math_configs,
        "math_limit": args.math_limit,
        "samples_per_problem": args.samples_per_problem,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "generation_temperature": args.generation_temperature,
        "reasoning_effort": args.reasoning_effort,
        "top_logprobs": args.top_logprobs,
        "math_skipped": math_load_result.skipped,
        "math_total_seen": math_load_result.total_seen,
        "math_loaded": len(math_load_result.items),
        "prm_type": "prompted_gpt_oss_20b",
        "solution_score": "product_step_scores_neutral_as_positive",
        "backend": "vllm-openai-compatible",
        "tensor_parallel_size": 4,
    }

    candidates: list[PRMCandidate] = []
    step_scores: list[StepScore] = []
    selections: list[PRMSelection] = []
    total_generations = len(math_load_result.items) * args.samples_per_problem
    progress = tqdm(total=total_generations, desc="PRM eval", unit="candidate")
    try:
        for item in math_load_result.items:
            problem_candidates: list[PRMCandidate] = []
            for sample_index in range(args.samples_per_problem):
                candidate = generate_candidate(
                    client=client,
                    item=item,
                    sample_index=sample_index,
                    temperature=args.generation_temperature,
                    max_tokens=args.max_tokens,
                )
                candidate, candidate_step_scores = score_candidate_steps(client, candidate, top_logprobs=args.top_logprobs)
                candidates.append(candidate)
                problem_candidates.append(candidate)
                step_scores.extend(candidate_step_scores)
                progress.update(1)
                if args.write_incremental:
                    partial_selections = selections + select_for_problem(problem_candidates)
                    write_outputs(output_dir, config, candidates, step_scores, partial_selections)
            selections.extend(select_for_problem(problem_candidates))
            if args.write_incremental:
                write_outputs(output_dir, config, candidates, step_scores, selections)
    finally:
        progress.close()

    write_outputs(output_dir, config, candidates, step_scores, selections)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prompted process supervision with GPT-OSS-20B.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"), help="API key for the endpoint.")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--math-limit", type=int, default=10)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--math-configs", type=parse_math_configs, default=MATH_CONFIGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--preflight-timeout", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results_prm"))
    parser.add_argument("--write-incremental", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = run_prm_evaluation(args)
    except EndpointUnavailableError as exc:
        print(f"Endpoint unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote PRM evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
