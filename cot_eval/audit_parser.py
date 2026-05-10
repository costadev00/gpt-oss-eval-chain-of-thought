from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from cot_eval.metrics import MetricRow, aggregate_metrics, write_metrics_csv
from cot_eval.scoring import answers_equal, parse_answer
from cot_eval.types import Prediction


def resolve_predictions_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path


def read_predictions(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: dict[str, Any] = json.loads(line)
                predictions.append(Prediction(**raw))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_number}: {exc}") from exc
    return predictions


def read_seed(run_dir: Path, default: int = 42) -> int:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return default
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    seed = config.get("seed", default)
    return seed if isinstance(seed, int) else default


def reparse_prediction(prediction: Prediction) -> Prediction:
    parsed = parse_answer(prediction.task, prediction.response)
    correct = answers_equal(prediction.task, parsed, prediction.gold)
    return replace(
        prediction,
        parsed_answer=parsed,
        correct=correct,
        parse_failed=parsed is None,
    )


def rescore_predictions(predictions: list[Prediction]) -> list[Prediction]:
    return [reparse_prediction(prediction) for prediction in predictions]


def metric_key(row: MetricRow) -> tuple[str, str]:
    return row.task, row.condition


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f}%"


def format_pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:+.1f} p.p."


def excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def markdown_cell(value: str) -> str:
    return value.replace("|", r"\|")


def changed_rows(old_predictions: list[Prediction], new_predictions: list[Prediction]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for old, new in zip(old_predictions, new_predictions, strict=True):
        parsed_changed = old.parsed_answer != new.parsed_answer
        correct_changed = old.correct != new.correct
        parse_failure_changed = old.parse_failed != new.parse_failed
        if not (parsed_changed or correct_changed or parse_failure_changed):
            continue
        rows.append(
            {
                "task": old.task,
                "condition": old.condition,
                "item_id": old.item_id,
                "gold": old.gold,
                "old_parsed": "" if old.parsed_answer is None else old.parsed_answer,
                "new_parsed": "" if new.parsed_answer is None else new.parsed_answer,
                "old_correct": str(old.correct),
                "new_correct": str(new.correct),
                "old_parse_failed": str(old.parse_failed),
                "new_parse_failed": str(new.parse_failed),
                "response_excerpt": excerpt(old.response),
            }
        )
    return rows


def write_changes_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "task",
        "condition",
        "item_id",
        "gold",
        "old_parsed",
        "new_parsed",
        "old_correct",
        "new_correct",
        "old_parse_failed",
        "new_parse_failed",
        "response_excerpt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rescored_predictions(path: Path, predictions: list[Prediction]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")


def write_markdown_audit(
    path: Path,
    predictions_path: Path,
    old_predictions: list[Prediction],
    new_predictions: list[Prediction],
    old_metrics: list[MetricRow],
    new_metrics: list[MetricRow],
    changes: list[dict[str, str]],
    sample_limit: int,
) -> None:
    old_lookup = {metric_key(row): row for row in old_metrics}
    new_lookup = {metric_key(row): row for row in new_metrics}
    task_order = {"gsm8k": 0, "last_letter": 1, "coin_flip": 2}
    condition_order = {"standard": 0, "cot": 1}
    keys = sorted(old_lookup.keys(), key=lambda key: (task_order.get(key[0], 99), condition_order.get(key[1], 99)))

    old_parse_failures = sum(prediction.parse_failed for prediction in old_predictions)
    new_parse_failures = sum(prediction.parse_failed for prediction in new_predictions)
    correctness_changes = sum(old.correct != new.correct for old, new in zip(old_predictions, new_predictions, strict=True))
    parsed_changes = sum(
        old.parsed_answer != new.parsed_answer for old, new in zip(old_predictions, new_predictions, strict=True)
    )

    lines = [
        "# Auditoria do parser",
        "",
        f"Arquivo avaliado: `{predictions_path}`",
        "",
        "## Resumo",
        "",
        f"- Predicoes avaliadas: {len(old_predictions)}",
        f"- Extracoes alteradas pelo parser atual: {parsed_changes}",
        f"- Correcoes/incorrecoes alteradas: {correctness_changes}",
        f"- Falhas de parse antigas: {old_parse_failures}",
        f"- Falhas de parse atuais: {new_parse_failures}",
        "",
        "## Metricas reprocessadas",
        "",
        "| Tarefa | Condicao | Acuracia antiga | Acuracia atual | Delta | Parse failures antigo | Parse failures atual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in keys:
        old = old_lookup[key]
        new = new_lookup[key]
        lines.append(
            f"| {old.task} | {old.condition} | {format_percent(old.accuracy)} | "
            f"{format_percent(new.accuracy)} | {format_pp(new.accuracy - old.accuracy)} | "
            f"{old.parse_failures} | {new.parse_failures} |"
        )

    lines.extend(["", "## Mudancas de extracao", ""])
    if not changes:
        lines.append("Nenhuma predicao mudou com o parser atual.")
    else:
        lines.extend(
            [
                "| Tarefa | Condicao | Item | Gold | Antigo | Atual | Correto antigo | Correto atual | Resposta |",
                "|---|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for row in changes[:sample_limit]:
            lines.append(
                f"| {markdown_cell(row['task'])} | {markdown_cell(row['condition'])} | "
                f"{markdown_cell(row['item_id'])} | {markdown_cell(row['gold'])} | "
                f"{markdown_cell(row['old_parsed'] or '-')} | {markdown_cell(row['new_parsed'] or '-')} | "
                f"{row['old_correct']} | {row['new_correct']} | {markdown_cell(row['response_excerpt'])} |"
            )
        if len(changes) > sample_limit:
            lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | mais {len(changes) - sample_limit} mudancas |")

    lines.extend(
        [
            "",
            "## Leitura",
            "",
            "Esta auditoria nao reexecuta o modelo; ela apenas reaplica o parser atual sobre as respostas ja salvas. "
            "Por isso, diferencas aqui medem sensibilidade da metrica ao parser, nao mudanca real no comportamento do modelo.",
            "Um parser bom deve preferir marcadores finais como `The answer is`, `Answer:`, `\\boxed{...}` e negrito terminal, "
            "mas ainda assim evitar capturar numeros de explicacoes posteriores ou de unidades contextuais.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_predictions(
    predictions_path: Path,
    output_dir: Path,
    seed: int,
    sample_limit: int,
    write_rescored: bool,
) -> tuple[int, int]:
    old_predictions = read_predictions(predictions_path)
    new_predictions = rescore_predictions(old_predictions)
    old_metrics = aggregate_metrics(old_predictions, seed=seed)
    new_metrics = aggregate_metrics(new_predictions, seed=seed)
    changes = changed_rows(old_predictions, new_predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(output_dir / "metrics_rescored.csv", new_metrics)
    write_changes_csv(output_dir / "parser_audit_changes.csv", changes)
    write_markdown_audit(
        output_dir / "parser_audit.md",
        predictions_path,
        old_predictions,
        new_predictions,
        old_metrics,
        new_metrics,
        changes,
        sample_limit,
    )
    if write_rescored:
        write_rescored_predictions(output_dir / "predictions_rescored.jsonl", new_predictions)
    correctness_changes = sum(old.correct != new.correct for old, new in zip(old_predictions, new_predictions, strict=True))
    return len(changes), correctness_changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit parser sensitivity for a saved evaluation run.")
    parser.add_argument("path", type=Path, help="Run directory or predictions.jsonl path.")
    parser.add_argument("--output-dir", type=Path, help="Where to write audit artifacts. Defaults to the run directory.")
    parser.add_argument("--seed", type=int, help="Bootstrap seed. Defaults to config.json seed or 42.")
    parser.add_argument("--sample-limit", type=int, default=20, help="Max changed examples shown in parser_audit.md.")
    parser.add_argument(
        "--no-rescored-predictions",
        action="store_true",
        help="Do not write predictions_rescored.jsonl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    predictions_path = resolve_predictions_path(args.path)
    if not predictions_path.exists():
        parser.error(f"Missing predictions file: {predictions_path}")

    run_dir = predictions_path.parent
    output_dir = args.output_dir or run_dir
    seed = args.seed if args.seed is not None else read_seed(run_dir)

    try:
        changed, correctness_changed = audit_predictions(
            predictions_path=predictions_path,
            output_dir=output_dir,
            seed=seed,
            sample_limit=args.sample_limit,
            write_rescored=not args.no_rescored_predictions,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"Wrote parser audit to {output_dir}. "
        f"Changed parses: {changed}; changed correctness labels: {correctness_changed}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
