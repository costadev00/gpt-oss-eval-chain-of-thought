from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from cot_eval.metrics import MetricRow
from cot_eval.types import Condition, Prediction, TaskName

TASK_LABELS = {
    "gsm8k": "GSM8K",
    "last_letter": "Last Letter",
    "coin_flip": "Coin Flip",
}

CONDITION_LABELS = {
    "standard": "Sem CoT",
    "cot": "CoT",
}


def percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f}%"


def signed_pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:+.1f} p.p."


def number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def metric_lookup(rows: list[MetricRow]) -> dict[tuple[TaskName, Condition], MetricRow]:
    return {(row.task, row.condition): row for row in rows}


def summarize_deltas(rows: list[MetricRow]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        if row.condition != "cot" or row.accuracy_delta_vs_standard is None:
            continue
        label = TASK_LABELS[row.task]
        delta_pp = row.accuracy_delta_vs_standard * 100
        if delta_pp > 2:
            verdict = "melhorou"
        elif delta_pp < -2:
            verdict = "piorou"
        else:
            verdict = "ficou praticamente empatado"
        lines.append(f"- {label}: CoT {verdict} em relacao ao baseline ({delta_pp:+.1f} p.p.).")
    return lines


def select_error_examples(predictions: list[Prediction], limit: int = 6) -> list[Prediction]:
    by_key: dict[tuple[TaskName, str], dict[Condition, Prediction]] = defaultdict(dict)
    for prediction in predictions:
        by_key[(prediction.task, prediction.item_id)][prediction.condition] = prediction

    examples: list[Prediction] = []
    for pair in by_key.values():
        standard = pair.get("standard")
        cot = pair.get("cot")
        if standard and cot and cot.correct and not standard.correct:
            examples.append(cot)
        if len(examples) >= limit:
            return examples[:limit]
    for prediction in predictions:
        if not prediction.correct:
            examples.append(prediction)
        if len(examples) >= limit:
            break
    return examples[:limit]


def write_markdown_analysis(path: Path, rows: list[MetricRow], predictions: list[Prediction]) -> None:
    lookup = metric_lookup(rows)
    tasks = sorted({row.task for row in rows})
    lines = [
        "# Analise critica: GPT-OSS-20B com e sem chain-of-thought",
        "",
        "## Resultados agregados",
        "",
        "| Tarefa | Sem CoT | CoT | Delta CoT | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        standard = lookup.get((task, "standard"))
        cot = lookup.get((task, "cot"))
        delta = cot.accuracy_delta_vs_standard if cot else None
        n = cot.n if cot else standard.n if standard else 0
        lines.append(
            f"| {TASK_LABELS[task]} | {percent(standard.accuracy if standard else None)} | "
            f"{percent(cot.accuracy if cot else None)} | {signed_pp(delta)} | {n} |"
        )

    lines.extend(["", "## Leitura critica", ""])
    deltas = summarize_deltas(rows)
    lines.extend(deltas or ["- Nao ha pares suficientes para comparar CoT contra o baseline."])
    lines.extend(
        [
            "- A comparacao isola o efeito do prompt: mesma API, mesmo modelo, mesma temperatura e mesmo esforco de reasoning configurado.",
            "- CoT tende a custar mais tokens e latencia; ganhos pequenos de acuracia devem ser lidos junto com a tabela de custo operacional.",
            "- Falhas de parsing contam como erro, pois em uso real uma resposta nao extraivel tambem quebra avaliacao automatica.",
            "- As tarefas simbolicas sao deterministicas e uteis para diagnostico, mas nao substituem benchmarks externos diversos.",
            "- O relatorio nao usa raw chain-of-thought interno do modelo; considera apenas resposta visivel, resposta extraida e metricas agregadas.",
        ]
    )

    lines.extend(["", "## Custo operacional", ""])
    lines.extend(["| Tarefa | Condicao | Latencia media (s) | Tokens prompt | Tokens resposta | Tokens totais |", "|---|---|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {TASK_LABELS[row.task]} | {CONDITION_LABELS[row.condition]} | "
            f"{number(row.avg_latency_s)} | {number(row.avg_prompt_tokens, 1)} | "
            f"{number(row.avg_completion_tokens, 1)} | {number(row.avg_total_tokens, 1)} |"
        )

    examples = select_error_examples(predictions)
    lines.extend(["", "## Exemplos para inspecao", ""])
    if not examples:
        lines.append("- Nenhum erro encontrado na amostra avaliada.")
    for example in examples:
        question = example.question.replace("\n", " ")
        if len(question) > 220:
            question = question[:217] + "..."
        lines.append(
            f"- {TASK_LABELS[example.task]} / {CONDITION_LABELS[example.condition]}: "
            f"gold={example.gold!r}, extraido={example.parsed_answer!r}, correto={example.correct}. "
            f"Pergunta: {question}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_latex_section(path: Path, rows: list[MetricRow]) -> None:
    lookup = metric_lookup(rows)
    tasks = sorted({row.task for row in rows})
    lines = [
        r"\section{Analise critica dos resultados}",
        "",
        "A Tabela~\\ref{tab:cot_accuracy} compara o desempenho do GPT-OSS-20B com prompting direto e com chain-of-thought. "
        "A Tabela~\\ref{tab:cot_cost} resume o custo operacional observado. "
        "A avaliacao considera apenas respostas visiveis e metricas agregadas; raw chain-of-thought interno do modelo nao foi exposto nem usado na analise.",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Acuracia por tarefa e condicao.}",
        r"\label{tab:cot_accuracy}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Tarefa & Sem CoT & CoT & Delta & IC 95\% CoT & N \\",
        r"\hline",
    ]
    for task in tasks:
        standard = lookup.get((task, "standard"))
        cot = lookup.get((task, "cot"))
        delta = cot.accuracy_delta_vs_standard if cot else None
        ci = "-" if cot is None else f"[{percent(cot.ci_low)}, {percent(cot.ci_high)}]"
        n = cot.n if cot else standard.n if standard else 0
        lines.append(
            f"{latex_escape(TASK_LABELS[task])} & {percent(standard.accuracy if standard else None)} & "
            f"{percent(cot.accuracy if cot else None)} & {latex_escape(signed_pp(delta))} & {latex_escape(ci)} & {n} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Custo operacional medio por item.}",
            r"\label{tab:cot_cost}",
            r"\begin{tabular}{llrrrr}",
            r"\hline",
            r"Tarefa & Condicao & Latencia (s) & Prompt tok. & Resposta tok. & Total tok. \\",
            r"\hline",
        ]
    )
    for row in rows:
        lines.append(
            f"{latex_escape(TASK_LABELS[row.task])} & {latex_escape(CONDITION_LABELS[row.condition])} & "
            f"{number(row.avg_latency_s)} & {number(row.avg_prompt_tokens, 1)} & "
            f"{number(row.avg_completion_tokens, 1)} & {number(row.avg_total_tokens, 1)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            "Criticamente, o ganho de chain-of-thought deve ser interpretado como uma troca entre desempenho e custo. "
            "Quando o delta e positivo em tarefas que exigem composicao de passos, o resultado e consistente com a hipotese do paper de que demonstracoes intermediarias ajudam o modelo a decompor problemas. "
            "Quando o delta e pequeno ou negativo, a explicacao mais plausivel e que a tarefa ja e simples o bastante para resposta direta, ou que o prompt longo introduz oportunidades adicionais de erro. "
            "As conclusoes tambem dependem do tamanho da amostra, da robustez do parser e da estabilidade do backend local em quatro GPUs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
