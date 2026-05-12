from __future__ import annotations

from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "first": "First sample",
    "majority_vote": "Majority vote",
    "prm_best_of_n": "PRM best-of-N",
    "oracle_best_of_n": "Oracle best-of-N",
}


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100 * value:.1f}%"


def num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != 0 and abs(numeric) < 10 ** -digits:
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def write_prm_markdown_analysis(
    path: Path,
    metrics: list[dict[str, Any]],
    candidate_stats: dict[str, Any],
    config: dict[str, Any],
) -> None:
    by_method = {str(row["method"]): row for row in metrics}
    prm_accuracy = float(by_method.get("prm_best_of_n", {}).get("accuracy", 0.0))
    majority_accuracy = float(by_method.get("majority_vote", {}).get("accuracy", 0.0))
    oracle_accuracy = float(by_method.get("oracle_best_of_n", {}).get("accuracy", 0.0))
    first_accuracy = float(by_method.get("first", {}).get("accuracy", 0.0))
    if prm_accuracy > majority_accuracy:
        prm_reading = "PRM best-of-N superou majority vote nesta rodada."
    elif prm_accuracy < majority_accuracy:
        prm_reading = "PRM best-of-N ficou abaixo de majority vote nesta rodada."
    else:
        prm_reading = "PRM best-of-N empatou com majority vote nesta rodada."

    lines = [
        "# Analise critica: Process Supervision promptado",
        "",
        "## Resultados agregados",
        "",
        "| Metodo | Acuracia | Corretas/N | Cobertura parse | Passos medios | Score PRM medio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        method = str(row["method"])
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {pct(float(row['accuracy']))} | "
            f"{row['correct']}/{row['n']} | {pct(float(row['parse_coverage']))} | "
            f"{num(row['avg_step_count'])} | {num(row['avg_solution_score'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Custo operacional",
            "",
            "| Metodo | Latencia geracao (s) | Latencia PRM (s) | Tokens geracao | Tokens PRM |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        method = str(row["method"])
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {num(row['avg_generation_latency_s'])} | "
            f"{num(row['avg_prm_latency_s'])} | {num(row['avg_generation_tokens'], 1)} | "
            f"{num(row['avg_prm_tokens'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Leitura critica",
            "",
            "- Esta implementacao usa um PRM promptado: o GPT-OSS-20B julga cada passo por prompt e logprobs; nenhum reward model foi fine-tunado em PRM800K.",
            "- O mesmo modelo gera e julga as solucoes, entao erros correlacionados podem fazer uma solucao incorreta parecer convincente ao proprio verificador.",
            "- A tarefa MATH foi filtrada para respostas numericas extraiveis de `\\boxed{...}`, uma simplificacao deliberada da v1.",
            f"- A rodada `{config['math_limit']}x{config['samples_per_problem']}` e exploratoria; resultados devem ser lidos como smoke experimental, nao como conclusao estatistica.",
            "- O baseline `oracle_best_of_n` e um teto superior: ele mostra o que seria possivel selecionar se o verificador fosse perfeito.",
            f"- {prm_reading} First={pct(first_accuracy)}, majority={pct(majority_accuracy)}, PRM={pct(prm_accuracy)}, oracle={pct(oracle_accuracy)}.",
            "",
            "## Diagnostico PRM",
            "",
            f"- Candidatos gerados: {candidate_stats['candidate_count']}.",
            f"- Candidatos corretos: {candidate_stats['correct_candidates']}.",
            f"- Score PRM medio de candidatos corretos: {num(candidate_stats['avg_score_correct'], 4)}.",
            f"- Score PRM medio de candidatos incorretos: {num(candidate_stats['avg_score_incorrect'], 4)}.",
            f"- Itens MATH pulados pelo filtro numeric-only: {config['math_skipped']} de {config['math_total_seen']} vistos.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def write_prm_latex_section(path: Path, metrics: list[dict[str, Any]], candidate_stats: dict[str, Any]) -> None:
    lines = [
        r"\section{Analise critica do Process Supervision promptado}",
        "",
        "A avaliacao implementa uma aproximacao promptada de PRM: o GPT-OSS-20B gera solucoes e tambem julga cada passo via logprobs. "
        "Assim, os resultados medem a utilidade de um verificador promptado, nao de um reward model treinado como em PRM800K.",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Acuracia por metodo de selecao.}",
        r"\label{tab:prm_accuracy}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Metodo & Acuracia & Corretas/N & Cobertura parse & Score PRM \\",
        r"\hline",
    ]
    for row in metrics:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        lines.append(
            f"{latex_escape(method)} & {pct(float(row['accuracy']))} & {row['correct']}/{row['n']} & "
            f"{pct(float(row['parse_coverage']))} & {num(row['avg_solution_score'], 4)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Custo medio por problema selecionado.}",
            r"\label{tab:prm_cost}",
            r"\begin{tabular}{lrrrr}",
            r"\hline",
            r"Metodo & Lat. geracao & Lat. PRM & Tok. geracao & Tok. PRM \\",
            r"\hline",
        ]
    )
    for row in metrics:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        lines.append(
            f"{latex_escape(method)} & {num(row['avg_generation_latency_s'])} & {num(row['avg_prm_latency_s'])} & "
            f"{num(row['avg_generation_tokens'], 1)} & {num(row['avg_prm_tokens'], 1)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
            "Criticamente, o PRM promptado deve ser interpretado como uma simulacao operacional de process supervision. "
            "Ele fornece feedback por passo e usa o produto dos scores de passos como no paper, mas nao foi treinado com labels humanos. "
            f"Nesta rodada, o score medio de candidatos corretos foi {num(candidate_stats['avg_score_correct'], 4)}, "
            f"contra {num(candidate_stats['avg_score_incorrect'], 4)} para candidatos incorretos. "
            "Se a separacao for pequena ou invertida, isso indica que o verificador promptado ainda nao e confiavel para best-of-N.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prm_final_report(
    path: Path,
    metrics: list[dict[str, Any]],
    candidate_stats: dict[str, Any],
    config: dict[str, Any],
) -> None:
    by_method = {str(row["method"]): row for row in metrics}
    prm_accuracy = float(by_method.get("prm_best_of_n", {}).get("accuracy", 0.0))
    majority_accuracy = float(by_method.get("majority_vote", {}).get("accuracy", 0.0))
    oracle_accuracy = float(by_method.get("oracle_best_of_n", {}).get("accuracy", 0.0))
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\title{Process Supervision promptado com GPT-OSS-20B}",
        r"\author{Evaluation harness local com vLLM}",
        rf"\date{{Rodada: {latex_escape(str(config['timestamp']))}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        "Este relatorio avalia uma aproximacao promptada de Process Supervision inspirada em \\emph{Let's Verify Step by Step}. "
        "O GPT-OSS-20B gera multiplas solucoes para problemas MATH e tambem julga cada passo como positivo, neutro ou negativo. "
        "O score de cada solucao e o produto dos scores dos passos, tratando neutro como positivo.",
        r"\end{abstract}",
        "",
        r"\section{Configuracao}",
        f"Foram avaliados {config['math_limit']} problemas MATH numeric-only, com {config['samples_per_problem']} solucoes por problema. "
        "A execucao usa vLLM local, temperatura de geracao configuravel e scoring PRM via logprobs no endpoint de completions.",
        "",
    ]
    section_path = path.with_name("_tmp_prm_section.tex")
    write_prm_latex_section(section_path, metrics, candidate_stats)
    section_text = section_path.read_text(encoding="utf-8")
    section_path.unlink(missing_ok=True)
    lines.append(section_text)
    lines.extend(
        [
            r"\section{Limitacoes}",
            "Esta v1 nao treina um PRM real e nao usa labels humanos de PRM800K. "
            "Como o mesmo modelo gera e julga, ha risco de vieses e erros correlacionados. "
            "A filtragem para respostas numericas simplifica MATH e reduz a cobertura do benchmark. "
            "A rodada inicial e pequena e deve ser ampliada para 50x8 ou mais antes de conclusoes robustas.",
            "",
            r"\section{Conclusao}",
            f"Nesta rodada, PRM best-of-N obteve {pct(prm_accuracy)}, majority vote obteve {pct(majority_accuracy)} "
            f"e oracle best-of-N obteve {pct(oracle_accuracy)}. "
            "Assim, o PRM promptado ainda nao demonstrou ganho seletivo sobre baselines simples nesta amostra. "
            "O resultado e util como diagnostico: o pipeline de process supervision funciona, mas o verificador promptado precisa de calibracao, mais amostras ou treinamento dedicado para se aproximar do comportamento de um PRM real.",
            r"\end{document}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
