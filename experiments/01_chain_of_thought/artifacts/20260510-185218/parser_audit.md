# Auditoria do parser

Arquivo bruto avaliado na execução original:
`results/20260510-185218/predictions.jsonl`.

Este arquivo bruto não é preservado no repositório enxuto. O resumo da auditoria
fica versionado aqui junto dos demais artefatos do experimento CoT.

## Resumo

- Predicoes avaliadas: 800
- Extracoes alteradas pelo parser atual: 0
- Correcoes/incorrecoes alteradas: 0
- Falhas de parse antigas: 5
- Falhas de parse atuais: 5

## Metricas reprocessadas

| Tarefa | Condicao | Acuracia antiga | Acuracia atual | Delta | Parse failures antigo | Parse failures atual |
|---|---|---:|---:|---:|---:|---:|
| gsm8k | standard | 94.5% | 94.5% | +0.0 p.p. | 2 | 2 |
| gsm8k | cot | 96.5% | 96.5% | +0.0 p.p. | 3 | 3 |
| last_letter | standard | 85.0% | 85.0% | +0.0 p.p. | 0 | 0 |
| last_letter | cot | 84.0% | 84.0% | +0.0 p.p. | 0 | 0 |
| coin_flip | standard | 100.0% | 100.0% | +0.0 p.p. | 0 | 0 |
| coin_flip | cot | 100.0% | 100.0% | +0.0 p.p. | 0 | 0 |

## Mudancas de extracao

Nenhuma predicao mudou com o parser atual.

## Leitura

Esta auditoria nao reexecuta o modelo; ela apenas reaplica o parser atual sobre as respostas ja salvas. Por isso, diferencas aqui medem sensibilidade da metrica ao parser, nao mudanca real no comportamento do modelo.
Um parser bom deve preferir marcadores finais como `The answer is`, `Answer:`, `\boxed{...}` e negrito terminal, mas ainda assim evitar capturar numeros de explicacoes posteriores ou de unidades contextuais.
