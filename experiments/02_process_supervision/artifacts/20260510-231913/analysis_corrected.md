# Analise corrigida: Process Supervision promptado

Esta analise consolida a rodada `20260510-231913` apos a correcao do parser
para respostas finais fracionarias, como `1/3`, `\frac{5}{9}` e expressoes com
`\displaystyle`. A fonte completa desta leitura e o relatorio consolidado em
`reports/gpt_oss_20b_cot_prm_report.tex`.

## Resultados agregados

| Metodo | Corretas/N | Acuracia | Falhas parse | Cobertura parse |
|---|---:|---:|---:|---:|
| First sample | 43/50 | 86.0% | 5 | 90.0% |
| Majority vote | 48/50 | 96.0% | 0 | 100.0% |
| PRM best-of-N | 48/50 | 96.0% | 0 | 100.0% |
| Oracle best-of-N | 49/50 | 98.0% | 1 | 98.0% |

## Custo operacional

| Metodo | Latencia geracao (s) | Latencia PRM (s) | Tokens geracao | Tokens PRM |
|---|---:|---:|---:|---:|
| First sample | 7.94 | 0.00 | 1794.4 | 0.0 |
| Majority vote | 61.67 | 0.00 | 14017.2 | 0.0 |
| PRM best-of-N | 61.67 | 2.14 | 14017.2 | 39707.4 |
| Oracle best-of-N | 61.67 | 0.00 | 14017.2 | 0.0 |

## Leitura critica

Gerar oito solucoes por problema aumentou substancialmente a acuracia em
relacao a usar apenas a primeira amostra: 86.0% para 96.0% com majority vote.
O PRM promptado empatou com majority vote em 48/50, mas adicionou cerca de
39707 tokens de verificacao por problema. Portanto, nesta configuracao, o
verificador por passos funcionou como demonstracao didatica de Process
Supervision, mas nao demonstrou ganho pratico sobre um baseline simples.

O oracle best-of-N chegou a 49/50, indicando que ainda havia margem para um
seletor melhor. Como o mesmo modelo gerou e julgou as solucoes, os scores do
PRM promptado devem ser lidos com cautela: eles nao substituem um reward model
treinado com labels humanos por passo.
