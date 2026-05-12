# Experimento 2: Process Supervision promptado

Esta pasta reúne o que pertence ao teste de Process Supervision promptado no
`openai/gpt-oss-20b`.

## Objetivo

Simular operacionalmente um PRM promptado: o modelo gera várias soluções para
problemas MATH, divide cada solução em passos, julga cada passo via `logprobs`
e escolhe uma solução por best-of-N.

Este experimento não treina um reward model real e não usa labels humanos de
PRM800K.

## Como executar

Com o servidor vLLM ativo em `http://localhost:8000/v1`:

```bash
python -m cot_eval.prm_run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --math-limit 50 \
  --samples-per-problem 8 \
  --math-configs intermediate_algebra,precalculus,geometry,number_theory \
  --seed 43 \
  --max-tokens 4096 \
  --generation-temperature 0.9 \
  --write-incremental
```

## Artefatos preservados

A rodada principal está em:

```text
experiments/02_process_supervision/artifacts/20260510-231913/
```

Arquivos:

- `config.json`: configuração da execução;
- `metrics_corrected.csv`: métricas corrigidas após ajuste do parser;
- `analysis_corrected.md`: análise crítica com os resultados corrigidos.

## Resultado resumido

| Método | Corretas/N | Acurácia | Cobertura parse |
|---|---:|---:|---:|
| First sample | 43/50 | 86,0% | 90,0% |
| Majority vote | 48/50 | 96,0% | 100,0% |
| PRM best-of-N | 48/50 | 96,0% | 100,0% |
| Oracle best-of-N | 49/50 | 98,0% | 98,0% |

Conclusão: best-of-8 ajudou muito sobre a primeira amostra, mas o PRM
promptado empatou com majority vote e adicionou custo de verificação.
