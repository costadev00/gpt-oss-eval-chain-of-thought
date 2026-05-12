# Experimento 1: Chain-of-Thought

Esta pasta reúne o que pertence à avaliação de Chain-of-Thought (CoT) contra
resposta direta no `openai/gpt-oss-20b`.

## Objetivo

Medir se exemplos few-shot com raciocínio intermediário melhoram a acurácia em
relação a exemplos few-shot com resposta direta, mantendo o mesmo modelo,
backend, temperatura e contrato de saída.

## Como executar

Com o servidor vLLM ativo em `http://localhost:8000/v1`:

```bash
python -m cot_eval.run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --tasks gsm8k,last_letter,coin_flip \
  --gsm8k-limit 200 \
  --symbolic-limit 100 \
  --seed 42
```

## Artefatos preservados

A rodada principal está em:

```text
experiments/01_chain_of_thought/artifacts/20260510-185218/
```

Arquivos:

- `config.json`: configuração da execução;
- `metrics_rescored.csv`: métricas após auditoria do parser;
- `analysis.md`: análise crítica gerada pela execução;
- `parser_audit.md`: auditoria do parser.

## Resultado resumido

| Tarefa | Sem CoT | CoT | Delta |
|---|---:|---:|---:|
| GSM8K | 94,5% | 96,5% | +2,0 p.p. |
| Last Letter | 85,0% | 84,0% | -1,0 p.p. |
| Coin Flip | 100,0% | 100,0% | +0,0 p.p. |

Conclusão: CoT ajudou em GSM8K, mas não trouxe ganho nas tarefas simbólicas
simples avaliadas.
