# GPT-OSS-20B: avaliação de Chain-of-Thought e Process Supervision

Este repositório contém um harness local para avaliar o modelo
`openai/gpt-oss-20b` em duas frentes:

- comparação entre prompting direto e prompting com Chain-of-Thought (CoT);
- experimento didático de Process Supervision promptado, no qual o próprio
  modelo gera soluções e julga passos intermediários via `logprobs`.

O objetivo é mostrar, de forma reprodutível, como os experimentos foram
executados, quais resultados foram obtidos e quais arquivos sustentam essas
conclusões.

## Arquivos importantes

| Caminho | Conteúdo |
|---|---|
| `cot_eval/run.py` | CLI do experimento CoT versus resposta direta |
| `cot_eval/prm_run.py` | CLI do experimento de Process Supervision promptado |
| `cot_eval/audit_parser.py` | Reprocessamento de respostas salvas para auditar o parser |
| `cot_eval/tasks.py` | Loader das tarefas GSM8K, Last Letter e Coin Flip |
| `cot_eval/math_tasks.py` | Loader e normalização do benchmark MATH |
| `cot_eval/prm_scoring.py` | Separação de passos, score por passo e majority vote |
| `experiments/01_chain_of_thought/` | Documentação e artefatos da avaliação CoT |
| `experiments/02_process_supervision/` | Documentação e artefatos do teste de Process Supervision |
| `reports/gpt_oss_20b_cot_prm_report.tex` | Relatório consolidado completo |
| `papers/2201.11903.pdf` | Paper de Chain-of-Thought |
| `papers/2305.20050.pdf` | Paper Let's Verify Step by Step |
| `tests/` | Testes unitários e smoke tests sem GPU |

As saídas brutas completas (`predictions.jsonl`, `candidates.jsonl`,
`step_scores.jsonl` etc.) não ficam versionadas. Elas podem ser regeneradas
pelos comandos abaixo.

## Máquina utilizada

Os resultados preservados foram produzidos localmente com vLLM em uma máquina
com a seguinte configuração:

| Componente | Valor |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 7975WX 32-Cores |
| Threads | 64 |
| RAM | 251 GiB |
| GPUs | 4x NVIDIA RTX 4000 Ada Generation |
| Memória por GPU | 20475 MiB |
| Driver NVIDIA | 580.126.20 |
| Python | 3.12.3 |
| vLLM | 0.20.2 |
| Backend | API local OpenAI-compatible via vLLM |

## Instalação

Crie um ambiente Python e instale o pacote em modo editável:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Para servir o modelo localmente com vLLM, instale também o extra de serving:

```bash
python -m pip install -e ".[serve]"
```

Se precisar autenticar no Hugging Face, crie um `.env` local a partir do
exemplo:

```bash
cp .env.example .env
```

Depois preencha `HF_TOKEN` e carregue as variáveis:

```bash
set -a
source .env
set +a
```

## Servir o GPT-OSS-20B

Suba o servidor vLLM nas quatro GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce
```

O endpoint fica disponível em:

```text
http://localhost:8000/v1
```

Em outro terminal, confirme que o servidor responde:

```bash
curl http://localhost:8000/v1/models
```

## Experimento 1: Chain-of-Thought

### Metodologia

Este experimento compara duas condições com o mesmo modelo, mesma temperatura,
mesmo `reasoning_effort=medium` e mesmo contrato de saída:

```text
Final answer: <answer>
```

As condições são:

- `standard`: exemplos few-shot com resposta direta;
- `cot`: exemplos few-shot com raciocínio intermediário antes da resposta
  final.

As tarefas avaliadas foram:

- GSM8K, com 200 itens do split de teste;
- Last Letter, tarefa simbólica local com 100 itens;
- Coin Flip, tarefa simbólica local com 100 itens.

Rodada principal preservada:

```text
experiments/01_chain_of_thought/artifacts/20260510-185218/
```

### Execução

```bash
python -m cot_eval.run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --tasks gsm8k,last_letter,coin_flip \
  --gsm8k-limit 200 \
  --symbolic-limit 100 \
  --seed 42
```

Para um smoke test menor:

```bash
python -m cot_eval.run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --tasks gsm8k,last_letter,coin_flip \
  --gsm8k-limit 5 \
  --symbolic-limit 5 \
  --seed 42
```

Cada execução cria uma pasta em `results/<timestamp>/`. Esse diretório é
ignorado pelo Git porque contém artefatos brutos de execução.

### Resultados

| Tarefa | Sem CoT | CoT | Delta |
|---|---:|---:|---:|
| GSM8K | 94,5% | 96,5% | +2,0 p.p. |
| Last Letter | 85,0% | 84,0% | -1,0 p.p. |
| Coin Flip | 100,0% | 100,0% | +0,0 p.p. |

Leitura: CoT trouxe ganho moderado em GSM8K, uma tarefa aritmética de múltiplas
etapas. Nas tarefas simbólicas simples, não houve ganho.

Consumo médio de tokens por item:

| Tarefa | Condição | Prompt | Resposta | Total |
|---|---|---:|---:|---:|
| GSM8K | Sem CoT | 584,4 | 202,1 | 786,5 |
| GSM8K | CoT | 892,4 | 219,0 | 1111,4 |
| Last Letter | Sem CoT | 303,4 | 72,8 | 376,1 |
| Last Letter | CoT | 428,4 | 146,7 | 575,1 |
| Coin Flip | Sem CoT | 502,3 | 298,2 | 800,5 |
| Coin Flip | CoT | 852,3 | 158,4 | 1010,8 |

Artefatos preservados:

- `experiments/01_chain_of_thought/artifacts/20260510-185218/config.json`;
- `experiments/01_chain_of_thought/artifacts/20260510-185218/metrics_rescored.csv`;
- `experiments/01_chain_of_thought/artifacts/20260510-185218/analysis.md`;
- `experiments/01_chain_of_thought/artifacts/20260510-185218/parser_audit.md`.

## Experimento 2: Process Supervision promptado

### Metodologia

Este experimento é inspirado no paper `Let's Verify Step by Step`, mas não
treina um PRM real. Em vez disso, usa uma aproximação promptada:

1. o GPT-OSS-20B gera várias soluções para cada problema MATH;
2. a solução é dividida em passos;
3. o mesmo modelo julga cada passo como `positive`, `neutral` ou `negative`
   usando o endpoint `/v1/completions` com `logprobs`;
4. `neutral` é tratado como positivo;
5. o score da solução é o produto dos scores dos passos;
6. o seletor escolhe uma solução entre os candidatos.

Foram comparados quatro métodos:

- `first`: usa a primeira amostra;
- `majority_vote`: escolhe a resposta final mais frequente;
- `prm_best_of_n`: escolhe a solução com maior score de processo;
- `oracle_best_of_n`: teto diagnóstico que escolhe uma solução correta quando
  ela existe entre os candidatos.

A rodada principal usou:

| Campo | Valor |
|---|---|
| Benchmark | `EleutherAI/hendrycks_math` |
| Configs | `intermediate_algebra`, `precalculus`, `geometry`, `number_theory` |
| Filtro | apenas respostas numéricas extraíveis |
| Problemas | 50 |
| Soluções por problema | 8 |
| Seed | 43 |
| Temperatura de geração | 0,9 |
| `max_tokens` | 4096 |

Rodada principal preservada:

```text
experiments/02_process_supervision/artifacts/20260510-231913/
```

### Execução

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

Para uma rodada mais barata:

```bash
python -m cot_eval.prm_run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --math-limit 10 \
  --samples-per-problem 4 \
  --seed 42 \
  --max-tokens 2048 \
  --generation-temperature 0.7 \
  --write-incremental
```

Cada execução cria uma pasta em `results_prm/<timestamp>/`. Esse diretório
também é ignorado pelo Git.

### Resultados corrigidos

Durante a análise posterior, o parser foi corrigido para tratar melhor respostas
fracionárias finais. Por isso, a tabela abaixo usa os números corrigidos do
relatório consolidado.

| Método | Corretas/N | Acurácia | Cobertura parse |
|---|---:|---:|---:|
| First sample | 43/50 | 86,0% | 90,0% |
| Majority vote | 48/50 | 96,0% | 100,0% |
| PRM best-of-N | 48/50 | 96,0% | 100,0% |
| Oracle best-of-N | 49/50 | 98,0% | 98,0% |

Leitura: gerar oito soluções por problema melhorou bastante o resultado em
relação à primeira amostra. Porém, o PRM promptado empatou com majority vote e
adicionou custo relevante de verificação, cerca de 39707 tokens de PRM por
problema. Nesta configuração, ele é útil como demonstração didática de Process
Supervision, mas não superou o baseline simples.

Artefatos preservados:

- `experiments/02_process_supervision/artifacts/20260510-231913/config.json`;
- `experiments/02_process_supervision/artifacts/20260510-231913/metrics_corrected.csv`;
- `experiments/02_process_supervision/artifacts/20260510-231913/analysis_corrected.md`.

## Auditoria do parser

Para reprocessar uma execução salva sem chamar o modelo novamente:

```bash
python -m cot_eval.audit_parser results/<timestamp>
```

O comando preserva `predictions.jsonl` e cria:

- `parser_audit.md`;
- `parser_audit_changes.csv`;
- `metrics_rescored.csv`;
- `predictions_rescored.jsonl`.

Na rodada principal de CoT, a versão auditada preservada está em
`experiments/01_chain_of_thought/artifacts/20260510-185218/parser_audit.md`.

## Testes

Os testes não precisam de GPU:

```bash
python -m pytest
```

Baseline antes da limpeza final do repositório:

```text
34 passed
```

## Relatório completo

O relatório LaTeX consolidado está em:

```text
reports/gpt_oss_20b_cot_prm_report.tex
```

Ele contém a discussão completa dos dois experimentos, tabelas de custo,
limitações e referências aos papers usados como base.

## Troubleshooting

Se `python -m cot_eval.run ...` ou `python -m cot_eval.prm_run ...` falhar com
`Connection refused`, o servidor vLLM provavelmente não está rodando em
`http://localhost:8000/v1`. Suba o servidor primeiro e valide com:

```bash
curl http://localhost:8000/v1/models
```

Se o shell disser `vllm: command not found`, ative a `.venv` ou use o caminho
explícito:

```bash
.venv/bin/vllm serve openai/gpt-oss-20b
```

Se o modelo exigir autenticação, confirme que `HF_TOKEN` está no `.env` local e
que as variáveis foram carregadas no terminal atual.

## Referências

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models:
  `papers/2201.11903.pdf`
- Let's Verify Step by Step: `papers/2305.20050.pdf`
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- MATH: https://huggingface.co/datasets/EleutherAI/hendrycks_math
