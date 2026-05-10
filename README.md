# GPT-OSS-20B CoT Evaluation

Harness para comparar `openai/gpt-oss-20b` com prompting direto e com
chain-of-thought, inspirado no paper `2201.11903.pdf`:
_Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_.

O escopo v1 avalia:

- `gsm8k`: amostra configuravel do split de teste de `openai/gsm8k`;
- `last_letter`: tarefa simbolica gerada deterministicamente;
- `coin_flip`: tarefa simbolica gerada deterministicamente.

## Setup

Crie um ambiente Python e instale o pacote local:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Se o modelo exigir autenticacao no Hugging Face, exporte o token antes de
subir o vLLM. O valor nao e usado nem impresso pelo harness.

```bash
set -a
source .env
set +a
```

## Servir GPT-OSS-20B nas 4 GPUs

Esta maquina foi planejada para rodar o modelo em modo distribuido nas quatro
GPUs NVIDIA RTX 4000 Ada Generation de aproximadamente 20 GB cada:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92
```

O endpoint exposto e OpenAI-compatible em `http://localhost:8000/v1`.

## Rodar a avaliacao

Com o servidor vLLM ativo:

```bash
python -m cot_eval.run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --tasks gsm8k,last_letter,coin_flip \
  --gsm8k-limit 200 \
  --symbolic-limit 100 \
  --seed 42
```

Para um teste pequeno:

```bash
python -m cot_eval.run \
  --base-url http://localhost:8000/v1 \
  --model openai/gpt-oss-20b \
  --tasks gsm8k,last_letter,coin_flip \
  --gsm8k-limit 5 \
  --symbolic-limit 5 \
  --seed 42
```

## Artefatos

Cada execucao cria uma pasta em `results/<timestamp>/` com:

- `config.json`: configuracao da execucao;
- `predictions.jsonl`: uma linha por item, tarefa e condicao;
- `metrics.csv`: acuracia, intervalo bootstrap, latencia e tokens;
- `analysis.md`: analise critica em Markdown;
- `analysis_section.tex`: secao LaTeX com tabelas e discussao critica.

O relatorio nao usa raw chain-of-thought interno do modelo. Ele considera a
resposta visivel, a resposta extraida pelo parser e metricas agregadas.

## Testes

```bash
python -m pytest
```

Os testes nao precisam de GPU. O smoke test usa um cliente fake para validar a
CLI e a escrita dos artefatos.

## Referencias

- Modelo: https://developers.openai.com/api/docs/models/gpt-oss-20b
- vLLM com gpt-oss: https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm
- Raw CoT: https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
