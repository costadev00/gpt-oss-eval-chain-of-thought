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

Para servir o `gpt-oss-20b` localmente com vLLM, instale tambem o extra de
serving:

```bash
python -m pip install -e ".[serve]"
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
  --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce
```

O endpoint exposto e OpenAI-compatible em `http://localhost:8000/v1`.
Se o shell ainda disser `vllm: command not found` depois da instalacao, rode
`hash -r` ou use o caminho explicito `.venv/bin/vllm`.

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

Por padrao, o harness envia tambem um `system` message identico para as duas
condicoes. Ele nao decide se a resposta deve ter CoT; isso continua sendo
controlado pelos exemplos few-shot. O papel do `system` message e apenas fixar
o contrato de saida:

```text
Final answer: <answer>
```

O parser prioriza esse marcador, o que reduz falsos erros quando uma resposta
CoT contem varios numeros ou frases intermediarias. Para sobrescrever esse
contrato, use `--system-prompt "..."`, mantendo o mesmo valor para as duas
condicoes se a comparacao precisa continuar justa.

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

## Auditar o parser

Para verificar se uma conclusao depende do parser, reprocesse uma execucao ja
salva sem chamar o modelo novamente:

```bash
python -m cot_eval.audit_parser results/<timestamp>
```

Esse comando preserva `predictions.jsonl` e cria:

- `parser_audit.md`: comparacao entre extracao antiga e extracao atual;
- `parser_audit_changes.csv`: exemplos em que a extracao ou corretude mudou;
- `metrics_rescored.csv`: metricas recalculadas com o parser atual;
- `predictions_rescored.jsonl`: predicoes reanotadas, sem substituir o arquivo original.

## Troubleshooting

Se `python -m cot_eval.run ...` falhar com `Connection refused` ou
`Endpoint unavailable`, o servidor vLLM nao esta rodando em
`http://localhost:8000/v1` ou esta em outra porta. Suba o servidor primeiro:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce
```

Em outro terminal, confirme que o endpoint responde:

```bash
curl http://localhost:8000/v1/models
```

Depois rode a avaliacao. Se o servidor estiver em outra porta ou maquina, use
`--base-url`, por exemplo `--base-url http://localhost:8001/v1`.

Se o vLLM ficar parado em `Starting to load model`, finalize o processo e baixe
os shards manualmente, um por vez:

```bash
set -a
source .env
set +a

hf download openai/gpt-oss-20b model-00000-of-00002.safetensors --max-workers 1
hf download openai/gpt-oss-20b model-00001-of-00002.safetensors --max-workers 1
hf download openai/gpt-oss-20b model-00002-of-00002.safetensors --max-workers 1
```

Depois suba o vLLM novamente com `--disable-custom-all-reduce`.

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
