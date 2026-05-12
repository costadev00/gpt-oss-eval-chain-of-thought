# Analise critica: GPT-OSS-20B com e sem chain-of-thought

## Resultados agregados

| Tarefa | Sem CoT | CoT | Delta CoT | N |
|---|---:|---:|---:|---:|
| Coin Flip | 100.0% | 100.0% | +0.0 p.p. | 100 |
| GSM8K | 94.5% | 96.5% | +2.0 p.p. | 200 |
| Last Letter | 85.0% | 84.0% | -1.0 p.p. | 100 |

## Leitura critica

- GSM8K: CoT melhorou em relacao ao baseline (+2.0 p.p.).
- Last Letter: CoT ficou praticamente empatado em relacao ao baseline (-1.0 p.p.).
- Coin Flip: CoT ficou praticamente empatado em relacao ao baseline (+0.0 p.p.).
- A comparacao isola o efeito do prompt: mesma API, mesmo modelo, mesma temperatura e mesmo esforco de reasoning configurado.
- CoT tende a custar mais tokens e latencia; ganhos pequenos de acuracia devem ser lidos junto com a tabela de custo operacional.
- Falhas de parsing contam como erro, pois em uso real uma resposta nao extraivel tambem quebra avaliacao automatica.
- As tarefas simbolicas sao deterministicas e uteis para diagnostico, mas nao substituem benchmarks externos diversos.
- O relatorio nao usa raw chain-of-thought interno do modelo; considera apenas resposta visivel, resposta extraida e metricas agregadas.

## Custo operacional

| Tarefa | Condicao | Latencia media (s) | Tokens prompt | Tokens resposta | Tokens totais |
|---|---|---:|---:|---:|---:|
| GSM8K | Sem CoT | 1.03 | 584.4 | 202.1 | 786.5 |
| GSM8K | CoT | 1.12 | 892.4 | 219.0 | 1111.4 |
| Last Letter | Sem CoT | 0.38 | 303.4 | 72.8 | 376.1 |
| Last Letter | CoT | 0.75 | 428.4 | 146.7 | 575.1 |
| Coin Flip | Sem CoT | 1.51 | 502.3 | 298.2 | 800.5 |
| Coin Flip | CoT | 0.82 | 852.3 | 158.4 | 1010.8 |

## Exemplos para inspecao

- GSM8K / CoT: gold='45', extraido='45', correto=True. Pergunta: While playing with her friends in their school playground, Katelyn saw 50 fairies flying above the nearby forest. After about twenty minutes, one of her friends saw half as many fairies as Katelyn saw come from the ea...
- GSM8K / CoT: gold='525', extraido='525', correto=True. Pergunta: For a New Year's resolution, Andy wants to lose 30 lbs. by his birthday, which is July 19th. Today is December 31st. If Andy needs to burn 3500 calories to lose a pound, how much of a calorie deficit (net amount of ca...
- GSM8K / CoT: gold='24', extraido='24', correto=True. Pergunta: Maria invited 4 of her friends over for a water balloon fight in the backyard. At the start of the game, Maria gave each of her friends 2 water balloons. She had one water balloon for herself.  Then her mom came out a...
- GSM8K / CoT: gold='80', extraido='80', correto=True. Pergunta: Blake and Kelly are having a contest to see who can run the most in 15 minutes. They decide to do it on a football field that is 100 yards long. Blake runs back and forth 15 times. Kelly runs back and forth once, and ...
- GSM8K / CoT: gold='6', extraido='6', correto=True. Pergunta: Dolly has two books. Pandora has one. If both Dolly and Pandora read each others' books as well as their own, how many books will they collectively read by the end?
- GSM8K / CoT: gold='175', extraido='175', correto=True. Pergunta: Jerome had 4 friends who came to visit him on a certain day. The first friend pressed on the doorbell 20 times before Jerome opened, the second friend pressed on the doorbell 1/4 times more than Jerome's first friend....
