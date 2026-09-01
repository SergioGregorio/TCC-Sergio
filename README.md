# TSP Genetic Algorithm Solver

Solucionador do **Problema do Caixeiro Viajante (TSP)** usando um **Algoritmo Memético**: um Algoritmo Genético (GA) combinado com busca local **2-opt** otimizada (lista de vizinhos KNN + don't-look bits). Suporta instâncias no formato **TSPLIB** e gera visualizações completas da solução.

## Recursos

- **Três modos de execução**: `GA_ONLY` (só GA), `LS_ONLY` (só 2-opt com random-restart) e `HYBRID` (GA + 2-opt).
- **2-opt escalável**: lista de K vizinhos mais próximos, don't-look bits e reversão do segmento mais curto — resolve instâncias de milhares de cidades em segundos.
- **Cálculo vetorizado** da matriz de distâncias (EUC_2D, ATT, CEIL_2D, GEO) e avaliação de rotas com NumPy.
- **Multiprocessing adaptativo**: paraleliza a avaliação para instâncias pequenas e usa execução serial para instâncias grandes (evitando overhead de serialização).
- **Parada antecipada** (early stopping) configurável.
- **Suporte TSPLIB**: `EUC_2D`, `ATT`, `CEIL_2D`, `GEO` e `EXPLICIT` (`UPPER_ROW`, `LOWER_ROW`, `FULL_MATRIX`).
- **Visualizações**: rota final numerada, evolução do fitness, dashboard completo, grade de evolução e comparação dos top 5.
- **Benchmark automatizado**: compara modos em várias instâncias e seeds, salvando resultados por rodada e um `summary.json` com estatísticas.

## Requisitos

- Python 3.10+
- Dependências em `requirements.txt`:
  - numpy, deap, matplotlib, networkx

## Instalação

```bash
git clone https://github.com/<seu-usuario>/tsp-genetic-algorithm.git
cd tsp-genetic-algorithm

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Uso

### Execução única

O arquivo de entrada e o modo são definidos em `config.py`:

```bash
python main.py
```

As saídas (5 imagens PNG) são salvas na pasta `output/`.

### Modos de execução

Edite `execution_mode` em `config.py` (`GeneticAlgorithmConfig`):

```python
execution_mode: ExecutionMode = ExecutionMode.HYBRID   # GA_ONLY | LS_ONLY | HYBRID
```

### Benchmark

Compara os três modos em várias instâncias, com múltiplas seeds:

```bash
# Padrão (att48, eil51, berlin52 x 3 modos x 5 rodadas)
python benchmark.py

# Personalizado
python benchmark.py --instances data/att48.tsp data/a280.tsp --runs 5 --generations 500 --population 300 --patience 100

# Com limite de tempo por rodada (evita execuções muito longas em instâncias grandes)
python benchmark.py --instances data/fnl4461.tsp --runs 5 --timeout 120
```

Opções do CLI: `--instances`, `--runs`, `--population`, `--generations`, `--patience`, `--timeout`, `--no-plots`.

Durante a execução, o benchmark exibe um **medidor de progresso** por rodada (contagem, porcentagem, tempo decorrido e ETA). Ao final, gera **gráficos comparativos** (use `--no-plots` para desativar).

Quando `--timeout <segundos>` é usado (> 0), cada rodada é executada em um **processo separado e serial** e é **terminada** se exceder o limite. Rodadas interrompidas são registradas com `timed_out: true` e contabilizadas em `timeout_runs` no `summary.json`.

Estrutura de saída:

```
resultados/
    <instancia>/
        <modo>/
            tentativa_1_seed_<X>.json
            ...
        comparison.png          # grafico comparativo da instancia
    summary.json
    summary_comparison.png      # comparacao entre instancias (se houver > 1)
```

O `summary.json` traz, por instância e modo: melhor (`best`), pior (`worst`), média (`mean`), desvio padrão (`std_dev`), gap do ótimo, tempo médio de execução e número de timeouts (`timeout_runs`).

Gráficos gerados:

- **`<instancia>/comparison.png`**: 4 painéis por instância — distância (best/mean +/- std), distribuição das distâncias (boxplot), gap do ótimo (%) e tempo médio de execução, comparando os três modos.
- **`summary_comparison.png`**: barras agrupadas do gap best (%) por modo em todas as instâncias (apenas quando há mais de uma instância).

## Estrutura do projeto

```
tsp-genetic-algorithm/
├── main.py               # Ponto de entrada (execução única)
├── benchmark.py          # Runner de experimentos
├── benchmark_visualizer.py # Gráficos comparativos do benchmark
├── config.py             # Configurações (dataclasses)
├── data_loader.py        # Parser TSPLIB + matriz de distâncias vetorizada
├── environment.py        # Avaliação de rotas
├── engine.py             # Função de fitness (DEAP)
├── orchestrator.py       # Loop evolutivo + modos de execução
├── local_search.py       # 2-opt (KNN + don't-look bits)
├── progress.py           # Barra de progresso de terminal (ETA)
├── solutions_reader.py   # Ótimos conhecidos + cálculo de gap
├── visualizer.py         # Geração das visualizações
├── data/                 # Instâncias TSPLIB (.tsp) + arquivo de soluções
├── requirements.txt
├── README.md
└── COMO_FUNCIONA.md      # Documentação técnica detalhada
```

## Documentação

Para detalhes de como cada módulo funciona internamente, veja [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md).

## Parâmetros principais (`config.py`)

| Parâmetro | Descrição |
|-----------|-----------|
| `population_size` | Tamanho da população |
| `number_of_generations` | Máximo de gerações (ou restarts no `LS_ONLY`) |
| `crossover_probability` | Probabilidade de cruzamento |
| `mutation_probability` | Probabilidade de mutação |
| `local_search_neighbors` | K de vizinhos para o 2-opt |
| `local_search_interval` | Intervalo (gerações) para aplicar 2-opt |
| `enable_early_stopping` / `early_stopping_patience` | Parada antecipada |
| `execution_mode` | `GA_ONLY`, `LS_ONLY` ou `HYBRID` |

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
