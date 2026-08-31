# Como o Código Opera

Este documento explica o funcionamento interno do solver de TSP (Problema do Caixeiro Viajante). O programa é um **Algoritmo Memético**: um Algoritmo Genético (GA) combinado com **busca local 2-opt**. Ele também suporta três **modos de execução** para comparação experimental (usado no benchmark do TCC).

---

## 1. Visão Geral do Fluxo

```
main.py
  │
  ├─ 1. DataConfig / ApplicationConfig ........ define parâmetros
  ├─ 2. TSPDataLoader ......................... lê o arquivo .tsp -> matriz de distâncias
  ├─ 3. TSPLibSolutionsReader ................. lê o ótimo conhecido (data/solutions)
  ├─ 4. TSPEnvironment ........................ avalia distância de uma rota
  ├─ 5. GeneticAlgorithmEngine ................ função de fitness
  ├─ 6. GeneticAlgorithmOrchestrator .......... executa o GA + 2-opt (por modo)
  └─ 7. TSPVisualizer ......................... gera os 5 PNGs de saída
```

Para experimentos comparativos existe um fluxo paralelo:

```
benchmark.py
  │
  ├─ Para cada instância x cada modo x N rodadas (seeds distintas):
  │     └─ instancia o orquestrador e roda uma vez
  ├─ Salva o resultado de cada rodada em resultados/<inst>/<modo>/tentativa_n_seed_X.json
  └─ Compila estatísticas em resultados/summary.json
```

---

## 2. Módulos e Responsabilidades

### `config.py`
Centraliza todos os parâmetros em dataclasses imutáveis (`frozen=True`):

- **`ExecutionMode`** (Enum) — `GA_ONLY`, `LS_ONLY`, `HYBRID`.
- **`GeneticAlgorithmConfig`** — parâmetros do GA, da busca local e o modo de execução.
- **`DataConfig`** — qual arquivo `.tsp` carregar e onde salvar saídas.
- **`VisualizationConfig`** — aparência dos gráficos.
- **`ApplicationConfig`** — agrega as três acima.

Parâmetros-chave do GA:

| Parâmetro | Função |
|-----------|--------|
| `population_size` | Número de rotas candidatas por geração |
| `number_of_generations` | Limite máximo de gerações (ou de restarts no modo LS_ONLY) |
| `crossover_probability` | Chance de cruzar um par de pais |
| `mutation_probability` | Chance de mutar um indivíduo |
| `tournament_size` | Tamanho do torneio de seleção |
| `elitism_count` | Quantos melhores são preservados intactos |
| `enable_local_search` | Liga/desliga o 2-opt (mestre) |
| `local_search_interval` | A cada quantas gerações aplicar 2-opt |
| `local_search_improvement_rate` | Fração da população refinada pelo 2-opt |
| `enable_early_stopping` | Liga/desliga parada antecipada |
| `early_stopping_patience` | Gerações sem melhora antes de parar |
| `execution_mode` | `GA_ONLY`, `LS_ONLY` ou `HYBRID` |

### `data_loader.py`
Lê arquivos TSPLIB e produz a **matriz de distâncias** `N x N`.

- Detecta `EDGE_WEIGHT_TYPE` e usa a fórmula de distância correta:
  - `EUC_2D` — euclidiana padrão
  - `ATT` — pseudo-euclidiana (fórmula especial com `sqrt(.../10)`)
  - `CEIL_2D` — euclidiana arredondada para cima
  - `GEO` — distância geográfica (lat/lon), com `PI=3.141592` e `RRR=6378.388`
  - `EXPLICIT` — matriz fornecida no arquivo (`UPPER_ROW`, `LOWER_ROW`, `FULL_MATRIX`)
- Coordenadas são indexadas em base 0 (alinhadas com os índices de rota do DEAP).
- Sinaliza se há coordenadas espaciais (`has_coordinates()`). Quando não há (formato EXPLICIT), o visualizador gera um layout artificial.

### `environment.py` — `TSPEnvironment`
Dada uma rota (lista de índices de cidades), calcula a **distância total** percorrida, incluindo o retorno à cidade inicial (ciclo fechado).

### `engine.py` — `GeneticAlgorithmEngine`
Adapta a avaliação para o DEAP: `compute_fitness(individual)` retorna `(distancia,)` — uma tupla, como o DEAP exige.

### `solutions_reader.py` — `TSPLibSolutionsReader`
Lê o arquivo `data/solutions` (formato `nome: valor`) com os ótimos conhecidos e calcula o **gap percentual** entre a solução encontrada e o ótimo.

### `local_search.py` — `TSPLocalSearch` (o 2-opt)
Refina uma rota removendo cruzamentos. A cada par de arestas `(a-b, c-d)`, testa trocá-las por `(a-c, b-d)` invertendo o segmento entre elas.

- **`two_opt(route, max_passes)`** — roda até convergir (polish final / random-restart).
- **`two_opt_fast(route, max_passes=2)`** — versão curta usada dentro do GA.
- O cálculo do ganho é incremental (delta), e a inversão é **in-place**:
  ```python
  best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
  ```
  Isso é o que torna o 2-opt rápido (varredura contínua, sem recriar listas).

### `orchestrator.py` — `GeneticAlgorithmOrchestrator`
O coração do algoritmo. Configura o DEAP e roda o fluxo conforme o modo (ver seção 3).

### `visualizer.py` — `TSPVisualizer`
Gera 5 imagens: rota final, evolução do fitness, dashboard completo, grade de evolução (12 snapshots) e comparação dos top 5. Usa coordenadas reais quando existem, ou layout Kamada-Kawai/Spring quando não há.

### `benchmark.py` — Runner de experimentos
Automatiza a coleta de resultados para o TCC (ver seção 5).

---

## 3. Modos de Execução

O campo `execution_mode` em `GeneticAlgorithmConfig` seleciona um de três comportamentos, isolados no `orchestrator.py`:

| Modo | O que roda | Método interno |
|------|------------|----------------|
| **`GA_ONLY`** | Apenas o Algoritmo Genético (sem 2-opt, nem in-loop nem polish final) | `_run_genetic` com `use_local_search=False` |
| **`LS_ONLY`** | Apenas Busca Local: random-restart 2-opt (gera rotas aleatórias e refina, guardando a melhor) | `_run_local_search_only` |
| **`HYBRID`** | GA + 2-opt (refina parte da população periodicamente + polish final) | `_run_genetic` com `use_local_search=True` |

No modo **`LS_ONLY`**, o parâmetro `number_of_generations` é reinterpretado como o **número máximo de restarts**.

Comportamento típico (att48, ótimo=10628):

| Modo | Gap aproximado |
|------|----------------|
| `GA_ONLY` | alto (dezenas a centenas de %) sem gerações suficientes |
| `LS_ONLY` | baixo (~1-3%) |
| `HYBRID` | mais baixo/estável com gerações adequadas (<1%) |

---

## 4. O Laço Evolutivo (modos GA_ONLY e HYBRID)

Em `GeneticAlgorithmOrchestrator._run_genetic()`:

1. **Inicialização** — gera `population_size` rotas aleatórias e avalia o fitness de cada uma (em paralelo com `multiprocessing.Pool`).

2. **Para cada geração:**
   1. **Seleção** — torneio seleciona `population_size - elitism_count` pais.
   2. **Cruzamento** — `cxOrdered` (Order Crossover) combina pares de pais respeitando a probabilidade de crossover.
   3. **Mutação** — `mutShuffleIndexes` embaralha genes conforme a probabilidade de mutação.
   4. **Reavaliação** — recalcula o fitness apenas dos indivíduos modificados.
   5. **Elitismo** — os `elitism_count` melhores da geração anterior são clonados e reinseridos: `population = elite + offspring`.
   6. **Busca local (2-opt)** — apenas se `HYBRID`: a cada `local_search_interval` gerações, refina a melhor fração (`local_search_improvement_rate`) dos descendentes.
   7. **Atualização do melhor** — se o melhor da geração superar o recorde, atualiza `best_fitness` (monotônico: só melhora).
   8. **Parada antecipada** — se `generations_without_improvement >= early_stopping_patience`, **interrompe o laço**.

3. **Polish final** — apenas se `HYBRID`: aplica o `two_opt` completo no melhor indivíduo encontrado.

4. **Retorno** — devolve a melhor rota e sua distância.

### Modo LS_ONLY

Em `_run_local_search_only()`: gera uma rota aleatória inicial, aplica `two_opt` completo, e então repete (random restart) por até `number_of_generations` tentativas, guardando sempre a melhor. Respeita a parada antecipada (para se não houver melhora por `early_stopping_patience` restarts).

---

## 5. Benchmark (`benchmark.py`)

Automatiza a coleta de resultados para análise estatística.

### O que faz
- Recebe uma lista de instâncias (`--instances`) e roda **cada instância × cada um dos 3 modos × N rodadas** (`--runs`, default 5).
- Cada rodada usa uma **seed aleatória distinta**, fixada para reprodutibilidade.
- Cada rodada é isolada com tratamento de erros — uma falha não aborta o benchmark.

### Saída
```
resultados/
    <instancia>/
        <modo>/
            tentativa_1_seed_<X>.json    # resultado bruto de uma rodada (inclui a seed)
            ...
    summary.json                          # estatísticas compiladas
```

Cada `tentativa_*.json` contém: instância, modo, tentativa, **seed**, melhor distância, ótimo, gap, tempo, gerações executadas, se houve early stopping e erro (se houver).

O `summary.json` traz, por instância e modo: `algorithm`, `instance`, `best`, `worst`, `mean`, `std_dev`, `best_gap_percent`, `mean_execution_time`, `successful_runs`, `failed_runs`.

### Como executar
```bash
# Padrão
python benchmark.py

# Personalizado
python benchmark.py --instances data/att48.tsp data/a280.tsp --runs 5 --generations 500 --population 300 --patience 100
```

Opções: `--instances`, `--runs`, `--population`, `--generations`, `--patience`.

---

## 6. Parada Antecipada (Early Stopping)

Quando passam `early_stopping_patience` gerações (ou restarts, no LS_ONLY) sem nenhuma melhora, o laço é interrompido e (no HYBRID) o polish final é aplicado. Isso:

- Elimina gerações desperdiçadas no platô.
- Reduz o tempo de execução.
- É reportado no relatório final (`Early stopping triggered at generation X`).

Controles em `config.py`: `enable_early_stopping` e `early_stopping_patience`.

---

## 7. Como Executar (single run)

```bash
python main.py
```

O arquivo `.tsp` e o modo são definidos em `config.py`. As saídas (5 PNGs) vão para a pasta `output/`.
