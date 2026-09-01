"""
Script de automação de benchmark para o TSP Genetic Algorithm.

Executa, para cada instância de teste e cada modo de execução (GA_ONLY, LS_ONLY,
HYBRID), múltiplas rodadas com seeds aleatórias distintas. Salva o resultado de
cada rodada individual em uma estrutura de pastas organizada e, ao final, compila
todas as métricas estatísticas em um único arquivo `summary.json`.

Estrutura de saída:
    resultados/
        <nome_instancia>/
            <modo>/
                tentativa_1_seed_<X>.json
                ...
        summary.json

Uso:
    python benchmark.py
    python benchmark.py --instances data/att48.tsp data/eil51.tsp --runs 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import random
import statistics
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ApplicationConfig, GeneticAlgorithmConfig, ExecutionMode
from data_loader import TSPDataLoader
from engine import GeneticAlgorithmEngine
from environment import TSPEnvironment
from orchestrator import GeneticAlgorithmOrchestrator
from solutions_reader import TSPLibSolutionsReader
from progress import format_duration
from benchmark_visualizer import generate_benchmark_reports


# Modos de execução a serem comparados no benchmark.
EXECUTION_MODES: List[ExecutionMode] = [
    ExecutionMode.GA_ONLY,
    ExecutionMode.LS_ONLY,
    ExecutionMode.HYBRID,
]

# Instâncias padrão (pequenas/médias) caso nenhuma seja informada via CLI.
DEFAULT_INSTANCES: List[str] = [
    "data/att48.tsp",
    "data/eil51.tsp",
    "data/berlin52.tsp",
]

DEFAULT_RUNS: int = 5
RESULTS_ROOT: Path = Path("resultados")


@dataclass
class RunResult:
    """Resultado de uma única rodada de execução."""

    instance: str
    mode: str
    attempt: int
    seed: int
    best_distance: float
    optimal_solution: Optional[int]
    gap_from_optimal: Optional[float]
    execution_time: float
    generations_run: int
    early_stopped: bool
    error: Optional[str] = None
    timed_out: bool = False

    def to_dict(self) -> Dict:
        data = dataclasses.asdict(self)
        if data["best_distance"] == float("inf"):
            data["best_distance"] = None
        return data


@dataclass
class BenchmarkConfig:
    """Parâmetros de alto nível que controlam o benchmark."""

    instances: List[str]
    runs: int = DEFAULT_RUNS
    results_root: Path = RESULTS_ROOT
    population_size: int = 300
    number_of_generations: int = 500
    early_stopping_patience: int = 100
    timeout_seconds: int = 0
    modes: List[ExecutionMode] = field(default_factory=lambda: list(EXECUTION_MODES))


def build_ga_config(
    benchmark_config: BenchmarkConfig,
    mode: ExecutionMode,
    seed: int,
    force_serial: bool = False,
) -> GeneticAlgorithmConfig:
    """Cria uma configuração de GA imutável para uma rodada específica.

    A visualização ao vivo é desativada para não poluir a saída do benchmark e a
    seed é fixada para garantir a reprodutibilidade de cada rodada. Quando
    ``force_serial`` é True (execução sob timeout), o número de processos é
    forçado para 1, evitando workers órfãos caso o processo seja terminado.
    """
    base_config = ApplicationConfig().genetic_algorithm
    processes = 1 if force_serial else base_config.number_of_processes
    return dataclasses.replace(
        base_config,
        population_size=benchmark_config.population_size,
        number_of_generations=benchmark_config.number_of_generations,
        early_stopping_patience=benchmark_config.early_stopping_patience,
        random_seed=seed,
        enable_live_visualization=False,
        number_of_processes=processes,
        execution_mode=mode,
    )


def run_single(
    instance_path: Path,
    mode: ExecutionMode,
    attempt: int,
    seed: int,
    benchmark_config: BenchmarkConfig,
    solutions_reader: TSPLibSolutionsReader,
    force_serial: bool = False,
) -> RunResult:
    """Executa uma única rodada do algoritmo e captura suas métricas.

    Qualquer exceção é capturada e registrada no próprio resultado para que uma
    falha isolada não interrompa todo o benchmark.
    """
    optimal_solution = solutions_reader.get_optimal_solution(instance_path)

    try:
        # Fixa as seeds globais antes de qualquer operação estocástica.
        random.seed(seed)

        data_loader = TSPDataLoader(instance_path)
        data_loader.load()

        environment = TSPEnvironment(data_loader.get_distance_matrix())
        engine = GeneticAlgorithmEngine(environment)

        ga_config = build_ga_config(benchmark_config, mode, seed, force_serial=force_serial)
        orchestrator = GeneticAlgorithmOrchestrator(
            engine=engine,
            config=ga_config,
            number_of_cities=data_loader.get_number_of_cities(),
        )

        start_time = time.time()
        _, best_distance = orchestrator.run_evolution()
        execution_time = time.time() - start_time

        gap = solutions_reader.calculate_gap(instance_path, best_distance)

        return RunResult(
            instance=instance_path.stem,
            mode=mode.value,
            attempt=attempt,
            seed=seed,
            best_distance=round(float(best_distance), 4),
            optimal_solution=optimal_solution,
            gap_from_optimal=round(gap, 4) if gap is not None else None,
            execution_time=round(execution_time, 4),
            generations_run=orchestrator.get_total_generations_run(),
            early_stopped=orchestrator.was_early_stopped(),
        )
    except Exception as exception:  # noqa: BLE001 - queremos registrar qualquer falha.
        return RunResult(
            instance=instance_path.stem,
            mode=mode.value,
            attempt=attempt,
            seed=seed,
            best_distance=float("inf"),
            optimal_solution=optimal_solution,
            gap_from_optimal=None,
            execution_time=0.0,
            generations_run=0,
            early_stopped=False,
            error=f"{type(exception).__name__}: {exception}\n{traceback.format_exc()}",
        )


def _timeout_worker(
    instance_path_str: str,
    mode_value: str,
    attempt: int,
    seed: int,
    benchmark_config: BenchmarkConfig,
    queue: "mp.Queue",
) -> None:
    """Executado em um processo separado; roda uma rodada e envia o resultado pela fila.

    Definido no n\u00edvel do m\u00f3dulo para ser pickl\u00e1vel (necess\u00e1rio no Windows com spawn).
    """
    reader = TSPLibSolutionsReader()
    result = run_single(
        instance_path=Path(instance_path_str),
        mode=ExecutionMode(mode_value),
        attempt=attempt,
        seed=seed,
        benchmark_config=benchmark_config,
        solutions_reader=reader,
        force_serial=True,
    )
    queue.put(result)


def run_with_timeout(
    instance_path: Path,
    mode: ExecutionMode,
    attempt: int,
    seed: int,
    benchmark_config: BenchmarkConfig,
    solutions_reader: TSPLibSolutionsReader,
) -> RunResult:
    """Executa uma rodada respeitando ``benchmark_config.timeout_seconds``.

    Se o timeout for 0 (ou negativo), executa normalmente no processo atual. Caso
    contr\u00e1rio, roda a rodada em um processo separado e o termina caso exceda o
    limite, retornando um ``RunResult`` marcado com ``timed_out=True``.
    """
    timeout = benchmark_config.timeout_seconds
    if not timeout or timeout <= 0:
        return run_single(instance_path, mode, attempt, seed, benchmark_config, solutions_reader)

    optimal_solution = solutions_reader.get_optimal_solution(instance_path)
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_timeout_worker,
        args=(str(instance_path), mode.value, attempt, seed, benchmark_config, queue),
    )
    
    start_time = time.time()
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return RunResult(
            instance=instance_path.stem,
            mode=mode.value,
            attempt=attempt,
            seed=seed,
            best_distance=float("inf"),
            optimal_solution=optimal_solution,
            gap_from_optimal=None,
            execution_time=round(time.time() - start_time, 4),
            generations_run=0,
            early_stopped=False,
            error=f"TIMEOUT after {timeout}s",
            timed_out=True,
        )
    
    try:
        return queue.get(timeout=10)
    except Exception:
        return RunResult(
            instance=instance_path.stem,
            mode=mode.value,
            attempt=attempt,
            seed=seed,
            best_distance=float("inf"),
            optimal_solution=optimal_solution,
            gap_from_optimal=None,
            execution_time=round(time.time() - start_time, 4),
            generations_run=0,
            early_stopped=False,
            error="No result returned by worker process",
        )


def save_run_result(result: RunResult, results_root: Path) -> Path:
    """Persiste o resultado de uma rodada individual em JSON.

    Caminho: resultados/<instancia>/<modo>/tentativa_<n>_seed_<seed>.json
    """
    output_dir = results_root / result.instance / result.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"tentativa_{result.attempt}_seed_{result.seed}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=4, ensure_ascii=False)

    return output_path


def compute_statistics(results: List[RunResult]) -> Dict:
    """Calcula Best, Worst, Mean, Std e tempo médio a partir das rodadas válidas."""
    valid_results = [r for r in results if r.error is None and r.best_distance != float("inf")]

    timeout_runs = sum(1 for r in results if r.timed_out)

    if not valid_results:
        return {
            "successful_runs": 0,
            "failed_runs": len(results),
            "timeout_runs": timeout_runs,
            "best": None,
            "worst": None,
            "mean": None,
            "std_dev": None,
            "mean_execution_time": None,
        }

    distances = [r.best_distance for r in valid_results]
    times = [r.execution_time for r in valid_results]
    optimal = valid_results[0].optimal_solution

    best = min(distances)
    mean = statistics.mean(distances)
    # pstdev com uma única amostra é 0.0; stdev exige >= 2 amostras.
    std_dev = statistics.stdev(distances) if len(distances) > 1 else 0.0

    best_gap = None
    if optimal:
        best_gap = round(((best - optimal) / optimal) * 100, 4)

    return {
        "successful_runs": len(valid_results),
        "failed_runs": len(results) - len(valid_results),
        "timeout_runs": timeout_runs,
        "optimal_solution": optimal,
        "best": round(best, 4),
        "worst": round(max(distances), 4),
        "mean": round(mean, 4),
        "std_dev": round(std_dev, 4),
        "best_gap_percent": best_gap,
        "mean_execution_time": round(statistics.mean(times), 4),
    }


def run_benchmark(benchmark_config: BenchmarkConfig) -> Tuple[Dict, Dict[str, Dict[str, List[RunResult]]]]:
    """Orquestra todo o benchmark: instâncias x modos x rodadas.

    Retorna uma tupla ``(summary, detailed)``:
    - ``summary``: objeto estruturado salvo em summary.json.
    - ``detailed``: mapeia instância -> modo -> lista de ``RunResult`` (para gráficos).
    """
    solutions_reader = TSPLibSolutionsReader()
    summary: Dict[str, Dict] = {}
    detailed: Dict[str, Dict[str, List[RunResult]]] = {}

    # Gera uma seed aleatória distinta por rodada.
    seed_generator = random.Random()

    total_runs = (
        len(benchmark_config.instances) * len(benchmark_config.modes) * benchmark_config.runs
    )
    completed_runs = 0
    benchmark_start = time.time()

    for instance_str in benchmark_config.instances:
        instance_path = Path(instance_str)

        if not instance_path.exists():
            print(f"  [AVISO] Instância não encontrada, pulando: {instance_path}")
            completed_runs += len(benchmark_config.modes) * benchmark_config.runs
            continue

        instance_name = instance_path.stem
        summary[instance_name] = {}
        detailed[instance_name] = {}

        print(f"\n{'=' * 80}")
        print(f" INSTÂNCIA: {instance_name}")
        print(f"{'=' * 80}")

        for mode in benchmark_config.modes:
            print(f"\n  Modo: {mode.value}")
            mode_results: List[RunResult] = []

            for attempt in range(1, benchmark_config.runs + 1):
                seed = seed_generator.randint(1, 1_000_000)

                result = run_with_timeout(
                    instance_path=instance_path,
                    mode=mode,
                    attempt=attempt,
                    seed=seed,
                    benchmark_config=benchmark_config,
                    solutions_reader=solutions_reader,
                )
                mode_results.append(result)
                save_run_result(result, benchmark_config.results_root)

                completed_runs += 1
                elapsed = time.time() - benchmark_start
                rate = completed_runs / elapsed if elapsed > 0 else 0.0
                eta = (total_runs - completed_runs) / rate if rate > 0 else 0.0
                fraction = completed_runs / total_runs * 100 if total_runs else 100.0
                progress = (
                    f" | [{completed_runs}/{total_runs} {fraction:.1f}% "
                    f"elapsed {format_duration(elapsed)} ETA {format_duration(eta)}]"
                )

                if result.timed_out:
                    print(
                        f"    Tentativa {attempt}/{benchmark_config.runs} "
                        f"(seed={seed}): TIMEOUT (> {benchmark_config.timeout_seconds}s){progress}"
                    )
                elif result.error is None:
                    print(
                        f"    Tentativa {attempt}/{benchmark_config.runs} "
                        f"(seed={seed}): dist={result.best_distance:.2f} "
                        f"gap={result.gap_from_optimal}% tempo={result.execution_time:.2f}s{progress}"
                    )
                else:
                    print(
                        f"    Tentativa {attempt}/{benchmark_config.runs} "
                        f"(seed={seed}): FALHOU{progress}"
                    )

            detailed[instance_name][mode.value] = mode_results
            stats = compute_statistics(mode_results)
            summary[instance_name][mode.value] = {
                "algorithm": mode.value,
                "instance": instance_name,
                **stats,
            }

    return summary, detailed


def save_summary(summary: Dict, results_root: Path) -> Path:
    """Salva o objeto de sumário completo em resultados/summary.json."""
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)
    return summary_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark do TSP Genetic Algorithm.")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=DEFAULT_INSTANCES,
        help="Lista de arquivos .tsp a serem testados.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Número de rodadas por (instância, modo).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=300,
        help="Tamanho da população do GA.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Número máximo de gerações (ou restarts no modo LS_ONLY).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Paciência para parada antecipada.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help=(
            "Tempo máximo por rodada em segundos (0 = sem limite). Ao ativar, cada "
            "rodada roda em um processo separado e serial, sendo terminada se exceder o limite."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desativa a geração dos gráficos comparativos ao final do benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    benchmark_config = BenchmarkConfig(
        instances=arguments.instances,
        runs=arguments.runs,
        population_size=arguments.population,
        number_of_generations=arguments.generations,
        early_stopping_patience=arguments.patience,
        timeout_seconds=arguments.timeout,
    )

    print("\n" + "=" * 80)
    print(" TSP BENCHMARK RUNNER")
    print("=" * 80)
    print(f" Instâncias: {benchmark_config.instances}")
    print(f" Modos:      {[m.value for m in benchmark_config.modes]}")
    print(f" Rodadas:    {benchmark_config.runs} por (instância, modo)")
    timeout_label = f"{benchmark_config.timeout_seconds}s" if benchmark_config.timeout_seconds > 0 else "sem limite"
    print(f" Timeout:    {timeout_label} por rodada")
    print(f" Total de execuções: "
          f"{len(benchmark_config.instances) * len(benchmark_config.modes) * benchmark_config.runs}")

    start_time = time.time()
    summary, detailed = run_benchmark(benchmark_config)
    total_time = time.time() - start_time

    summary_path = save_summary(summary, benchmark_config.results_root)

    print("\n" + "=" * 80)
    print(" BENCHMARK CONCLUÍDO")
    print("=" * 80)
    print(f" Tempo total: {format_duration(total_time)} ({total_time:.2f}s)")
    print(f" Sumário salvo em: {summary_path.absolute()}")
    print(f" Resultados individuais em: {benchmark_config.results_root.absolute()}")

    if not arguments.no_plots:
        print("\n Gerando gráficos comparativos...")
        try:
            generated = generate_benchmark_reports(summary, detailed, benchmark_config.results_root)
            for path in generated:
                print(f"   - {path.absolute()}")
        except Exception as exception:
            print(f"   [AVISO] Falha ao gerar gráficos: {exception}")


if __name__ == "__main__":
    main()
