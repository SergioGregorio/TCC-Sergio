from pathlib import Path
from typing import List
import time
from datetime import datetime
from config import ApplicationConfig
from data_loader import TSPDataLoader
from environment import TSPEnvironment
from engine import GeneticAlgorithmEngine
from orchestrator import GeneticAlgorithmOrchestrator
from visualizer import TSPVisualizer
from solutions_reader import TSPLibSolutionsReader


BANNER_WIDTH = 80
BOX_INNER = 60


def print_banner(title: str) -> None:
    print("\n" + "=" * BANNER_WIDTH)
    print("#" * BANNER_WIDTH)
    print("#" + title.center(BANNER_WIDTH - 2) + "#")
    print("#" * BANNER_WIDTH)
    print("=" * BANNER_WIDTH)


def print_section(title: str) -> None:
    print("\n" + "=" * BANNER_WIDTH)
    print(f" {title}")
    print("=" * BANNER_WIDTH)


def box_top() -> str:
    return "+" + "-" * (BOX_INNER + 2) + "+"


def box_sep() -> str:
    return "+" + "-" * (BOX_INNER + 2) + "+"


def box_bottom() -> str:
    return "+" + "-" * (BOX_INNER + 2) + "+"


def box_title(text: str) -> str:
    return "| " + text.center(BOX_INNER) + " |"


def box_row(label: str, value: str) -> str:
    spacing = max(1, BOX_INNER - len(label) - len(value))
    return "| " + label + " " * spacing + value + " |"


def main() -> None:
    config = ApplicationConfig()
    
    print_banner("TSP GENETIC ALGORITHM SOLVER")
    
    print(f"\nLoading TSP data from: {config.data.tsp_file_path}")
    
    data_loader = TSPDataLoader(config.data.tsp_file_path)
    data_loader.load()
    
    solutions_reader = TSPLibSolutionsReader()
    optimal_solution = solutions_reader.get_optimal_solution(config.data.tsp_file_path)
    
    print()
    print(box_top())
    print(box_title("CONFIGURATION SUMMARY"))
    print(box_sep())
    print(box_row("Number of cities:", str(data_loader.get_number_of_cities())))
    print(box_row("Population size:", str(config.genetic_algorithm.population_size)))
    print(box_row("Generations:", str(config.genetic_algorithm.number_of_generations)))
    print(box_row("Crossover probability:", f"{config.genetic_algorithm.crossover_probability:.1%}"))
    print(box_row("Mutation probability:", f"{config.genetic_algorithm.mutation_probability:.1%}"))
    print(box_row("Parallel processes:", str(config.genetic_algorithm.number_of_processes)))
    print(box_row("Execution mode:", config.genetic_algorithm.execution_mode.value))
    if optimal_solution:
        print(box_row("TSPLIB optimal solution:", str(optimal_solution)))
    print(box_bottom())
    
    environment = TSPEnvironment(data_loader.get_distance_matrix())
    engine = GeneticAlgorithmEngine(environment)
    
    orchestrator = GeneticAlgorithmOrchestrator(
        engine=engine,
        config=config.genetic_algorithm,
        number_of_cities=data_loader.get_number_of_cities()
    )
    
    visualizer = TSPVisualizer(
        coordinates=data_loader.get_coordinates_as_list(),
        config=config.visualization,
        distance_matrix=data_loader.get_distance_matrix(),
        num_cities=data_loader.get_number_of_cities()
    )
    
    if visualizer.uses_artificial_layout:
        print("\nNote: Using Kamada-Kawai graph layout (no spatial coordinates in file)")
    
    print_section("STARTING GENETIC ALGORITHM EVOLUTION")
    
    def progress_callback(generation: int, total: int, fitness: float, route: List[int]) -> None:
        visualizer.print_progress_bar(generation, total, fitness)
    
    start_time = time.time()
    
    if config.genetic_algorithm.enable_live_visualization:
        best_route, best_distance = orchestrator.run_evolution(progress_callback=progress_callback)
    else:
        best_route, best_distance = orchestrator.run_evolution()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print()
    print_section("EVOLUTION COMPLETED SUCCESSFULLY")
    
    if orchestrator.was_early_stopped():
        print(f"\nEarly stopping triggered at generation {orchestrator.get_early_stopped_generation()} "
              f"(no improvement for {config.genetic_algorithm.early_stopping_patience} generations)")
    
    initial_distance = orchestrator.get_initial_fitness()
    improvement = initial_distance - best_distance
    improvement_percent = (improvement / initial_distance) * 100
    
    gap_from_optimal = None
    if optimal_solution:
        gap_from_optimal = solutions_reader.calculate_gap(config.data.tsp_file_path, best_distance)
    
    print()
    print(box_top())
    print(box_title("FINAL RESULTS"))
    print(box_sep())
    print(box_row("Best distance found:", f"{best_distance:.2f} units"))
    if optimal_solution:
        print(box_row("TSPLIB optimal solution:", f"{optimal_solution} units"))
        if gap_from_optimal is not None:
            print(box_row("Gap from optimal:", f"{gap_from_optimal:.2f}%"))
    print(box_row("Initial distance:", f"{initial_distance:.2f} units"))
    print(box_row("Total improvement:", f"{improvement:.2f} units"))
    print(box_row("Improvement percentage:", f"{improvement_percent:.2f}%"))
    print(box_row("Best found at generation:", str(orchestrator.get_best_generation())))
    print(box_row("Generations executed:", str(orchestrator.get_total_generations_run())))
    print(box_row("Execution time:", f"{execution_time:.2f} seconds"))
    print(box_row("Generations/second:", f"{orchestrator.get_total_generations_run()/execution_time:.2f}"))
    print(box_bottom())
    
    print(f"\nOptimal route (first 10): {best_route[:10]}{'...' if len(best_route) > 10 else ''}")
    
    print_section("GENERATING VISUALIZATIONS")
    
    solution_output_path = config.data.output_directory / "tsp_solution.png"
    fitness_output_path = config.data.output_directory / "fitness_evolution.png"
    dashboard_output_path = config.data.output_directory / "complete_dashboard.png"
    evolution_grid_path = config.data.output_directory / "evolution_grid.png"
    top5_comparison_path = config.data.output_directory / "top5_comparison.png"
    
    print("\n  [1/5] Generating clean route visualization...")
    visualizer.plot_solution(best_route, best_distance, solution_output_path)
    print(f"        Saved to: {solution_output_path}")
    
    print("\n  [2/5] Generating fitness evolution plot...")
    visualizer.plot_fitness_evolution(orchestrator.get_fitness_history(), fitness_output_path)
    print(f"        Saved to: {fitness_output_path}")
    
    print("\n  [3/5] Generating comprehensive dashboard...")
    
    statistics = {
        'population_size': config.genetic_algorithm.population_size,
        'generations': config.genetic_algorithm.number_of_generations,
        'crossover_prob': config.genetic_algorithm.crossover_probability,
        'mutation_prob': config.genetic_algorithm.mutation_probability,
        'tournament_size': config.genetic_algorithm.tournament_size,
        'initial_distance': initial_distance,
        'improvement_percent': improvement_percent,
        'convergence_gen': orchestrator.get_convergence_generation(),
        'execution_time': execution_time,
        'gen_per_sec': orchestrator.get_total_generations_run() / execution_time,
        'avg_improvement': improvement / orchestrator.get_total_generations_run(),
        'best_generation': orchestrator.get_best_generation(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimal_solution': optimal_solution if optimal_solution else 0,
        'gap_from_optimal': gap_from_optimal if gap_from_optimal is not None else 0
    }
    
    visualizer.create_comprehensive_dashboard(
        best_route,
        best_distance,
        orchestrator.get_fitness_history(),
        statistics,
        dashboard_output_path
    )
    print(f"        Saved to: {dashboard_output_path}")
    
    print("\n  [4/5] Generating evolution grid (12 snapshots)...")
    top_individuals_history = orchestrator.get_top_individuals_history()
    visualizer.plot_generation_evolution_grid(
        top_individuals_history,
        evolution_grid_path,
        num_snapshots=12
    )
    print(f"        Saved to: {evolution_grid_path}")
    
    print("\n  [5/5] Generating top 5 comparison (key generations)...")
    total_gens = config.genetic_algorithm.number_of_generations
    key_generations = [
        1,
        total_gens // 4,
        total_gens // 2,
        3 * total_gens // 4,
        total_gens
    ]
    key_generations = [g for g in key_generations if g in top_individuals_history]
    
    visualizer.plot_top_5_comparison(
        top_individuals_history,
        key_generations,
        top5_comparison_path
    )
    print(f"        Saved to: {top5_comparison_path}")
    
    print_section("ALL PROCESSES COMPLETED SUCCESSFULLY")
    print(f"\nAll outputs saved to: {config.data.output_directory.absolute()}")
    print("\nGenerated 5 visualization files:")
    print(f"   1. {solution_output_path.name} - Clean optimal route")
    print(f"   2. {fitness_output_path.name} - Fitness evolution chart")
    print(f"   3. {dashboard_output_path.name} - Complete statistics dashboard")
    print(f"   4. {evolution_grid_path.name} - Evolution grid (12 snapshots)")
    print(f"   5. {top5_comparison_path.name} - Top 5 individuals comparison\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise
