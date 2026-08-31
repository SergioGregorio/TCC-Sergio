from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import randint


class ExecutionMode(str, Enum):
    GA_ONLY = "GA_ONLY"
    LS_ONLY = "LS_ONLY"
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 300
    number_of_generations: int = 500
    crossover_probability: float = 0.85
    mutation_probability: float = 0.20
    tournament_size: int = 3
    elitism_count: int = 5
    number_of_processes: int = 4
    random_seed: int = randint(0, 10000)
    display_interval: int = 25
    enable_live_visualization: bool = True
    enable_local_search: bool = True
    local_search_interval: int = 20
    local_search_improvement_rate: float = 0.1
    local_search_neighbors: int = 10
    enable_early_stopping: bool = True
    early_stopping_patience: int = 100
    execution_mode: ExecutionMode = ExecutionMode.HYBRID


@dataclass(frozen=True)
class DataConfig:
    tsp_file_path: Path = Path("data/att48.tsp")
    output_directory: Path = Path("output")


@dataclass(frozen=True)
class VisualizationConfig:
    figure_width: int = 12
    figure_height: int = 10
    node_size: int = 50
    edge_width: float = 0.5
    route_width: float = 2.5
    background_edge_alpha: float = 0.2
    route_edge_alpha: float = 1.0
    background_edge_color: str = "gray"
    route_edge_color: str = "red"
    node_color: str = "lightblue"
    dpi: int = 150
    start_node_color: str = "green"
    start_node_size: int = 150
    city_label_font_size: int = 8


@dataclass(frozen=True)
class ApplicationConfig:
    genetic_algorithm: GeneticAlgorithmConfig = GeneticAlgorithmConfig()
    data: DataConfig = DataConfig()
    visualization: VisualizationConfig = VisualizationConfig()
