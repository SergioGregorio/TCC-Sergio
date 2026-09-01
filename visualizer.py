from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import numpy.typing as npt
import datetime
from config import VisualizationConfig


class TSPVisualizer:
    
    def __init__(
        self,
        coordinates: Optional[List[Tuple[float, float]]],
        config: VisualizationConfig,
        distance_matrix: npt.NDArray[np.float64],
        num_cities: int
    ) -> None:
        self.original_coordinates = coordinates
        self.config = config
        self.graph = nx.Graph()
        self.distance_matrix = distance_matrix
        self.num_cities = num_cities
        self.coordinates: Optional[List[Tuple[float, float]]] = None
        self.uses_artificial_layout = False
        
        self._build_complete_graph()
        self._resolve_coordinates()
    
    def _build_complete_graph(self) -> None:
        cities_count = self.num_cities if self.num_cities else len(self.original_coordinates)
        
        for i in range(cities_count):
            self.graph.add_node(i)
        
        for i in range(cities_count):
            for j in range(i + 1, cities_count):
                weight = self.distance_matrix[i][j]
                self.graph.add_edge(i, j, weight=weight)
    
    def _resolve_coordinates(self) -> None:
        if self.original_coordinates:
            self.coordinates = self.original_coordinates
        else:
            self.uses_artificial_layout = True
            self._generate_artificial_layout()
    
    def _generate_artificial_layout(self) -> None:
        try:
            pos = nx.kamada_kawai_layout(self.graph, weight="weight")
        except Exception:
            pos = nx.spring_layout(self.graph, seed=42)
        
        scale_factor = 1000.0
        self.coordinates = [
            (pos[i][0] * scale_factor, pos[i][1] * scale_factor)
            for i in range(len(pos))
        ]
    
    def plot_solution(self, best_route: List[int], best_distance: float, output_path: Path) -> None:
        self.plot_enhanced_solution(best_route, best_distance, output_path)
    
    def plot_fitness_evolution(self, fitness_history: List[float], output_path: Path) -> None:
        if not fitness_history:
            return
        
        generations = list(range(len(fitness_history)))
        initial_value = fitness_history[0]
        final_value = fitness_history[-1]
        best_index = int(np.argmin(fitness_history))
        best_value = fitness_history[best_index]
        improvement_percent = (
            (initial_value - final_value) / initial_value * 100 if initial_value else 0.0
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, fitness_history, color="#1f77b4", linewidth=2, label="Best distance")
        ax.fill_between(generations, fitness_history, final_value, color="#1f77b4", alpha=0.08)
        
        ax.scatter([0], [initial_value], color="#d62728", zorder=5, label=f"Initial: {initial_value:.2f}")
        ax.scatter([best_index], [best_value], color="#2ca02c", zorder=5, label=f"Best: {best_value:.2f}")
        ax.annotate(
            f"Best @ gen {best_index}\n{best_value:.2f}",
            (best_index, best_value),
            textcoords="offset points",
            xytext=(10, 20),
            fontsize=9,
            fontweight="bold",
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ca02c"),
            arrowprops=dict(arrowstyle="->", color="#2ca02c")
        )
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness (Distance)")
        ax.set_title(
            f"Convergence Curve  |  Improvement: {improvement_percent:.2f}%",
            fontweight="bold"
        )
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="upper right")
        fig.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.config.dpi)
        plt.close(fig)
    
    def plot_enhanced_solution(self, best_route: List[int], best_distance: float, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        route_coords = []
        for city in best_route:
            coord = self.coordinates[city]
            route_coords.append(coord)
        route_coords.append(self.coordinates[best_route[0]])
        
        layout_info = " (Artificial Layout)" if self.uses_artificial_layout else ""
        
        x_coords = [coord[0] for coord in route_coords]
        y_coords = [coord[1] for coord in route_coords]
        
        ax.plot(
            x_coords, y_coords, "o-",
            color=self.config.route_edge_color,
            linewidth=self.config.route_width,
            markerfacecolor=self.config.node_color,
            markeredgecolor="black",
            markersize=6
        )
        
        start_coord = self.coordinates[best_route[0]]
        ax.plot(start_coord[0], start_coord[1], "o", color=self.config.start_node_color, markersize=14, zorder=5)
        ax.annotate(
            "Start: City " + str(best_route[0]),
            (start_coord[0], start_coord[1]),
            textcoords="offset points",
            xytext=(28, 28),
            ha="left",
            va="bottom",
            fontweight="bold",
            color="navy",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="navy"),
            arrowprops=dict(arrowstyle="->", color="navy"),
            zorder=6
        )
        
        for city in best_route:
            x, y = self.coordinates[city]
            ax.annotate(
                str(city),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                va="bottom",
                fontsize=self.config.city_label_font_size,
                fontweight="bold",
                color="black"
            )
        
        ax.set_xlim(min(x_coords) - abs(min(x_coords)) * 0.05 - 1, max(x_coords) + abs(max(x_coords)) * 0.05 + 1)
        ax.set_ylim(min(y_coords) - abs(min(y_coords)) * 0.05 - 1, max(y_coords) + abs(max(y_coords)) * 0.05 + 1)
        ax.set_title(
            "Optimal TSP Route" + layout_info +
            "\nDistance: " + f"{best_distance:.2f}" + " units | Cities: " + str(len(best_route)),
            fontweight="bold"
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(["Route", "Start"], loc="upper right")
        ax.set_aspect("equal", "box")
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi)
        plt.close()
    
    def create_comprehensive_dashboard(
        self,
        best_route: List[int],
        best_distance: float,
        fitness_history: List[float],
        statistics: Dict[str, Any],
        output_path: Path
    ) -> None:
        fig = plt.figure(figsize=(22, 16))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.15], width_ratios=[1.25, 1])
        
        # --- Mapa da rota (esquerda, ocupando as duas primeiras linhas) ---
        ax_route = fig.add_subplot(gs[0:2, 0])
        positions = {i: self.coordinates[i] for i in range(len(self.coordinates))}
        route_edges = [
            (best_route[i], best_route[(i + 1) % len(best_route)])
            for i in range(len(best_route))
        ]
        
        nx.draw_networkx_edges(
            self.graph, positions, ax=ax_route,
            width=self.config.edge_width,
            alpha=self.config.background_edge_alpha,
            edge_color=self.config.background_edge_color
        )
        nx.draw_networkx_edges(
            self.graph, positions, ax=ax_route, edgelist=route_edges,
            width=self.config.route_width,
            alpha=self.config.route_edge_alpha,
            edge_color=self.config.route_edge_color
        )
        nx.draw_networkx_nodes(
            self.graph, positions, ax=ax_route,
            node_size=self.config.node_size,
            node_color=self.config.node_color
        )
        nx.draw_networkx_nodes(
            self.graph, positions, ax=ax_route, nodelist=[best_route[0]],
            node_size=self.config.start_node_size,
            node_color=self.config.start_node_color
        )
        ax_route.set_title("Optimal Route Visualization", fontweight="bold", color="darkgreen")
        ax_route.axis("off")
        
        # --- Evolução do fitness (direita, topo) ---
        ax_fitness = fig.add_subplot(gs[0, 1])
        generations = list(range(len(fitness_history)))
        ax_fitness.plot(generations, fitness_history, "-", color="#2E86AB")
        ax_fitness.set_xlabel("Generation")
        ax_fitness.set_ylabel("Best Distance")
        ax_fitness.set_title("Fitness Evolution", fontweight="bold")
        ax_fitness.grid(True, alpha=0.3)
        
        # --- Melhoria por geração (direita, meio) ---
        ax_improvement = fig.add_subplot(gs[1, 1])
        improvements = [
            fitness_history[i - 1] - fitness_history[i]
            for i in range(1, len(fitness_history))
        ]
        ax_improvement.bar(range(len(improvements)), improvements, color="#A23B72")
        ax_improvement.set_xlabel("Generation")
        ax_improvement.set_ylabel("Improvement")
        ax_improvement.set_title("Generation-to-Generation Improvement", fontweight="bold")
        ax_improvement.grid(True, axis="y", alpha=0.3)
        
        # --- Resumo (embaixo, largura total) ---
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis("off")
        stats_text = self._build_report_text(best_distance, statistics)
        ax_stats.text(
            0.5, 0.5, stats_text,
            family="monospace", ha="center", va="center",
            transform=ax_stats.transAxes, fontsize=10
        )
        
        dashboard_title = "TSP Genetic Algorithm - Complete Analysis Dashboard"
        if self.uses_artificial_layout:
            dashboard_title = dashboard_title + " [Graph-Based Layout]"
        fig.suptitle(dashboard_title, fontweight="bold", fontsize=16)
        
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()
    
    def _build_report_text(self, best_distance: float, statistics: Dict[str, Any]) -> str:
        col = 40
        
        def cell(label: str, value: str) -> str:
            spacing = max(1, col - len(label) - len(value))
            return label + " " * spacing + value
        
        def border() -> str:
            return "+" + "-" * (col + 2) + "+" + "-" * (col + 2) + "+"
        
        def full(text: str) -> str:
            return "| " + text.center(col * 2 + 3) + " |"
        
        def two(left_label: str, left_value: str, right_label: str, right_value: str) -> str:
            return "| " + cell(left_label, left_value) + " | " + cell(right_label, right_value) + " |"
        
        optimal = statistics.get("optimal_solution")
        gap = statistics.get("gap_from_optimal")
        
        lines = [
            border(),
            full("GENETIC ALGORITHM EXECUTION REPORT"),
            border(),
            full("SOLUTION METRICS" + " " * (col - 16) + "ALGORITHM PARAMETERS"),
            border(),
            two("Best distance found:", f"{best_distance:.2f} units",
                "Population size:", str(statistics.get("population_size"))),
            two("Number of cities:", str(self.num_cities),
                "Generations:", str(statistics.get("generations"))),
            two("Initial distance:", f"{statistics.get('initial_distance'):.2f} units",
                "Crossover rate:", f"{statistics.get('crossover_prob'):.1%}"),
            two("Improvement:", f"{statistics.get('improvement_percent'):.2f}%",
                "Mutation rate:", f"{statistics.get('mutation_prob'):.1%}"),
            two("Convergence generation:", str(statistics.get("convergence_gen")),
                "Tournament size:", str(statistics.get("tournament_size"))),
        ]
        
        if optimal:
            gap_value = f"{gap:.2f}%" if gap is not None else "N/A"
            lines.append(
                two("TSPLIB optimal:", f"{optimal} units",
                    "Gap from optimal:", gap_value)
            )
        
        lines += [
            border(),
            full("PERFORMANCE STATISTICS"),
            border(),
            two("Execution time:", f"{statistics.get('execution_time'):.2f} seconds",
                "Generations/second:", f"{statistics.get('gen_per_sec'):.2f}"),
            two("Avg improvement/gen:", f"{statistics.get('avg_improvement'):.4f}",
                "Best generation:", str(statistics.get("best_generation"))),
            border(),
            full("Generated: " + str(statistics.get("timestamp"))),
            border(),
        ]
        
        return "\n".join(lines)
    
    def plot_top_5_comparison(
        self,
        top_individuals_history: Dict[int, List[Tuple[List[int], float]]],
        selected_generations: List[int],
        output_path: Path
    ) -> None:
        num_generations = len(selected_generations)
        if num_generations == 0:
            return
        
        fig = plt.figure(figsize=(20, num_generations * 4))
        gs = GridSpec(num_generations, 5, figure=fig)
        
        for gen_idx, generation in enumerate(selected_generations):
            top_5 = top_individuals_history[generation]
            for rank, (route, distance) in enumerate(top_5):
                ax = fig.add_subplot(gs[gen_idx, rank])
                
                route_coords = []
                for city in route:
                    coord = self.coordinates[city]
                    route_coords.append(coord)
                route_coords.append(self.coordinates[route[0]])
                
                x_coords = [coord[0] for coord in route_coords]
                y_coords = [coord[1] for coord in route_coords]
                
                ax.plot(x_coords, y_coords, "o-", color="black", linewidth=1.5, markersize=3)
                start_coord = self.coordinates[route[0]]
                ax.plot(start_coord[0], start_coord[1], "o", color="green", markersize=8)
                ax.set_title(
                    "Gen " + str(generation) + " - Rank #" + str(rank + 1) + "\nDist: " + f"{distance:.1f}",
                    fontweight="bold"
                )
                ax.set_aspect("equal", "box")
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.tick_params(labelsize=8)
        
        fig.suptitle("Top 5 Individuals Evolution Across Generations", fontweight="bold", fontsize=16, y=0.995)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()
    
    def plot_generation_evolution_grid(
        self,
        top_individuals_history: Dict[int, List[Tuple[List[int], float]]],
        output_path: Path,
        num_snapshots: int = 12
    ) -> None:
        total_generations = max(top_individuals_history.keys())
        step = max(1, total_generations // num_snapshots)
        selected_gens = [g for g in range(1, total_generations + 1, step) if g in top_individuals_history]
        selected_gens = selected_gens[:num_snapshots]
        
        rows = (len(selected_gens) + 3) // 4
        cols = 4
        
        fig = plt.figure(figsize=(20, rows * 5))
        gs = GridSpec(rows, cols, figure=fig)
        
        for idx, generation in enumerate(selected_gens):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col])
            
            best_route, best_distance = top_individuals_history[generation][0]
            
            route_coords = []
            for city in best_route:
                coord = self.coordinates[city]
                route_coords.append(coord)
            route_coords.append(self.coordinates[best_route[0]])
            
            x_coords = [coord[0] for coord in route_coords]
            y_coords = [coord[1] for coord in route_coords]
            
            progress = generation / total_generations
            color = plt.cm.RdYlGn(progress)
            
            ax.plot(x_coords, y_coords, "o-", color=color, linewidth=1.5, markersize=3, markeredgecolor="black")
            start_coord = self.coordinates[best_route[0]]
            ax.plot(start_coord[0], start_coord[1], "o", color="lime", markersize=8)
            ax.set_title(
                "Generation " + str(generation) + "\nDistance: " + f"{best_distance:.2f}",
                fontweight="bold"
            )
            ax.set_aspect("equal", "box")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(labelsize=8)
            margin = 0.5
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        fig.suptitle("Best Individual Evolution Over Generations", fontweight="bold", fontsize=16, y=0.995)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close()
