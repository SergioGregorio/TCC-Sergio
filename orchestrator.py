from typing import List, Tuple, Any, Callable, Optional, Dict
import random
import contextlib
import numpy as np
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from config import GeneticAlgorithmConfig, ExecutionMode
from engine import GeneticAlgorithmEngine
from local_search import TSPLocalSearch


PARALLEL_MAX_CITIES = 800


class GeneticAlgorithmOrchestrator:
    
    def __init__(
        self, 
        engine: GeneticAlgorithmEngine, 
        config: GeneticAlgorithmConfig,
        number_of_cities: int
    ) -> None:
        self.engine = engine
        self.config = config
        self.number_of_cities = number_of_cities
        self.toolbox = base.Toolbox()
        self.best_individual: List[int] = []
        self.best_fitness: float = float('inf')
        self.fitness_history: List[float] = []
        self.initial_fitness: float = 0.0
        self.best_generation: int = 0
        self.convergence_generation: int = 0
        self.convergence_threshold: float = 0.001
        self.generations_without_improvement: int = 0
        self.early_stopped_generation: int = 0
        self.total_generations_run: int = 0
        self.top_individuals_history: Dict[int, List[Tuple[List[int], float]]] = {}
        self.local_search = TSPLocalSearch(
            self.engine.environment.distance_matrix,
            neighbor_count=config.local_search_neighbors
        )
        
        self._setup_deap_framework()
        
    def _clone_individual(self, individual: List[int]) -> List[int]:
        return creator.Individual(individual[:])
    
    def _setup_deap_framework(self) -> None:
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox.register(
            "indices", 
            random.sample, 
            range(self.number_of_cities), 
            self.number_of_cities
        )
        self.toolbox.register(
            "individual", 
            tools.initIterate, 
            creator.Individual, 
            self.toolbox.indices
        )
        self.toolbox.register(
            "population", 
            tools.initRepeat, 
            list, 
            self.toolbox.individual
        )
        
        self.toolbox.register("evaluate", self.engine.compute_fitness)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        self.toolbox.register(
            "select", 
            tools.selTournament, 
            tournsize=self.config.tournament_size
        )
        self.toolbox.register("clone", self._clone_individual)
        
    def run_evolution(
        self,
        progress_callback: Optional[Callable[[int, int, float, List[int]], None]] = None
    ) -> Tuple[List[int], float]:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        if self.config.execution_mode == ExecutionMode.LS_ONLY:
            return self._run_local_search_only(progress_callback)
        
        return self._run_genetic(progress_callback)
    
    def _run_genetic(
        self,
        progress_callback: Optional[Callable[[int, int, float, List[int]], None]] = None
    ) -> Tuple[List[int], float]:
        use_local_search = (
            self.config.execution_mode == ExecutionMode.HYBRID
            and self.config.enable_local_search
        )
        
        population = self.toolbox.population(n=self.config.population_size)
        
        use_parallel = (
            self.config.number_of_processes > 1
            and self.number_of_cities <= PARALLEL_MAX_CITIES
        )
        pool_context = (
            Pool(processes=self.config.number_of_processes)
            if use_parallel else contextlib.nullcontext(None)
        )
        
        with pool_context as pool:
            self.toolbox.register("map", pool.map if pool is not None else map)
            
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit
            
            initial_best = tools.selBest(population, 1)[0]
            self.initial_fitness = initial_best.fitness.values[0]
            self.best_fitness = self.initial_fitness
            self.best_individual = list(initial_best)
            self.fitness_history.append(self.initial_fitness)
                
            for generation in range(self.config.number_of_generations):
                offspring_size = len(population) - self.config.elitism_count
                offspring = self.toolbox.select(population, offspring_size)
                offspring = list(map(self.toolbox.clone, offspring))
                
                for i in range(0, len(offspring) - 1, 2):
                    if random.random() < self.config.crossover_probability:
                        self.toolbox.mate(offspring[i], offspring[i + 1])
                        del offspring[i].fitness.values
                        del offspring[i + 1].fitness.values
                        
                for mutant in offspring:
                    if random.random() < self.config.mutation_probability:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        
                invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
                for ind, fit in zip(invalid_individuals, fitnesses):
                    ind.fitness.values = fit
                    
                elite = tools.selBest(population, self.config.elitism_count)
                elite_clones = [self.toolbox.clone(ind) for ind in elite]
                
                if use_local_search and (generation + 1) % self.config.local_search_interval == 0:
                    self._apply_local_search(offspring)
                
                population[:] = elite_clones + offspring
                
                generation_best = tools.selBest(population, 1)[0]
                generation_best_fitness = generation_best.fitness.values[0]
                self.fitness_history.append(generation_best_fitness)
                
                previous_best = self.best_fitness
                if generation_best_fitness < self.best_fitness:
                    self.best_fitness = generation_best_fitness
                    self.best_individual = list(generation_best)
                    self.best_generation = generation + 1
                    self.generations_without_improvement = 0
                else:
                    self.generations_without_improvement += 1
                
                if self.convergence_generation == 0:
                    if self.generations_without_improvement >= 50:
                        self.convergence_generation = generation + 1 - 50
                
                top_5 = tools.selBest(population, min(5, len(population)))
                self.top_individuals_history[generation + 1] = [
                    (list(ind), ind.fitness.values[0]) for ind in top_5
                ]
                
                if progress_callback and (generation + 1) % self.config.display_interval == 0:
                    progress_callback(generation + 1, self.config.number_of_generations, self.best_fitness, self.best_individual)
                
                self.total_generations_run = generation + 1
                
                if self.config.enable_early_stopping:
                    if self.generations_without_improvement >= self.config.early_stopping_patience:
                        self.early_stopped_generation = generation + 1
                        break
            
            if use_local_search:
                final_route, final_distance = self.local_search.two_opt(self.best_individual)
                if final_distance < self.best_fitness:
                    self.best_fitness = final_distance
                    self.best_individual = final_route
                    
        return self.best_individual, self.best_fitness
    
    def _run_local_search_only(
        self,
        progress_callback: Optional[Callable[[int, int, float, List[int]], None]] = None
    ) -> Tuple[List[int], float]:
        max_restarts = self.config.number_of_generations
        patience = self.config.early_stopping_patience
        
        initial_route = list(np.random.permutation(self.number_of_cities))
        self.initial_fitness = self.local_search._calculate_route_distance(initial_route)
        
        best_route, best_distance = self.local_search.two_opt(initial_route)
        self.best_fitness = best_distance
        self.best_individual = best_route
        self.best_generation = 1
        self.fitness_history.append(self.best_fitness)
        self.top_individuals_history[1] = [(best_route, best_distance)]
        self.total_generations_run = 1
        
        without_improvement = 0
        for restart in range(1, max_restarts):
            candidate = list(np.random.permutation(self.number_of_cities))
            candidate_route, candidate_distance = self.local_search.two_opt(candidate)
            
            if candidate_distance < self.best_fitness:
                self.best_fitness = candidate_distance
                self.best_individual = candidate_route
                self.best_generation = restart + 1
                without_improvement = 0
            else:
                without_improvement += 1
            
            self.fitness_history.append(self.best_fitness)
            self.top_individuals_history[restart + 1] = [(candidate_route, candidate_distance)]
            self.total_generations_run = restart + 1
            
            if progress_callback and (restart + 1) % self.config.display_interval == 0:
                progress_callback(restart + 1, max_restarts, self.best_fitness, self.best_individual)
            
            if self.config.enable_early_stopping and without_improvement >= patience:
                self.early_stopped_generation = restart + 1
                break
        
        return self.best_individual, self.best_fitness
    
    def _apply_local_search(self, population: List[Any]) -> None:
        num_to_improve = max(1, int(len(population) * self.config.local_search_improvement_rate))
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])
        
        for i in range(num_to_improve):
            individual = sorted_population[i]
            improved_route, improved_distance = self.local_search.two_opt_fast(list(individual))
            
            if improved_distance < individual.fitness.values[0]:
                individual[:] = improved_route
                individual.fitness.values = (improved_distance,)
    
    def get_fitness_history(self) -> List[float]:
        return self.fitness_history
    
    def get_initial_fitness(self) -> float:
        return self.initial_fitness
    
    def get_best_generation(self) -> int:
        return self.best_generation
    
    def get_convergence_generation(self) -> int:
        if self.convergence_generation == 0:
            return self.config.number_of_generations
        return self.convergence_generation
    
    def get_top_individuals_history(self) -> Dict[int, List[Tuple[List[int], float]]]:
        return self.top_individuals_history
    
    def get_total_generations_run(self) -> int:
        return self.total_generations_run
    
    def was_early_stopped(self) -> bool:
        return self.early_stopped_generation > 0
    
    def get_early_stopped_generation(self) -> int:
        return self.early_stopped_generation
