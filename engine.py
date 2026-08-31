from typing import List, Tuple
from environment import TSPEnvironment


class GeneticAlgorithmEngine:
    
    def __init__(self, environment: TSPEnvironment) -> None:
        self.environment = environment
        
    def compute_fitness(self, individual: List[int]) -> Tuple[float]:
        route_distance = self.environment.evaluate_route(individual)
        return (route_distance,)
