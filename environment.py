from typing import List
import numpy as np
import numpy.typing as npt


class TSPEnvironment:
    
    def __init__(self, distance_matrix: npt.NDArray[np.float64]) -> None:
        self.distance_matrix = distance_matrix
        self.number_of_cities = distance_matrix.shape[0]
        
    def evaluate_route(self, route: List[int]) -> float:
        indices = np.asarray(route, dtype=np.int64)
        return float(self.distance_matrix[indices, np.roll(indices, -1)].sum())
    
    def get_number_of_cities(self) -> int:
        return self.number_of_cities
