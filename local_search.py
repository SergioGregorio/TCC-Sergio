from typing import List, Tuple, Optional
from collections import deque
import numpy as np
import numpy.typing as npt


class TSPLocalSearch:
    
    def __init__(self, distance_matrix: npt.NDArray[np.float64], neighbor_count: int = 10) -> None:
        self.distance_matrix = distance_matrix
        self.number_of_cities = distance_matrix.shape[0]
        self.neighbor_count = min(neighbor_count, max(1, self.number_of_cities - 1))
        self.neighbors = self._build_neighbor_lists(self.neighbor_count)
    
    def _build_neighbor_lists(self, k: int) -> npt.NDArray[np.int32]:
        n = self.number_of_cities
        neighbors = np.empty((n, k), dtype=np.int32)
        distance_matrix = self.distance_matrix
        chunk = 512
        
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = distance_matrix[start:end].copy()
            local_rows = np.arange(end - start)
            block[local_rows, np.arange(start, end)] = np.inf
            
            nearest = np.argpartition(block, kth=k - 1, axis=1)[:, :k]
            nearest_distances = np.take_along_axis(block, nearest, axis=1)
            order = np.argsort(nearest_distances, axis=1)
            neighbors[start:end] = np.take_along_axis(nearest, order, axis=1)
        
        return neighbors
    
    def two_opt(self, route: List[int], max_passes: Optional[int] = None) -> Tuple[List[int], float]:
        n = self.number_of_cities
        tour = np.asarray(route, dtype=np.int64).copy()
        position = np.empty(n, dtype=np.int64)
        position[tour] = np.arange(n)
        
        iteration_cap = 0 if max_passes is None else max_passes * n
        
        queue = deque(int(city) for city in tour)
        in_queue = np.ones(n, dtype=bool)
        iterations = 0
        
        while queue:
            if iteration_cap and iterations >= iteration_cap:
                break
            city = queue.popleft()
            in_queue[city] = False
            iterations += 1
            self._improve_city(city, tour, position, queue, in_queue)
        
        best = tour.tolist()
        return best, self._calculate_route_distance(best)
    
    def _improve_city(
        self,
        c1: int,
        tour: npt.NDArray[np.int64],
        position: npt.NDArray[np.int64],
        queue: deque,
        in_queue: npt.NDArray[np.bool_]
    ) -> bool:
        n = self.number_of_cities
        distance_matrix = self.distance_matrix
        p1 = position[c1]
        succ1 = tour[(p1 + 1) % n]
        pred1 = tour[(p1 - 1) % n]
        row = distance_matrix[c1]
        d_succ = row[succ1]
        d_pred = row[pred1]
        radius = d_succ if d_succ > d_pred else d_pred
        
        for c3 in self.neighbors[c1]:
            d13 = row[c3]
            if d13 >= radius:
                break
            
            if d13 < d_succ:
                succ3 = tour[(position[c3] + 1) % n]
                if c3 != succ1 and succ3 != c1:
                    gain = (d_succ + distance_matrix[c3, succ3]) - (d13 + distance_matrix[succ1, succ3])
                    if gain > 1e-9:
                        self._apply_move(tour, position, int(p1), int(position[c3]))
                        self._activate((c1, succ1, c3, succ3), queue, in_queue)
                        return True
            
            if d13 < d_pred:
                pred3 = tour[(position[c3] - 1) % n]
                if c3 != pred1 and pred3 != c1:
                    gain = (d_pred + distance_matrix[c3, pred3]) - (d13 + distance_matrix[pred1, pred3])
                    if gain > 1e-9:
                        self._apply_move(tour, position, int((p1 - 1) % n), int((position[c3] - 1) % n))
                        self._activate((c1, pred1, c3, pred3), queue, in_queue)
                        return True
        
        return False
    
    def _apply_move(
        self,
        tour: npt.NDArray[np.int64],
        position: npt.NDArray[np.int64],
        i: int,
        j: int
    ) -> None:
        n = self.number_of_cities
        a = (i + 1) % n
        b = j % n
        segment_length = (b - a) % n + 1
        
        if segment_length * 2 <= n:
            self._reverse(tour, position, a, b, segment_length)
        else:
            a2 = (j + 1) % n
            b2 = i % n
            self._reverse(tour, position, a2, b2, (b2 - a2) % n + 1)
    
    def _reverse(
        self,
        tour: npt.NDArray[np.int64],
        position: npt.NDArray[np.int64],
        a: int,
        b: int,
        segment_length: int
    ) -> None:
        n = self.number_of_cities
        for _ in range(segment_length // 2):
            tour[a], tour[b] = tour[b], tour[a]
            position[tour[a]] = a
            position[tour[b]] = b
            a = (a + 1) % n
            b = (b - 1) % n
    
    def _activate(
        self,
        cities: Tuple[int, ...],
        queue: deque,
        in_queue: npt.NDArray[np.bool_]
    ) -> None:
        for city in cities:
            city = int(city)
            if not in_queue[city]:
                in_queue[city] = True
                queue.append(city)
    
    def two_opt_fast(self, route: List[int], max_passes: int = 2) -> Tuple[List[int], float]:
        return self.two_opt(route, max_passes=max_passes)
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        indices = np.asarray(route, dtype=np.int64)
        return float(self.distance_matrix[indices, np.roll(indices, -1)].sum())
    
    def improve_population(
        self,
        population: List[List[int]],
        improvement_rate: float = 0.2,
        max_passes: int = 2
    ) -> List[List[int]]:
        improved_population = []
        improvement_count = int(len(population) * improvement_rate)
        
        for i, individual in enumerate(population):
            if i < improvement_count:
                improved_route, _ = self.two_opt(individual, max_passes=max_passes)
                improved_population.append(improved_route)
            else:
                improved_population.append(individual)
        
        return improved_population
