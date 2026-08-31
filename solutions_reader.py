from pathlib import Path
from typing import Optional, Dict


class TSPLibSolutionsReader:
    
    def __init__(self, solutions_file_path: Path = Path("data/solutions")) -> None:
        self.solutions_file_path = solutions_file_path
        self.solutions: Dict[str, int] = {}
        self._load_solutions()
    
    def _load_solutions(self) -> None:
        if not self.solutions_file_path.exists():
            return
        
        with open(self.solutions_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                
                parts = line.split(':')
                if len(parts) >= 2:
                    problem_name = parts[0].strip()
                    distance_str = parts[1].strip()
                    
                    distance_str = distance_str.split('(')[0].strip()
                    
                    try:
                        optimal_distance = int(distance_str)
                        self.solutions[problem_name] = optimal_distance
                    except ValueError:
                        continue
    
    def get_optimal_solution(self, tsp_file_path: Path) -> Optional[int]:
        problem_name = tsp_file_path.stem
        return self.solutions.get(problem_name)
    
    def calculate_gap(self, tsp_file_path: Path, found_distance: float) -> Optional[float]:
        optimal = self.get_optimal_solution(tsp_file_path)
        if optimal is None:
            return None
        
        gap = ((found_distance - optimal) / optimal) * 100
        return gap
