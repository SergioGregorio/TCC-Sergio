from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import numpy.typing as npt


class TSPDataLoader:
    
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.coordinates: Dict[int, Tuple[float, float]] = {}
        self.distance_matrix: npt.NDArray[np.float64] = np.array([])
        self.dimension: int = 0
        self.edge_weight_type: str = ""
        self.edge_weight_format: str = ""
        self.has_spatial_coordinates: bool = False
    
    def load(self) -> None:
        self._parse_tsp_file()
        
        if self.edge_weight_type == "EXPLICIT":
            self.has_spatial_coordinates = False
            self.coordinates = {}
        else:
            self._compute_distance_matrix()
    
    def _parse_tsp_file(self) -> None:
        with open(self.file_path, "r") as file:
            lines = file.readlines()
        
        node_coord_section_started = False
        
        for position in range(len(lines)):
            line = lines[position].strip()
            
            if not line:
                continue
            
            if line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[1].strip())
                continue
            
            if line.startswith("EDGE_WEIGHT_TYPE"):
                self.edge_weight_type = line.split(":")[1].strip()
                continue
            
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                self.edge_weight_format = line.split(":")[1].strip()
                continue
            
            if line.startswith("NODE_COORD_SECTION"):
                node_coord_section_started = True
                continue
            
            if line.startswith("EDGE_WEIGHT_SECTION"):
                node_coord_section_started = False
                self._parse_explicit_weights(lines[position + 1:])
                continue
            
            if line.startswith("EOF"):
                break
            
            if node_coord_section_started:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x_coordinate = float(parts[1])
                    y_coordinate = float(parts[2])
                    self.coordinates[node_id - 1] = (x_coordinate, y_coordinate)
                    self.has_spatial_coordinates = True
    
    def _parse_explicit_weights(self, lines: List[str]) -> None:
        weights: List[float] = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("EOF") or line.startswith("DISPLAY_DATA_SECTION"):
                break
            if not line:
                continue
            parts = line.split()
            for part in parts:
                try:
                    weights.append(int(part))
                except ValueError:
                    continue
        
        self.distance_matrix = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        idx = 0
        
        if self.edge_weight_format == "UPPER_ROW":
            for i in range(self.dimension):
                for j in range(i + 1, self.dimension):
                    self.distance_matrix[i][j] = weights[idx]
                    self.distance_matrix[j][i] = weights[idx]
                    idx += 1
        elif self.edge_weight_format == "LOWER_ROW":
            for i in range(1, self.dimension):
                for j in range(i):
                    self.distance_matrix[i][j] = weights[idx]
                    self.distance_matrix[j][i] = weights[idx]
                    idx += 1
        elif self.edge_weight_format == "FULL_MATRIX":
            for i in range(self.dimension):
                for j in range(self.dimension):
                    self.distance_matrix[i][j] = weights[idx]
                    idx += 1
        else:
            for i in range(self.dimension):
                for j in range(i + 1, self.dimension):
                    if idx < len(weights):
                        self.distance_matrix[i][j] = weights[idx]
                        self.distance_matrix[j][i] = weights[idx]
                        idx += 1
    
    def _compute_distance_matrix(self) -> None:
        num_cities = len(self.coordinates)
        coords = np.array([self.coordinates[i] for i in range(num_cities)], dtype=np.float64)
        
        if self.edge_weight_type == "GEO":
            self.distance_matrix = self._compute_geo_matrix(coords)
            return
        
        squared_norms = np.einsum("ij,ij->i", coords, coords)
        squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (coords @ coords.T)
        np.maximum(squared_distances, 0.0, out=squared_distances)
        
        if self.edge_weight_type == "ATT":
            rij = np.sqrt(squared_distances / 10.0)
            tij = np.round(rij)
            matrix = np.where(tij < rij, tij + 1.0, tij)
        elif self.edge_weight_type == "CEIL_2D":
            matrix = np.ceil(np.sqrt(squared_distances))
        else:
            matrix = np.sqrt(squared_distances)
        
        np.fill_diagonal(matrix, 0.0)
        self.distance_matrix = matrix
    
    def _compute_geo_matrix(self, coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        PI = 3.141592
        RRR = 6378.388
        
        deg_x = coords[:, 0].astype(int)
        min_x = coords[:, 0] - deg_x
        latitudes = PI * (deg_x + 5.0 * min_x / 3.0) / 180.0
        
        deg_y = coords[:, 1].astype(int)
        min_y = coords[:, 1] - deg_y
        longitudes = PI * (deg_y + 5.0 * min_y / 3.0) / 180.0
        
        q1 = np.cos(longitudes[:, None] - longitudes[None, :])
        q2 = np.cos(latitudes[:, None] - latitudes[None, :])
        q3 = np.cos(latitudes[:, None] + latitudes[None, :])
        argument = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
        matrix = (RRR * np.arccos(argument) + 1.0).astype(int).astype(np.float64)
        
        np.fill_diagonal(matrix, 0.0)
        return matrix
    
    def _calculate_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        if self.edge_weight_type == "ATT":
            return self._calculate_att_distance(point_a, point_b)
        if self.edge_weight_type == "CEIL_2D":
            return self._calculate_ceil_2d_distance(point_a, point_b)
        if self.edge_weight_type == "GEO":
            return self._calculate_geo_distance(point_a, point_b)
        return self._calculate_euclidean_distance(point_a, point_b)
    
    def _calculate_euclidean_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        x_diff = point_a[0] - point_b[0]
        y_diff = point_a[1] - point_b[1]
        return float(np.sqrt(x_diff * x_diff + y_diff * y_diff))
    
    def _calculate_att_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        x_diff = point_a[0] - point_b[0]
        y_diff = point_a[1] - point_b[1]
        rij = np.sqrt((x_diff * x_diff + y_diff * y_diff) / 10.0)
        tij = int(round(rij))
        if tij < rij:
            return float(tij + 1)
        return float(tij)
    
    def _calculate_ceil_2d_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        x_diff = point_a[0] - point_b[0]
        y_diff = point_a[1] - point_b[1]
        return float(np.ceil(np.sqrt(x_diff * x_diff + y_diff * y_diff)))
    
    def _calculate_geo_distance(self, point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
        PI = 3.141592
        RRR = 6378.388
        
        deg_a_x = int(point_a[0])
        min_a_x = point_a[0] - deg_a_x
        lat_a = PI * (deg_a_x + 5.0 * min_a_x / 3.0) / 180.0
        
        deg_a_y = int(point_a[1])
        min_a_y = point_a[1] - deg_a_y
        lon_a = PI * (deg_a_y + 5.0 * min_a_y / 3.0) / 180.0
        
        deg_b_x = int(point_b[0])
        min_b_x = point_b[0] - deg_b_x
        lat_b = PI * (deg_b_x + 5.0 * min_b_x / 3.0) / 180.0
        
        deg_b_y = int(point_b[1])
        min_b_y = point_b[1] - deg_b_y
        lon_b = PI * (deg_b_y + 5.0 * min_b_y / 3.0) / 180.0
        
        q1 = np.cos(lon_a - lon_b)
        q2 = np.cos(lat_a - lat_b)
        q3 = np.cos(lat_a + lat_b)
        
        return float(int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0))
    
    def get_coordinates_as_list(self) -> Optional[List[Tuple[float, float]]]:
        if not self.coordinates:
            return None
        return [self.coordinates[i] for i in range(len(self.coordinates))]
    
    def get_distance_matrix(self) -> npt.NDArray[np.float64]:
        return self.distance_matrix
    
    def get_number_of_cities(self) -> int:
        if self.dimension > 0:
            return self.dimension
        return len(self.coordinates)
    
    def has_coordinates(self) -> bool:
        return self.has_spatial_coordinates and len(self.coordinates) > 0
