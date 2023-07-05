from itertools import product
import numpy as np

def count_alive_neighbours(site: np.ndarray, state: np.ndarray) -> int:
        
    neighbor_indices = (site + np.array(list(product(np.array([0, -1, 1]), repeat=len(site))))[1:]) % len(state)
    
    neighbors = state[tuple(neighbor_indices.T)]
    
    num_alive_neighbors = np.sum(neighbors)
    
    return num_alive_neighbors
