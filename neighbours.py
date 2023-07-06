from itertools import product
import numpy as np
from numba import njit

@njit(parallel = True)
def count_alive_neighbours_1d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
    
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    
    return state[left] + state[right]

@njit(parallel = True)
def count_alive_neighbours_2d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
        
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    top = (site[1] - 1 + N) % N
    bottom = (site[1] + 1) % N
                
    return state[left, site[1]] + state[right, site[1]] + state[site[0], top] + state[site[0], bottom] + state[left, top] + state[right, top] + state[left, bottom] + state[right, bottom]

@njit(parallel = True)
def count_alive_neighbours_3d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
        
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    top = (site[1] - 1 + N) % N
    bottom = (site[1] + 1) % N
    front = (site[2] - 1 + N) % N
    back = (site[2] + 1) % N
    
    total_sum = 0

    x = site[0]
    y = site[1]
    z = site[2]

    total_sum += state[left, y, z]
    total_sum += state[right, y, z]
    total_sum += state[x, top, z]
    total_sum += state[x, bottom, z]
    total_sum += state[left, top, z]
    total_sum += state[right, top, z]
    total_sum += state[left, bottom, z]
    total_sum += state[right, bottom, z]
    total_sum += state[left, y, front]
    total_sum += state[right, y, front]
    total_sum += state[left, y, back]
    total_sum += state[right, y, back]
    total_sum += state[x, top, front]
    total_sum += state[x, top, back]
    total_sum += state[x, bottom, front]
    total_sum += state[x, bottom, back]
    total_sum += state[left, top, front]
    total_sum += state[right, top, front]
    total_sum += state[left, bottom, front]
    total_sum += state[right, bottom, front]
    total_sum += state[left, top, back]
    total_sum += state[right, top, back]
    total_sum += state[left, bottom, back]
    total_sum += state[right, bottom, back]
    total_sum += state[x, top-1, z]
    total_sum += state[x, bottom+1, z]
    total_sum += state[x, top, front-1]
    total_sum += state[x, bottom, back+1]
    
    return total_sum