from itertools import product
import numpy as np
from numba import njit

@njit()
def count_alive_neighbours_1d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
    
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    
    return state[left] + state[right]

@njit()
def count_alive_neighbours_2d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
        
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    top = (site[1] - 1 + N) % N
    bottom = (site[1] + 1) % N
                
    return state[left, site[1]] + state[right, site[1]] + state[site[0], top] + state[site[0], bottom] + state[left, top] + state[right, top] + state[left, bottom] + state[right, bottom]

@njit()
def count_alive_neighbours_3d(site: np.ndarray, state: np.ndarray) -> int:
    
    N = len(state)
        
    left = (site[0] - 1 + N) % N
    right = (site[0] + 1) % N
    top = (site[1] - 1 + N) % N
    bottom = (site[1] + 1) % N
    front = (site[2] - 1 + N) % N
    back = (site[2] + 1) % N

    x = site[0]
    y = site[1]
    z = site[2]

    return state[left, y, z] + state[right, y, z] + state[x, top, z] + state[x, bottom, z] + state[left, top, z] + state[right, top, z] + state[left, bottom, z] + state[right, bottom, z] + state[left, y, front] + state[right, y, front] + state[left, y, back] + state[right, y, back] + state[x, top, front] + state[x, top, back] + state[x, bottom, front] + state[x, bottom, back] + state[left, top, front] + state[right, top, front] + state[left, bottom, front] + state[right, bottom, front] + state[left, top, back] + state[right, top, back] + state[left, bottom, back] + state[right, bottom, back] + state[x, top-1, z] + state[x, bottom+1, z] + state[x, top, front-1] + state[x, bottom, back+1]
    