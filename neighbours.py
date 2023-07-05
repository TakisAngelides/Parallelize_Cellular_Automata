from itertools import product
import numpy as np
from numba import njit, prange

@njit(parallel = True)
def count_alive_neighbours(site: np.ndarray, state: np.ndarray) -> int:
        
    d = len(state.shape)
    
    if d == 1:
        
        sum = 0
        for i in prange(-1, 2):
            if i == 0:
                continue
            if state[site[0] + i]:
                sum += 1
        return sum
        
    elif d == 2:
        
        sum = 0
        for i in prange(-1, 2):
            for j in prange(-1, 2):
                if i == 0 and j == 0:
                    continue
                if state[site[0] + i, site[1] + j]:
                    sum += 1
        return sum
    
    elif d == 3:
        
        sum = 0
        for i in prange(-1, 2):
            for j in prange(-1, 2):
                for k in prange(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    if state[site[0] + i, site[1] + j, site[2] + k]:
                        sum += 1
        return sum
