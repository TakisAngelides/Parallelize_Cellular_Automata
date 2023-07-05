from itertools import product
import numpy as np
from numba import njit

#@njit() is the decorater for Numba, if  we have @njit(parallel=True) it enables Numba's Just-in-Time (JIT) compilation and parallel execution for the apply_rules function.
@njit()
def count_alive_neighbours(site: np.ndarray, state: np.ndarray) -> int:
    
    """
    Comment: 
        For given given index we iterate over all nearest neighbours and count the number of alive ones, we iterate over the respective index as well,
        to compensate for this we initialize sum=-state[site[0]]. In that way we avoid an if statement
         
    Inputs:
    
        site : 1-dimensional numpy array of length d - where d is the number of dimensions of the hypercube - specifying the site of a given cell

        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    Returns:
    
        int : number of alive neighbours
    """
    
    d = len(state.shape)
    
    if d == 1:
        
        sum = 0
        for i in [-1, 1]:
            if state[site[0] + i]:
                sum += 1
        return sum
        
    elif d == 2:
        
        sum = -state[site[0]]
        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                if state[site[0] + i, site[1] + j]:
                    sum += 1
        return sum
    
    elif d == 3:
        
        sum = -state[site[0]]
        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    
                    if state[site[0] + i, site[1] + j, site[2] + k]:
                        sum += 1
        return sum
