from itertools import product
import numpy as np
from numba import njit

@njit()
def count_alive_neighbours(site: np.ndarray, state: np.ndarray) -> int:
    
    """
         
    Inputs:
    
        site : 1-dimensional numpy array of length d - where d is the number of dimensions of the hypercube - specifying the site of a given cell

        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    product(np.array([0, -1, 1]), repeat = len(site)) is equivalent to [[x, y, z] for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]] for d = 3
    
    from the above list we omit the first element [0, 0, 0] since we want to add these lists to site = np.array([i, j, k]) to get the neighbour indices hence the [1:]
    
    then we do broadcasting of site which is of shape (d,) and this product array which is of shape (3^d, d) to get the tuples which represent the indices of the neighbours
    
    and then we access the state with these and add them up to see how many are alive and return the integer of alive neighbours

    be careful that this assumes by summing that the elements of the state are boolean-like eg 0, 1 or False, True and not something like 2, 3

    Returns:
    
        int : number of alive neighbours

    """
    
    d = len(state.shape)
    N = len(state)
    
    if d == 1:
        
        sum = 0
        
        left = (site[0] - 1 + N) % N
        right = (site[0] + 1) % N
        
        return state[left] + state[right]
        
    elif d == 2:
        
        left = (site[0] - 1 + N) % N
        right = (site[0] + 1) % N
        top = (site[1] - 1 + N) % N
        bottom = (site[1] + 1) % N
                
        return state[left, site[1]] + state[right, site[1]] + state[site[0], top] + state[site[0], bottom] + state[left, top] + state[right, top] + state[left, bottom] + state[right, bottom]
    
    elif d == 3:
        
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
