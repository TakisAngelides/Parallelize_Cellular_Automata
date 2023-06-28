from itertools import product
import numpy as np

def count_alive_neighbours(site : np.ndarray, state : np.ndarray) -> int:
    
    """
    
    Inputs:
    
        site : 1-dimensional numpy array of length d - where d is the number of dimensions of the hypercube - specifying the site of a given cell
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    product(np.array([0, -1, 1]), repeat = len(site)) is equivalent to [[x, y, z] for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]] for d = 3
    
    from the above list we omit the first element [0, 0, 0] since we want to add these lists to site = np.array([i, j, k]) to get the neighbour indices hence the [1:]
    
    then we do broadcasting of site which is of shape (d,) and this product array which is of shape (3^d, d) to get the tuples which represent the indices of the neighbours
    
    and then we access the state with these and add them up to see how many are alive and return the integer of alive neighbours

    be careful that this assumes by summing that the elements of the state are boolean-like eg 0, 1 or False, True and not something like 2, 3

    # TODO: Is there a way to avoid the for loop here? Eg by accessing the elements silmutaneously with parallelization and doing the sum by sharing the memory?
    # TODO: Will this function work with Numba and JIT since I am using itertools?
    
    Returns:
    
        int : number of alive neighbours
    
    """
       
    return np.sum([state[element.reshape(1, 1, -1)] for element in site+np.array(list(product(np.array([0, -1, 1]), repeat = len(site)))[1:])])
