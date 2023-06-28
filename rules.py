from neighbours import *

def apply_rules(state : np.ndarray) -> np.ndarray:
    
    """

    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    The function trivially iterates over all cells and applies the Clouds 1 rule (see https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
    
    # TODO: Parallelize this function
    # TODO: Can Numba JIT be applied to this function since it calles the count_alive_neighbours function which is not certain whether it can be applied to Numba JIT itself?

    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    
    """
    
    I, J, K = state.shape
    new_state = state.copy()
    for i in range(I):
        for j in range(J):
            for k in range(K):
                current_state = state[i, j, k]
                alive = count_alive_neighbours(np.array([i, j, k]), state)
                if (alive >= 13) and (alive <= 19) and (not current_state):
                    new_state[i, j, k] = True
                if current_state and (alive < 13):
                    new_state[i, j, k] = False
                
    return new_state