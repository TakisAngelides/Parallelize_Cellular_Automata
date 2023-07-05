from neighbours import *

#@njit() is the decorater for Numba, if  we have @njit(parallel=True) it enables Numba's Just-in-Time (JIT) compilation and parallel execution for the apply_rules function.
@njit()
def apply_rules(state : np.ndarray) -> np.ndarray:
    
    """
    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

        The function iterates over all cells and applies the rules of Conway's game of life rule (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    
    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    """
    
    
    new_state = np.full(state.shape, False)
    
    for site_index, current_cell_value in np.ndenumerate(state):
    
        alive : int = count_alive_neighbours(site_index, state)
        
        if current_cell_value and alive >= 2 and alive < 4:
            new_state[site_index] = True
        if not current_cell_value and alive == 3:
            new_state[site_index]= True
    
                
    return new_state