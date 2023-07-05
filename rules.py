from neighbours import *

@njit(parallel = True)
def apply_rules(state : np.ndarray) -> np.ndarray:
    
    """


    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

        The function iterates over all cells and applies the rules of Conway's game of life rule (see https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    
    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    
    """
    
    new_state = np.full(state.shape, False)
    
    d = len(state.shape)
    dims = state.shape
    
    if d == 1:
        
        for i in prange(dims[0]):
    
            current_cell_value = state[i]
    
            alive : int = count_alive_neighbours(np.array([i]), state)
            
            if current_cell_value and alive >= 1:
                new_state[i] = True
            if not current_cell_value and alive == 2:
                new_state[i]= True
        
    elif d == 2:
        
        for i in prange(dims[0]):
            for j in prange(dims[1]):
    
                current_cell_value = state[i, j]
        
                alive : int = count_alive_neighbours(np.array([i, j]), state)
                
                if current_cell_value and alive >= 2 and alive < 4:
                    new_state[i, j] = True
                if not current_cell_value and alive == 3:
                    new_state[i, j]= True
    
    elif d == 3:
        
        for i in prange(dims[0]):
            for j in prange(dims[1]):
                for k in prange(dims[2]):
    
                    current_cell_value = state[i, j, k]
            
                    alive : int = count_alive_neighbours(np.array([i, j, k]), state)
                    
                    if current_cell_value and alive >= 9 and alive < 23:
                        new_state[i, j, k] = True
                    if not current_cell_value and alive == 3:
                        new_state[i, j, k]= True
                
    return new_state