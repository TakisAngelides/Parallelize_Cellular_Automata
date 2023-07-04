from neighbours import *

@cp.fuse(kernel_name = 'apply_rules')
def apply_rules_1d(state : cp.ndarray, state_shape : int) -> cp.ndarray:
    
    """

    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    The function trivially iterates over all cells and applies the Clouds 1 rule (see https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
    
    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    
    """
    
    new_state = cp.full((state_shape), 0)
    
    dims = state_shape
    d = len(dims)
    
    if d == 1:
        
        # Get the thread indices
        thread_x = cp.threadIdx.x
        
        # Get the block indices
        block_x = cp.blockIdx.x
        
        # Calculate the global indices
        global_x = block_x * cp.blockDim.x + thread_x
        
        current_cell_value = state[global_x]

        alive : int = count_alive_neighbours(cp.array([global_x]), state)
        
        if current_cell_value and alive < 2:
            new_state[global_x] = 1
        if not current_cell_value and alive == 2:
            new_state[global_x]= 1
        
    # elif d == 2:
        
    #     for i in prange(dims[0]):
    #         for j in prange(dims[1]):
    
    #             current_cell_value = state[i, j]
        
    #             alive : int = count_alive_neighbours(cp.array([i, j]), state)
                
    #             if current_cell_value and alive >= 2 and alive < 4:
    #                 new_state[i, j] = True
    #             if not current_cell_value and alive == 3:
    #                 new_state[i, j]= True
    
    # elif d == 3:
        
    #     for i in prange(dims[0]):
    #         for j in prange(dims[1]):
    #             for k in prange(dims[2]):
    
    #                 current_cell_value = state[i, j, k]
            
    #                 alive : int = count_alive_neighbours(cp.array([i, j, k]), state)
                    
    #                 if current_cell_value and alive >= 9 and alive < 23:
    #                     new_state[i, j, k] = True
    #                 if not current_cell_value and alive == 3:
    #                     new_state[i, j, k]= True
                
    return new_state