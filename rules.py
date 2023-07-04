from neighbours import *
import random as rnd

def apply_rules(state : np.ndarray) -> np.ndarray:
    
    """

    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    The function trivially iterates over all cells and applies the Clouds 1 rule (see https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
    
    # TODO: Parallelize this function

    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    
    """
    
    new_state = np.full(state.shape, False)
    
    for site_index, current_cell_value in np.ndenumerate(state):
    
        alive = count_alive_neighbours(site_index, state)
        
        # print(alive)
    
        if current_cell_value and alive >= 2 and alive < 4:
            new_state[site_index] = True
        if not current_cell_value and alive == 3:
            new_state[site_index]= True
    
                
    return new_state

def apply_rules_surface_growths(state : np.ndarray,height) -> np.ndarray:
    
    """

    Inputs:
    
        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    The function trivially iterates over all cells and applies the Clouds 1 rule (see https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/)
    
    # TODO: Parallelize this function

    Returns:
    
        new_state : d-dimensional numpy array specifying the state of the cellular automata after 1 time evolution of rules, the value at each element is boolean-like
    
    """
    
    new_state = np.full(state.shape, False)
    
    for site_index, current_cell_value in np.ndenumerate(state):
        
        #parameter
        kappa=0.1
        rho=0.5
        
        #calculate T_i
        T_i=height[(site_index[0]-1)%len(height)]+height[(site_index[0]+1)%len(height)]-2*height[site_index[0]]
        #calculate T_max
        T_max=(-1/kappa)*np.log(rho)
        if(T_i>=T_max):
            new_state[site_index] = True
        else:
            #calculate p_i
            p_i=rho*np.exp(kappa*T_i)
            random_number=rnd.random()
            if(random_number<p_i):
                new_state[site_index] = True



    
                
    return new_state