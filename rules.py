from neighbours import *
from numba import prange

@njit(parallel = True)
def apply_rules_1d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
        
    N = len(state)
    new_state = np.full(state.shape, False)
    
    if which_rules == '54':
    
        for idx in prange(N):
            
            site_index = site_indices[idx]
            x = site_index[0]
            current_cell_value = state[x]
                    
            alive = count_alive_neighbours_1d(site_index, state)

            new_state[site_index] = (((current_cell_value) and (alive == 0)) or ((current_cell_value) and (alive > 0)))

    if which_rules == '90':
        
        for idx in prange(N):
            
            site_index = site_indices[idx]
            x = site_index[0]
            current_cell_value = state[x]
            
            alive = count_alive_neighbours_1d(site_index, state)

            new_state[site_index] = (((current_cell_value) and (alive == 1)) or ((current_cell_value) and (alive == 1)))

    return new_state


@njit(parallel = True)
def apply_rules_2d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
    
    N = len(state)
    new_state = np.full(state.shape, False)
    
    if which_rules == 'game_of_life':
        
        for idx in prange(N**2):
        
            site_index = site_indices[idx]
            x, y = site_index
            current_cell_value = state[x, y]
                
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = ((current_cell_value) and ((alive == 2) or (alive == 3))) or ((current_cell_value == False) and (alive == 3))

    if which_rules == 'tumor_growth':
        
        for idx in prange(N**2):
        
            site_index = site_indices[idx]
            x, y = site_index
            current_cell_value = state[x, y]
                
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = ((current_cell_value == True) or ((current_cell_value == False) and (alive >= 3) and (np.random.rand() < 0.2)))

    return new_state

@njit(parallel = True)
def apply_rules_3d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
    
    N = len(state)
    new_state = np.full(state.shape, False)
    
    if which_rules == 'clouds_I':
        
        for idx in prange(N**3):
        
            site_index = site_indices[idx]
            x, y, z = site_index
            current_cell_value = state[x, y, z]

            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = ((current_cell_value == True) and (alive <= 26) and (alive >= 13)) or ((current_cell_value == False) and (((alive <= 14) and (alive >=13)) or ((alive <= 19) and (alive >=17))))

    if which_rules == 'builder_II':
        
        for idx in prange(N**3):
        
            site_index = site_indices[idx]
            x, y, z = site_index
            current_cell_value = state[x, y, z]
                        
            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = (((current_cell_value == True) and (alive <= 26) and (alive >= 13)) or ((current_cell_value == False) and (alive == 1)))

    return new_state
