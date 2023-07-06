from neighbours import *
from numba import prange

# @njit(parallel = True)
def apply_rules_1d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
        
    N = len(state)
    new_state = np.full(state.shape, False)
    
    for idx in range(N):
        
        site_index = site_indices[idx]
        current_cell_value = state[site_index]
        
        if which_rules == '54':
            
            alive = count_alive_neighbours_1d(site_index, state)
            print(alive)

            new_state[site_index] = ((current_cell_value == True and alive == 0 ) or ((current_cell_value == False) and (alive > 0)))

        if which_rules == '90':
            
            alive = count_alive_neighbours_1d(site_index, state)

            new_state[site_index] = (((current_cell_value == True) and (alive == 1)) or ((current_cell_value == False) and (alive == 1)))

    return new_state


@njit(parallel = True)
def apply_rules_2d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
    
    N = len(state)
    new_state = np.full(state.shape, False)
    
    for idx in prange(N**2):
        
        site_index = site_indices[idx]
        current_cell_value = state[site_index]

        if which_rules == 'game_of_life':
            
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = ((current_cell_value) and ((alive == 2) or (alive == 3))) or ((not current_cell_value) and (alive == 3))

        if which_rules == 'tumor_growth':
            
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = ((current_cell_value == True) or ((current_cell_value == False) and (alive >= 3) and (np.random.rand() < 0.2)))

    return new_state

@njit(parallel = True)
def apply_rules_3d(state : np.ndarray, which_rules : str, site_indices : np.ndarray) -> np.ndarray:
    
    N = len(state)
    new_state = np.full(state.shape, False)
    
    for idx in prange(N**3):
        
        site_index = site_indices[idx]
        current_cell_value = state[site_index]

        if which_rules == 'clouds_I':
            
            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = ((current_cell_value == True) and (alive <= 26) and (alive >= 13)) or ((current_cell_value == False) and (((alive <= 14) and (alive >=13)) or ((alive <= 19) and (alive >=17))))

        if which_rules == 'builder_II':
            
            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = (((current_cell_value == True) and (alive <= 26) and (alive >= 13)) or ((current_cell_value == False) and (alive == 1)))

    return new_state
