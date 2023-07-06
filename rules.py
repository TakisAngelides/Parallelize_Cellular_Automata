from neighbours import *

@njit()
def apply_rules(state : np.ndarray, which_rules : str) -> np.ndarray:
        
    new_state = np.full(state.shape, False)
    
    for site_index, current_cell_value in np.ndenumerate(state):
        
        if which_rules == '54':
            
            alive = count_alive_neighbours_1d(site_index, state)

            new_state[site_index] = ((current_cell_value == True and alive == 0 ) or (current_cell_value == False and alive > 0))

        if which_rules == '90':
            
            alive = count_alive_neighbours_1d(site_index, state)

            new_state[site_index] = ((current_cell_value == True and alive == 1 ) or (current_cell_value == False and alive == 1))

        if which_rules == 'game_of_life':
            
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = ((current_cell_value) and ((alive == 2) or (alive == 3))) or ((not current_cell_value) and (alive == 3))

        if which_rules == 'tumor_growth':
            
            alive = count_alive_neighbours_2d(site_index, state)

            new_state[site_index] = (current_cell_value == True or (current_cell_value == False and alive >= 3 and np.random.rand() < 0.2))

        if which_rules == 'clouds_I':
            
            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = (current_cell_value == True and alive <= 26 and alive >= 13) or (current_cell_value == False and ((alive <= 14 and alive >=13) or (alive <= 19 and alive >=17)))

        if which_rules == 'builder_II':
            
            alive = count_alive_neighbours_3d(site_index, state)

            new_state[site_index] = ((current_cell_value == True and alive <= 26 and alive >= 13) or (current_cell_value == False and alive ==1))
    
    return new_state