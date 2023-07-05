from neighbours import *

@njit()
def apply_rules(state : np.ndarray, which_rules : str) -> np.ndarray:
        
    new_state = np.full(state.shape, False)
    
    for site_index, current_cell_value in np.ndenumerate(state):
    
        alive = count_alive_neighbours(site_index, state)
        
        if which_rules == '54':

            new_state[site_index] = ((current_cell_value == True and alive == 0 ) or (current_cell_value == False and alive > 0))

        if which_rules == '90':

            new_state[site_index] = ((current_cell_value == True and alive == 1 ) or (current_cell_value == False and alive == 1))

        if which_rules == 'game_of_life':

            new_state[site_index] = ((current_cell_value) and ((alive == 2) or (alive == 3))) or ((not current_cell_value) and (alive == 3))

        if which_rules == 'tumor_growth':

            new_state[site_index] = (current_cell_value == True or (current_cell_value == False and alive >= 3 and np.random.rand() < 0.2))

        if which_rules == 'clouds_I':

            new_state[site_index] = (current_cell_value == True and alive <= 26 and alive >= 13) or (current_cell_value == False and ((alive <= 14 and alive >=13) or (alive <= 19 and alive >=17)))

        if which_rules == 'builder_II':

            new_state[site_index] = ((current_cell_value == True and alive <= 26 and alive >= 13) or (current_cell_value == False and alive ==1))
    
    return new_state