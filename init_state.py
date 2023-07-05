import numpy as np

def get_initial_state(shape, which_rules):
    
    if (which_rules == '54'  or  which_rules == '90'):
        initial_state = np.zeros(shape, dtype = bool)
        initial_state[len(initial_state)//2] = True
        return initial_state

    if (which_rules == 'game_of_life'):

        initial_state = np.zeros(shape, dtype = bool)
        initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
        initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = True
        initial_state[(len(initial_state)//2)+1, (initial_state.shape[0]//2)+1] = True
        initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
        initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)] = True
        return initial_state

    if (which_rules == 'tumor_growth'):
        initial_state = np.zeros(shape, dtype=bool)
        initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
        return initial_state

    if (which_rules == 'clouds_1'):
        initial_state = np.zeros(shape, dtype=bool)
        initial_state = np.random.choice([True,False],shape, p = [0.45,0.55])
        return initial_state

    if (which_rules == 'builder_II'):
        initial_state = np.zeros(shape, dtype = bool)
        initial_state[ initial_state.shape[0]//2, initial_state.shape[1]//2, initial_state.shape[2]//2] = True

        return initial_state