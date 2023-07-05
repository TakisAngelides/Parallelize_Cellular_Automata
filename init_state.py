import cellpylib as cpl
import cupy as cp
import numpy as np

def initialize_array(shape, initial_value = 0):
    
    return cp.full(shape, initial_value)

def initialize_random_array(shape):
    
    return cp.random.randint(0, 2, size = shape, dtype = cp.uint8)

def initialize_cellpy_array(shape):
    
    return cpl.init_simple2d(*shape, dtype = bool)

def initialize_glider(shape):
    
    initial_state = cp.zeros(shape, dtype = bool)
    initial_state[:,initial_state.shape[0]//2] = True
    initial_state[0,initial_state.shape[0]//2] = False
    initial_state[len(initial_state)//2,initial_state.shape[0]//2] = False
    
    return initial_state

def initialize_two_glider_octomino(shape):
    
    initial_state = cp.zeros(shape)
    
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
    initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)+1, initial_state.shape[0]//2] = True
    
    return initial_state

def get_initial_state(shape,which_rules):
    if (which_rules == '54'  or  which_rules == '90'):
        initial_state = np.zeros(shape, dtype = bool)
        initial_state[len(initial_state)//2] = True

    if (which_rules == 'game_of_life'):

        initial_state = np.zeros(shape, dtype = bool)
        initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
        initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = True
        initial_state[(len(initial_state)//2)+1, (initial_state.shape[0]//2)+1] = True
        initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
        initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)] = True

    if (which_rules == 'tumor_growth'):
        initial_state = np.zeros(shape, dtype=bool)
        initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True

    if (which_rules == 'clouds_I'):
        initial_state = np.zeros(shape, dtype=bool)
        initial_state = np.random.choice([True,False],shape, p = [0.47,0.53])

    if (which_rules == 'builder_II'):
        initial_state = np.zeros(shape, dtype = bool)
        initial_state[ initial_state.shape[0]//2, initial_state.shape[1]//2, initial_state.shape[2]//2] = True


    return initial_state
