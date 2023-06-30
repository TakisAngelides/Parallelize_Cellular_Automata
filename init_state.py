import numpy as np
import cellpylib as cpl

def initialize_array(shape, initial_value=0, dtype=np.bool):
    
    return np.full(shape, initial_value, dtype=dtype)

def initialize_random_array(shape, values=[True, False], p = [0.5, 0.5]):
    
    return np.random.choice(values, size = shape, p = p)

def initialize_cellpy_array(shape):
    
    return cpl.init_simple2d(*shape, dtype = bool)

def initialize_glider(shape):
    
    initial_state = np.zeros(shape, dtype = bool)
    initial_state[:,initial_state.shape[0]//2] = True
    initial_state[0,initial_state.shape[0]//2] = False
    initial_state[len(initial_state)//2,initial_state.shape[0]//2] = False
    
    return initial_state

def initialize_two_glider_octomino(shape):
    
    initial_state = np.zeros(shape)
    
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
    initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)+1, initial_state.shape[0]//2] = True
    
    return initial_state