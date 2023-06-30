from init_state import *
from rules import *
from dumpGIF import *
import os
import numba

numba.set_num_threads(16)
print('Number of numba threads is set to:', numba.get_num_threads(), flush = True)

def get_configurations(time_steps, initial_state):

    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    configurations[0] = initial_state

    state = initial_state
    for t in range(1, time_steps):
        print(f'Running time step {t}', flush = True)
        state = apply_rules(state)
        configurations[t] = state
        
    return configurations
        

time_steps = 16
shape = (16, 16, 16)
# initial_state = initialize_two_glider_octomino(shape)
initial_state = initialize_random_array(shape)
print('Have written the initial state and now calling to get configurations.', flush = True)
configurations = get_configurations(time_steps, initial_state)

dumpGIF(configurations, 'test.gif')
