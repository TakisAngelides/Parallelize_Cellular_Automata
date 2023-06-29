from init_state import *
from rules import *
from dumpGIF import *

def get_configurations(time_steps, initial_state):

    #history[i, :, :] is the i-th timeslice of the evolution
    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    #initialize
    configurations[0] = initial_state

    state = initial_state
    for t in range(1, time_steps):
        state = apply_rules(state)
        configurations[1] = state
        
    return configurations
        

time_steps = 100
initial_state = initialize_random_array((4, 4, 4))
configurations = get_configurations(time_steps, initial_state)

dumpGIF(configurations, 'test.gif')
