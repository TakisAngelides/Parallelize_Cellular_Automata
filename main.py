from init_state import *
from rules import *
from dumpGIF import *

def get_configurations(time_steps, initial_state):

    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    configurations[0] = initial_state

    state = initial_state
    for t in range(1, time_steps):
        state = apply_rules(state)
        configurations[t] = state
        
    return configurations
        

time_steps = 10
initial_state = initialize_random_array((4, 4, 4))
configurations = get_configurations(time_steps, initial_state)

dumpGIF(configurations, 'test.gif')
