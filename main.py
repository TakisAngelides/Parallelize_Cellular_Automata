from init_state import *
from rules import *
from dumpGIF import *
import os
import numba
from datetime import datetime
import cupy as cp

numba.set_num_threads(1)
print('Number of numba threads is set to:', numba.get_num_threads(), flush = True)

def get_configurations(configurations_shape, initial_state):
    
    start = datetime.now()

    configurations = cp.full(configurations_shape, 0, dtype = cp.uint8)

    configurations[0] = initial_state

    state = initial_state
    state_shape = cp.array(state.shape)
    for t in range(1, time_steps):
        print(f'Running time step {t} starting at {datetime.now()}', flush = True)
        state = apply_rules_1d(state, state_shape)
        configurations[t] = state
        
    print(f'Time taken to calculate the configurations on GPU is {datetime.now()-start}. (Does not account for GPU back to CPU transfer time)', flush = True)
    
    # Transfer from gpu to cpu the configurations
    configurations_cpu = cp.asnumpy(configurations)
        
    return configurations_cpu
        

time_steps = 2
shape = [3]
# initial_state = initialize_two_glider_octomino(shape)
initial_state = initialize_random_array(shape)
configurations_shape = tuple([time_steps] + shape)
print(f'Have written the initial state with shape {shape} and now calling to get configurations for {time_steps} time steps.', flush = True)
configurations = get_configurations(configurations_shape, initial_state)

# print('Now calling to get the gif and save it.')
# start = datetime.now()
# dumpGIF(configurations, 'test.gif')
# print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
