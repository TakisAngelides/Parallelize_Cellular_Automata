from init_state import *
from rules import *
from dumpGIF import *
import os
import numba
from datetime import datetime

def get_configurations(time_steps, initial_state):
    """"
    Arguments:
        time_steps: the number of time steps in the evolution of the cellular automata
        initial_state: initial state for the evolution of the cellular automata
    
    Returns:
        Numpy array of size time_steps x initial_state.shape
        
    """
    #initialize configurations as numpy array of shape time_steps x initial_state.shape and with None entry
    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    #first timeslice of configurations equals initial_state
    configurations[0] = initial_state

    #from state the state of the next timeslice will be determined
    state = initial_state
    #loop to create timeslices in configurations
    for t in range(1, time_steps):
        #start timing for one time step
        print(f'Running time step {t} starting at {datetime.now()}', flush = True)
        #aply rules to state to obtain the next state
        state = apply_rules(state)
        #copy new state to configurations
        configurations[t] = state
        
    return configurations
        

#number of time steps in the evolution of the cellular automata
time_steps = 100
#shape of the intitial state of the cellular automata
shape = (100,100)

#call right intitial configurations
initial_state = initialize_glider_gun(shape)
#initial_state = initialize_random_array(shape)
print(f'Have written the initial state with shape {shape} and now calling to get configurations for {time_steps} time steps.', flush = True)
start = datetime.now()
configurations = get_configurations(time_steps, initial_state)
print(f'Time taken to get configurations is {datetime.now()-start}.', flush = True)

# print('Now calling to get the gif and save it.')
# start = datetime.now()
dumpGIF(configurations, 'test.gif')
# print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
