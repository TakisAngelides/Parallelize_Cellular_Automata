from init_state import *
from rules import *
from dumpGIF import *

def get_height(state, height):


    for i in range(len(height)):

        if(state[i]):
            height[i]+=1

    return height


def calculate_length(arr):
    if arr is None:
        return []  # or any other appropriate default value
    return len(arr)



def get_configurations(time_steps, initial_state):

    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    configurations[0] = initial_state
    height=np.zeros(initial_state.shape[0])

    state = initial_state
    for t in range(1, time_steps):
        state = apply_rules_surface_growths(state,height)
        configurations[t] = state
        height=get_height(state,height)
        


    return configurations, height
        

time_steps = 1000
shape = (100)

#initial_state = initialize_two_glider_octomino(shape)
initial_state =initialize_array(shape)
sim = get_configurations(time_steps, initial_state)
configurations=sim[0]
height_profile=sim[1]

np.savetxt('output.txt', height_profile, fmt='%d')

#dumpGIF(configurations, 'test.gif')
