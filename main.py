from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dumpGIF import *

@cuda.jit
def update_state(state, new_state, width, height, configurations, iteration):
    
    i, j = cuda.grid(2)
    
    left = (j - 1 + width) % width
    right = (j + 1) % width
    top = (i - 1 + height) % height
    bottom = (i + 1) % height
    
    alive = state[i, left] + state[i, right] + state[top, j] + state[bottom, j]
    
    if state[i, j] == 1:  # Current cell is live
        if alive < 2 or alive > 3:
            # Any live cell with fewer than two or more than three live neighbors dies
            new_state[i, j] = 0
    else:  # Current cell is dead
        if alive == 3:
            # Any dead cell with exactly three live neighbors becomes a live cell
            new_state[i, j] = 1

    configurations[iteration, i, j] = new_state[i, j]  # Save the current state in the configurations array

def get_configurations(initial_state, num_iterations, width, height):
    block_size = (1, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    state_dev = cuda.to_device(initial_state)  # Copy the initial state to the GPU
    new_state_dev = cuda.device_array_like(state_dev)
    configurations = np.empty((num_iterations + 1, width, height), dtype = np.uint8)  # Array to store configurations on CPU
    configurations[0] = initial_state  # Store the initial state in the configurations array
    
    configurations_dev = cuda.to_device(configurations)  # Copy configurations array to the GPU
    
    for t in range(num_iterations):
        
        update_state[grid_size, block_size](state_dev, new_state_dev, width, height, configurations_dev, t + 1) # on GPU
        cuda.synchronize()  # Ensure all computations on GPU are completed    
        
        # Swap state and new_state arrays
        state_dev, new_state_dev = new_state_dev, state_dev
    
    # Copy the configurations array from GPU to CPU
    configurations = configurations_dev.copy_to_host()
    
    return configurations

# Set the size of the grid
width = 16
height = 16

# Set the number of iterations
num_iterations = 10

# Create the initial state randomly
# initial_state = np.random.randint(0, 2, size=(width, height), dtype = np.uint8)
initial_state = np.zeros((width, height), dtype = np.uint8)
initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = 1
initial_state[len(initial_state)//2, initial_state.shape[0]//2] = 1
initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = 1
initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+1] = 1
initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+1] = 1
initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+2] = 1
initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+2] = 1
initial_state[(len(initial_state)//2)+1, initial_state.shape[0]//2] = 1

# Run the cellular automaton and get the configurations
configurations = get_configurations(initial_state, num_iterations, width, height)

print(configurations)

print('Now calling to get the gif and save it.')
start = datetime.now()
dumpGIF(configurations, 'test.gif')
print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
