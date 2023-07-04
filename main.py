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
    
    new_state[i, j] = (state[i, left] + state[i, right] + state[top, j] + state[bottom, j]) % 2
    
    configurations[i, j, iteration] = new_state[i, j]  # Save the current state in the configurations array

def get_configurations(initial_state, num_iterations, width, height):
    block_size = (1, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    state_dev = cuda.to_device(initial_state)  # Copy the initial state to the GPU
    new_state_dev = cuda.device_array_like(state_dev)
    configurations = np.empty((width, height, num_iterations + 1), dtype=np.uint8)  # Array to store configurations on CPU
    configurations[:, :, 0] = initial_state  # Store the initial state in the configurations array
    
    configurations_dev = cuda.to_device(configurations)  # Copy configurations array to the GPU
    
    for t in range(num_iterations):
        update_state[grid_size, block_size](state_dev, new_state_dev, width, height, configurations_dev, t + 1) # on GPU
    
    cuda.synchronize()  # Ensure all computations on GPU are completed
    
    # Copy the configurations array from GPU to CPU
    configurations = configurations_dev.copy_to_host()
    
    return configurations

# Set the size of the grid
width = 16
height = 16

# Set the number of iterations
num_iterations = 5

# Create the initial state randomly
initial_state = np.random.randint(0, 2, size=(width, height), dtype = np.uint8)

# Run the cellular automaton and get the configurations
configurations = get_configurations(initial_state, num_iterations, width, height)

# # Plot the final state (last configuration)
# final_state = configurations[:, :, -1]
# plt.imshow(final_state, cmap='binary')
# plt.savefig('test.png', bbox_inches='tight')

print('Now calling to get the gif and save it.')
start = datetime.now()
dumpGIF(configurations, 'test.gif')
print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
