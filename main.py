from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.animation as animation
from dumpGIF import *

@cuda.jit
def update_state(width, height, configurations_dev, iteration):
    

    x, y = cuda.grid(2)
    
    left = (x - 1 + width) % width
    right = (x + 1) % width
    top = (y - 1 + height) % height
    bottom = (y + 1) % height
    
    # Moore neighbours
    alive = (
    configurations_dev[iteration-1, left, y] + configurations_dev[iteration-1, right, y] + configurations_dev[iteration-1, x, top] +configurations_dev[iteration-1, x, bottom] +
    configurations_dev[iteration-1, left, top] +configurations_dev[iteration-1, right, top] +configurations_dev[iteration-1, left, bottom] + configurations_dev[iteration-1, right, bottom]
)
      
    configurations_dev[iteration, x, y] = ((configurations_dev[iteration-1, x, y]) and (alive >= 2) and (alive < 4)) or ((not configurations_dev[iteration-1, x, y]) and (alive == 3))
    
def get_configurations(num_iterations, width, height):
    
    block_size = (1, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])
    
    configurations = np.empty((num_iterations + 1, width, height), dtype = bool)  # Array to store configurations on CPU
    configurations[0, :, :] = initial_state
        
    configurations_dev = cuda.to_device(configurations)  # Copy configurations array to the GPU
    
    for t in range(num_iterations):

        update_state[grid_size, block_size](width, height, configurations_dev, t + 1) # on GPU
        cuda.synchronize()  # Ensure all computations on GPU are completed    
        
    # Copy the configurations array from GPU to CPU
    configurations = configurations_dev.copy_to_host()
    
    return configurations
    
# Set the size of the grid
width = 16
height = 16

shape = (width, height)
which_rules = 'tumor_growth'

# Set the number of iterations
num_iterations = 100

# Create the initial state
initial_state = get_initial_state(shape, which_rules)

# Run the cellular automaton and get the configurations
configurations = get_configurations(num_iterations, width, height)

print('Now calling to get the gif and save it.')
start = datetime.now()
dumpGIF(configurations, 'test.gif')
print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
