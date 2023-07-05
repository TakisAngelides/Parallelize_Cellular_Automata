from datetime import datetime
from init_state import *
from dumpGIF import *
from update_state import *

def get_configurations(num_iterations, shape):
    d = len(shape)
    
    if d == 1 :
        width = shape[0]
        block_size = 1
        grid_size = (width + block_size - 1) // block_size
    
        configurations = np.empty((num_iterations + 1, width), dtype=bool)
        configurations[0, :] = initial_state
    
        configurations_dev = cuda.to_device(configurations)
    
        for t in range(num_iterations):
            update_state_1D[grid_size, block_size](width, configurations_dev, t + 1)
            print(t)
            cuda.synchronize()
    
        configurations = configurations_dev.copy_to_host()
    
        return configurations

    if d == 2 :    
        width = shape[0]
        height = shape[1]
        
        
        block_size = (1, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])
        
        configurations = np.empty((num_iterations + 1, width, height), dtype = bool)  # Array to store configurations on CPU
        configurations[0, :, :] = initial_state
            
        configurations_dev = cuda.to_device(configurations)  # Copy configurations array to the GPU
        
        for t in range(num_iterations):
    
            update_state_2D[grid_size, block_size](width, height, configurations_dev, t + 1) # on GPU
            cuda.synchronize()  # Ensure all computations on GPU are completed    
            
        # Copy the configurations array from GPU to CPU
        configurations = configurations_dev.copy_to_host()
        
        return configurations


    if d == 3:
        width = shape[0]
        height = shape[1]
        depth = shape[2]
        
        block_size = (1, 1, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1],
            (depth + block_size[2] - 1) // block_size[2]
        )
    
        configurations = np.empty((num_iterations + 1, width, height, depth), dtype=bool)
        configurations[0, :, :, :] = initial_state
    
        configurations_dev = cuda.to_device(configurations)
    
        for t in range(num_iterations):
            update_state_3D[grid_size, block_size](width, height, depth, configurations_dev, t + 1)
            
            cuda.synchronize()
    
        configurations = configurations_dev.copy_to_host()
    
        return configurations
        
# Set the size of the grid
L = 16

shape = (L, L, L)

# which_rules can be '54' for 1D, 'game_of_life' for 2D, 'clouds_I' for 3D
which_rules = 'game_of_life'

# Set the number of iterations
num_iterations = 100

# Create the initial state
initial_state = get_initial_state(shape, which_rules)

# Run the cellular automaton and get the configurations
configurations = get_configurations(num_iterations, shape)

print('Now calling to get the gif and save it.')
start = datetime.now()
dumpGIF(configurations, 'test.gif')
print(f'Time taken to save the gif is {datetime.now()-start}.', flush = True)
