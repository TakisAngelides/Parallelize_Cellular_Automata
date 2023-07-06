from datetime import datetime
from init_state import get_initial_state
from dumpGIF import *
from update_state import *

def get_configurations(num_iterations, shape, which_rules):
    d = len(shape)
    
    if d == 1 :
        width = shape[0]
        block_size = 1
        grid_size = (width + block_size - 1) // block_size
    
        configurations = np.empty((num_iterations + 1, width), dtype=bool)
        configurations[0, :] = initial_state
    
        configurations_dev = cuda.to_device(configurations)
        if which_rules == '90':    
            for t in range(num_iterations):
                update_state_1D_r90[grid_size, block_size](width, configurations_dev, t + 1)
                cuda.synchronize()
        elif which_rules == '54':
            for t in range(num_iterations):
                update_state_1D_r54[grid_size, block_size](width, configurations_dev, t + 1)
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
        if which_rules == 'tumor_growth':
            for t in range(num_iterations):
                update_state_2D_rTM[grid_size, block_size](width, height, configurations_dev, t + 1) # on GPU
                cuda.synchronize()  # Ensure all computations on GPU are completed   
        elif which_rules == 'game_of_life':
            for t in range(num_iterations):
                update_state_2D_rGOL[grid_size, block_size](width, height, configurations_dev, t + 1) # on GPU
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
        if which_rules == 'builder_II':
            for t in range(num_iterations):
                update_state_3D_rBII[grid_size, block_size](width, height, depth, configurations_dev, t + 1)
                
                cuda.synchronize()
        elif which_rules == 'clouds_I':
            for t in range(num_iterations):
                update_state_3D_rCI[grid_size, block_size](width, height, depth, configurations_dev, t + 1)
                
                cuda.synchronize()            
            
        configurations = configurations_dev.copy_to_host()
    
        return configurations

