from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def update_state(state, new_state, width, height, configurations, iteration):
    
    i, j = cuda.grid(2) # thread id 
    print(i, j)
    
    left = (j - 1 + width) % width
    right = (j + 1) % width
    top = (i - 1 + height) % height
    bottom = (i + 1) % height
    
    new_state[i, j] = (state[i, left] + state[i, right] + state[top, j] + state[bottom, j]) % 2
    
    configurations[i, j, iteration] = new_state[i, j]  # Save the current state in the configurations array

def get_configurations(initial_state, num_iterations, width, height):
    block_size = (1, 1)  # number of threads within a block
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1]) # number of blocks

    state = np.ascontiguousarray(initial_state)  # Create a contiguous copy of the initial state
    new_state = np.empty_like(state)
    configurations = np.empty((width, height, num_iterations), dtype=np.uint8)  # Array to store configurations on CPU
    
    configurations_dev = cuda.to_device(configurations)  # Copy configurations array to the GPU
    
    for i in range(num_iterations):
        update_state[grid_size, block_size](state, new_state, width, height, configurations_dev, i) # on GPU 
        state, new_state = new_state, state
    
    cuda.synchronize()  # Ensure all computations on GPU are completed
    
    # Copy the configurations array from GPU to CPU
    configurations = configurations_dev.copy_to_host()
    
    return configurations

# Set the size of the grid
width = 100
height = 100

# Set the number of iterations
num_iterations = 10  # Increase the number of iterations as per your requirement

# Create the initial state randomly
initial_state = np.random.randint(0, 2, size=(width, height), dtype=np.uint8)

# Run the cellular automaton and get the configurations
configurations = get_configurations(initial_state, num_iterations, width, height)

# Plot the final state (last configuration)
final_state = configurations[:, :, -1]
plt.imshow(final_state, cmap='binary')
plt.savefig('test.png', bbox_inches='tight')

# Save all the configurations in a separate file or process them as needed
# configurations[:, :, i] contains the state of the automaton at iteration i
