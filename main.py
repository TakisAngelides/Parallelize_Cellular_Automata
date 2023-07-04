from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def update_state(state, new_state):
    
    """
    
    # Define the kernel to run on GPU
    # This is called num_iteration number of times and each thread takes care of updating the corresponding index in new_state
    
    """
    
    i = cuda.grid(1)
        
    N = len(state)
    left = (i - 1 + N) % N
    right = (i + 1) % N
            
    new_state[i] = (state[i-1] + state[i+1]) % 2

def run_cellular_automaton(initial_state, num_iterations):
    
    state = initial_state.copy()
    new_state = np.empty_like(state)
    block_size = 128
    grid_size = (state.shape[0] + block_size - 1) // block_size

    for _ in range(num_iterations):
        update_state[grid_size, block_size](state, new_state)
        state, new_state = new_state, state

    return state

# Set the size of the grid
width = 100

# Set the number of iterations
num_iterations = 1

# Create the initial state randomly
initial_state = np.random.randint(0, 2, size=width, dtype=np.uint8)

# Run the cellular automaton
final_state = run_cellular_automaton(initial_state, num_iterations)

# Plot the final state
plt.imshow(final_state.reshape(1, -1), cmap='binary')
plt.savefig('test.png', bbox_inches = 'tight')
