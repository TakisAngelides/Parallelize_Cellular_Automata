from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def update_state(state, new_state):
    i, j = cuda.grid(2)
    
    N, M = state.shape
    left = (j - 1 + M) % M
    right = (j + 1) % M
    top = (i - 1 + N) % N
    bottom = (i + 1) % N
    
    new_state[i, j] = (state[i, left] + state[i, right] + state[top, j] + state[bottom, j]) % 2

def run_cellular_automaton(initial_state, num_iterations):
    state = initial_state.copy()
    new_state = np.empty_like(state)
    block_size = (16, 16)
    grid_size = ((state.shape[0] + block_size[0] - 1) // block_size[0], (state.shape[1] + block_size[1] - 1) // block_size[1])

    for _ in range(num_iterations):
        update_state[grid_size, block_size](state, new_state)
        state, new_state = new_state, state

    return state

# Set the size of the grid
width = 100
height = 100

# Set the number of iterations
num_iterations = 1

# Create the initial state randomly
initial_state = np.random.randint(0, 2, size=(width, height), dtype=np.uint8)

# Run the cellular automaton
final_state = run_cellular_automaton(initial_state, num_iterations)

# Plot the final state
plt.imshow(final_state, cmap='binary')
plt.savefig('test.png', bbox_inches='tight')

