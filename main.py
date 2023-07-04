import cupy as cp
import matplotlib.pyplot as plt

@cp.fuse
def update_state(state, new_state):
    height, width = 100, 100

    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                new_state[i, j] = state[i, j]
            else:
                new_state[i, j] = (state[i-1, j-1] + state[i-1, j] + state[i-1, j+1] +
                                   state[i, j-1] + state[i, j] + state[i, j+1] +
                                   state[i+1, j-1] + state[i+1, j] + state[i+1, j+1]) // 9

def run_cellular_automaton(initial_state):
    state = initial_state

    for _ in range(2):
        new_state = cp.empty_like(state)
        update_state(state, new_state)
        state = new_state

    return state

# Set the size of the grid
width = 100
height = 100

# Set the number of iterations
num_iterations = 2

# Create the initial state randomly
initial_state = cp.random.randint(0, 2, size=(height, width), dtype=cp.int32)

# Run the cellular automaton
final_state = run_cellular_automaton(initial_state)

# Convert the final state to a NumPy array for visualization
final_state_np = cp.asnumpy(final_state)

# Plot the final state
plt.imshow(final_state_np, cmap='binary')
plt.savefig('test.png', bbox_inches = 'tight')
