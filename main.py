import cupy as cp
import matplotlib.pyplot as plt

def update_state(state):
    new_state = cp.empty((100, 100), dtype=state.dtype)
    new_state[1:-1, 1:-1] = (state[:-2, :-2] + state[:-2, 1:-1] + state[:-2, 2:] +
                             state[1:-1, :-2] + state[1:-1, 2:] +
                             state[2:, :-2] + state[2:, 1:-1] + state[2:, 2:]) // 8

    new_state[0, :] = new_state[1, :]
    new_state[-1, :] = new_state[-2, :]
    new_state[:, 0] = new_state[:, 1]
    new_state[:, -1] = new_state[:, -2]

    return new_state

@cp.fuse
def run_cellular_automaton(initial_state):
    state = initial_state

    for _ in range(2):
        state = update_state(state)

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
