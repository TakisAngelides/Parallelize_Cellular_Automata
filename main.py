import cupy as cp
import matplotlib.pyplot as plt

def run_cellular_automaton(initial_state, num_iterations):
    state = initial_state
    new_state = cp.empty_like(state)

    for _ in range(num_iterations):
        # Apply rules to update the state
        new_state[1:-1, 1:-1] = (state[:-2, :-2] + state[:-2, 1:-1] + state[:-2, 2:] +
                                 state[1:-1, :-2] + state[1:-1, 2:] +
                                 state[2:, :-2] + state[2:, 1:-1] + state[2:, 2:]) // 8

        # Update the boundary cells with the nearest neighbors
        new_state[0, :] = new_state[1, :]
        new_state[-1, :] = new_state[-2, :]
        new_state[:, 0] = new_state[:, 1]
        new_state[:, -1] = new_state[:, -2]

        # Swap the state and new_state arrays
        state, new_state = new_state, state

    return state

# Set the size of the grid
width = 100
height = 100

# Set the number of iterations
num_iterations = 100

# Create the initial state randomly
initial_state = cp.random.randint(0, 2, size=(height, width), dtype=cp.int32)

# Run the cellular automaton
final_state = run_cellular_automaton(initial_state, num_iterations)

# Convert the final state to a NumPy array for visualization
final_state_np = cp.asnumpy(final_state)

# Plot the final state
plt.imshow(final_state_np, cmap='binary')
plt.savefig('test.png', bbox_inches = 'tight')
