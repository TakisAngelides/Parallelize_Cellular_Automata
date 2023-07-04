import numpy as np
from numba import cuda

@cuda.jit
def cellular_automaton_kernel(arr, new_arr):
    # Calculate the global indices in 2D space
    x, y = cuda.grid(2)

    # Get the dimensions of the array
    dim_x, dim_y = arr.shape

    # Define the neighborhood indices
    neighbor_indices = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    # Apply the cellular automaton rules
    if x < dim_x and y < dim_y:
        cell_state = arr[x, y]
        neighbor_count = 0

        # Count the number of alive neighbors
        for dx, dy in neighbor_indices:
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within the array boundaries
            if 0 <= nx < dim_x and 0 <= ny < dim_y:
                neighbor_state = arr[nx, ny]
                neighbor_count += neighbor_state

        # Apply the rules of the cellular automaton
        if cell_state == 1 and neighbor_count < 2:
            new_arr[x, y] = 0  # Cell dies due to underpopulation
        elif cell_state == 1 and neighbor_count > 3:
            new_arr[x, y] = 0  # Cell dies due to overpopulation
        elif cell_state == 0 and neighbor_count == 3:
            new_arr[x, y] = 1  # Cell is born due to reproduction
        else:
            new_arr[x, y] = cell_state  # Cell remains unchanged

def run_cellular_automaton(arr, num_iterations):
    # Get the dimensions of the array
    dim_x, dim_y = arr.shape

    # Allocate GPU memory for the array
    d_arr = cuda.to_device(arr)
    d_new_arr = cuda.device_array_like(d_arr)

    # Define the block and grid sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = (dim_x + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (dim_y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Run the cellular automaton for the specified number of iterations
    for _ in range(num_iterations):
        cellular_automaton_kernel[blocks_per_grid, threads_per_block](d_arr, d_new_arr)
        d_arr, d_new_arr = d_new_arr, d_arr

    # Transfer the final array back to the CPU
    final_arr = d_arr.copy_to_host()

    return final_arr


# Example usage
arr = np.random.randint(0, 2, size=(100, 100))
num_iterations = 10

final_arr = run_cellular_automaton(arr, num_iterations)
