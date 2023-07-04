from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

@cuda.jit
def update_state(state, new_state, width, height):
    
    i, j = cuda.grid(2) # each thread is specified by two indices
    
    left = (j - 1 + width) % width
    right = (j + 1) % width
    top = (i - 1 + height) % height
    bottom = (i + 1) % height
    
    new_state[i, j] = (state[i, left] + state[i, right] + state[top, j] + state[bottom, j]) % 2

@cuda.jit
def get_configurations(initial_state, num_iterations, width, height):
    
    # The grid is the grid of blocks where each block contains threads arranged in a two dimensional grid
    
    # state.shape[0] represents the size of the first dimension of the state array. In the case of a 2D cellular automaton, it corresponds to the number of rows.
    # block_size[0] is the size of a block in the first dimension of the CUDA grid. It represents the number of threads in a block along the rows.
    # (state.shape[0] + block_size[0] - 1) calculates the total number of threads required to cover the rows of the state array, considering that the last block may have fewer threads if the number of rows is not divisible by the block size.
    # (state.shape[0] + block_size[0] - 1) // block_size[0] performs integer division, rounding down to determine the number of blocks needed to cover the rows. This gives the grid size in the first dimension.
    
    # Now, let's consider the second dimension:

    # state.shape[1] represents the size of the second dimension of the state array. In the case of a 2D cellular automaton, it corresponds to the number of columns.
    # block_size[1] is the size of a block in the second dimension of the CUDA grid. It represents the number of threads in a block along the columns.
    # (state.shape[1] + block_size[1] - 1) calculates the total number of threads required to cover the columns of the state array, considering that the last block may have fewer threads if the number of columns is not divisible by the block size.
    # (state.shape[1] + block_size[1] - 1) // block_size[1] performs integer division, rounding down to determine the number of blocks needed to cover the columns. This gives the grid size in the second dimension.
    # Therefore, (grid_size_x, grid_size_y) represents the grid size in the X and Y dimensions, respectively, and it determines the number of blocks required to cover the entire state array in both dimensions.
    
    # In CUDA programming, it is common to divide the data into blocks and assign each block to a thread. However, the total number of elements in the data may not be evenly divisible by the block size. In such cases, the last block may have fewer elements than the other blocks.
    # To ensure that all elements are covered, we need to calculate the total number of threads required, taking into account the possibility of an incomplete last block.
    # In the context of the calculation (state.shape[0] + block_size[0] - 1), we add block_size[0] - 1 to the total number of elements in the first dimension of the state array. This accounts for the additional threads needed to cover any remaining elements in the last block.
    # Let's consider an example:
    # Suppose the state array has 100 rows (state.shape[0] = 100) and the block size is 32 rows (block_size[0] = 32).
    # The calculation (state.shape[0] + block_size[0] - 1) would be (100 + 32 - 1) = 131, which represents the total number of threads required to cover the rows of the state array, considering the last block may have fewer threads.
    # By performing integer division (state.shape[0] + block_size[0] - 1) // block_size[0], we obtain 131 // 32 = 4. This means that we need 4 blocks in the first dimension to cover the rows of the state array.
    # The same logic applies to the second dimension of the state array when calculating the grid size in the Y dimension.
    
    block_size = (1, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1])

    state = initial_state.copy()
    new_state = np.empty_like(state)

    for _ in range(num_iterations):
        update_state[grid_size, block_size](state, new_state, width, height) # on GPU 
        state, new_state = new_state, state

    return state

# Set the size of the grid
width = 100
height = 100

# Set the number of iterations
num_iterations = 1

# Create the initial state randomly
initial_state = np.random.randint(0, 2, size = (width, height), dtype = np.uint8)

# Run the cellular automaton
final_state = get_configurations[1, 1](initial_state, num_iterations, width, height)

# Plot the final state
plt.imshow(final_state, cmap='binary')
plt.savefig('test.png', bbox_inches='tight')

