from itertools import product
import cupy as cp
from numba import njit, prange, cuda


@cp.fuse(kernel_name = 'addition')
def addition(x, y):
    return x + y

@cp.fuse(kernel_name = 'count_alive_neighbours')
def count_alive_neighbours(site: cp.ndarray, state: cp.ndarray) -> int:
    
    """
         
    Inputs:
    
        site : 1-dimensional numpy array of length d - where d is the number of dimensions of the hypercube - specifying the site of a given cell

        state : d-dimensional numpy array specifying the state of the cellular automata, the value at each element is boolean-like

    product(cp.array([0, -1, 1]), repeat = len(site)) is equivalent to [[x, y, z] for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]] for d = 3
    
    from the above list we omit the first element [0, 0, 0] since we want to add these lists to site = cp.array([i, j, k]) to get the neighbour indices hence the [1:]
    
    then we do broadcasting of site which is of shape (d,) and this product array which is of shape (3^d, d) to get the tuples which represent the indices of the neighbours
    
    and then we access the state with these and add them up to see how many are alive and return the integer of alive neighbours

    be careful that this assumes by summing that the elements of the state are boolean-like eg 0, 1 or False, True and not something like 2, 3
    
    Returns:
    
        int : number of alive neighbours

    """
    
    state_shape = state.shape
    d = len(state_shape)
    
    if d == 1:
        
        # Get the thread indices
        thread_x = cp.threadIdx.x
        
        # Get the block indices
        block_x = cp.blockIdx.x
        
        # Get the dimensions of the array
        dim_x = state_shape

        # Calculate the global indices
        global_x = block_x * cp.blockDim.x + thread_x
        
        # Handle edge cases
        right = (global_x + 1) % dim_x 
        left = (global_x - 1 + dim_x) % dim_x 
                
        return addition(state[left], state[right])
        
    # elif d == 2:
        
    #     # Get the thread indices
    #     thread_x = cp.threadIdx.x
    #     thread_y = cp.threadIdx.y

    #     # Get the block indices
    #     block_x = cp.blockIdx.x
    #     block_y = cp.blockIdx.y

    #     # Get the dimensions of the array
    #     dim_x, dim_y = state.shape

    #     # Calculate the global indices
    #     global_x = block_x * cp.blockDim.x + thread_x
    #     global_y = block_y * cp.blockDim.y + thread_y
        
    #     sum = -state[site]
    #     for i in prange(-1, 2):
    #         for j in prange(-1, 2):
    #             sum = addition(sum, state[global_x + i, global_y + j])
    #     return sum
    
    # elif d == 3:
        
    #     # Get the thread indices
    #     thread_x = cp.threadIdx.x
    #     thread_y = cp.threadIdx.y
    #     thread_z = cp.threadIdx.z

    #     # Get the block indices
    #     block_x = cp.blockIdx.x
    #     block_y = cp.blockIdx.y
    #     block_z = cp.blockIdx.z

    #     # Get the dimensions of the array
    #     dim_x, dim_y, dim_z = state.shape

    #     # Calculate the global indices
    #     global_x = block_x * cp.blockDim.x + thread_x
    #     global_y = block_y * cp.blockDim.y + thread_y
    #     global_z = block_z * cp.blockDim.z + thread_z
        
    #     sum = -state[site]
    #     for i in prange(-1, 2):
    #         for j in prange(-1, 2):
    #             for k in prange(-1, 2):
    #                 sum = addition(sum, state[global_x + i, global_y + j, global_z + k])
    #     return sum
