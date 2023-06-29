import numpy as np
import init_state


# Example usage
def my_function():
    print("Hello")

def call_function_multiple_times(func,N, init):


    #history[i, :, :] is the i-th timeslice of the evolution
    history=np.full(tuple(np.append(N,init.shape)),1,dtype=init.dtype)

    #initialize
    history[0, :, :]=init


    for _ in range(1,N):
        func()

call_function_multiple_times(my_function, 10 ,init_state.initialize_array((2,2)))