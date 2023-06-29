import numpy as np
import init_state




def get_configurations(func,N, init):


    #history[i, :, :] is the i-th timeslice of the evolution
    history=np.full(tuple(np.append(N,init.shape)),1,dtype=init.dtype)

    #initialize
    history[0, :, :]=init


    for _ in range(1,N):
        func()

