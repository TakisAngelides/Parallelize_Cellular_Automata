import numpy as np


def initialize_array(shape, initial_value=0, dtype=np.bool):
    
    return np.full(shape, initial_value, dtype=dtype)




def initialize_random_array(shape,values=[True, False]):
    
    return np.random.choice(values, size=shape)


