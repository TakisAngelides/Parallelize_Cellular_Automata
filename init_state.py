import numpy as np
import cellpylib as cpl

def initialize_array(shape, initial_value=0, dtype=np.bool):
    """"
    Arguments:
        shape: the dimension of the array to to be created in the function
        initial_value: value for every entry of the array, by default 0
        dtype: type of initial_value, by default np.bool

    Return:
        array of shape shape, values are by default 0 and entries have type np.bool
    """
    
    return np.full(shape, initial_value, dtype=dtype)

def initialize_random_array(shape, values=[True, False], p = [0.5, 0.5]):
    """"
    Arguments:
        shape: the dimension of the array 
        values: these list give the possible entries of the array 
        p: p gives the probabilites that entries have a value of the list values 

    Return:
        array of shape shape, entries have values from values, which are distributed according to p. By default values=[True, False] and p = [0.5, 0.5]
    """
    
    return np.random.choice(values, size = shape, p = p)

def initialize_cellpy_array(shape):
    """
    Initializes a cellpy array with the specified shape.

    Parameters:
        shape (tuple): A tuple specifying the dimensions of the array.

    Returns:
        cpl.array: A cellpy array initialized with the specified shape and boolean data type.

    Example:
        >>> initialize_cellpy_array((3, 4))
        cpl.array([[False, False, False, False],
                   [False, False, False, False],
                   [False, False, False, False]], dtype=bool)
    """  
    
    return cpl.init_simple2d(*shape, dtype = bool)

def initialize_glider(shape):
    """
    Initializes a glider pattern in a 2D numpy array with the shape shape.

    Parameters:
        shape (tuple): A tuple specifying the dimensions of the array.

    Returns:
        numpy.array: A 2D numpy array with the glider pattern.

    Example:
        >>> initialize_glider((5, 5))
        array([[False, False, False, False, False],
               [False, False, False,  True, False],
                [False, False, False, False,  True],
                [False,  True,  True,  True,  True],
                [False, False, False, False, False]], dtype=bool)
    """
    
    initial_state = np.zeros(shape, dtype = bool)
    initial_state[:,initial_state.shape[0]//2] = True
    initial_state[0,initial_state.shape[0]//2] = False
    initial_state[len(initial_state)//2,initial_state.shape[0]//2] = False
    
    return initial_state

def initialize_two_glider_octomino(shape):

    """
    Gives the pattern glider, it evolves indefintltey 
    Arguments:
        shape: a tuple which gives the shape of the initial configuration
    """
    
    initial_state = np.zeros(shape)
    
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
    initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+1] = True
    initial_state[(len(initial_state)//2)-1, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)-2, (initial_state.shape[0]//2)+2] = True
    initial_state[(len(initial_state)//2)+1, initial_state.shape[0]//2] = True
    
    return initial_state


def initialize_glider(shape):

    """
    Gives the pattern glider, it evolves indefintltey 
    (see: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    Arguments:
        shape: a tuple which gives the shape of the initial configuration
    """
    
    initial_state = np.zeros(shape)

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
    initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
    initial_state[(len(initial_state)//2)-1, initial_state.shape[0]//2] = True
    initial_state[(len(initial_state)//2)-2, initial_state.shape[0]//2] = True
    initial_state[len(initial_state)//2-1, (initial_state.shape[0]//2)-2] = True


    return initial_state


def initialize_glider_gun(shape):
    """
    Gives the pattern glider gun, it evolves indefintltey 
    (see: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
    Arguments:
        shape: a tuple which gives the shape of the initial configuration
    """

    initial_state = np.zeros(shape)

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)-1] = True
    initial_state[len(initial_state)//2, initial_state.shape[0]//2] = True
    initial_state[(len(initial_state)//2)-1, initial_state.shape[0]//2] = True
    initial_state[(len(initial_state)//2)-1, initial_state.shape[0]//2-1] = True

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+9] = True
    initial_state[len(initial_state)//2-1, (initial_state.shape[0]//2)+9] = True
    initial_state[len(initial_state)//2+1, (initial_state.shape[0]//2)+9] = True

    initial_state[len(initial_state)//2+2, (initial_state.shape[0]//2)+10] = True
    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+10] = True

    initial_state[len(initial_state)//2+3, (initial_state.shape[0]//2)+11] = True
    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+11] = True


    initial_state[len(initial_state)//2+3, (initial_state.shape[0]//2)+12] = True
    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+12] = True

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+13] = True

    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+14] = True
    initial_state[len(initial_state)//2+2, (initial_state.shape[0]//2)+14] = True

    
    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+15] = True
    initial_state[len(initial_state)//2+1, (initial_state.shape[0]//2)+15] = True
    initial_state[len(initial_state)//2-1, (initial_state.shape[0]//2)+15] = True

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+16] = True


    initial_state[len(initial_state)//2-1, (initial_state.shape[0]//2)+19] = True
    initial_state[len(initial_state)//2-1, (initial_state.shape[0]//2)+20] = True

    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+19] = True
    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+20] = True

    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+19] = True
    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+20] = True

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+21] = True
    initial_state[len(initial_state)//2-4, (initial_state.shape[0]//2)+21] = True

    initial_state[len(initial_state)//2, (initial_state.shape[0]//2)+23] = True
    initial_state[len(initial_state)//2-4, (initial_state.shape[0]//2)+23] = True

    initial_state[len(initial_state)//2+1, (initial_state.shape[0]//2)+23] = True
    initial_state[len(initial_state)//2-5, (initial_state.shape[0]//2)+23] = True

    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+33] = True
    initial_state[len(initial_state)//2-2, (initial_state.shape[0]//2)+34] = True

    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+33] = True
    initial_state[len(initial_state)//2-3, (initial_state.shape[0]//2)+34] = True

    return initial_state





