from numba import cuda

@cuda.jit
def update_state_1D(width, configurations_dev, iteration):
    x = cuda.grid(1)

    left = (x - 1 + width) % width
    right = (x + 1) % width

    alive = (
        configurations_dev[iteration-1, left] +
        configurations_dev[iteration-1, right]
    )

    configurations_dev[iteration, x] = (
        (configurations_dev[iteration-1, x]) and (alive == 0)
    ) or ((not configurations_dev[iteration-1, x]) and (alive > 0))




@cuda.jit
def update_state_2D(width, height, configurations_dev, iteration):
    

    x, y = cuda.grid(2)
    
    left = (x - 1 + width) % width
    right = (x + 1) % width
    top = (y - 1 + height) % height
    bottom = (y + 1) % height
    
    # Moore neighbours
    alive = (
    configurations_dev[iteration-1, left, y] + configurations_dev[iteration-1, right, y] + configurations_dev[iteration-1, x, top] +configurations_dev[iteration-1, x, bottom] +
    configurations_dev[iteration-1, left, top] +configurations_dev[iteration-1, right, top] +configurations_dev[iteration-1, left, bottom] + configurations_dev[iteration-1, right, bottom]
)
      
    configurations_dev[iteration, x, y] = ((configurations_dev[iteration-1, x, y]) and (alive >= 2) and (alive < 4)) or ((not configurations_dev[iteration-1, x, y]) and (alive == 3))



@cuda.jit
def update_state_3D(width, height, depth, configurations_dev, iteration):
    x, y, z = cuda.grid(3)

    left = (x - 1 + width) % width
    right = (x + 1) % width
    top = (y - 1 + height) % height
    bottom = (y + 1) % height
    front = (z - 1 + depth) % depth
    back = (z + 1) % depth

    alive = (
        configurations_dev[iteration-1, left, y, z] +
        configurations_dev[iteration-1, right, y, z] +
        configurations_dev[iteration-1, x, top, z] +
        configurations_dev[iteration-1, x, bottom, z] +
        configurations_dev[iteration-1, left, top, z] +
        configurations_dev[iteration-1, right, top, z] +
        configurations_dev[iteration-1, left, bottom, z] +
        configurations_dev[iteration-1, right, bottom, z] +
        configurations_dev[iteration-1, left, y, front] +
        configurations_dev[iteration-1, right, y, front] +
        configurations_dev[iteration-1, left, y, back] +
        configurations_dev[iteration-1, right, y, back] +
        configurations_dev[iteration-1, x, top, front] +
        configurations_dev[iteration-1, x, top, back] +
        configurations_dev[iteration-1, x, bottom, front] +
        configurations_dev[iteration-1, x, bottom, back] +
        configurations_dev[iteration-1, left, top, front] +
        configurations_dev[iteration-1, right, top, front] +
        configurations_dev[iteration-1, left, bottom, front] +
        configurations_dev[iteration-1, right, bottom, front] +
        configurations_dev[iteration-1, left, top, back] +
        configurations_dev[iteration-1, right, top, back] +
        configurations_dev[iteration-1, left, bottom, back] +
        configurations_dev[iteration-1, right, bottom, back] +
        configurations_dev[iteration-1, x, top-1, z] +
        configurations_dev[iteration-1, x, bottom+1, z] +
        configurations_dev[iteration-1, x, top, front-1] +
        configurations_dev[iteration-1, x, bottom, back+1]
    )

    configurations_dev[iteration, x, y, z] = (    (   (configurations_dev[iteration-1, x, y, z])    and      (alive <= 26)     and     (alive >= 13)    ) or (     (not configurations_dev[iteration-1, x, y, z])    and    (    (alive <= 14 and alive >=13 )     or     (alive <= 19 and alive >=17)    )))
