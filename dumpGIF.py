import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
def dumpGIF(states, filename):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title(f'{0}')
    
    def animate(frame):
        
        ax.set_title(f'{frame}')
        ax.clear()
        ax.set_axis_off()
        
        return ax.voxels(states[frame%len(states)], edgecolor="k")
    
    ani = animation.FuncAnimation(fig, animate, frames = len(states), interval = 0.2)
    ani.save(filename, writer='pillow')
    plt.show()
