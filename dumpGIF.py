import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def dumpGIF(states, filename):
    
    fig = plt.figure()
    ax = fig.add_subplot()
        
    def animate(frame):
        
        ax.clear()
        ax.set_title(f"Iteration: {frame}")
        ax.imshow(states[frame], cmap = 'binary')
        ax.set_xticks([])
        ax.set_yticks([])
    
    ani = animation.FuncAnimation(fig, animate, frames = len(states), interval = 200)
    ani.save(filename, writer='pillow')
