import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
def dumpGIF(states, filename):
    """
    Arguments:
    
    states: d dimensional cube of the evolution of the cellular automata
    filename: name of the saved file
    Returns:
        Shows and saves a gif under filename
    """
    
    fig = plt.figure()
    
    if len(states[0].shape) == 3:
        
        ax = fig.add_subplot(111, projection = '3d')
    
    elif len(states[0].shape) == 1:
        
        ax = fig.add_subplot()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    elif len(states[0].shape) == 2:
    
        ax = fig.add_subplot()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
    
    def animate(frame):
        
        ax.clear()
        ax.set_axis_off()
        ax.set_title(f'{frame}')

        if len(states[0].shape) == 3:
            
            ax.clear()
            ax.set_axis_off()
            ax.set_title(f'{frame}')
            ax.voxels(states[frame % len(states)], edgecolor = "k")

        elif len(states[0].shape) == 2:
            
            ax.clear()
            ax.set_title(f"Iteration: {frame}")
            ax.imshow(states[frame], cmap = 'binary')
            ax.set_xticks([])
            ax.set_yticks([])

        elif len(states[0].shape) == 1:
            
            ax.clear()
            ax.set_xlim(0, len(states[0]))
            ax.set_ylim(-0.5, 1.5)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_title(f"Iteration: {frame}")
            ax.set_aspect('equal')
                        
            for x, box in enumerate(states[frame % len(states)]):
                if box == 1:
                    ax.add_patch(plt.Rectangle((x, 0), 1, 1, edgecolor='black', facecolor='red'))
                else:
                    ax.add_patch(plt.Rectangle((x, 0), 1, 1, edgecolor='black', facecolor='white'))
            
    ani = animation.FuncAnimation(fig, animate, frames = len(states), interval = 10)
    ani.save(filename, writer='pillow')
    plt.show()
