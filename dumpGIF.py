import matplotlib.pyplot as plt
import matplotlib.animation as animation



def dumpGIF(states,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(frame):
        ax.clear()
        ax.set_axis_off()
        ax.voxels(states[frame], edgecolor="k")

    ani = animation.FuncAnimation(fig, animate, frames=len(states), interval=200)
    ani.save(filename, writer='pillow')
    plt.show()
