from datetime import datetime
# import multiprocessing
from get_configurations import *

def get_simulation(items):

    time_steps, initial_state, which_rules, trial_num = items

    # Set the size of the grid
    shape = initial_state.shape

    N = shape[0]

    d = len(shape)

    print(d, time_steps, N, which_rules, trial_num, datetime.now(), flush = True)


    start = datetime.now()

    # Run the cellular automaton and get the configurations
    configurations = get_configurations(time_steps, initial_state, shape, which_rules)

    end = datetime.now()

    duration = end - start
    
    print(duration, flush = True)

    with open(f'Timing_Results/numba_gpu_timed/{d}_{time_steps}_{N}_{trial_num}_{which_rules}.txt', 'w') as f:
        f.write(f'{duration.total_seconds()}')


trial_num = 5
time_steps_list = [10, 20, 30, 40, 50]
N_list = [4, 8, 16, 32, 64, 128]
which_rules_list = ['54', '90', 'game_of_life', 'clouds_I', 'builder_II'] # 'tumor_growth'
process_items = []

for i in range(trial_num):
    for time_steps in time_steps_list:
        for N in N_list:
            for idx, which_rules in enumerate(which_rules_list):

                if idx < 2:
                    shape = (N)
                elif idx >= 2 and idx < 4:
                    shape = (N, N)
                elif idx >= 4:
                    shape = (N, N, N)

                initial_state = get_initial_state(shape, which_rules)

                process_items.append([time_steps, initial_state, which_rules, i])

# pool = multiprocessing.Pool(processes = 64)
# results = pool.map(get_simulation, process_items)
# pool.close()
# pool.join()

for item in process_items:
    get_simulation(item)
