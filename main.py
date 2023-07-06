from rules import *
import numba
from datetime import datetime
from init_state import get_initial_state
import sys

num_threads = int(sys.argv[1])

numba.set_num_threads(num_threads)
print('Number of numba threads is set to:', numba.get_num_threads(), flush = True)


def get_configurations(items):
    
    time_steps, initial_state, which_rules, trial_num = items
    
    shape = initial_state.shape
    
    N = shape[0]
    d = len(shape)

    print(d, time_steps, N, which_rules, trial_num, datetime.now(), flush = True)
    
    state = initial_state
    
    if d == 1:
    
        site_indices = np.array([np.unravel_index(idx, state.shape) for idx in range(N)])
    
    if d == 2:
    
        site_indices = np.array([np.unravel_index(idx, state.shape) for idx in range(N**2)])
    
    if d == 3:
    
        site_indices = np.array([np.unravel_index(idx, state.shape) for idx in range(N**3)])

    start = datetime.now()

    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    configurations[0] = initial_state

    for t in range(1, time_steps):
        
        if d == 1:
            
            state = apply_rules_1d(state, which_rules, site_indices)
            
        if d == 2:
        
            state = apply_rules_2d(state, which_rules, site_indices)
            
        if d == 3:
            
            state = apply_rules_3d(state, which_rules, site_indices)
        
        
        configurations[t] = state
        
    end = datetime.now()
    
    duration = end - start
    
    print(duration, flush = True)
    
    with open(f'Timing_Results/njit_parallel_timed/{d}_{time_steps}_{N}_{trial_num}_{which_rules}.txt', 'w') as f:
        f.write(f'{duration.total_seconds()}')

trial_num = 5
time_steps_list = [10, 20, 30, 40, 50]
N_list = [16, 32, 64, 128]
which_rules_list = ['54', '90', 'game_of_life', 'tumor_growth', 'clouds_I', 'builder_II']
process_items = []

for i in range(trial_num):
    for time_steps in time_steps_list:
        for N in N_list:
            for idx, which_rules in enumerate(which_rules_list):
                
                if idx < 2:
                    shape = (N)
                elif idx >= 2 and idx < 4:
                    shape = (N, N)
                else:
                    shape = (N, N, N)
                
                initial_state = get_initial_state(shape, which_rules)  
                
                process_items.append([time_steps, initial_state, which_rules, i])

# pool = multiprocessing.Pool(processes = 1)
# results = pool.map(get_configurations, process_items)       
# pool.close()
# pool.join() 

# for item in process_items:
#     get_configurations(item)

initial_state = get_initial_state((128, 128, 128), 'clouds_I')  
get_configurations([50, initial_state, 'clouds_I', 0])
