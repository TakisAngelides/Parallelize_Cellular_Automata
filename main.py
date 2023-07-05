from init_state import get_initial_state
from rules import *
from datetime import datetime
import multiprocessing

def get_configurations(items):
    
    time_steps, initial_state, which_rules, trial_num = items
    
    shape = initial_state.shape
    
    N = shape[0]
    d = len(shape)

    print(d, time_steps, N, which_rules, trial_num, datetime.now(), flush = True)

    start = datetime.now()

    configurations = np.full(tuple(np.append(time_steps, initial_state.shape)), None, dtype=initial_state.dtype)

    configurations[0] = initial_state

    state = initial_state
    
    for t in range(1, time_steps):
        
        state = apply_rules(state, which_rules)
        
        configurations[t] = state
        
    end = datetime.now()
    
    duration = end - start
    
    with open(f'Timing_Results/njit_timed/{d}_{time_steps}_{N}_{trial_num}_{which_rules}.txt', 'w') as f:
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
                elif idx > 2 and idx < 4:
                    shape = (N, N)
                else:
                    shape = (N, N, N)
                
                initial_state = get_initial_state(shape, which_rules)  
                
                process_items.append([time_steps, initial_state, which_rules, i])

pool = multiprocessing.Pool(processes = 64)
results = pool.map(get_configurations, process_items)       
pool.close()
pool.join() 
