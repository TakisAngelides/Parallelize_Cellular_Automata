import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['d', 'time', 'N', 'trial', 'rules']
df = pd.DataFrame(columns = columns)
for element in os.listdir('Timing_Results/main_timed'):
    
    split_parts = element.split("_")
    split_parts = [int(part) if part.isdigit() else part for part in split_parts]
    file_parts = split_parts[-1].split(".")
    split_parts[-1:] = file_parts[:-1] + ["builder_" + file_parts[-1]]
    
    d, time, N, trial, rules = split_parts
    
    with open(f'Timing_Results/main_timed/{element}', 'r') as f:
        duration = f.read()

    data_tmp = [d, time, N, trial, rules, duration]
    df_tmp = pd.DataFrame([data_tmp])
    df = pd.concat([df, df_tmp], ignore_index=True)


d = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration'].mean()
d = d.rename(columns = {'duration' : 'average_duration'})

x, y = d[(d.N == 64) & (d.d == 3) & (d.rules == 'clouds_I')]
plt.plot(x, y, '-x')
plt.savefig('sharon.png', bbox_inches = 'tight')
    

