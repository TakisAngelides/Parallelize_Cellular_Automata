import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['d', 'time', 'N', 'trial', 'rules', 'duration']
df = pd.DataFrame(columns = columns)
for element in os.listdir('Timing_Results/main_timed'):
    
    row = element.split('.')[0]
    row = row.split('_')
    d = int(row[0])
    time = int(row[1])
    N = int(row[2])
    trial = int(row[3])
    rules = row[4]
    
    with open(f'Timing_Results/main_timed/{element}', 'r') as f:
        duration = float(f.read())

    data_tmp = [d, time, N, trial, rules, duration]
    df_tmp = pd.DataFrame([data_tmp], columns = columns)
    df = pd.concat([df, df_tmp], ignore_index=True)


df_std = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration'].std()
df = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration'].mean()


for N in df.N:
    for d in df.d:
        for rules in df.rules:
            
            d_tmp = df[(df.N == N) & (df.d == d) & (df.rules == rules)]
            d_std_tmp = df_std[(df.N == N) & (df.d == d) & (df.rules == rules)]

            x = d_tmp.time
            y = d_tmp.duration
            yerr = d_std_tmp.duration
            
            plt.errorbar(x, y, yerr = yerr, fmt = 'x')
            plt.plot(x, y)
            plt.ylabel('Duration (s)')
            plt.xlabel('Evolution Steps')
            plt.savefig(f'Plots/main_timed/duration_vs_time/{d}_{N}_{rules}.png', bbox_inches = 'tight')
    

