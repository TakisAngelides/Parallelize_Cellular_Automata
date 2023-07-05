import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = ['d', 'time', 'N', 'trial', 'rules', 'duration']
df = pd.DataFrame(columns = columns)
for element in os.listdir('Timing_Results/main_timed'):
    
    row = element.split('_')
    d = int(row[0])
    time = int(row[1])
    N = int(row[2])
    trial = int(row[3])
    rules = row[4]
    
    with open(f'Timing_Results/main_timed/{element}', 'r') as f:
        duration = f.read()

    data_tmp = [d, time, N, trial, rules, duration]
    df_tmp = pd.DataFrame([data_tmp])
    df = pd.concat([df, df_tmp], ignore_index=True)
    
d = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration'].mean()
d = d.rename(columns = {'duration' : 'average_duration'})

d_tmp = d[(d.N == 64) & (d.d == 3) & (d.rules == 'clouds')]
x = d_tmp.time
y = d_tmp.average_duration
plt.plot(x, y, '-x')
plt.savefig('sharon.png', bbox_inches = 'tight')
    

