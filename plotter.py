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
    df_tmp = pd.DataFrame([data_tmp], columns = columns)
    df = pd.concat([df, df_tmp], ignore_index=True)

print(df)
df = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration']
print(df)
df = df.mean()
print(df)
df = df.rename(columns = {'duration' : 'average_duration'})
print(df)

d_tmp = df[(df.N == 64) & (df.d == 3) & (df.rules == 'clouds')]

x = d_tmp.time
y = d_tmp.average_duration
plt.plot(x, y, '-x')
plt.savefig('sharon.png', bbox_inches = 'tight')
    

