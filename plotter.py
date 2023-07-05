import os
import numpy as np
import pandas as pd

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


df_averaged = df.groupby(['d', 'time', 'N', 'rules'], as_index = False)['duration'].mean()
df_averaged = df_averaged.rename(columns = {'duration' : 'average_duration'})


    

