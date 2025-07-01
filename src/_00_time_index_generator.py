import pandas as pd
import os
from pathlib import Path


def generate_time_index_csv(frequency_in_minutes=60, path=None,
                            start_time='2025-01-01 00:00', end_time='2025-04-30 23:59',
                            filename='time_index.csv', timestamp_format='%d.%m.%Y %H:%M'):
    '''suplementary function allowing to create the df'''
    if path is None:
        path = os.path.join(Path.home(), 'Desktop')

    time_index = pd.date_range(start=start_time, end=end_time, freq=str(frequency_in_minutes) + 'T')    # date range
    df = pd.DataFrame(time_index, columns=['Time'])         # convert to df
    df['Time'] = df['Time'].dt.strftime(timestamp_format)   # format the 'Time' column
    full_path = os.path.join(path, filename)    # full file path
    df.to_csv(full_path, index=False)       # write to csv file
    print('csv time-index file created')    # confirmation


'''APPLICATION'''
frequency = 10      # desired frequency in mins

generate_time_index_csv(frequency, filename=f'time_index_{frequency}.csv') # generate the timestamp

