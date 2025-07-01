import pandas as pd
import os
from _00_CEAM_run import directory, known_format


'''GLOBAL ARGUMENTS AND FUNCTIONS'''
climate_classes = [
    {
        'name': 'AA',
        'Ti_outer_min': 10, 'Ti_outer_max': 25,
        'RHi_outer_min': 35, 'RHi_outer_max': 65,
        'Ti_seasonal_offset_p': 5, 'RHi_seasonal_offset_p': 0,
        'Ti_seasonal_offset_n': 5, 'RHi_seasonal_offset_n': 0,
        'Ti_short_offset': 2, 'RHi_short_offset': 5
    },
    {
        'name': 'A1',
        'Ti_outer_min': 10, 'Ti_outer_max': 25,
        'RHi_outer_min': 35, 'RHi_outer_max': 65,
        'Ti_seasonal_offset_p': 5, 'RHi_seasonal_offset_p': 10,
        'Ti_seasonal_offset_n': 10, 'RHi_seasonal_offset_n': 10,
        'Ti_short_offset': 2, 'RHi_short_offset': 5
    },
    {
        'name': 'A2',
        'Ti_outer_min': 10, 'Ti_outer_max': 25,
        'RHi_outer_min': 35, 'RHi_outer_max': 65,
        'Ti_seasonal_offset_p': 5, 'RHi_seasonal_offset_p': 0,
        'Ti_seasonal_offset_n': 10, 'RHi_seasonal_offset_n': 0,
        'Ti_short_offset': 2, 'RHi_short_offset': 10
    },
    {
        'name': 'B',
        'Ti_outer_min': None, 'Ti_outer_max': 30,
        'RHi_outer_min': 30, 'RHi_outer_max': 70,
        'Ti_seasonal_offset_p': 10, 'RHi_seasonal_offset_p': 10,
        'Ti_seasonal_offset_n': 20, 'RHi_seasonal_offset_n': 10,
        'Ti_short_offset': 5, 'RHi_short_offset': 10
    },
    {
        'name': 'C',
        'Ti_outer_min': None, 'Ti_outer_max': 40,
        'RHi_outer_min': 25, 'RHi_outer_max': 75,
        'Ti_seasonal_offset_p': None, 'RHi_seasonal_offset_p': None,
        'Ti_seasonal_offset_n': None, 'RHi_seasonal_offset_n': None,
        'Ti_short_offset': None, 'RHi_short_offset': None
    },
    {
        'name': 'D',
        'Ti_outer_min': None, 'Ti_outer_max': None,
        'RHi_outer_min': None, 'RHi_outer_max': 75,
        'Ti_seasonal_offset_p': None, 'RHi_seasonal_offset_p': None,
        'Ti_seasonal_offset_n': None, 'RHi_seasonal_offset_n': None,
        'Ti_short_offset': None, 'RHi_short_offset': None
    },
]


'''FUNCTIONALITIES'''
def main():
    h_comb_path = os.path.join(directory, 'h_comb.csv')  # path to hourly input
    d_comb_path = os.path.join(directory, 'd_comb.csv')  # path to daily input
    w_comb_path = os.path.join(directory, 'w_comb.csv')  # path to weekly input

    h_comb = pd.read_csv(h_comb_path)  # load the hourly input
    d_comb = pd.read_csv(d_comb_path)  # load the daily input
    w_comb = pd.read_csv(w_comb_path)  # load the weekly input

    h_comb['Time'] = pd.to_datetime(h_comb['Time'], format=known_format)  # convert 'Time' column to df
    d_comb['Time'] = pd.to_datetime(d_comb['Time'], format='%d.%m.%Y')  # convert 'Time' column to df
    h_comb['Date'] = h_comb['Time'].dt.date  # extract date from 'Time'; required for merging
    d_comb['Date'] = d_comb['Time'].dt.date  # extract date from 'Time'; required for merging

    d_comb = d_comb[['Date', 'Ti', 'RHi']]  # the necessary columns from d_comb

    merged = pd.merge(h_comb, d_comb, on='Date', how='left', suffixes=('', '_d'))  # merge the data on the 'Date' column

    # handle empty records in d_comb
    mask_empty = merged[['Ti_d', 'RHi_d']].isna().any(axis=1)
    merged.loc[mask_empty, ['Ti_d_ave', 'RHi_d_ave']] = None

    merged['Ti_d_ave'] = merged['Ti_d']  # new column for daily average value
    merged['RHi_d_ave'] = merged['RHi_d']  # new column for daily average value

    merged.drop(columns=['Date', 'Ti_d', 'RHi_d'], inplace=True)  # drop the selected columns

    # monthly averages
    merged['Year_Month'] = merged['Time'].dt.strftime('%Y-%m')
    monthly_averages = merged.groupby('Year_Month').agg({'Ti': 'mean', 'RHi': 'mean'}).reset_index()
    monthly_averages.rename(columns={'Ti': 'Ti_m_ave', 'RHi': 'RHi_m_ave'}, inplace=True)

    merged = pd.merge(merged, monthly_averages, left_on='Year_Month', right_on='Year_Month', how='left')  # merge monthly averages

    merged.drop(columns=['Year_Month'], inplace=True)  # drop the selected columns

    # merge weekly data
    w_comb['start'] = pd.to_datetime(w_comb['start'], format=known_format)
    w_comb['end'] = pd.to_datetime(w_comb['end'], format=known_format)

    # initialize the weekly columns with NaN
    merged['Te_w'] = float('nan')
    merged['RHe_w'] = float('nan')
    merged['Ti_w'] = float('nan')
    merged['RHi_w'] = float('nan')

    # assign weekly data to hourly data
    for _, row in w_comb.iterrows():
        mask = (merged['Time'] >= row['start']) & (merged['Time'] < row['end'])
        merged.loc[mask, 'Te_w'] = row['Te']
        merged.loc[mask, 'RHe_w'] = row['RHe']
        merged.loc[mask, 'Ti_w'] = row['Ti']
        merged.loc[mask, 'RHi_w'] = row['RHi']

    # calculate MA for weekly temperature (Ti_MA)
    rolling_window_T = 7 * 24  # 7 days in hours
    merged['Ti_MA-7d'] = float('nan')   # initialize a column for the MA-7d

    # calculate the constant average for the first 7 days (168 records)
    initial_average = merged['Ti'][:rolling_window_T].mean()
    merged.loc[:rolling_window_T-1, 'Ti_MA-7d'] = initial_average

    # apply moving average for every subsequent 24 records
    for start in range(rolling_window_T, len(merged)+24, 24):   # iterate in steps of 24 hours
        window_start = start - rolling_window_T                 # define the 7-days window
        window_end = start
        moving_avg = merged['Ti'][window_start:window_end].mean()   # compute over the previous 168 records
        merged.loc[start-24:start-1, 'Ti_MA-7d'] = moving_avg       # assign the MA to the next 24 records

    merged['Ti_MA_h'] = float('nan')    # initialize a column for the MA_h

    # calculate the average for the first 7 days (168 records); used for MA_h
    initial_average_h = merged['Ti'][:rolling_window_T].mean()
    merged.loc[:rolling_window_T - 1, 'Ti_MA_h'] = initial_average_h

    # apply rolling average starting from the 169th record; used for MA_h
    merged.loc[rolling_window_T:, 'Ti_MA_h'] = (
        merged['Ti']
        .rolling(window=rolling_window_T, min_periods=1)
        .mean()
        .iloc[rolling_window_T:]
    )

    # calculate MA for monthly humidity (RHi_MA)
    rolling_window_RH = 30 * 24  # 30 days in hours
    merged['RHi_MA-30d'] = float('nan')     # initialize a column for the RHi_MA-30d

    # calculate the constant average for the first 30 days (720 records)
    initial_average = merged['RHi'][:rolling_window_RH].mean()
    merged.loc[:rolling_window_RH - 1, 'RHi_MA-30d'] = initial_average

    # apply MA for every subsequent 24 records
    for start in range(rolling_window_RH, len(merged)+24, 24):  # iterate in steps of 24 hours
        window_start = start - rolling_window_RH                # define the 30-days window
        window_end = start
        moving_avg = merged['RHi'][window_start:window_end].mean()  # compute over the previous 720 records
        merged.loc[start-24:start-1, 'RHi_MA-30d'] = moving_avg     # assign the MA to the next 24 records

    merged['RHi_MA_h'] = float('nan')       # initialize a column for the RHi_MA_h (over 30 days)

    # calculate the average for the first 30 days (720 records); used for RHi_MA_h
    initial_average_h = merged['RHi'][:rolling_window_RH].mean()
    merged.loc[:rolling_window_RH - 1, 'RHi_MA_h'] = initial_average_h

    # apply rolling average starting from the 721st record; used for RHi_MA_h
    merged.loc[rolling_window_RH:, 'RHi_MA_h'] = (
        merged['RHi']
        .rolling(window=rolling_window_RH, min_periods=1)
        .mean()
        .iloc[rolling_window_RH:]
    )

    merged['Time'] = merged['Time'].dt.strftime(known_format)  # convert 'Time' back to '%d.%m.%Y %H:%M' format

    output_path = os.path.join(directory, 'h_comb_CC.csv')  # output path
    merged.to_csv(output_path, index=False)  # save the modified df


'''APPLICATION'''
if __name__ == "__main__":
    main()
