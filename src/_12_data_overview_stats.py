import pandas as pd
import os
import numpy as np
import importlib
from _00_CEAM_run import directory, known_format, solar_radiation, demands_list


'''FUNCTIONALITIES'''
def main():
    if solar_radiation == 'n':
        exclude_solar = True
    else:
        exclude_solar = False

    def check_exceptions(df, column, file):
        '''check the exceptions in input files'''
        if exclude_solar and column == 'Is':  # skip 'Is' column if solar_radiation is 'n'
            return
        max_val = round(df[column].max(), 3)    # max value with 3 decimal points
        min_val = round(df[column].min(), 3)    # min value with 3 decimal points
        if file == 'ECD_ready.csv' and column == 'RH' and (min_val < 0 or max_val > 100):   # range for RH in ECD file
            print(f'Data in column {column} of file {file} is incorrect!!\n'
                  f'RH must be within 0-100 range!!')
        if file == 'ECD_ready.csv' and column == 'Is' and min_val < 0:      # only positive radiation in ECD file
            print(f'Data in column {column} of file {file} is incorrect!!\n'
                  f'Solar radiation cannot be negative!!')
        if file == 'ICD_ready.csv' and column == 'RH' and (min_val < 0 or max_val > 100):   # range for RH in ICD file
            print(f'Data in column {column} of file {file} is incorrect!!\n'
                  f'RH must be within 0-100 range!!')
        if file == 'ED_ready.csv' and min_val < 0:      # no negative values in ED file
            print(f'Data in column {column} of file {file} is incorrect!!\n'
                  f'Energy demand cannot be negative!!')

    def calculate_statistics(df, column):
        '''calculate statistics for columns in data files'''
        if exclude_solar and column == 'Is':        # skip 'Is' column if solar_radiation is 'n'
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if df.empty or df[column].isnull().all():   # check if the df or column is empty
            max_val = np.nan
            min_val = np.nan
            avg_val = np.nan
            median_val = np.nan
            q1_val = np.nan
            q3_val = np.nan
            max_time = np.nan
            min_time = np.nan
        else:
            max_val = round(df[column].max(), 3)  # max value with 3 decimal points
            min_val = round(df[column].min(), 3)  # min value with 3 decimal points
            avg_val = round(df[column].mean(), 3)  # average value with 3 decimal points
            median_val = round(df[column].median(), 3)  # median with 3 decimal points
            q1_val = round(df[column].quantile(0.25), 3)  # Q1 with 3 decimal points
            q3_val = round(df[column].quantile(0.75), 3)  # Q3 with 3 decimal points
            max_time = df[df[column] == df[column].max()]['Time'].values[0]  # time corresponding to max value
            min_time = df[df[column] == df[column].min()]['Time'].values[0]  # time corresponding to min value

        return max_val, min_val, avg_val, median_val, q1_val, q3_val, max_time, min_time

    def process_files(directory, files):
        '''process csv files in a given directory'''
        # set columns to exclude
        all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
        exclude_columns = all_demands.difference(demands_list)

        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f'\nSummary for data in {file}:')
                total_rows = len(df)
                for column in df.columns:
                    if column in exclude_columns or (column == 'Is' and exclude_solar):
                        continue
                    if column != 'Time':
                        if column in ['HDD', 'CDD']:
                            column_sum = df[column].sum()
                            column_count = df[column].count()
                            print(
                                f'Column: {column}\n'
                                f'Sum: {column_sum}\n'
                                f'Count: {column_count} (out of {total_rows} rows of input file)'
                            )
                        else:
                            check_exceptions(df, column, file)  # check the exceptions
                            max_val, min_val, avg_val, median_val, q1_val, q3_val, max_time, min_time = (
                                calculate_statistics(df, column))    # call the stat function with calculated variables
                            if column == 'Is' and file == 'ECD_ready.csv' and not exclude_solar:  # stats for Is
                                print(
                                    f'Column: {column}\n'
                                    f'Max: {max_val}, Average: {avg_val}, Median: {median_val}, Q1: {q1_val}, Q3: {q3_val}\n'
                                    f'Max: {max_val} recorded on {max_time}\n'
                                )
                            else:  # stats for other columns
                                print(
                                    f'Column: {column}\n'
                                    f'Max: {max_val}, Min: {min_val}, Average: {avg_val}, Median: {median_val}, Q1: {q1_val}, Q3: {q3_val}\n'
                                    f'Max: {max_val} recorded on {max_time}\n'
                                    f'Min: {min_val} recorded on {min_time}'
                                )
                if 'Time' in df.columns:
                    df = df.drop(['Time'], axis=1)
            else:
                print(f'\nSummary for data in {file}:\nFile {file} does not exist in the directory {directory}')    # confirmation

    def compute_daily_averages(input_dir, output_dir, operational_files=None, exception_files=None):
        '''function performing daily assessments: averages and sums'''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if operational_files is not None:
            files = operational_files
        if exception_files is not None:
            files = [f for f in files if f not in exception_files]

        for file in files:
            file_path = os.path.join(input_dir, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)
                df.set_index('Time', inplace=True)

                # ensure 'Time' is always in original_columns
                original_columns = df.columns.tolist()
                if 'Time' in original_columns:
                    original_columns.remove('Time')         # temperary remove 'Time'

                if file == 'ED_ready.csv':
                    daily_df = df.resample('D').sum(min_count=1)  # sum with minimum count of 1 to avoid zero-filling empty periods
                else:
                    if 'Is' in df.columns and 'HDD' in df.columns and 'CDD' in df.columns and not exclude_solar:
                        is_col = df[['Is']].resample('D').sum(min_count=1)    # sum with minimum count of 1 for 'Is' column
                        hdd_col = df[['HDD']].resample('D').sum(min_count=1)  # sum with minimum count of 1 for 'HDD' column
                        cdd_col = df[['CDD']].resample('D').sum(min_count=1)  # sum with minimum count of 1 for 'CDD' column
                        other_cols = df.drop(columns=['Is','HDD','CDD']).resample('D').mean()  # mean for other columns
                        daily_df = pd.concat([is_col, hdd_col, cdd_col, other_cols], axis=1)
                    else:
                        daily_df = df.resample('D').mean()  # mean for all columns

                if daily_df.empty:
                    daily_df = pd.DataFrame(columns=['Time'] + original_columns)    # create empty df with original columns, including 'Time'
                else:
                    daily_df.reset_index(inplace=True)
                    daily_df['Time'] = daily_df['Time'].dt.strftime(known_d_format)
                    daily_df = daily_df[['Time'] + original_columns]        # reinsert 'Time' column as the first column

                output_file_name = file.replace('.csv', '_d.csv')
                output_file = os.path.join(output_dir, output_file_name)
                daily_df.to_csv(output_file, index=False)
                print(f"File {output_file_name} generated and saved in {output_dir}")
            else:
                print(f"No {file} in the given path {input_dir}")

    def compute_monthly_averages(input_dir, output_dir, operational_files=None, exception_files=None):
        '''function performing monthly assessments: averages and sums'''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if operational_files is not None:
            files = operational_files
        if exception_files is not None:
            files = [f for f in files if f not in exception_files]

        for file in files:
            file_path = os.path.join(input_dir, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)
                df.set_index('Time', inplace=True)

                original_columns = df.columns.tolist()
                if 'Time' in original_columns:
                    original_columns.remove('Time')

                if file == 'ED_ready.csv':
                    monthly_df = df.resample('M').sum(min_count=1)
                else:
                    if 'Is' in df.columns and not exclude_solar:
                        is_col = df[['Is']].resample('M').sum(min_count=1)
                        hdd_col = df[['HDD']].resample('M').sum(min_count=1)
                        cdd_col = df[['CDD']].resample('M').sum(min_count=1)
                        other_cols = df.drop(columns=['Is','HDD','CDD']).resample('M').mean()
                        monthly_df = pd.concat([is_col, hdd_col, cdd_col, other_cols], axis=1)
                    else:
                        monthly_df = df.resample('M').mean()

                if monthly_df.empty:
                    monthly_df = pd.DataFrame(columns=['Time'] + original_columns)
                else:
                    monthly_df.reset_index(inplace=True)
                    monthly_df['Time'] = monthly_df['Time'].dt.strftime(known_m_format)
                    monthly_df = monthly_df[['Time'] + original_columns]

                output_file_name = file.replace('.csv', '_m.csv')
                output_file = os.path.join(output_dir, output_file_name)
                monthly_df.to_csv(output_file, index=False)
                print(f"File {output_file_name} generated and saved in {output_dir}")
            else:
                print(f"No {file} in the given path {input_dir}")

    def compute_weekly_averages(input_dir, output_dir, operational_files=None, exception_files=None):
        '''function performing weekly assessments: averages and sums'''
        if not os.path.exists(output_dir):  # if output path exists
            os.makedirs(output_dir)  # create if not
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]  # filter daily averaged files
        if operational_files is not None:  # work on operational files
            files = operational_files
        if exception_files is not None:  # include the exception files
            files = [f for f in files if f not in exception_files]

        for file in files:  # iteration over files
            file_path = os.path.join(input_dir, file)  # complete path to files
            if os.path.exists(file_path):  # if the given files exist = True
                df = pd.read_csv(file_path)  # read the file as df
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)  # check if 'Time' is df
                df.set_index('Time', inplace=True)

                if file == 'ED_ready.csv':  # exception for ED_ready_d.csv file
                    weekly_df = df.resample('7D').sum(min_count=1)  # resample columns: compute sum every 7 days
                else:
                    if 'Is' in df.columns and not exclude_solar:  # exception for 'Is' column
                        is_col = df[['Is']].resample('7D').sum(min_count=1)  # resample 'Is' column to weekly frequency: compute sum
                        hdd_col = df[['HDD']].resample('7D').sum(min_count=1)
                        cdd_col = df[['CDD']].resample('7D').sum(min_count=1)
                        other_cols = df.drop(columns=['Is','HDD','CDD']).resample('7D').mean()  # resample other columns: compute mean
                        weekly_df = pd.concat([is_col, hdd_col, cdd_col, other_cols], axis=1)  # concatenate the results
                    else:
                        weekly_df = df.resample('7D').mean()  # resample data to weekly frequency: compute mean

                weekly_df['start'] = weekly_df.index        # compute start time of each week
                weekly_df['end'] = weekly_df.index + pd.Timedelta(days=7)   # compute end time of each week
                last_end_time = weekly_df.index[-1] + pd.Timedelta(days=7)  # calculate the nominal end of the last week
                if last_end_time > df.index.max():
                    last_end_time = df.index.max() + pd.Timedelta(hours=1)  # extend to 00:00 of the next day
                weekly_df.loc[weekly_df.index[-1], 'end'] = last_end_time
                weekly_df['start'] = weekly_df['start'].dt.strftime(known_format)   # format 'start' column
                weekly_df['end'] = weekly_df['end'].dt.strftime(known_format)       # format 'end' column
                weekly_df.reset_index(inplace=True)     # reset index to the 'Time' column
                weekly_df = weekly_df[['Time','start','end'] + [col for col in df.columns if col not in ['Time','start','end']]] # reorder columns
                weekly_df['Time'] = range(1, len(weekly_df) + 1)    # reset the 'Time' column (consecutive numbers)

                output_file_name = file.replace('.csv', '_w.csv')  # output name
                output_file = os.path.join(output_dir, output_file_name)  # output path
                weekly_df.to_csv(output_file, index=False)  # save as csv
                print(f"File {output_file_name} generated and saved in {output_dir}")  # confirmation
            else:
                print(f"No {file} in the given path {input_dir}")  # confirmation

    print(solar_radiation)

    files = ['ECD_ready.csv', 'ICD_ready.csv', 'ED_ready.csv']  # list of operational input files
    known_d_format = '%d.%m.%Y'         # daily format for averaging
    known_w_format = '%W.%Y'            # weekly format for averaging
    known_m_format = '%m.%Y'            # monthly format for averaging
    output_log_file = os.path.join(directory, '02_stats_report.txt')

    compute_daily_averages(directory, directory, files, None)
    compute_monthly_averages(directory, directory, files, None)
    compute_weekly_averages(directory, directory, files, None)

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
    _11_initial_csv_check.save_output_to_file(output_log_file, process_files, directory, files)


'''APPLICATION'''
if __name__ == "__main__":
    main()
