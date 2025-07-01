import os
import pandas as pd
import importlib
from _00_CEAM_run import directory, known_format, solar_radiation
from _11_initial_csv_check import directory_others


'''FUNCTIONALITIES'''
def main():
        def get_weekly_extremes(directory, filename, exceptions=None):
            '''function to define the extreme weeks'''
            if exceptions is None:
                exceptions = []     # list to store all exceptions

            def calculate_weekly_stats(week_df):
                '''calculate weekly statistics used for extreme weeks'''
                if solar_radiation == 'n':
                    avg_Te = week_df['Te'].mean()
                    avg_RHe = week_df['RHe'].mean()
                    avg_pve = week_df['pve'].mean()
                    return avg_Te, avg_RHe, avg_pve
                else:
                    avg_Te = week_df['Te'].mean()
                    avg_RHe = week_df['RHe'].mean()
                    avg_pve = week_df['pve'].mean()
                    sum_Is = week_df['Is'].sum(min_count=1)
                    return avg_Te, avg_RHe, avg_pve, sum_Is

            def iterate_files(directory, filename):
                '''iterate over files to find extreme weeks'''
                extreme_results = {}
                file_path = os.path.join(directory, filename)
                df = pd.read_csv(file_path)
                df['Time'] = pd.to_datetime(df['Time'], format=known_d_format)
                df.set_index('Time', inplace=True)
                all_weeks = []  # list to store all examined weeks

                print(f"\nExamined file: {filename}\nStart Date: {df.index[0].strftime(known_d_format)}, "
                      f"End Date: {df.index[-1].strftime(known_d_format)}\n")

                for i in range(len(df) - 6):
                    if solar_radiation == 'n':
                        week_df = df.iloc[i:i + 7]
                        start_week = week_df.index[0].strftime(known_d_format)
                        end_week = week_df.index[-1].strftime(known_d_format)
                        avg_T, avg_RH, avg_pv = calculate_weekly_stats(week_df)
                        all_weeks.append([start_week, end_week, avg_T, avg_RH, avg_pv])

                        all_weeks_df = pd.DataFrame(all_weeks, columns=['Start', 'End', 'Te', 'RHe', 'pve'])
                        all_weeks_df.index.name = 'Index'
                        all_weeks_file_path = os.path.join(directory, f"{filename.split('.')[0]}_all_weeks.csv")
                        all_weeks_df.to_csv(all_weeks_file_path)

                        min_Te, max_Te = float('inf'), float('-inf')
                        max_RHe, min_RHe = float('-inf'), float('inf')
                        extreme_weeks = {}

                        avg_Te, avg_RHe, avg_pve = calculate_weekly_stats(week_df)
                        if avg_Te is not None:
                            if avg_Te < min_Te:
                                min_Te = avg_Te
                                extreme_weeks['EWW'] = week_df
                            if avg_Te > max_Te:
                                max_Te = avg_Te
                                extreme_weeks['ESW'] = week_df
                            if avg_RHe > max_RHe:
                                max_RHe = avg_RHe
                                extreme_weeks['EHW'] = week_df
                            if avg_RHe < min_RHe:
                                min_RHe = avg_RHe
                                extreme_weeks['EDW'] = week_df
                    else:
                        week_df = df.iloc[i:i + 7]
                        start_week = week_df.index[0].strftime(known_d_format)
                        end_week = week_df.index[-1].strftime(known_d_format)
                        avg_T, avg_RH, avg_pv, sum_Is = calculate_weekly_stats(week_df)
                        all_weeks.append([start_week, end_week, avg_T, avg_RH, sum_Is, avg_pv])

                        all_weeks_df = pd.DataFrame(all_weeks, columns=['Start', 'End', 'Te', 'RHe', 'Is', 'pve'])
                        all_weeks_df.index.name = 'Index'
                        all_weeks_file_path = os.path.join(directory, f"{filename.split('.')[0]}_all_weeks.csv")
                        all_weeks_df.to_csv(all_weeks_file_path)

                        min_Te, max_Te = float('inf'), float('-inf')
                        max_RHe, min_RHe = float('-inf'), float('inf')
                        max_Is, min_Is = float('-inf'), float('inf')
                        extreme_weeks = {}

                        avg_Te, avg_RHe, avg_pve, sum_Is = calculate_weekly_stats(week_df)
                        if avg_Te is not None:
                            if avg_Te < min_Te:
                                min_Te = avg_Te
                                extreme_weeks['EWW'] = week_df
                            if avg_Te > max_Te:
                                max_Te = avg_Te
                                extreme_weeks['ESW'] = week_df
                            if avg_RHe > max_RHe:
                                max_RHe = avg_RHe
                                extreme_weeks['EHW'] = week_df
                            if avg_RHe < min_RHe:
                                min_RHe = avg_RHe
                                extreme_weeks['EDW'] = week_df
                            if sum_Is > max_Is:
                                max_Is = sum_Is
                                extreme_weeks['ESunW_p'] = week_df
                            if sum_Is < min_Is:
                                min_Is = sum_Is
                                extreme_weeks['ESunW_n'] = week_df

                for key, week_df in extreme_weeks.items():
                    start_date = week_df.index[0].strftime(known_format)
                    end_date = (week_df.index[-1] + pd.Timedelta(hours=23)).strftime(known_format)
                    if key in ['EWW', 'ESW']:
                        avg_Te = week_df['Te'].mean()
                        print(f"Extreme {key}: Start Date: {start_date}, End Date: {end_date}, Avg Te: {avg_Te:.2f}")
                        extreme_results[key] = {
                            'start_date': start_date,
                            'end_date': end_date,
                        }
                    elif key in ['EHW', 'EDW']:
                        avg_RHe = week_df['RHe'].mean()
                        print(f"Extreme {key}: Start Date: {start_date}, End Date: {end_date}, Avg RH: {avg_RHe:.2f}")
                        extreme_results[key] = {
                            'start_date': start_date,
                            'end_date': end_date,
                        }
                    elif key in ['ESunW_p', 'ESunW_n']:
                        total_Is = week_df['Is'].sum(min_count=1)
                        print(f"Extreme {key}: Start Date: {start_date}, End Date: {end_date}, Total Is: {total_Is:.2f}")
                        extreme_results[key] = {
                            'start_date': start_date,
                            'end_date': end_date,
                        }
                return extreme_results
            return iterate_files(directory, filename)

        def save_extreme_results_to_csv(extreme_results, directory):
            '''function to save extreme results to a CSV file'''
            keys = extreme_results.keys()
            start_dates = [extreme_results[key]['start_date'] for key in keys]
            end_dates = [extreme_results[key]['end_date'] for key in keys]

            df = pd.DataFrame({
                'Key': keys,
                'Start Date': start_dates,
                'End Date': end_dates
            })
            df_transposed = df.set_index('Key').T

            output_file_path = os.path.join(directory, 'extreme_results.csv')
            df_transposed.to_csv(output_file_path, sep=',', header=True, index=False)

            print(f"Extreme results saved to {output_file_path}")

        def extreme_weeks_inputs(directory, directory_extremes, input_files):
            '''function to create input files based on extreme weeks from multiple input files'''
            extreme_results = pd.read_csv(directory_extremes)

            for input_file in input_files:
                input_file_path = os.path.join(directory, input_file)
                file_data = pd.read_csv(input_file_path)
                file_prefix = os.path.splitext(os.path.basename(input_file))[0].split('_')[0]
                file_data['Time'] = pd.to_datetime(file_data['Time'], format=known_format)

                for header in extreme_results.columns:
                    start_time = pd.to_datetime(extreme_results[header][0], format=known_format)
                    end_time = pd.to_datetime(extreme_results[header][1], format=known_format)
                    filtered_data = file_data[(file_data['Time'] >= start_time) & (file_data['Time'] <= end_time)]
                    filtered_data['Time'] = filtered_data['Time'].dt.strftime(known_format)
                    output_filename = os.path.join(directory, f'{header}_{file_prefix}_data.csv')
                    filtered_data.to_csv(output_filename, index=False)

            print("Data extraction and saving completed.")

        def combine_csv_files(input_directory, output_directory, starts_with):
            csv_files = [f for f in os.listdir(input_directory) if f.startswith(starts_with) and f.endswith('.csv')]
            data_frames = []        # initialize an empty dictionary

            for csv_file in csv_files:
                file_path = os.path.join(input_directory, csv_file)
                df = pd.read_csv(file_path)

                # ensure 'Time' column is the first column
                if 'Time' in df.columns:
                    columns = ['Time'] + [col for col in df.columns if col != 'Time']
                    df = df[columns]

                data_frames.append(df)      # append df to the list

            combined_df = data_frames[0]
            for df in data_frames[1:]:
                combined_df = pd.merge(combined_df, df, on='Time', how='outer')

            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]     # remove duplicate columns if any

            output_file_path = os.path.join(output_directory, f'{starts_with}.csv')
            combined_df.to_csv(output_file_path, index=False)

            print(f"Combined file saved to: {output_file_path}")        # confirmation

        if solar_radiation == 'y':
            prefixes = ['EWW', 'ESW', 'EHW', 'EDW', 'ESunW_p', 'ESunW_n']
        else:
            prefixes = ['EWW', 'ESW', 'EHW', 'EDW']
        filename_d = 'ECD_ready_d.csv'
        base_files = ['ECD_ready.csv', 'ICD_ready.csv', 'ED_ready.csv']

        known_d_format = '%d.%m.%Y'

        _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
        _11_initial_csv_check.create_folders(directory_others, folders=['to_use'])
        others_to_use = os.path.join(directory_others, 'to_use')

        output_filename = os.path.join(directory, '03_extremes_report.txt')
        _11_initial_csv_check.save_output_to_file(output_filename, get_weekly_extremes, directory, filename_d)

        extremes = get_weekly_extremes(directory, filename_d)
        save_extreme_results_to_csv(extremes, directory)

        extremes_path = os.path.join(directory, 'extreme_results.csv')
        extreme_weeks_inputs(directory, extremes_path, base_files)
        _11_initial_csv_check.move_files(directory, directory_others, file_name='extreme_results.csv')

        extreme_weeks_files = []  # list for supporting files
        for file_name in os.listdir(directory):  # iterate over files
            if file_name.endswith('.csv') and '_data' in file_name:  # arguments: .png and '_data'
                extreme_weeks_files.append(file_name)  # extend the list
        for extreme_weeks_name in extreme_weeks_files:  # loop for daily files
            _11_initial_csv_check.move_files(directory, directory_others,
                                                 file_name=extreme_weeks_name)  # moving files
        for prefix in prefixes:
            combine_csv_files(directory_others, others_to_use, prefix)


'''APPLICATION'''
if __name__ == "__main__":
    main()