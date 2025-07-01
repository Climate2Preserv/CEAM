import os
import csv
import pandas as pd
import glob
import shutil
import importlib
import sys
from datetime import datetime
from _00_CEAM_run import (directory, known_format, solar_radiation, demands_list,
                          date_format_icd, date_format_ecd, date_format_ed, outlier_value,
                          frequency_icd, frequency_ecd, frequency_ed, T_HDD, T_CDD)


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
directory_reports = os.path.join(directory, 'reports')      # path to 'reports' folder
directory_graphs = os.path.join(directory, 'graphs')        # path to 'graphs' folder
directory_raw_data = os.path.join(directory, 'raw_data')    # path to 'raw_data' folder
directory_others = os.path.join(directory, 'others')        # path to 'others' folder

directory_graphs_01 = os.path.join(directory_graphs, '01_input overview')
directory_graphs_02 = os.path.join(directory_graphs, '02_correlations')
directory_graphs_03 = os.path.join(directory_graphs, '03_energy signature')
directory_graphs_04 = os.path.join(directory_graphs, '04_climate class overview')
directory_graphs_05 = os.path.join(directory_graphs, '05_predictions')

output_file_names = []  # needed empty list
modified_files = []  # list to store the names of modified files

sys.stdout.reconfigure(encoding='utf-8')        # for greek symbols

_03_degree_days = importlib.import_module('_03_degree_days')


def save_output_to_file(output_filename, func, *args, **kwargs):
    '''function to save print outputs to a text file'''
    original_stdout = sys.stdout  # save a reference to the original standard output

    try:
        with open(output_filename, 'a', encoding="utf-8") as file:
            sys.stdout = file  # change the standard output to the file
            func_output = func(*args, **kwargs)  # get the output from the function
            sys.stdout = original_stdout  # reset the standard output to its original value

            if func_output is not None:
                file.write(str(func_output))
                file.write('\n')

    except Exception as e:
        sys.stdout = original_stdout  # make sure to reset stdout even if there is an error
        with open(output_filename, 'a', encoding="utf-8") as file:
            file.write(f"Error occurred while saving output: {e}\n")

def create_folders(dir_path, folders=[]):
    '''create a set of provided folders in a given path if they do not exist
    if the folders exist they are not changed, with the content inside'''
    for folder in folders:
        folder_path = os.path.join(dir_path, folder)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder created at {folder_path}")
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e}")

def move_files(src_dir, dest_dir, file_extension=None, file_name=None, exception_files=None, filename_include=None,
               move_folder=None):
    '''move specified file(s) from the source to the destination directory'''
    if file_name:
        if filename_include and filename_include not in file_name:
            print(f"{file_name} does not include the string '{filename_include}', skipping.")
            return
        try:        # only the specified file
            src_file_path = os.path.join(src_dir, file_name)
            dest_file_path = os.path.join(dest_dir, file_name)
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved {file_name} to {dest_dir}")   # confirmation
        except OSError as e:
            print(f"Error moving {file_name}: {e}")
    elif move_folder:
        for folder_name in move_folder:                 # loop through each folder in move_folder
            src_folder_path = os.path.join(src_dir, folder_name)
            dest_folder_path = os.path.join(dest_dir, folder_name)

            if os.path.isdir(src_folder_path):          # check if folder exists
                try:
                    shutil.move(src_folder_path, dest_folder_path)
                    print(f"Moved the folder {folder_name} to {dest_folder_path}")  # confirmation
                except OSError as e:
                    print(f"Error moving the folder {folder_name}: {e}")
            else:
                print(f"Folder {folder_name} does not exist in {src_dir}. Skipping.")
    else:           # move all files with the specified extension
        pattern = os.path.join(src_dir, '*' if file_extension is None else f'*{file_extension}')
        files = glob.glob(pattern)
        for file_path in files:
            base_name = os.path.basename(file_path)
            if exception_files and base_name in exception_files:
                continue
            if filename_include and filename_include not in base_name:
                continue
            try:
                dest_file_path = os.path.join(dest_dir, base_name)
                shutil.move(file_path, dest_file_path)
                print(f"Moved {base_name} to {dest_dir}")  # confirmation
            except OSError as e:
                print(f"Error moving {base_name}: {e}")

def rename_files(directory, substrings=None, suffix='_ready', input_names=None, output_names=None):
    '''function to rename file(s)'''
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if substrings is None or any(substring in filename for substring in substrings):
                if input_names and output_names:
                    if filename in input_names:
                        idx = input_names.index(filename)
                        new_filename = output_names[idx]
                    else:
                        continue  # skip renaming if filename not in input_names
                else:
                    new_filename = filename.split('_')[0] + suffix + '.csv'

                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))  # rename with path
                print(f"Renamed {filename} to {new_filename}")  # confirmation

def combine_txt_files(directory, output_filename, file_type='.txt', file_exception=None, prefix=None):
    '''combines several txt files with an optional prefix'''
    # ensure file_exception is a list or None
    if file_exception is None:
        file_exception = []

    combined_filepath = os.path.join(directory, output_filename)  # output file path
    try:
        with open(combined_filepath, 'w') as outfile:
            for filename in os.listdir(directory):
                # filter files by type, exception list, and prefix
                if filename.endswith(file_type) and filename not in file_exception:
                    if prefix is None or filename.startswith(prefix):  # check for prefix if given
                        filepath = os.path.join(directory, filename)
                        with open(filepath, 'r') as infile:
                            outfile.write(infile.read())
        print(f'Combined files into: {combined_filepath}')
        return combined_filepath

    except FileNotFoundError:
        print(f'Error: Directory "{directory}" not found.')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None

def delete_files(directory, file_type='*', exception_files=None):
    '''deleting files'''
    deleted_files = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(file_type):
                if exception_files and filename in exception_files:
                    continue  # skip if exception_files
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                deleted_files.append(filename)
                print(f'Deleted: {filename}')
        return deleted_files

    except FileNotFoundError:
        print(f'Error: Directory "{directory}" not found.')
        return []
    except Exception as e:
        print(f'Error: {e}')
        return []

def copy_files(src_dir, dest_dir, file_extension=None, file_name=None, exception_files=None, filename_including=None):
    '''copy specified file(s) from the source to the destination directory'''
    if file_name:
        try:  # only the specified files
            src_file_path = os.path.join(src_dir, file_name)
            dest_file_path = os.path.join(dest_dir, file_name)
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {file_name} to {dest_dir}")  # confirmation
        except OSError as e:
            print(f"Error copying {file_name}: {e}")  # confirmation

    elif filename_including:  # files containing part of the name
        pattern = os.path.join(src_dir, f'*{filename_including}*')
        files = glob.glob(pattern)
        for file_path in files:
            if exception_files and os.path.basename(file_path) in exception_files:
                continue
            try:
                dest_file_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.copy(file_path, dest_file_path)
                print(f"Copied {os.path.basename(file_path)} to {dest_dir}")  # confirmation
            except OSError as e:
                print(f"Error copying {os.path.basename(file_path)}: {e}")  # confirmation

    else:  # all files with the specified extension
        pattern = os.path.join(src_dir, f'*.{file_extension}')
        files = glob.glob(pattern)
        for file_path in files:
            if exception_files and os.path.basename(file_path) in exception_files:
                continue
            try:
                dest_file_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.copy(file_path, dest_file_path)
                print(f"Copied {os.path.basename(file_path)} to {dest_dir}")  # confirmation
            except OSError as e:
                print(f"Error copying {os.path.basename(file_path)}: {e}")  # confirmation


'''FUNCTIONALITIES'''
def main():
    def check_and_change_separator(directory):
        '''function checking the used seperator in csv file(s) and changing it to comma'''
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    print(f"File: {filename}, Separator: {repr(dialect.delimiter)}")
                    if dialect.delimiter != ',':
                        df = pd.read_csv(filepath, sep=dialect.delimiter)
                        df.to_csv(filepath, index=False)
                        print(f"Separator in {filename} changed to ','")

    def check_and_fix_date_format_in_directory(directory, file_name, date_format, frequency):
        '''check and fix date format in the 'Time' column of csv input files; only for hourly frequencies'''
        # only process if frequency is 'h'
        if frequency != 'h':
            print(f"Skipping file {file_name} due to frequency: {frequency}")
            return

        # generate known formats
        examine_formats = []
        time_formats = ["%H:%M:%S", "%H:%M", "%H"]
        space_variations = [" ", "  ", "   "]

        # for date in date_formats:
        for space in space_variations:
            for time in time_formats:
                examine_formats.append(f"{date_format}{space}{time}")

        examine_formats.extend(date_format)
        print(f"Known formats: {examine_formats}")
        csv_path = os.path.join(directory, file_name)       # full path

        if not os.path.exists(csv_path):
            print(f"File {csv_path} not found.")
            return
        df = pd.read_csv(csv_path)

        if 'Time' not in df.columns:
            print(f"Error: 'Time' column not found in {file_name}. Skipping.")
            return

        detected_formats = set()

        def parse_date(value):
            nonlocal detected_formats
            for fmt in examine_formats:
                try:
                    parsed = datetime.strptime(value, fmt)
                    detected_formats.add(fmt)
                    return parsed.strftime(known_format)    # convert to standard format
                except ValueError:
                    continue
            print(f"Unrecognized date format in {file_name}: {value}")
            return None                                     # return None for unrecognized formats

        df['Time'] = df['Time'].apply(parse_date)           # conversion

        new_filename = file_name.split('.')[0] + '_modified.csv'  # saving
        new_filepath = os.path.join(directory, new_filename)
        df.to_csv(new_filepath, index=False)

        print(f"Detected formats in 'Time' column for {file_name}: {detected_formats}")
        print(f"Modified CSV saved at: {new_filepath}")
        modified_files.append(new_filename)  # add the new filename to the list

    def ed_file_adjustment_based_on_climate(directory, ecd_file, ed_file, output_file):
        '''function to adjust m/d ED input to the h-frequency based on climate data, in particular HDD/CDD'''
        if frequency_ed == 'h':
            pass
        elif frequency_ed in ['d', 'm']:
            if frequency_ecd == 'h':
                if os.path.exists(ecd_file) and os.path.exists(ed_file):    # the presence of ECD.csv and ED.csv
                    print('yes')

                    df_ecd = pd.read_csv(ecd_file)
                    # filter columns with any case for 'Time' and 'Te'
                    time_column = [col for col in df_ecd.columns if col.lower() == 'time']
                    te_column = [col for col in df_ecd.columns if col.lower() == 'te']

                    if time_column and te_column:
                        filtered_data = df_ecd[[time_column[0], te_column[0]]]  # use the first match
                        # write the filtered data to a new CSV file
                        filtered_data.to_csv(output_file, index=False)
                        print(f"New file generated: {output_file}")
                        # perform HDD/CDD calculations
                        _03_degree_days.process_temperature_data(directory, os.path.basename(output_file),
                                                                 T_HDD, T_CDD, known_format)
                    else:
                        print("Required columns 'Time' or 'Te' not found in ECD.csv.")
                else:
                    print("Either 'ECD.csv' or 'ED.csv' not found in the directory.")

        demand_to_metric = {
            'D1': 'HDD',  # D1 is linked with HDD
            'D2': 'CDD'  # D2 is linked with CDD
        }

        # load the filtered hourly data
        filtered_data = pd.read_csv(output_file, parse_dates=['Time'], date_format=known_format)
        ed_data = pd.read_csv(ed_file, parse_dates=['Time'], date_format=date_format_ed)

        # iterate through each demand in the demands list
        for demand in demands_list:
            if demand in demand_to_metric and demand in ed_data.columns:
                adjustment_metric = demand_to_metric[demand]
                filtered_data['Period'] = filtered_data['Time'].dt.to_period('M')  # Monthly adjustment by default

                if frequency_ed == 'm':
                    # monthly adjustment
                    period_totals = filtered_data.groupby('Period')[adjustment_metric].sum().rename(
                        f'Total_{adjustment_metric}')
                    # merge the totals back to the filtered data
                    filtered_data = filtered_data.merge(period_totals, on='Period', how='left')

                    # match the demand value for each month from ED.csv
                    ed_data['Period'] = ed_data['Time'].dt.to_period('M')  # monthly adjustment
                    filtered_data = filtered_data.merge(ed_data[['Period', demand]], on='Period', how='left')

                    # adjust the hourly demand values based on HDD/CDD proportion for the monthly frequency
                    filtered_data[f'Adjusted_{demand}'] = 0
                    valid_rows = (
                            (filtered_data[demand].notnull()) &
                            (filtered_data[adjustment_metric] > 0) &
                            (filtered_data[f'Total_{adjustment_metric}'] > 0)
                    )
                    filtered_data.loc[valid_rows, f'Adjusted_{demand}'] = (
                            filtered_data.loc[valid_rows, demand] *
                            (filtered_data.loc[valid_rows, adjustment_metric] /
                             filtered_data.loc[valid_rows, f'Total_{adjustment_metric}'])
                    )
                elif frequency_ed == 'd':
                    # daily adjustment
                    filtered_data['Day'] = filtered_data['Time'].dt.date
                    period_totals = filtered_data.groupby('Day')[adjustment_metric].sum().rename(
                        f'Total_{adjustment_metric}')
                    # merge the totals back to the filtered data
                    filtered_data = filtered_data.merge(period_totals, on='Day', how='left')

                    # match the demand value for each day from ED.csv
                    ed_data['Day'] = ed_data['Time'].dt.date  # Extract the day from ED data
                    filtered_data = filtered_data.merge(ed_data[['Day', demand]], on='Day', how='left')

                    # adjust the hourly demand values based on HDD/CDD proportion for the daily frequency
                    filtered_data[f'Adjusted_{demand}'] = 0
                    valid_rows = (
                            (filtered_data[demand].notnull()) &
                            (filtered_data[adjustment_metric] > 0) &
                            (filtered_data[f'Total_{adjustment_metric}'] > 0)
                    )
                    filtered_data.loc[valid_rows, f'Adjusted_{demand}'] = (
                            filtered_data.loc[valid_rows, demand] *
                            (filtered_data.loc[valid_rows, adjustment_metric] /
                             filtered_data.loc[valid_rows, f'Total_{adjustment_metric}'])
                    )

        # adjusting the filtered file
        original_copy_path = os.path.join(directory, 'filtered_data_og.csv')
        shutil.copyfile(output_file, original_copy_path)
        print(f"Original file copied to {original_copy_path}")

        columns_to_drop = [col for col in filtered_data.columns if col.startswith('Total_') or col == 'Period']
        filtered_data.drop(columns=columns_to_drop, inplace=True)

        columns_to_keep = ['Time'] + [f'Adjusted_{d}' for d in demands_list if f'Adjusted_{d}' in filtered_data.columns]
        filtered_data = filtered_data[columns_to_keep]

        for demand in demands_list:
            if f'Adjusted_{demand}' in filtered_data.columns:
                filtered_data.rename(columns={f'Adjusted_{demand}': demand}, inplace=True)

        for demand in ['D1', 'D2', 'D3', 'D4', 'D5']:
            if demand not in filtered_data.columns:
                filtered_data[demand] = None  # Add empty column

        if 'Time' in filtered_data.columns:
            filtered_data['Time'] = pd.to_datetime(filtered_data['Time']).dt.strftime(known_format)

        filtered_data.to_csv(output_file, index=False)
        print(f"Adjusted data saved to {output_file}")

    def process_csv_files(directory):
        '''function to rename and reorder headers in input CSV files to match the required headers'''
        if solar_radiation == 'y':
            required_headers = {
                'ECD.csv': ['Time', 'Te', 'RHe', 'Is'],
                'ICD.csv': ['Time', 'Ti', 'RHi'],
                'ED.csv': ['Time', 'D1', 'D2', 'D3', 'D4', 'D5']
            }
        else:
            required_headers = {
                'ECD.csv': ['Time', 'Te', 'RHe'],
                'ICD.csv': ['Time', 'Ti', 'RHi'],
                'ED.csv': ['Time', 'D1', 'D2', 'D3', 'D4', 'D5']
            }

        def normalize_header(header):
            return header.strip().lower()

        for filename in os.listdir(directory):
            if filename in required_headers:
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8-sig') as file:
                        csv_reader = csv.reader(file)
                        headers = next(csv_reader)
                        rows = list(csv_reader)
                        normalized_headers = [normalize_header(header) for header in headers]

                        required = required_headers[filename]
                        header_mapping = {}
                        for orig_header in headers:
                            normalized_orig = normalize_header(orig_header)
                            for req_header in required:
                                if normalized_orig == normalize_header(req_header):
                                    header_mapping[orig_header] = req_header
                                    break

                        for req_header in required:
                            if req_header not in header_mapping.values():
                                header_mapping[req_header] = req_header

                        print(f"\nProcessing file: {filename}")
                        print(f"Original headers: {headers}")
                        print(f"Required headers: {required}")
                        print(f"Header mapping: {header_mapping}")

                        renamed_headers = [header_mapping.get(header, header) for header in headers]
                        print(f"Renamed headers: {renamed_headers}")

                        reordered_headers = required
                        header_index_map = {header: renamed_headers.index(header) for header in renamed_headers if
                                            header in reordered_headers}

                        print(f"Reordered headers: {reordered_headers}")
                        print(f"Header index map: {header_index_map}")

                        reordered_rows = [
                            [
                                row[header_index_map[header]] if header in header_index_map and header_index_map[
                                    header] < len(row) else ''
                                for header in reordered_headers
                            ]
                            for row in rows
                        ]

                        updated_content = [reordered_headers] + reordered_rows

                    with open(filepath, 'w', newline='', encoding='utf-8') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerows(updated_content)

                    print(f"File: {filename}, Headers renamed and reordered.")

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

    def get_most_common_frequency(file_path, filename, known_format):
        '''function checking the file for its frequency'''
        full_path = f'{file_path}/{filename}'
        df = pd.read_csv(full_path)
        df['Time'] = pd.to_datetime(df['Time'], format=known_format)
        df.set_index('Time', inplace=True)
        sample_minutes = (df.index.to_series().diff().dt.total_seconds() // 60).mode()[0]  # chceck the common frequency
        print(f'timestamp for the {filename} is {sample_minutes} min')
        return int(sample_minutes)

    def fill_missing_timestamps(file_path, filename, known_format, sample_minutes):
        '''function checking the input file(s) for missing timestamps
        and replacing it with a complete timestamp with empty rows'''
        full_path = f'{file_path}/{filename}'
        df = pd.read_csv(full_path)
        df['Time'] = pd.to_datetime(df['Time'], format=known_format)
        df.set_index('Time', inplace=True)
        all_times = pd.date_range(start=df.index.min(), end=df.index.max(),
                                  freq=f'{sample_minutes}T')  # create a complete df
        df_all = pd.DataFrame(index=all_times)
        missing_timestamps = df_all.index.difference(df.index)  # identify the missing timestamps
        total_missing_records = len(missing_timestamps)
        print(f"Total number of missing records in {filename}: {total_missing_records}")
        print(df.index.min())
        print(df.index.max())

        if total_missing_records > 0:  # if missing timestamps, fill them with NaN values
            df = df_all.join(df).reset_index()
            df.rename(columns={'index': 'Time'}, inplace=True)
            df['Time'] = df['Time'].dt.strftime(known_format)
            output_file_path = f'{file_path}/{filename.split(".")[0]}_filled.csv'
            df.to_csv(output_file_path, index=False)
            missing_timestamps = missing_timestamps.to_series().reset_index(drop=True)  # range of missing timestamps
            start_time = missing_timestamps[0]
            end_time = start_time
            count = 1

            for i in range(1, len(missing_timestamps)):
                if missing_timestamps[i] - end_time > pd.Timedelta(minutes=sample_minutes):
                    print(
                        f"{count} record(s) missing starts at {start_time.strftime(known_format)} and ends at {end_time.strftime(known_format)}")
                    start_time = missing_timestamps[i]
                    count = 1
                else:
                    count += 1
                end_time = missing_timestamps[i]
            print(
                f"{count} record(s) missing starts at {start_time.strftime(known_format)} and ends at {end_time.strftime(known_format)}")
        else:
            print(f"No missing timestamps in {filename}")  # confirmation

    def examine_files_outliers(direction, examined_files):
        '''function to distinguish the outliers from the given input files'''
        print('\nOUTLIERS ASSESSMENT')
        for file_name in examined_files:                        # iterate over the files in examined_files
            file_path = os.path.join(direction, file_name)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            print(f"\nProcessing file: {file_name}")

            try:
                data = pd.read_csv(file_path)                   # read the file into a df
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue

            columns_to_examine = data.columns
            if demands_list and file_name == 'ED_modified.csv':         # adjust the columns to demands_list
                columns_to_examine = [col for col in demands_list if col in data.columns]
                missing_columns = [col for col in demands_list if col not in data.columns]
                if missing_columns:
                    print(f"Warning: Missing columns in {file_name}: {missing_columns}")

            if outlier_value == None:      # skip processing if outliers are not examined
                print(f"Skipping {file_name} analysis as outliers are not examined.")
                continue

            for column in columns_to_examine:
                if column == 'Time':
                    continue

                values = data[column].dropna()      # drop NaN values
                total_records = len(values)
                if total_records == 0:
                    print(f"\nNo valid data in column {column} of {file_name}")
                    continue

                # statistics calculations
                mean = round(values.mean(), 2)
                std = round(values.std(), 2)
                median = round(values.median(), 2)
                within_1_std = values[(values >= mean - std) & (values <= mean + std)]
                within_2_std = values[(values >= mean - 2 * std) & (values <= mean + 2 * std)]
                within_3_std = values[(values >= mean - 3 * std) & (values <= mean + 3 * std)]
                lower_bound = round(mean - outlier_value * std, 2)
                upper_bound = round(mean + outlier_value * std, 2)

                data[column] = data[column].where((data[column] >= lower_bound) & (data[column] <= upper_bound))    # filter data

                # output print statements
                if outlier_value != None:
                    print(f"  Column {column}:")
                    print(f"    Total records: {total_records}")
                    print(f"    Median (\u03b7): {median:.2f}")
                    print(f"    Mean (\u03bc): {mean:.2f}, Std Dev (\u03c3): {std:.2f}")
                    print(f"    Upper bound: {upper_bound:.2f}; Lower bound: {lower_bound:.2f}")
                    print(f"    Records within -\u03c3:+\u03c3: {len(within_1_std)} ({len(within_1_std) / total_records * 100:.2f}%)")
                    print(f"    Records within -2\u03c3:+2\u03c3: {len(within_2_std)} ({len(within_2_std) / total_records * 100:.2f}%)")
                    print(f"    Records within -3\u03c3:+3\u03c3: {len(within_3_std)} ({len(within_3_std) / total_records * 100:.2f}%)")

            output_file_path = os.path.join(direction, file_name)
            data = data.round(2)
            data.to_csv(output_file_path, index=False)

    def average_data(file_path, filename, known_format, sample_ave_minutes=60):
        '''function averaging the df based on the given averaging sample'''
        # global output_file_names
        full_path = f'{file_path}/{filename}'
        df = pd.read_csv(full_path)  # load the file
        df['Time'] = pd.to_datetime(df['Time'], format=known_format)  # convert the 'Time' column to df
        df.set_index('Time', inplace=True)  # set 'Time' as the index
        df_resampled = df.resample(f'{sample_ave_minutes}T').mean()  # resampling due to interval
        df_resampled.reset_index(inplace=True)  # convert the datetime index and format it
        df_resampled['Time'] = df_resampled['Time'].dt.strftime(known_format)

        output_file_name = f'{filename.split(".")[0]}_{sample_ave_minutes}_average.csv'
        output_file_path = os.path.join(file_path, output_file_name)
        df_resampled.to_csv(output_file_path, index=False)
        output_file_names.append(output_file_name)

    def check_missing_values(file_path, filename):
        '''function checking the input file(s) for missing values in all columns'''
        full_path = f'{file_path}/{filename}'
        df = pd.read_csv(full_path)

        # set columns to exclude
        all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
        exclude_columns = all_demands.difference(demands_list)

        df['missing'] = df.drop(columns='Time', errors='ignore').isnull().any(axis=1)
        df['block'] = (df['missing'].shift(1) != df['missing']).astype(int).cumsum()

        output = []
        output.append(f"\nExamined file: {filename}")

        def process_columns(condition):
            '''function to process missing values by condition'''
            for column in df.columns:
                if column != 'Time' and column not in exclude_columns and condition(column):
                    missing_count = df[column].isnull().sum()
                    output.append(f"Column: {column}, total missing Values: {missing_count}")
                    if missing_count > 0:
                        for name, group in df[df[column].isnull()].groupby('block'):
                            output.append(
                                f"{len(group)} missing values start at {group['Time'].min()} and end at {group['Time'].max()}"
                            )

        if solar_radiation == 'y':
            process_columns(lambda col: True)                           # include all columns except excluded ones
        else:
            process_columns(lambda col: not col.startswith('Is'))       # exclude columns starting with 'Is'

        filtered_output = [line for line in output if
                           not line.startswith('Column: missing,') and not line.startswith('Column: block,')]

        return '\n'.join(filtered_output)

    substrings = ['ECD', 'ICD', 'ED']  # recommended prefixes; will be used further
    report_filename = '01_missings_report.txt'

    output_summary_1 = os.path.join(directory, '01_missings_report1.txt')
    output_summary_2 = os.path.join(directory, '01_missings_report2.txt')
    output_summary_3 = os.path.join(directory, '01_missings_report3.txt')

    create_folders(directory, folders=['reports', 'graphs', 'others', 'raw_data'])  # make necessary folders in dir
    create_folders(directory_graphs, folders=['01_input overview', '02_correlations',
                                              '03_energy signature', '04_climate class overview', '05_predictions'])

    copy_files(directory, directory_raw_data, file_extension='csv')  # copy raw files
    check_and_change_separator(directory)  # verifying the separator
    save_output_to_file(output_summary_1, process_csv_files, directory)  # checking the headers

    check_and_fix_date_format_in_directory(directory, 'ICD.csv', date_format_icd, frequency_icd)
    check_and_fix_date_format_in_directory(directory, 'ECD.csv', date_format_ecd, frequency_ecd)
    check_and_fix_date_format_in_directory(directory, 'ED.csv', date_format_ed, frequency_ed)
    save_output_to_file(output_summary_3, examine_files_outliers, directory, modified_files)

    ecd_modified_file = os.path.join(directory, 'ECD_modified.csv')
    if frequency_ed != 'h':
        ed_modified_file = os.path.join(directory, 'ED.csv')
        output_temp_file = os.path.join(directory, 'filtered_data.csv')
        ed_file_adjustment_based_on_climate(directory, ecd_modified_file, ed_modified_file, output_temp_file)
        rename_files(directory, input_names=['filtered_data.csv'], output_names=['ED_modified.csv'])  # renaming
        modified_files.append('ED_modified.csv')  # add the new filename to the list

    # examine_files(directory, modified_files)
    for filename in modified_files:  # work on all csv files in directory
        save_output_to_file(output_summary_2, check_missing_values, directory, filename)  # process the check_missing_values and save the report
        sample_minutes = get_most_common_frequency(directory, filename, known_format)  # get the frequency of the df
        fill_missing_timestamps(directory, filename, known_format, sample_minutes)  # filling the missing timestamps
        average_data(directory, filename, known_format, 60)  # averaging with given sample
    move_files(directory, directory_others, '*.csv', None,
               exception_files=output_file_names)  # move all the temporary files to others folder
    copy_files(directory, directory_others, file_extension='*.csv')  # copy final files
    rename_files(directory, substrings)  # renaming

    _01_convert_units = importlib.import_module('_01_convert_units')
    _01_convert_units.main_conversion_logic()           # units conversion to SI

    combine_txt_files(directory, report_filename, file_type='.txt', file_exception=['var.txt', 'terminal_records.txt'])
    delete_files(directory, file_type='.txt', exception_files=[report_filename, 'var.txt', 'terminal_records.txt'])


'''APPLICATION'''
if __name__ == "__main__":
    main()
