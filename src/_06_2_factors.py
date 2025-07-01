import os
import pandas as pd
import importlib
from _00_CEAM_run import directory, T_HDD, T_CDD
from _11_initial_csv_check import directory_reports


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
def process_factors_base(directory, filename=None, exceptions=[], parameters=[]):
    '''function to calculate the factors for the climate assessment from base file'''

    # symbolic parameter names
    parm_1 = 'Ti'
    parm_2 = 'Te'
    parm_3 = 'RHi'
    parm_4 = 'RHe'
    parm_5 = 'Tdi'
    parm_6 = 'Tde'
    parm_7 = 'Twbi'
    parm_8 = 'Twbe'
    parm_9 = 'Is'

    if filename:  # file(s) selection
        filenames = [filename] if filename.endswith('.csv') else []
    else:
        filenames = [f for f in os.listdir(directory) if f.endswith('.csv') and f not in exceptions]

    for fname in filenames:  # loop through all files in the directory
        file_path = os.path.join(directory, fname)
        try:
            df = pd.read_csv(file_path)  # read the csv into a df

            # verified allowed calculations
            if parm_1 in parameters and parm_2 in parameters:
                if parm_1 in df.columns and parm_2 in df.columns:
                    df['deltaT'] = df[parm_1] - df[parm_2]

            if parm_3 in parameters and parm_4 in parameters:
                if parm_3 in df.columns and parm_4 in df.columns:
                    df['deltaRH'] = df[parm_3] - df[parm_4]

            if parm_5 in parameters and parm_6 in parameters:
                if parm_5 in df.columns and parm_6 in df.columns:
                    df['deltaTd'] = df[parm_5] - df[parm_6]

            if parm_7 in parameters and parm_8 in parameters:
                if parm_7 in df.columns and parm_8 in df.columns:
                    df['deltaTwb'] = df[parm_7] - df[parm_8]

            if parm_9 in parameters:
                if parm_9 in df.columns:
                    max_Is = df[parm_9].max()

                    if parm_1 in parameters and parm_2 in parameters and 'deltaT' in df.columns:
                        df['TIST'] = df.apply(lambda row: (
                            row['deltaT'] - row['deltaT'] * (row[parm_9] / max_Is) if row[parm_2] < T_HDD
                            else row['deltaT'] + row['deltaT'] * (row[parm_9] / max_Is) if row[parm_2] > T_CDD
                            else row['deltaT']) if pd.notnull(row[parm_9]) else None, axis=1)

                    if parm_5 in parameters and parm_6 in parameters and 'deltaTd' in df.columns:
                        df['TISTd'] = df.apply(lambda row: (
                            row['deltaTd'] - row['deltaTd'] * (row[parm_9] / max_Is) if row[parm_6] < Td_HDD
                            else row['deltaTd'] + row['deltaTd'] * (row[parm_9] / max_Is) if row[parm_6] > Td_CDD
                            else row['deltaTd']) if pd.notnull(row[parm_9]) else None, axis=1)

                    if parm_7 in parameters and parm_8 in parameters and 'deltaTwb' in df.columns:
                        df['TISTwb'] = df.apply(lambda row: (
                            row['deltaTwb'] - row['deltaTwb'] * (row[parm_9] / max_Is) if row[parm_8] < Twb_HDD
                            else row['deltaTwb'] + row['deltaTwb'] * (row[parm_9] / max_Is) if row[parm_8] > Twb_CDD
                            else row['deltaTwb']) if pd.notnull(row[parm_9]) else None, axis=1)

            df.to_csv(file_path, index=False)  # save the df to csv

        except Exception as e:
            print(f"Error processing file {fname}: {e}")

def process_factors_param(directory, filename=None, exceptions=[], parameters=[]):
    '''function to calculate the factors for the climate assessment for predictions
    used in other scripts'''

    # assign parameters based on the provided list
    parm_1 = parameters[0] if len(parameters) > 0 else None
    parm_2 = parameters[1] if len(parameters) > 1 else None
    parm_3 = parameters[2] if len(parameters) > 2 else None
    parm_4 = parameters[3] if len(parameters) > 3 else None
    parm_5 = parameters[4] if len(parameters) > 4 else None
    parm_6 = parameters[5] if len(parameters) > 5 else None
    parm_7 = parameters[6] if len(parameters) > 6 else None
    parm_8 = parameters[7] if len(parameters) > 7 else None
    parm_9 = parameters[8] if len(parameters) > 8 else None

    if filename:  # file(s) selection
        filenames = [filename] if filename.endswith('.csv') else []
    else:
        filenames = [f for f in os.listdir(directory) if f.endswith('.csv') and f not in exceptions]

    for fname in filenames:  # loop through all files in the directory
        file_path = os.path.join(directory, fname)
        try:
            df = pd.read_csv(file_path)  # read the csv into a df

            # verified allowed calculations
            if parm_1 and parm_2:
                if parm_1 in df.columns and parm_2 in df.columns:
                    df[f'deltaT_{parm_1}'] = df[parm_1] - df[parm_2]

            if parm_3 and parm_4:
                if parm_3 in df.columns and parm_4 in df.columns:
                    if parm_4 == 'RHe':
                        df[f'deltaRH_{parm_3}'] = df[parm_3] - df[parm_4]
                    else:
                        df[f'deltaAbsH_{parm_3}'] = df[parm_3] - df[parm_4]

            if parm_5 and parm_6:
                if parm_5 in df.columns and parm_6 in df.columns:
                    df[f'deltaTd_{parm_5}'] = df[parm_5] - df[parm_6]

            if parm_7 and parm_8:
                if parm_7 in df.columns and parm_8 in df.columns:
                    df[f'deltaTwb_{parm_7}'] = df[parm_7] - df[parm_8]

            if parm_9:
                if parm_9 in df.columns:
                    max_Is = df[parm_9].max()

                    if parm_1 and parm_2 and f'deltaT_{parm_1}' in df.columns:
                        df[f'TIST_{parm_1}'] = df.apply(lambda row: (
                            row[f'deltaT_{parm_1}'] - row[f'deltaT_{parm_1}'] * (row[parm_9] / max_Is)
                            if row[parm_2] < T_HDD
                            else row[f'deltaT_{parm_1}'] + row[f'deltaT_{parm_1}'] * (row[parm_9] / max_Is) if
                            row[parm_2] > T_CDD
                            else row[f'deltaT_{parm_1}']) if pd.notnull(row[parm_9]) else None, axis=1)

                    if parm_5 and parm_6 and f'deltaTd_{parm_5}' in df.columns:
                        df[f'TISTd_{parm_5}'] = df.apply(lambda row: (
                            row[f'deltaTd_{parm_5}'] - row[f'deltaTd_{parm_5}'] * (row[parm_9] / max_Is)
                            if row[parm_6] < Td_HDD
                            else row[f'deltaTd_{parm_5}'] + row[f'deltaTd_{parm_5}'] * (row[parm_9] / max_Is)
                            if row[parm_6] > Td_CDD
                            else row[f'deltaTd_{parm_5}']) if pd.notnull(row[parm_9]) else None, axis=1)

                    if parm_7 and parm_8 and f'deltaTwb_{parm_7}' in df.columns:
                        df[f'TISTwb_{parm_7}'] = df.apply(lambda row: (
                            row[f'deltaTwb_{parm_7}'] - row[f'deltaTwb_{parm_7}'] * (row[parm_9] / max_Is)
                            if row[parm_8] < Twb_HDD
                            else row[f'deltaTwb_{parm_7}'] + row[f'deltaTwb_{parm_7}'] * (row[parm_9] / max_Is)
                            if row[parm_8] > Twb_CDD
                            else row[f'deltaTwb_{parm_7}']) if pd.notnull(row[parm_9]) else None, axis=1)

            df.to_csv(file_path, index=False)  # save the df to csv

        except Exception as e:
            print(f"Error processing file {fname}: {e}")

def calculate_average_RH(file_path):
    '''calculate RH_ave for HDD/CDD periods'''
    try:
        df = pd.read_csv(file_path)  # read the csv into a df

        avg_RHe_HDD = df[df['HDD'] > 0]['RHe'].dropna().mean() if 'HDD' in df.columns else None
        avg_RHe_CDD = df[df['CDD'] > 0]['RHe'].dropna().mean() if 'CDD' in df.columns else None

        return avg_RHe_HDD, avg_RHe_CDD

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None


file_name = 'h_comb.csv'
path = os.path.join(directory, file_name)
_06_1_Td_Twb_calc = importlib.import_module('_06_1_Td_Twb_calc')

avg_RHe_HDD, avg_RHe_CDD = calculate_average_RH(path)  # calculate necessary RH averages
if avg_RHe_HDD is not None:  # confirmation
    print(f"Average RHe for records with HDD > 0: {avg_RHe_HDD}")
if avg_RHe_CDD is not None:  # confirmation
    print(f"Average RHe for records with CDD > 0: {avg_RHe_CDD}")

Td_HDD = _06_1_Td_Twb_calc.calculate_single_td(T_HDD, avg_RHe_HDD)
Td_CDD = _06_1_Td_Twb_calc.calculate_single_td(T_CDD, avg_RHe_CDD)
print(Td_HDD, Td_CDD)  # confirmation

Twb_HDD = _06_1_Td_Twb_calc.calculate_single_twb(T_HDD, avg_RHe_HDD)
Twb_CDD = _06_1_Td_Twb_calc.calculate_single_twb(T_CDD, avg_RHe_CDD)
print(Twb_HDD, Twb_CDD)  # confirmation


'''FUNCTIONALITIES'''
def main():
    process_factors_base(directory, exceptions=[],
                         parameters=['Ti', 'Te', 'RHi', 'RHe', 'Tdi', 'Tde', 'Twbi', 'Twbe', 'Is'])

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
    _11_initial_csv_check.move_files(directory, directory_reports, file_extension='*.txt',
                                     exception_files=['var.txt', 'terminal_records.txt'])


'''APPLICATION'''
if __name__ == "__main__":
    main()
