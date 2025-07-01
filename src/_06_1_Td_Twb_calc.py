import pandas as pd
import numpy as np
import os
from _00_CEAM_run import directory


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
E0 = 0.611      # unit [kPa]
L_Rv = 5423     # unit [K]
T0 = 273.15     # unit [K]

def calculate_es(T):
    '''function to calculate saturation vapor pressure'''
    T_kelvin = T + T0
    Es = E0 * np.exp((L_Rv) * (1 / T0 - 1 / T_kelvin))
    return Es

def calculate_single_td(T, RH):
    '''function to calculate dew point'''
    # T_kelvin = T + T0
    Es = calculate_es(T)
    E = Es * (RH / 100)
    Td = 1 / ((1 / T0) - (np.log(E / E0) / L_Rv))
    Td_celsius = Td - T0
    return Td_celsius

def calculate_single_twb(T, RH):
    '''function to calculate wet bulb temperature'''
    Twb = (
        T * np.arctan(0.151977 * ((RH + 8.313659)**0.5)) +
        np.arctan(T + RH) -
        np.arctan(RH - 1.676331) +
        0.00391838 * (RH**1.5) * np.arctan(0.023101 * RH) -
        4.686035
    )
    return Twb

def calculate_td(input_file, params):
    '''function to calculate dew points'''
    data = pd.read_csv(input_file)                              # read the input
    print(f"Processing file: {input_file}")                     # confirmation
    print("Headers from the input file:", list(data.columns))   # confirmation

    for param1, param2 in params:       # presence of params
        if param1 not in data.columns or param2 not in data.columns:
            print(f"Columns {param1} or {param2} not found in the input file. Skipping this pair.")
            continue

        if 'i' in param1 and 'i' in param2:
            Td_column = 'Tdi'
        elif 'e' in param1 and 'e' in param2:
            Td_column = 'Tde'
        else:
            Td_column = f'Td_{param1}_{param2}'

        print(f"Calculating dew point for columns: {param1}, {param2} -> {Td_column}")      # confirmation
        data[Td_column] = data.apply(lambda row: calculate_single_td(row[param1], row[param2]), axis=1)
        data[Td_column] = data[Td_column].round(3)          # rounding up to 3 decimal points
        print(data[[param1, param2, Td_column]].head())     # confirmation

    data.to_csv(input_file, index=False)    # save the updated data back to the input file
    print(f"Dew point temperatures calculated and saved to the input file: {input_file}")  # confirmation

def calculate_twb(input_file, params):
    '''function to calculate wet bulb temperatures'''
    data = pd.read_csv(input_file)                              # read the input
    print(f"Processing file: {input_file}")                     # confirmation
    print("Headers from the input file:", list(data.columns))   # confirmation

    for param1, param2 in params:       # presence of params
        if param1 not in data.columns or param2 not in data.columns:
            print(f"Columns {param1} or {param2} not found in the input file. Skipping this pair.")
            continue

        if 'i' in param1 and 'i' in param2:
            Twb_column = 'Twbi'
        elif 'e' in param1 and 'e' in param2:
            Twb_column = 'Twbe'
        else:
            Twb_column = f'Twb_{param1}_{param2}'

        print(f"Calculating wet bulb temperature for columns: {param1}, {param2} -> {Twb_column}")  # confirmation
        data[Twb_column] = data.apply(lambda row: calculate_single_twb(row[param1], row[param2]), axis=1)
        data[Twb_column] = data[Twb_column].round(3)        # rounding up to 3 decimal points
        print(data[[param1, param2, Twb_column]].head())    # confirmation

    data.to_csv(input_file, index=False)    # save the updated data back to the input file
    print(f"Wet bulb temperatures calculated and saved to the input file: {input_file}")  # confirmation

def process_files(directory, include_files=None, exclude_files=None):
    '''process csv files based on inclusion and exclusion criteria'''
    include_files_set = set(include_files) if include_files else None       # convert include_files to set
    exclude_files_set = set(exclude_files) if exclude_files else set()      # convert exclude_files to set
    selected_files = []     # selected files list

    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):          # all csv files
            if file_name in exclude_files_set:  # skip the exclude list files
                continue
            if include_files_set and file_name not in include_files_set:    # process only include files
                continue
            selected_files.append(file_name)    # add the file name to the list

    return selected_files   # return the filled list


'''FUNCTIONALITIES'''
def main():
    params = [('Ti', 'RHi'), ('Te', 'RHe')]     # pairs of columns to be examined

    selected_files = process_files(directory, include_files=None, exclude_files=None)  # get all selected files
    print(f"Files selected for analysis: {selected_files}")     # confirmation

    for file_name in selected_files:
        input_file = os.path.join(directory, file_name)
        calculate_td(input_file, params)        # compute td
        calculate_twb(input_file, params)       # compute twb


'''APPLICATION'''
if __name__ == "__main__":
    main()
