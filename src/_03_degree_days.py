import pandas as pd
import os
from _00_CEAM_run import directory, known_format, T_HDD, T_CDD


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
def process_temperature_data(directory, input_file, hdd_base_temperature, cdd_base_temperature, known_format):
    '''function to examine T data in input file for HDD/CDD calculations'''

    def calculate_hdd(temperature, hdd_base_temperature, records_per_day):
        '''function to calculate HDD records'''
        if pd.isnull(temperature):
            return None
        return max(0, hdd_base_temperature - temperature) * (1 / records_per_day)

    def calculate_cdd(temperature, cdd_base_temperature, records_per_day):
        '''function to calculate CDD records'''
        if pd.isnull(temperature):
            return None
        return max(0, temperature - cdd_base_temperature) * (1 / records_per_day)

    file_path = os.path.join(directory, input_file)         # path to the input file
    print(f"Reading data from: {file_path}")                # confirmation
    df = pd.read_csv(file_path)                             # read the input file as df

    if 'Time' not in df.columns or 'Te' not in df.columns:  # validation
        raise ValueError("Input file must contain 'Time' and 'Te' columns.")

    df['Time'] = pd.to_datetime(df['Time'], format=known_format, errors='coerce')
    if df['Time'].isnull().any():
        raise ValueError("Some 'Time' values could not be converted to datetime. Please check the format.")

    df['HDD'] = None  # add HDD column
    df['CDD'] = None  # add CDD column

    # grouping the date part to get daily records
    df['Date'] = df['Time'].dt.date
    daily_groups = df.groupby('Date')

    for date, group in daily_groups:
        records_per_day = group['Te'].notnull().sum()           # empty records
        if records_per_day > 0:                                 # non-empty days
            df.loc[group.index, 'HDD'] = group['Te'].apply(
                lambda temp: calculate_hdd(temp, hdd_base_temperature, records_per_day))  # calculate HDD
            df.loc[group.index, 'CDD'] = group['Te'].apply(
                lambda temp: calculate_cdd(temp, cdd_base_temperature, records_per_day))  # calculate CDD

    df.drop(columns=['Date'], inplace=True)
    df['Time'] = df['Time'].dt.strftime(known_format)  # format the 'Time' back to the known_format

    # save the updated df to csv
    output_file = os.path.join(directory, input_file)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


'''FUNCTIONALITIES'''
def main():
    input_file = 'ECD_ready.csv'
    process_temperature_data(directory, input_file, T_HDD, T_CDD, known_format)


'''APPLICATION'''
if __name__ == "__main__":
    main()
