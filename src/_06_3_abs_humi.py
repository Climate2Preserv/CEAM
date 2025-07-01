import os
import pandas as pd
import numpy as np
from _00_CEAM_run import directory


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
RW = 461.5      # specific gas constant for water vapor [J/(kg*K)]
PC = 22.064e6   # critical pressure for water [Pa]
TC = 647.096    # critical temperature for water [K]
A_CONSTANTS = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502,]  # empirical constants [-]

def calculate_saturation_vapor_pressure(temp_kelvin):
    '''calculates saturation vapor pressure using Wagner&Pruss equation'''
    tau = 1 - (temp_kelvin / TC)
    exponent_term = (
            A_CONSTANTS[0] * tau
            + A_CONSTANTS[1] * tau ** 1.5
            + A_CONSTANTS[2] * tau ** 3
            + A_CONSTANTS[3] * tau ** 3.5
            + A_CONSTANTS[4] * tau ** 4
            + A_CONSTANTS[5] * tau ** 7.5
    )
    return PC * np.exp((TC / temp_kelvin) * exponent_term)                          # saturation vapor pressure [Pa]

def calculate_absolute_humidity(temp_celsius, rh):
    '''calculates absolute humidity'''
    temp_kelvin = temp_celsius + 273.15                                             # air absolute temperature [K]
    saturation_pressure = calculate_saturation_vapor_pressure(temp_kelvin)          # saturation vapor pressure [Pa]
    partial_pressure = (rh / 100) * saturation_pressure                             # actual vapor pressure [Pa]

    abs_humi = (partial_pressure / (RW * temp_kelvin)) * 1000                       # absolute humidity [g/m^3]
    return abs_humi

def process_files_in_directory(directory):
    '''function to process the csv files to calculate absolute humidity'''
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            print(f"Processing file: {filepath}")

            df = pd.read_csv(filepath, sep=',')
            print(f"Data read from {filename}:{df.head()}")

            if 'Ti' in df.columns and 'RHi' in df.columns:
                df['AbsHi'] = df.apply(lambda row: calculate_absolute_humidity(row['Ti'], row['RHi'])
                if pd.notnull(row['Ti']) and pd.notnull(row['RHi']) else None, axis=1)
            df['AbsHi'] = df['AbsHi'].round(5)

            if 'Te' in df.columns and 'RHe' in df.columns:
                df['AbsHe'] = df.apply(lambda row: calculate_absolute_humidity(row['Te'], row['RHe'])
                if pd.notnull(row['Te']) and pd.notnull(row['RHe']) else None, axis=1)
            df['AbsHe'] = df['AbsHe'].round(5)

            if 'AbsHi' in df.columns and 'AbsHe' in df.columns:
                df['deltaAbsH'] = df['AbsHi'] - df['AbsHe']
            df['deltaAbsH'] = df['deltaAbsH'].round(5)

            print(f"Updated data from {filename}:{df[['AbsHi', 'AbsHe', 'deltaAbsH']].head()}")

            df.to_csv(filepath, sep=',', index=False)
            print(f"File updated: {filename}")


'''FUNCTIONALITIES'''
def main():
    process_files_in_directory(directory)


'''APPLICATION'''
if __name__ == "__main__":
    main()
