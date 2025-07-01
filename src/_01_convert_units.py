import os
import pandas as pd
from _00_CEAM_run import directory, units


'''FUNCTIONALITIES'''
def fahrenheit_to_celsius(f):
    '''convert Fahrenheit (F) to Celsius (C)'''
    return (f - 32) * 5.0 / 9.0

def btu_to_kwh(btu):
    '''convert BTU to kWh'''
    return btu * 0.000293071

def convert_units(directory, filename, column, function):
    '''for the selected files apply units conversion and update the files with SI units'''
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)

    if column in df.columns:                # verify columns for conversion to be applied
        df[column] = df[column].apply(function)
        df.to_csv(file_path, index=False)   # save the modified DF back to the same CSV
        print(f"Processed column '{column}' in {filename}.")
    else:
        print(f"Column '{column}' not found in {filename}.")

def main_conversion_logic():
    '''specified logic to handle unit conversions'''
    if units == 'IU':       # if IU
        convert_units(directory, 'ECD_ready.csv', 'Te', fahrenheit_to_celsius)
        convert_units(directory, 'ICD_ready.csv', 'Ti', fahrenheit_to_celsius)
        for col in ['D1', 'D2', 'D3', 'D4', 'D5']:
            convert_units(directory, 'ED_ready.csv', col, btu_to_kwh)
    else:                   # if SI
        print("Units are SI. No conversion needed.")


'''APPLICATION'''
if __name__ == "__main__":
    main_conversion_logic()