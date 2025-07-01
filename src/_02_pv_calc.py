import pandas as pd
import os
import numpy as np
from _00_CEAM_run import directory


'''FUNCTIONALITIES'''
def main():
    def calculate_es(temp_c):
        '''calculate saturation vapor pressure (in Pa) following the Magnus-Tetens equation'''
        A, B, C = 6.112, 17.67, 243.5   # constants
        es = A * np.exp((B * temp_c) / (C + temp_c))    # calculate saturation vapor pressure
        return es * 100     # convert from hPa (mbar) to Pa

    def calculate_e(rh, es):
        '''calculate vapor pressure in Pa'''
        return (rh / 100) * es

    def pv_calc_csv(input_dir, input_file, columns):
        '''calculate and save vapor pressure for the input file'''
        input_path = os.path.join(input_dir, input_file)  # full path to file; needed for function
        df = pd.read_csv(input_path)  # load the input file
        # calculate the vapor pressure for a new column 'pv' only if 'RH' and 'T' are not missing
        df[columns[2]] = df.apply(
            lambda row: calculate_e(row[columns[1]], calculate_es(row[columns[0]])) if pd.notna(row[columns[1]]) and pd.notna(
                row[columns[0]]) else np.nan, axis=1)
        df.to_csv(input_path, index=False)  # save the updated input file
        print(f"The vapor pressure has been calculated and added to {input_file} in {input_dir}")  # confirmation

    input_ECD_file = 'ECD_ready.csv'  # input ECD file
    input_ICD_file = 'ICD_ready.csv'  # input IC file

    ECD_cols = ['Te', 'RHe', 'pve']     # columns for ECD file
    ICD_cols = ['Ti', 'RHi', 'pvi']     # columns for ICD file

    pv_calc_csv(directory, input_ECD_file, ECD_cols)  # calculate vapor pressure for ECD file
    pv_calc_csv(directory, input_ICD_file, ICD_cols)  # calculate vapor pressure for ICD file


'''APPLICATION'''
if __name__ == "__main__":
    main()