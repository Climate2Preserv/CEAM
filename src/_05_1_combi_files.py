import os
import pandas as pd
import importlib
from _00_CEAM_run import directory, solar_radiation
from _11_initial_csv_check import directory_others


'''FUNCTIONALITIES'''
def main():
    def combine_files(directory):
        '''function to combine input files into the so-called combi_files;
        combi_files will be used furhter in the process'''
        freq_dict = {}      # dictionary needed to hold data frames grouped by frequencies

        for filename in os.listdir(directory):      # loop through files in the directory
            if filename.endswith('.csv'):           # work on csv files
                frequency = filename.split('_')[-1].split('.')[0]   # extract frequency from the filename
                df = pd.read_csv(os.path.join(directory, filename)) # read the file into df
                df = df.loc[:, ~df.columns.duplicated()]            # remove duplicates
                if frequency not in freq_dict:
                    freq_dict[frequency] = df                       # add the df to the dictionary
                else:
                    freq_dict[frequency] = pd.merge(freq_dict[frequency], df, on='Time', how='outer')   # merge the df with the same frequency

        combi_files = []    # list needed to hold combi_files names
        for frequency, df in freq_dict.items():         # argument for frequency
            if frequency == 'm':                        # argument for monthly files
                df['Time'] = df['Time'].astype(str)     # ensure 'Time' is a str
                df['Time'] = df['Time'].apply(lambda x: '{:02d}.{}'.format(int(x.split('.')[0]), x.split('.')[1]))  # use '%m.%Y' format
            elif frequency == 'w':                      # argument for weekly files
                if solar_radiation == 'y':
                    columns_to_keep = ['Time', 'start', 'end', 'Te', 'RHe', 'Is', 'pve', 'HDD', 'CDD', 'D1', 'D2',
                                       'D3', 'D4', 'D5', 'Ti', 'RHi', 'pvi']  # restructure the df
                else:
                    columns_to_keep = ['Time', 'start', 'end', 'Te', 'RHe', 'pve', 'HDD', 'CDD', 'D1', 'D2',
                                       'D3', 'D4', 'D5', 'Ti', 'RHi', 'pvi']  # restructure the df
                df = df.loc[:, columns_to_keep]

            output_filename = f'{frequency}_comb.csv'   # generate output filename
            df.to_csv(os.path.join(directory, output_filename), index=False)     # save df to csv
            combi_files.append(output_filename)     # append the combi_files list with filename(s)

        return combi_files  # return the list

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')  # import

    combi_files = combine_files(directory)
    _11_initial_csv_check.create_folders(directory_others, folders=['old'])  # create 'to_use' folder
    directory_others_old = os.path.join(directory_others, 'old')  # path to 'others' folder
    _11_initial_csv_check.move_files(directory_others, directory_others_old, file_extension='*.csv', exception_files='extreme_results.csv')  # moving files
    _11_initial_csv_check.move_files(directory, directory_others, file_extension='*.csv', exception_files=combi_files)  # moving files
    _11_initial_csv_check.rename_files(directory, input_names=['ready_comb.csv'], output_names=['h_comb.csv'])  # rename the given file

    _11_initial_csv_check.move_files(directory, directory_others, file_name='weeks_comb.csv')


'''APPLICATION'''
if __name__ == "__main__":
    main()