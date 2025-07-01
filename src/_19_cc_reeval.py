import pandas as pd
import os
import sys
import shutil
import importlib
import matplotlib.pyplot as plt
import numpy as np
import psychrolib
import gc
from matplotlib.dates import DateFormatter
from _07_CC_file import climate_classes
from _00_CEAM_run import directory, known_format, demands_list, optimization_schedule
from _11_initial_csv_check import directory_reports, directory_others
from _00_CEAM_run import cc_reeval


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
climate_class_symbol = ['AA', 'A1', 'A2', 'B', 'C', 'D']
_11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
_08_psychrometric = importlib.import_module('_08_psychrometric')

def generate_demand_files(input_file_path, demands_list, scenarios_list, additional_data_path):
    df = pd.read_csv(input_file_path, parse_dates=['Time'], dayfirst=True)
    df = df[['Time'] + [col for col in df.columns if col != 'Time']]
    df['Time'] = df['Time'].dt.strftime(known_format)

    df_additional = pd.read_csv(additional_data_path, parse_dates=['Time'], dayfirst=True)
    additional_columns = ['Ti_d_ave', 'RHi_d_ave', 'Ti_m_ave', 'RHi_m_ave', 'Ti_w', 'RHi_w', 'Ti_MA-7d', 'Ti_MA_h',
                          'RHi_MA-30d', 'RHi_MA_h']
    df_additional['Time'] = df_additional['Time'].dt.strftime(known_format)

    columns_to_add = [col for col in additional_columns if col not in df.columns]
    if columns_to_add:
        df = df.merge(df_additional[['Time'] + columns_to_add], on='Time', how='left')

    input_dir = os.path.dirname(input_file_path)

    # process each demand in the demands_list
    for demand in demands_list:
        for scenario in scenarios_list:
            if demand == 'D1':
                if f'Heating_neg_T-D1_2_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d1 = df[['Time', f'Heating_neg_T-D1_2_{scenario}', 'Ti', 'RHi'] + additional_columns].copy()
                    df_d1.rename(columns={
                        f'Heating_neg_T-D1_2_{scenario}': 'Ti',
                        'Ti': 'Ti_og',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d1['RHi'] = df_d1['RHi_og']
                    df_d1.to_csv(os.path.join(input_dir, f'D1_T2_{scenario}.csv'), index=False)
                if f'Heating_neg_T-D1_5_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d1 = df[['Time', f'Heating_neg_T-D1_5_{scenario}', 'Ti', 'RHi'] + additional_columns].copy()
                    df_d1.rename(columns={
                        f'Heating_neg_T-D1_5_{scenario}': 'Ti',
                        'Ti': 'Ti_og',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d1['RHi'] = df_d1['RHi_og']
                    df_d1.to_csv(os.path.join(input_dir, f'D1_T5_{scenario}.csv'), index=False)

            elif demand == 'D2':
                if f'Cooling_pos_T-D2_2_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d2 = df[['Time', f'Cooling_pos_T-D2_2_{scenario}', 'Ti', 'RHi'] + additional_columns].copy()
                    df_d2.rename(columns={
                        f'Cooling_pos_T-D2_2_{scenario}': 'Ti',
                        'Ti': 'Ti_og',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d2['RHi'] = df_d2['RHi_og']
                    df_d2.to_csv(os.path.join(input_dir, f'D2_T2_{scenario}.csv'), index=False)
                if f'Cooling_pos_T-D2_5_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d2 = df[['Time', f'Cooling_pos_T-D2_5_{scenario}', 'Ti', 'RHi'] + additional_columns].copy()
                    df_d2.rename(columns={
                        f'Cooling_pos_T-D2_5_{scenario}': 'Ti',
                        'Ti': 'Ti_og',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d2['RHi'] = df_d2['RHi_og']
                    df_d2.to_csv(os.path.join(input_dir, f'D2_T5_{scenario}.csv'), index=False)

            elif demand == 'D4':
                if 'Ti' in df.columns and f'Humidify_mix_RH-D4_5_{scenario}' in df.columns and 'RHi' in df.columns:
                    df_d4 = df[
                        ['Time', 'Ti', f'Humidify_mix_RH-D4_5_{scenario}', 'RHi'] + list(additional_columns)].copy()
                    df_d4.rename(columns={
                        'Ti': 'Ti_og',
                        f'Humidify_mix_RH-D4_5_{scenario}': 'RHi',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d4['Ti'] = df_d4['Ti_og']
                    df_d4.to_csv(os.path.join(input_dir, f'D4_RH5_{scenario}.csv'), index=False)
                if 'Ti' in df.columns and f'Humidify_mix_RH-D4_10_{scenario}' in df.columns and 'RHi' in df.columns:
                    df_d4 = df[
                        ['Time', 'Ti', f'Humidify_mix_RH-D4_10_{scenario}', 'RHi'] + list(additional_columns)].copy()
                    df_d4.rename(columns={
                        'Ti': 'Ti_og',
                        f'Humidify_mix_RH-D4_10_{scenario}': 'RHi',
                        'RHi': 'RHi_og'
                    }, inplace=True)
                    df_d4['Ti'] = df_d4['Ti_og']
                    df_d4.to_csv(os.path.join(input_dir, f'D4_RH10_{scenario}.csv'), index=False)

            elif demand == 'D5':
                if f'Total_mix_T-D5_2_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d5_1 = df[['Time', f'Total_mix_T-D5_2_{scenario}', 'RHi', 'Ti'] + additional_columns].copy()
                    df_d5_1.rename(columns={'Ti': 'Ti_og', f'Total_mix_T-D5_2_{scenario}': 'Ti', 'RHi': 'RHi_og'},
                                   inplace=True)
                    df_d5_1['RHi'] = df_d5_1['RHi_og']
                    df_d5_1.to_csv(os.path.join(input_dir, f'D5_T2_{scenario}.csv'), index=False)
                if f'Total_mix_T-D5_5_{scenario}' in df.columns and 'RHi' in df.columns and 'Ti' in df.columns:
                    df_d5_1 = df[['Time', f'Total_mix_T-D5_5_{scenario}', 'RHi', 'Ti'] + additional_columns].copy()
                    df_d5_1.rename(columns={'Ti': 'Ti_og', f'Total_mix_T-D5_5_{scenario}': 'Ti', 'RHi': 'RHi_og'},
                                   inplace=True)
                    df_d5_1['RHi'] = df_d5_1['RHi_og']
                    df_d5_1.to_csv(os.path.join(input_dir, f'D5_T5_{scenario}.csv'), index=False)
                if (f'Total_mix_T-D5_2_{scenario}' in df.columns and f'Humidify_mix_RH-D4_5_{scenario}' in df.columns
                        and 'Ti' in df.columns and 'RHi' in df.columns):
                    df_d5_2 = df[['Time', f'Total_mix_T-D5_2_{scenario}', f'Humidify_mix_RH-D4_5_{scenario}', 'Ti',
                                  'RHi'] + additional_columns].copy()
                    df_d5_2.rename(columns={'Ti': 'Ti_og', f'Total_mix_T-D5_2_{scenario}': 'Ti', 'RHi': 'RHi_og',
                                            f'Humidify_mix_RH-D4_5_{scenario}': 'RHi'}, inplace=True)
                    df_d5_2.to_csv(os.path.join(input_dir, f'D5_T2RH5_{scenario}.csv'), index=False)
                if (f'Total_mix_T-D5_5_{scenario}' in df.columns and f'Humidify_mix_RH-D4_10_{scenario}' in df.columns
                        and 'Ti' in df.columns and 'RHi' in df.columns):
                    df_d5_2 = df[['Time', f'Total_mix_T-D5_5_{scenario}', f'Humidify_mix_RH-D4_10_{scenario}', 'Ti',
                                  'RHi'] + additional_columns].copy()
                    df_d5_2.rename(columns={'Ti': 'Ti_og', f'Total_mix_T-D5_5_{scenario}': 'Ti', 'RHi': 'RHi_og',
                                            f'Humidify_mix_RH-D4_10_{scenario}': 'RHi'}, inplace=True)
                    df_d5_2.to_csv(os.path.join(input_dir, f'D5_T5RH10_{scenario}.csv'), index=False)

def process_csv(directory_path, exception_files=[]):
    '''function to process all CSV files in a directory'''
    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    files_to_process = [f for f in files if f not in exception_files]

    if not files_to_process:
        print("No files to process.")
        return
    folder_paths = []

    for file_name in files_to_process:
        file_path = os.path.join(directory_path, file_name)
        folder_name = os.path.splitext(file_name)[0]
        folder_path = os.path.join(directory_path, folder_name)
        folder_paths.append(folder_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Create the folder if it doesn't exist

        shutil.move(file_path, os.path.join(folder_path, file_name))

        file_path = os.path.join(folder_path, file_name)

        # create the report file
        report_file_path = os.path.join(folder_path, f'report_CC_re-eval_{file_name}.txt')
        with open(report_file_path, 'w') as report_file:
            sys.stdout = report_file

            print(f"\nProcessing file: {file_name} in folder: {folder_name}")
            sys.stdout.flush()

            df = pd.read_csv(file_path)

            if 'Ti' not in df.columns or 'RHi' not in df.columns or 'Time' not in df.columns:
                print("The CSV file must contain 'Ti', 'RHi', and 'Time' columns.")
                return
            if 'Ti_m_ave' not in df.columns or 'RHi_m_ave' not in df.columns:
                print("The CSV file must contain 'Ti_m_ave' and 'RHi_m_ave' columns for short-term assessments.")
                return

            try:
                df['Time'] = pd.to_datetime(df['Time'], format=known_format)
            except ValueError as e:
                print(f"Error converting 'Time' column to datetime: {e}")
                return

            total_Ti = df['Ti'].count()         # the total count of Ti records
            total_RHi = df['RHi'].count()       # the total count of RHi records
            Ti_annual_ave = df['Ti'].mean()     # annual Ti mean
            RHi_annual_ave = df['RHi'].mean()   # annual RHi mean

            total_Ti_RHi = df.dropna(subset=['Ti', 'RHi']).shape[0]    # the total count of both records

            # print information statements
            print(f"\nInput data summary:")
            print(f"  Total Ti records: {total_Ti}")
            print(f"  Total RHi records: {total_RHi}")
            print(f"  Total (simultaneous) Ti and RHi records: {total_Ti_RHi}")
            print(f"  Annual average Ti: {Ti_annual_ave:.2f}")
            print(f"  Annual average RHi: {RHi_annual_ave:.2f}")
            print(f"\nCLIMATE CLASS ASSESSMENT")

            for climate_class in climate_classes:
                name = climate_class['name']

                if climate_class['Ti_outer_min'] is None and climate_class['Ti_outer_max'] is None:
                    Ti_within_outer_range = total_Ti
                else:
                    Ti_within_outer_range = df[
                        (df['Ti'] >= (climate_class['Ti_outer_min'] or float('-inf'))) &
                        (df['Ti'] <= (climate_class['Ti_outer_max'] or float('inf')))
                        ].shape[0]

                if climate_class['RHi_outer_min'] is None and climate_class['RHi_outer_max'] is None:
                    RHi_within_outer_range = total_RHi
                else:
                    RHi_within_outer_range = df[
                        (df['RHi'] >= (climate_class['RHi_outer_min'] or float('-inf'))) &
                        (df['RHi'] <= (climate_class['RHi_outer_max'] or float('inf')))
                        ].shape[0]

                if climate_class['Ti_outer_min'] is None and climate_class['Ti_outer_max'] is None:
                    Ti_within_annual_range = total_Ti
                else:
                    Ti_within_annual_range = df[
                        (df['Ti'] >= Ti_annual_ave - (climate_class['Ti_seasonal_offset_n'] or float('inf'))) &
                        (df['Ti'] <= Ti_annual_ave + (climate_class['Ti_seasonal_offset_p'] or float('inf')))
                        ].shape[0]

                if (climate_class['Ti_outer_min'] is None and climate_class['Ti_outer_max'] is None and
                        climate_class['RHi_outer_min'] is None and climate_class['RHi_outer_max'] is None):
                    Ti_RHi_within_outer_range = total_Ti_RHi
                else:
                    Ti_RHi_within_outer_range = df[
                        (df['Ti'] >= (climate_class['Ti_outer_min'] or float('-inf'))) &
                        (df['Ti'] <= (climate_class['Ti_outer_max'] or float('inf'))) &
                        (df['RHi'] >= (climate_class['RHi_outer_min'] or float('-inf'))) &
                        (df['RHi'] <= (climate_class['RHi_outer_max'] or float('inf')))
                        ].shape[0]

                if (climate_class['Ti_seasonal_offset_n'] is None and climate_class['Ti_seasonal_offset_p'] is None and
                        climate_class['RHi_seasonal_offset_n'] is None and climate_class['RHi_seasonal_offset_p'] is None):
                    Ti_RHi_within_annual_range = total_Ti_RHi
                else:
                    Ti_lower_bound = Ti_annual_ave - (
                        climate_class['Ti_seasonal_offset_n'] if climate_class['Ti_seasonal_offset_n'] is not None
                        else float('inf'))
                    Ti_upper_bound = Ti_annual_ave + (
                        climate_class['Ti_seasonal_offset_p'] if climate_class['Ti_seasonal_offset_p'] is not None
                        else float('inf'))
                    RHi_lower_bound = RHi_annual_ave - (
                        climate_class['RHi_seasonal_offset_n'] if climate_class['RHi_seasonal_offset_n'] is not None
                        else float('inf'))
                    RHi_upper_bound = RHi_annual_ave + (
                        climate_class['RHi_seasonal_offset_p'] if climate_class['RHi_seasonal_offset_p'] is not None
                        else float('inf'))

                    Ti_RHi_within_annual_range = df[
                        (df['Ti'] >= Ti_lower_bound) &
                        (df['Ti'] <= Ti_upper_bound) &
                        (df['RHi'] >= RHi_lower_bound) &
                        (df['RHi'] <= RHi_upper_bound)
                        ].shape[0]

                if climate_class['RHi_seasonal_offset_n'] is None and climate_class['RHi_seasonal_offset_p'] is None:
                    RHi_within_annual_range = total_RHi
                else:
                    RHi_within_annual_range = df[
                        ((climate_class['RHi_seasonal_offset_n'] != 0) &
                         (df['RHi'] >= RHi_annual_ave - climate_class['RHi_seasonal_offset_n']))
                        |
                        ((climate_class['RHi_seasonal_offset_p'] != 0) &
                         (df['RHi'] <= RHi_annual_ave + climate_class['RHi_seasonal_offset_p']))
                        |
                        ((climate_class['RHi_seasonal_offset_n'] == 0) &
                         (climate_class['RHi_seasonal_offset_p'] == 0) &
                         (df['RHi'] == RHi_annual_ave))
                        ].shape[0]

                if climate_class['Ti_short_offset'] is None:
                    Ti_within_short_range = total_Ti
                    Ti_within_short_range_MA_7d = total_Ti
                    Ti_within_short_range_MA_h = total_Ti
                else:
                    Ti_within_short_range = df[
                        (df['Ti'] >= df['Ti_w'] - climate_class['Ti_short_offset']) &
                        (df['Ti'] <= df['Ti_w'] + climate_class['Ti_short_offset'])
                        ].shape[0]
                    Ti_within_short_range_MA_7d = df[
                        (df['Ti'] >= df['Ti_MA-7d'] - climate_class['Ti_short_offset']) &
                        (df['Ti'] <= df['Ti_MA-7d'] + climate_class['Ti_short_offset'])
                        ].shape[0]
                    Ti_within_short_range_MA_h = df[
                        (df['Ti'] >= df['Ti_MA_h'] - climate_class['Ti_short_offset']) &
                        (df['Ti'] <= df['Ti_MA_h'] + climate_class['Ti_short_offset'])
                        ].shape[0]

                if climate_class['RHi_short_offset'] is None:
                    RHi_within_short_range = total_RHi
                    RHi_within_short_range_MA_30d = total_RHi
                    RHi_within_short_range_MA_h = total_RHi
                else:
                    RHi_within_short_range = df[
                        (df['RHi'] >= df['RHi_m_ave'] - climate_class['RHi_short_offset']) &
                        (df['RHi'] <= df['RHi_m_ave'] + climate_class['RHi_short_offset'])
                        ].shape[0]
                    RHi_within_short_range_MA_30d = df[
                        (df['RHi'] >= df['RHi_MA-30d'] - climate_class['RHi_short_offset']) &
                        (df['RHi'] <= df['RHi_MA-30d'] + climate_class['RHi_short_offset'])
                        ].shape[0]
                    RHi_within_short_range_MA_h = df[
                        (df['RHi'] >= df['RHi_MA_h'] - climate_class['RHi_short_offset']) &
                        (df['RHi'] <= df['RHi_MA_h'] + climate_class['RHi_short_offset'])
                        ].shape[0]

                if (climate_class['Ti_short_offset'] is None and climate_class['RHi_short_offset'] is None):
                    Ti_RHi_within_short_range = total_Ti_RHi
                else:
                    Ti_lower_short_bound = df['Ti_w'] - (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    Ti_upper_short_bound = df['Ti_w'] + (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    RHi_lower_short_bound = df['RHi_m_ave'] - (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))
                    RHi_upper_short_bound = df['RHi_m_ave'] + (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))

                    Ti_RHi_within_short_range = df[
                        (df['Ti'] >= Ti_lower_short_bound) &
                        (df['Ti'] <= Ti_upper_short_bound) &
                        (df['RHi'] >= RHi_lower_short_bound) &
                        (df['RHi'] <= RHi_upper_short_bound)
                        ].shape[0]

                if (climate_class['Ti_short_offset'] is None and climate_class['RHi_short_offset'] is None):
                    Ti_RHi_within_short_range_MA_d = total_Ti_RHi
                else:
                    Ti_lower_short_bound_MA_d = df['Ti_MA-7d'] - (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    Ti_upper_short_bound_MA_d = df['Ti_MA-7d'] + (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    RHi_lower_short_bound_MA_d = df['RHi_MA-30d'] - (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))
                    RHi_upper_short_bound_MA_d = df['RHi_MA-30d'] + (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))

                    Ti_RHi_within_short_range_MA_d = df[
                        (df['Ti'] >= Ti_lower_short_bound_MA_d) &
                        (df['Ti'] <= Ti_upper_short_bound_MA_d) &
                        (df['RHi'] >= RHi_lower_short_bound_MA_d) &
                        (df['RHi'] <= RHi_upper_short_bound_MA_d)
                        ].shape[0]

                if (climate_class['Ti_short_offset'] is None and climate_class['RHi_short_offset'] is None):
                    Ti_RHi_within_short_range_MA_h = total_Ti_RHi
                else:
                    Ti_lower_short_bound_MA_h = df['Ti_MA_h'] - (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    Ti_upper_short_bound_MA_h = df['Ti_MA_h'] + (
                        climate_class['Ti_short_offset'] if climate_class['Ti_short_offset'] is not None else float('inf'))
                    RHi_lower_short_bound_MA_h = df['RHi_MA_h'] - (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))
                    RHi_upper_short_bound_MA_h = df['RHi_MA_h'] + (
                        climate_class['RHi_short_offset'] if climate_class['RHi_short_offset'] is not None else float('inf'))

                    Ti_RHi_within_short_range_MA_h = df[
                        (df['Ti'] >= Ti_lower_short_bound_MA_h) &
                        (df['Ti'] <= Ti_upper_short_bound_MA_h) &
                        (df['RHi'] >= RHi_lower_short_bound_MA_h) &
                        (df['RHi'] <= RHi_upper_short_bound_MA_h)
                        ].shape[0]

                Ti_outer_range_percentage = (Ti_within_outer_range / total_Ti) * 100
                RHi_outer_range_percentage = (RHi_within_outer_range / total_RHi) * 100
                Ti_annual_range_percentage = (Ti_within_annual_range / total_Ti) * 100
                RHi_annual_range_percentage = (RHi_within_annual_range / total_RHi) * 100
                Ti_RHi_outer_range_percentage = (Ti_RHi_within_outer_range / total_Ti_RHi) * 100
                Ti_RHi_annual_range_percentage = (Ti_RHi_within_annual_range / total_Ti_RHi) * 100
                Ti_short_range_percentage = (Ti_within_short_range / total_Ti) * 100

                Ti_short_range_percentage_MA_7d = (Ti_within_short_range_MA_7d / total_Ti) * 100
                Ti_short_range_percentage_MA_h = (Ti_within_short_range_MA_h / total_Ti) * 100

                RHi_short_range_percentage = (RHi_within_short_range / total_RHi) * 100

                RHi_short_range_percentage_MA_30d = (RHi_within_short_range_MA_30d / total_RHi) * 100
                RHi_short_range_percentage_MA_h = (RHi_within_short_range_MA_h / total_RHi) * 100

                Ti_RHi_short_range_percentage = (Ti_RHi_within_short_range / total_Ti_RHi) * 100

                Ti_RHi_short_range_percentage_MA_d = (Ti_RHi_within_short_range_MA_d / total_Ti_RHi) * 100
                Ti_RHi_short_range_percentage_MA_h = (Ti_RHi_within_short_range_MA_h / total_Ti_RHi) * 100

                # print information statements
                print(f"\nClimate Class '{name}':")
                print(f"  Long-term fulfillment: {Ti_RHi_outer_range_percentage:.2f}% of the time")
                print(f"  Seasonal-term fulfillment: {Ti_RHi_annual_range_percentage:.2f}% of the time")
                print(f"  Short-term fulfillment: {Ti_RHi_short_range_percentage:.2f}% of the time")
                print(f"  Short-term fulfillment for daily moving average: {Ti_RHi_short_range_percentage_MA_d:.2f}% of the time")
                print(f"  Short-term fulfillment for hourly moving average: {Ti_RHi_short_range_percentage_MA_h:.2f}% of the time")
                print(f"\n  Details:")
                print(f"    Ti records within outer min and max range: {Ti_within_outer_range} ({Ti_outer_range_percentage:.2f}%)")
                print(f"    RHi records within outer min and max range: {RHi_within_outer_range} ({RHi_outer_range_percentage:.2f}%)")
                print(f"    Ti and RHi records (simultaneously) within outer min and max range: {Ti_RHi_within_outer_range} ({Ti_RHi_outer_range_percentage:.2f}%)")
                print(f"\n    Ti records within annual average ± seasonal offset: {Ti_within_annual_range} ({Ti_annual_range_percentage:.2f}%)")
                print(f"    RHi records within annual average ± seasonal offset: {RHi_within_annual_range} ({RHi_annual_range_percentage:.2f}%)")
                print(f"    Ti and RHi records (simultaneously) within annual average ± seasonal offset: {Ti_RHi_within_annual_range} ({Ti_RHi_annual_range_percentage:.2f}%)")
                print(f"\n    Ti records within short-term offset: {Ti_within_short_range} ({Ti_short_range_percentage:.2f}%)")
                print(f"    RHi records within short-term offset: {RHi_within_short_range} ({RHi_short_range_percentage:.2f}%)")
                print(f"    Ti and RHi records (simultaneously) within short-term offset: {Ti_RHi_within_short_range} ({Ti_RHi_short_range_percentage:.2f}%)")
                print(f"\n    Ti_MA_d records within short-term offset: {Ti_within_short_range_MA_7d} ({Ti_short_range_percentage_MA_7d:.2f}%)")
                print(f"    RHi_MA_d records within short-term offset: {RHi_within_short_range_MA_30d} ({RHi_short_range_percentage_MA_30d:.2f}%)")
                print(f"    Ti_MA_d and RHi_MA_d records (simultaneously) within short-term offset: {Ti_RHi_within_short_range_MA_d} ({Ti_RHi_short_range_percentage_MA_d:.2f}%)")
                print(f"\n    Ti_MA_h records within short-term offset: {Ti_within_short_range_MA_h} ({Ti_short_range_percentage_MA_h:.2f}%)")
                print(f"    RHi_MA_h records within short-term offset: {RHi_within_short_range_MA_h} ({RHi_short_range_percentage_MA_h:.2f}%)")
                print(f"    Ti_MA_h and RHi_MA_h records (simultaneously) within short-term offset: {Ti_RHi_within_short_range_MA_h} ({Ti_RHi_short_range_percentage_MA_h:.2f}%)")

                plot_climate_class_assessment(df, climate_class, Ti_annual_ave, RHi_annual_ave, folder_path)  # for plots
            for name_class in climate_class_symbol:
                _08_psychrometric.plot_psychrometric_chart(folder_path, file_name, 'Ti', 'RHi', 'Ti_og', 'RHi_og', name_class)

            sys.stdout = sys.__stdout__

    return folder_paths

def plot_climate_class_assessment(df, climate_class, Ti_annual_ave, RHi_annual_ave, directory):
    '''function to plot Ti and RHi with Climate Classes requirements'''
    _13_data_overview_plots = importlib.import_module('_13_data_overview_plots')

    Ti_outer_min = climate_class['Ti_outer_min']
    Ti_outer_max = climate_class['Ti_outer_max']
    RHi_outer_min = climate_class['RHi_outer_min']
    RHi_outer_max = climate_class['RHi_outer_max']
    Ti_seasonal_offset_p = climate_class['Ti_seasonal_offset_p']
    Ti_seasonal_offset_n = climate_class['Ti_seasonal_offset_n']
    RHi_seasonal_offset_p = climate_class['RHi_seasonal_offset_p']
    RHi_seasonal_offset_n = climate_class['RHi_seasonal_offset_n']
    Ti_short_offset = climate_class['Ti_short_offset']
    RHi_short_offset = climate_class['RHi_short_offset']

    time_values = df['Time']
    Ti_values = df['Ti']
    RHi_values = df['RHi']

    Ti_w = df['Ti_w']
    Ti_MA_d = df['Ti_MA-7d']
    Ti_MA_h = df['Ti_MA_h']
    if Ti_short_offset is not None:
        Ti_short_offset_upper = Ti_w + Ti_short_offset
        Ti_short_offset_lower = Ti_w - Ti_short_offset

        Ti_short_offset_upper_MA_d = Ti_MA_d + Ti_short_offset
        Ti_short_offset_lower_MA_d = Ti_MA_d - Ti_short_offset

        Ti_short_offset_upper_MA_h = Ti_MA_h + Ti_short_offset
        Ti_short_offset_lower_MA_h = Ti_MA_h - Ti_short_offset

    RHi_m_ave = df['RHi_m_ave']
    RHi_MA_d = df['RHi_MA-30d']
    RHi_MA_h = df['RHi_MA_h']
    if RHi_short_offset is not None:
        RHi_short_offset_upper = RHi_m_ave + RHi_short_offset
        RHi_short_offset_lower = RHi_m_ave - RHi_short_offset

        RHi_short_offset_upper_MA_d = RHi_MA_d + RHi_short_offset
        RHi_short_offset_lower_MA_d = RHi_MA_d - RHi_short_offset

        RHi_short_offset_upper_MA_h = RHi_MA_h + RHi_short_offset
        RHi_short_offset_lower_MA_h = RHi_MA_h - RHi_short_offset

    plot_height, plot_width = 25, 15
    dpi = 300

    # Ti Plot
    fig, ax1 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(Ti_values), dtype=bool)

    if Ti_outer_max is not None:
        outer_points_mask |= (Ti_values > Ti_outer_max)
    if Ti_outer_min is not None:
        outer_points_mask |= (Ti_values < Ti_outer_min)
    if Ti_seasonal_offset_p is not None:
        outer_points_mask |= (Ti_values > (Ti_annual_ave + Ti_seasonal_offset_p))
    if Ti_seasonal_offset_n is not None:
        outer_points_mask |= (Ti_values < (Ti_annual_ave - Ti_seasonal_offset_n))
    if Ti_short_offset is not None:
        Ti_short_offset_upper = Ti_w + Ti_short_offset
        Ti_short_offset_lower = Ti_w - Ti_short_offset
        outer_points_mask |= (Ti_values > Ti_short_offset_upper)
        outer_points_mask |= (Ti_values < Ti_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    lower_limit = 0 if Ti_values.min() >= 0 else (np.floor(Ti_values.min() / 5) * 5)
    upper_limit = 35 if Ti_values.min() <= 30 else (np.ceil(Ti_values.max() / 5) * 5)

    ax1.scatter(time_values[inner_points_mask], Ti_values[inner_points_mask],
                label='Ti (inside data)', marker='d', s=35, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax1.scatter(time_values[outer_points_mask], Ti_values[outer_points_mask],
                label='Ti (outside data)', marker='d', s=35, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax1.plot(time_values, Ti_values, linestyle='-', color='dimgray', linewidth=0.5)

    if Ti_short_offset is not None:
        ax1.plot(time_values, Ti_short_offset_upper, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset')
        ax1.plot(time_values, Ti_short_offset_lower, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)

    ax1.plot(time_values, Ti_w, linestyle='-', color='#82B446', linewidth=2.5, label='weekly average')

    ax1.axhline(y=Ti_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if Ti_outer_min is not None:
        ax1.axhline(y=Ti_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if Ti_outer_max is not None:
        ax1.axhline(y=Ti_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if Ti_seasonal_offset_n is not None:
        ax1.axhline(y=Ti_annual_ave - Ti_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if Ti_seasonal_offset_p is not None:
        ax1.axhline(y=Ti_annual_ave + Ti_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax1, time_values, Ti_values)

    ax1.set_ylabel('Ti [°C]', fontsize=35, fontweight='bold')
    ax1.set_title(f"Ti Assessment - Climate Class '{climate_class['name']}'", fontsize=35, fontweight='bold')
    ax1.set_ylim(lower_limit, upper_limit)
    ax1.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax1.set_xlim(df['Time'].min(), df['Time'].max())
    ax1.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax1.tick_params(axis='y', which='major', labelsize=15)
    ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend1 = ax1.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
        )
    for text in legend1.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_T.png"))
    plt.close(fig)
    del fig
    gc.collect()

    # Ti_MA-d plot
    fig, ax1 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(Ti_values), dtype=bool)

    if Ti_outer_max is not None:
        outer_points_mask |= (Ti_values > Ti_outer_max)
    if Ti_outer_min is not None:
        outer_points_mask |= (Ti_values < Ti_outer_min)
    if Ti_seasonal_offset_p is not None:
        outer_points_mask |= (Ti_values > (Ti_annual_ave + Ti_seasonal_offset_p))
    if Ti_seasonal_offset_n is not None:
        outer_points_mask |= (Ti_values < (Ti_annual_ave - Ti_seasonal_offset_n))
    if Ti_short_offset is not None:
        Ti_short_offset_upper = Ti_MA_d + Ti_short_offset
        Ti_short_offset_lower = Ti_MA_d - Ti_short_offset
        outer_points_mask |= (Ti_values > Ti_short_offset_upper)
        outer_points_mask |= (Ti_values < Ti_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    lower_limit = 0 if Ti_values.min() >= 0 else (np.floor(Ti_values.min() / 5) * 5)
    upper_limit = 35 if Ti_values.min() <= 30 else (np.ceil(Ti_values.max() / 5) * 5)

    ax1.scatter(time_values[inner_points_mask], Ti_values[inner_points_mask],
                label='Ti (inside data)', marker='d', s=35, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax1.scatter(time_values[outer_points_mask], Ti_values[outer_points_mask],
                label='Ti (outside data)', marker='d', s=35, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax1.plot(time_values, Ti_values, linestyle='-', color='dimgray', linewidth=0.5)

    if Ti_short_offset is not None:
        ax1.plot(time_values, Ti_short_offset_upper_MA_d, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset MA-d')
        ax1.plot(time_values, Ti_short_offset_lower_MA_d, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)

    ax1.plot(time_values, Ti_MA_d, linestyle='-', color='#82B446', linewidth=2.5, label='MA-7d')

    ax1.axhline(y=Ti_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if Ti_outer_min is not None:
        ax1.axhline(y=Ti_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if Ti_outer_max is not None:
        ax1.axhline(y=Ti_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if Ti_seasonal_offset_n is not None:
        ax1.axhline(y=Ti_annual_ave - Ti_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if Ti_seasonal_offset_p is not None:
        ax1.axhline(y=Ti_annual_ave + Ti_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax1, time_values, Ti_values)

    ax1.set_ylabel('Ti [°C]', fontsize=35, fontweight='bold')
    ax1.set_title(f"Ti Assessment - Climate Class '{climate_class['name']}' - daily MA", fontsize=35, fontweight='bold')
    ax1.set_ylim(lower_limit, upper_limit)
    ax1.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax1.set_xlim(df['Time'].min(), df['Time'].max())
    ax1.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax1.tick_params(axis='y', which='major', labelsize=15)
    ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend1 = ax1.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
    )
    for text in legend1.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_T_MA_d.png"))
    plt.close(fig)
    del fig
    gc.collect()

    # Ti_MA-h plot
    fig, ax1 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(Ti_values), dtype=bool)

    if Ti_outer_max is not None:
        outer_points_mask |= (Ti_values > Ti_outer_max)
    if Ti_outer_min is not None:
        outer_points_mask |= (Ti_values < Ti_outer_min)
    if Ti_seasonal_offset_p is not None:
        outer_points_mask |= (Ti_values > (Ti_annual_ave + Ti_seasonal_offset_p))
    if Ti_seasonal_offset_n is not None:
        outer_points_mask |= (Ti_values < (Ti_annual_ave - Ti_seasonal_offset_n))
    if Ti_short_offset is not None:
        Ti_short_offset_upper = Ti_MA_h + Ti_short_offset
        Ti_short_offset_lower = Ti_MA_h - Ti_short_offset
        outer_points_mask |= (Ti_values > Ti_short_offset_upper)
        outer_points_mask |= (Ti_values < Ti_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    lower_limit = 0 if Ti_values.min() >= 0 else (np.floor(Ti_values.min() / 5) * 5)
    upper_limit = 35 if Ti_values.min() <= 30 else (np.ceil(Ti_values.max() / 5) * 5)

    ax1.scatter(time_values[inner_points_mask], Ti_values[inner_points_mask],
                label='Ti (inside data)', marker='d', s=35, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax1.scatter(time_values[outer_points_mask], Ti_values[outer_points_mask],
                label='Ti (outside data)', marker='d', s=35, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax1.plot(time_values, Ti_values, linestyle='-', color='dimgray', linewidth=0.5)

    if Ti_short_offset is not None:
        ax1.plot(time_values, Ti_short_offset_upper_MA_h, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset MA-h')
        ax1.plot(time_values, Ti_short_offset_lower_MA_h, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)

    ax1.plot(time_values, Ti_MA_h, linestyle='-', color='#82B446', linewidth=2.5, label='MA-h')

    ax1.axhline(y=Ti_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if Ti_outer_min is not None:
        ax1.axhline(y=Ti_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if Ti_outer_max is not None:
        ax1.axhline(y=Ti_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if Ti_seasonal_offset_n is not None:
        ax1.axhline(y=Ti_annual_ave - Ti_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if Ti_seasonal_offset_p is not None:
        ax1.axhline(y=Ti_annual_ave + Ti_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax1, time_values, Ti_values)

    ax1.set_ylabel('Ti [°C]', fontsize=35, fontweight='bold')
    ax1.set_title(f"Ti Assessment - Climate Class '{climate_class['name']}' - hourly MA", fontsize=35, fontweight='bold')
    ax1.set_ylim(lower_limit, upper_limit)
    ax1.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax1.set_xlim(df['Time'].min(), df['Time'].max())
    ax1.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax1.tick_params(axis='y', which='major', labelsize=15)
    ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend1 = ax1.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
    )
    for text in legend1.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_T_MA_h.png"))
    plt.close(fig)
    del fig
    gc.collect()

    # RHi plot
    fig, ax2 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(RHi_values), dtype=bool)

    if RHi_outer_max is not None:
        outer_points_mask |= (RHi_values > RHi_outer_max)
    if RHi_outer_min is not None:
        outer_points_mask |= (RHi_values < RHi_outer_min)
    if RHi_seasonal_offset_p is not None:
        outer_points_mask |= (RHi_values > (RHi_annual_ave + RHi_seasonal_offset_p))
    if RHi_seasonal_offset_n is not None:
        outer_points_mask |= (RHi_values < (RHi_annual_ave - RHi_seasonal_offset_n))
    if RHi_short_offset is not None:
        RHi_short_offset_upper = RHi_m_ave + RHi_short_offset
        RHi_short_offset_lower = RHi_m_ave - RHi_short_offset
        outer_points_mask |= (RHi_values > RHi_short_offset_upper)
        outer_points_mask |= (RHi_values < RHi_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    ax2.scatter(time_values[inner_points_mask], RHi_values[inner_points_mask],
                label='RHi (inside data)', linestyle='-', color='dimgray', marker='d', s=30, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax2.scatter(time_values[outer_points_mask], RHi_values[outer_points_mask],
                label='RHi (outside data)', linestyle='-', color='#C4A36A', marker='d', s=30, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax2.plot(time_values, RHi_values, linestyle='-', color='dimgray', linewidth=0.5)

    if RHi_short_offset is not None:
        ax2.plot(time_values, RHi_short_offset_upper, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset')
        ax2.plot(time_values, RHi_short_offset_lower, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)
    ax2.plot(time_values, RHi_m_ave, linestyle='-', color='seagreen', linewidth=2, label='monthly average')
    ax2.axhline(y=RHi_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if RHi_outer_min is not None:
        ax2.axhline(y=RHi_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if RHi_outer_max is not None:
        ax2.axhline(y=RHi_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if RHi_seasonal_offset_n is not None:
        ax2.axhline(y=RHi_annual_ave - RHi_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if RHi_seasonal_offset_p is not None:
        ax2.axhline(y=RHi_annual_ave + RHi_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax2, time_values, RHi_values)

    ax2.set_ylabel('RHi [%]', fontsize=35, fontweight='bold')
    ax2.set_title(f"RHi Assessment - Climate Class '{climate_class['name']}'", fontsize=35, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax2.set_xlim(df['Time'].min(), df['Time'].max())
    ax2.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=15)
    ax2.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend2 = ax2.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
        )
    for text in legend2.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_RH.png"))
    plt.close(fig)
    del fig
    gc.collect()

    # RHi MA-daily plot
    fig, ax2 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(RHi_values), dtype=bool)

    if RHi_outer_max is not None:
        outer_points_mask |= (RHi_values > RHi_outer_max)
    if RHi_outer_min is not None:
        outer_points_mask |= (RHi_values < RHi_outer_min)
    if RHi_seasonal_offset_p is not None:
        outer_points_mask |= (RHi_values > (RHi_annual_ave + RHi_seasonal_offset_p))
    if RHi_seasonal_offset_n is not None:
        outer_points_mask |= (RHi_values < (RHi_annual_ave - RHi_seasonal_offset_n))
    if RHi_short_offset is not None:
        RHi_short_offset_upper = RHi_MA_d + RHi_short_offset
        RHi_short_offset_lower = RHi_MA_d - RHi_short_offset
        outer_points_mask |= (RHi_values > RHi_short_offset_upper)
        outer_points_mask |= (RHi_values < RHi_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    ax2.scatter(time_values[inner_points_mask], RHi_values[inner_points_mask],
                label='RHi (inside data)', linestyle='-', color='dimgray', marker='d', s=30, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax2.scatter(time_values[outer_points_mask], RHi_values[outer_points_mask],
                label='RHi (outside data)', linestyle='-', color='#C4A36A', marker='d', s=30, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax2.plot(time_values, RHi_values, linestyle='-', color='dimgray', linewidth=0.5)

    if RHi_short_offset is not None:
        ax2.plot(time_values, RHi_short_offset_upper_MA_d, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset MA-d')
        ax2.plot(time_values, RHi_short_offset_lower_MA_d, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)
    ax2.plot(time_values, RHi_MA_d, linestyle='-', color='seagreen', linewidth=2, label='MA-30d')
    # ax2.plot(time_values, RHi_MA_h, linestyle='-', color='pink', linewidth=2, label='MA-h')
    ax2.axhline(y=RHi_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if RHi_outer_min is not None:
        ax2.axhline(y=RHi_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if RHi_outer_max is not None:
        ax2.axhline(y=RHi_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if RHi_seasonal_offset_n is not None:
        ax2.axhline(y=RHi_annual_ave - RHi_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if RHi_seasonal_offset_p is not None:
        ax2.axhline(y=RHi_annual_ave + RHi_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax2, time_values, RHi_values)

    ax2.set_ylabel('RHi [%]', fontsize=35, fontweight='bold')
    ax2.set_title(f"RHi Assessment - Climate Class '{climate_class['name']}' - daily MA", fontsize=35, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax2.set_xlim(df['Time'].min(), df['Time'].max())
    ax2.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=15)
    ax2.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend2 = ax2.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
        )
    for text in legend2.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_RH_MA_d.png"))
    plt.close(fig)
    del fig
    gc.collect()

    # RHi MA-hourly plot
    fig, ax2 = plt.subplots(figsize=(plot_height, plot_width), dpi=dpi)
    outer_points_mask = np.zeros(len(RHi_values), dtype=bool)

    if RHi_outer_max is not None:
        outer_points_mask |= (RHi_values > RHi_outer_max)
    if RHi_outer_min is not None:
        outer_points_mask |= (RHi_values < RHi_outer_min)
    if RHi_seasonal_offset_p is not None:
        outer_points_mask |= (RHi_values > (RHi_annual_ave + RHi_seasonal_offset_p))
    if RHi_seasonal_offset_n is not None:
        outer_points_mask |= (RHi_values < (RHi_annual_ave - RHi_seasonal_offset_n))
    if RHi_short_offset is not None:
        RHi_short_offset_upper = RHi_MA_h + RHi_short_offset
        RHi_short_offset_lower = RHi_MA_h - RHi_short_offset
        outer_points_mask |= (RHi_values > RHi_short_offset_upper)
        outer_points_mask |= (RHi_values < RHi_short_offset_lower)

    inner_points_mask = ~outer_points_mask

    ax2.scatter(time_values[inner_points_mask], RHi_values[inner_points_mask],
                label='RHi (inside data)', linestyle='-', color='dimgray', marker='d', s=30, linewidths=0.5,
                facecolor='lightgrey', edgecolor='dimgray')
    ax2.scatter(time_values[outer_points_mask], RHi_values[outer_points_mask],
                label='RHi (outside data)', linestyle='-', color='#C4A36A', marker='d', s=30, linewidths=0.5,
                facecolor='#FAF3D3', edgecolor='#C4A36A')

    ax2.plot(time_values, RHi_values, linestyle='-', color='dimgray', linewidth=0.5)

    if RHi_short_offset is not None:
        ax2.plot(time_values, RHi_short_offset_upper_MA_h, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5,
                 label='Short Offset MA-h')
        ax2.plot(time_values, RHi_short_offset_lower_MA_h, linestyle=(0, (1, 0.3)), color='#82B446', linewidth=3.5)
    ax2.plot(time_values, RHi_MA_h, linestyle='-', color='seagreen', linewidth=2, label='MA-h')
    ax2.axhline(y=RHi_annual_ave, linestyle='-', linewidth=2.5, color='steelblue', label='Annual Avg')
    if RHi_outer_min is not None:
        ax2.axhline(y=RHi_outer_min, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Min')
    if RHi_outer_max is not None:
        ax2.axhline(y=RHi_outer_max, linestyle=(0, (2.5, 0.5)), linewidth=7.5, color='firebrick', label='Outer Max')
    if RHi_seasonal_offset_n is not None:
        ax2.axhline(y=RHi_annual_ave - RHi_seasonal_offset_n, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9',
                    label='Seasonal Offset')
    if RHi_seasonal_offset_p is not None:
        ax2.axhline(y=RHi_annual_ave + RHi_seasonal_offset_p, linestyle=(0, (2.5, 0.5)), linewidth=5, color='#77A5C9')

    _13_data_overview_plots.highlight_missing_periods(ax2, time_values, RHi_values)

    ax2.set_ylabel('RHi [%]', fontsize=35, fontweight='bold')
    ax2.set_title(f"RHi Assessment - Climate Class '{climate_class['name']}' - hourly MA", fontsize=35, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.xaxis.set_major_formatter(DateFormatter('%m.%Y'))
    ax2.set_xlim(df['Time'].min(), df['Time'].max())
    ax2.set_xlabel('Time', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=15)
    ax2.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)
    legend2 = ax2.legend(
        loc='lower right',
        fontsize=15,
        edgecolor='black',
        facecolor='white',
        fancybox=False,
        borderpad=1,
        labelspacing=1,
        handlelength=5
        )
    for text in legend2.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_fontsize(15)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{climate_class['name']}_assessment_RH_MA_h.png"))
    plt.close(fig)
    del fig
    gc.collect()


'''FUNCTIONALITIES'''
def main():
    if cc_reeval == 'y':
        _11_initial_csv_check.create_folders(directory_reports, folders=['08_CC_ReEval'])
        cc_reeval_folder_path = os.path.join(directory_reports, '08_CC_ReEval')

        _11_initial_csv_check.copy_files(directory, cc_reeval_folder_path, file_name='h_comb_CC.csv')
        _11_initial_csv_check.copy_files(directory_others, cc_reeval_folder_path, file_name='prediction_inputs_all.csv')

        additional_data_path = os.path.join(cc_reeval_folder_path, 'h_comb_CC.csv')
        prediction_inputs_all_path = os.path.join(cc_reeval_folder_path, 'prediction_inputs_all.csv')

        generate_demand_files(prediction_inputs_all_path, demands_list, optimization_schedule, additional_data_path)
        folder_paths = process_csv(cc_reeval_folder_path, ['h_comb_CC.csv', 'prediction_inputs_all.csv'])

        for folder_path in folder_paths:
            report_filename = 'CC_re-eval_report.txt'
            output_log_file = os.path.join(folder_path, report_filename)
            _11_initial_csv_check.create_folders(folder_path, folders=['01_overview', '02_psychrometric'])
            directory_graphs_CC_re_01 = os.path.join(folder_path, '01_overview')
            directory_graphs_CC_re_02 = os.path.join(folder_path, '02_psychrometric')

            _11_initial_csv_check.create_folders(directory_graphs_CC_re_01, folders=['MA_daily', 'MA_hourly'])
            _11_initial_csv_check.move_files(folder_path, os.path.join(directory_graphs_CC_re_01, 'MA_daily'),
                                             filename_include='_MA_d')
            _11_initial_csv_check.move_files(folder_path, os.path.join(directory_graphs_CC_re_01, 'MA_hourly'),
                                             filename_include='_MA_h')
            _11_initial_csv_check.move_files(folder_path, directory_graphs_CC_re_02,
                                                         file_extension='*.png',
                                                         filename_include='psychrometric_')
            _11_initial_csv_check.move_files(folder_path, directory_graphs_CC_re_01,
                                             file_extension='*.png')
    else:
        print('no CC re-evaluation.')


'''APPLICATION'''
if __name__ == "__main__":
    main()
