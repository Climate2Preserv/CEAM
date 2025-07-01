import os
import pandas as pd
import holidays
import shutil
import importlib
import re
import numpy as np
from _00_CEAM_run import (directory, known_format, optimization_schedule, operation_schedule,
                          solar_radiation, country_name, demands_list)
from _07_CC_file import climate_classes


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
country_abbreviations = {
    'Albania': 'AL', 'Algeria': 'DZ', 'Argentina': 'AR', 'Armenia': 'AM', 'Australia': 'AU', 'Austria': 'AT', 'Azerbaijan': 'AZ',
    'Bangladesh': 'BD', 'Belarus': 'BY', 'Belgium': 'BE', 'Bolivia': 'BO', 'Bosnia and Herzegovina': 'BA', 'Brazil': 'BR', 'Bulgaria': 'BG',
    'Canada': 'CA', 'Chile': 'CL', 'China': 'CN', 'Colombia': 'CO', 'Costa Rica': 'CR', 'Croatia': 'HR', 'Czech Republic': 'CZ', 'Czechia': 'CZ',
    'Denmark': 'DK', 'Dominican Republic': 'DO',
    'Ecuador': 'EC', 'Egypt': 'EG', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR',
    'Georgia': 'GE', 'Germany': 'DE', 'Greece': 'GR',
    'Honduras': 'HN', 'Hong Kong': 'HK', 'Hungary': 'HU',
    'Iceland': 'IS', 'India': 'IN', 'Indonesia': 'ID', 'Ireland': 'IE', 'Israel': 'IL', 'Italy': 'IT',
    'Jamaica': 'JM', 'Japan': 'JP',
    'Kazakhstan': 'KZ', 'Kenya': 'KE', 'Kyrgyzstan': 'KG',
    'Latvia': 'LV', 'Liechtenstein': 'LI', 'Lithuania': 'LT', 'Luxembourg': 'LU',
    'Malaysia': 'MY', 'Malta': 'MT', 'Mexico': 'MX', 'Moldova': 'MD', 'Mongolia': 'MN', 'Morocco': 'MA',
    'Netherlands': 'NL', 'New Zealand': 'NZ', 'Nicaragua': 'NI', 'Nigeria': 'NG', 'North Macedonia': 'MK', 'Norway': 'NO',
    'Pakistan': 'PK', 'Panama': 'PA', 'Paraguay': 'PY', 'Peru': 'PE', 'Philippines': 'PH', 'Poland': 'PL', 'Portugal': 'PT',
    'Romania': 'RO', 'Russia': 'RU',
    'San Marino': 'SM', 'Saudi Arabia': 'SA', 'Serbia': 'RS', 'Singapore': 'SG', 'Slovakia': 'SK', 'Slovenia': 'SI', 'South Africa': 'ZA', 'South Korea': 'KR', 'Spain': 'ES', 'Sri Lanka': 'LK', 'Sweden': 'SE', 'Switzerland': 'CH',
    'Taiwan': 'TW', 'Thailand': 'TH', 'Tunisia': 'TN', 'Turkey': 'TR',
    'Ukraine': 'UA', 'United Arab Emirates': 'AE', 'United Kingdom': 'GB', 'United States': 'US', 'Uruguay': 'UY', 'Uzbekistan': 'UZ',
    'Venezuela': 'VE', 'Vietnam': 'VN',
    'Zambia': 'ZM', 'Zimbabwe': 'ZW'
}

# some constants
E0 = 0.611      # unit [kPa]
L_Rv = 5423     # unit [K]
T0 = 273.15     # unit [K]
RW = 461.5      # specific gas constant for water vapor [J/(kg*K)]
PC = 22.064e6   # critical pressure for water [Pa]
TC = 647.096    # critical temperature for water [K]
A_CONSTANTS = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502,]  # empirical constants [-]

all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
exclude_columns = all_demands.difference(demands_list)

def get_country_symbol(country_name):
    '''function to get the country symbol'''
    country_name = country_name.strip()         # remove leading/trailing spaces
    country_name_upper = country_name.upper()   # convert to uppercase

    # check if country_name is a key in country_abbreviations
    if country_name_upper in country_abbreviations:
        return country_abbreviations[country_name_upper]
    # check if country_name is a value in country_abbreviations
    elif country_name_upper in country_abbreviations.values():
        return country_name_upper
    else:
        return 'BE'     # default to 'BE' if not found

country_symbol = get_country_symbol(country_name)

def calculate_es(T):
    '''function to calculate saturation vapor pressure'''
    T_kelvin = T + T0
    Es = E0 * np.exp((L_Rv) * (1 / T0 - 1 / T_kelvin))
    return Es

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
    return PC * np.exp((TC / temp_kelvin) * exponent_term)      # saturation vapor pressure [Pa]

def extract_columns(input_file_path):
    '''function creating the initial file for predictions with working_time column'''
    if solar_radiation == 'y':
        demand_columns = {
            'D1': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D1', 'HDD', 'deltaT', 'TIST', 'Ti_w', 'RHi_m_ave'],
            'D2': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D2', 'CDD', 'deltaT', 'TIST', 'Twbi', 'Twbe', 'deltaTwb',
                   'TISTwb', 'Ti_w', 'RHi_m_ave'],
            'D4': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D4', 'deltaRH', 'AbsHi', 'AbsHe', 'deltaAbsH', 'Ti_w', 'RHi_m_ave'],
            'D5': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D5', 'HDD', 'CDD', 'deltaT', 'TIST', 'Twbi', 'Twbe', 'deltaTwb',
                   'TISTwb', 'Tdi', 'Tde', 'deltaTd', 'TISTd', 'Ti_w', 'RHi_m_ave']
        }
    else:
        demand_columns = {
            'D1': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D1', 'HDD', 'deltaT', 'Ti_w', 'RHi_m_ave'],
            'D2': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D2', 'CDD', 'deltaT', 'Twbi', 'Twbe', 'deltaTwb', 'Ti_w', 'RHi_m_ave'],
            'D4': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D4', 'deltaRH', 'AbsHi', 'AbsHe', 'deltaAbsH', 'Ti_w', 'RHi_m_ave'],
            'D5': ['Time', 'Te', 'RHe', 'Ti', 'RHi', 'D5', 'HDD', 'CDD', 'deltaT', 'Twbi', 'Twbe', 'deltaTwb', 'Tdi',
                   'Tde', 'deltaTd', 'Ti_w', 'RHi_m_ave']
        }

    # initialize required_columns with unique values from selected demands
    required_columns = set()
    for demand in demands_list:
        if demand in demand_columns:
            required_columns.update(demand_columns[demand])

    # add solar radiation-related columns if applicable
    if solar_radiation == 'y':
        required_columns.update(['Is'])

    required_columns = list(required_columns)

    df = pd.read_csv(input_file_path)   # read the input csv
    df = df[required_columns]           # drop columns
    df['Time'] = pd.to_datetime(df['Time'], format=known_format, errors='coerce')

    na_rows = df[df['Time'].isna()]     # rows verification with NaT values
    if not na_rows.empty:
        print("Rows with remaining conversion issues:\n", na_rows)
        raise ValueError(
            "There are remaining NaT values in the 'Time' column after conversion. Please check the input format.")

    df['week_day'] = df['Time'].dt.day_name()       # extract weekday names
    df['working_time'] = 0                          # initialize 'working_time' column with 0

    # populate 'working_time' based on operation_schedule
    for schedule in operation_schedule:
        operation_day = schedule['Operation day']
        from_hour = int(schedule['from'])
        till_hour = int(schedule['till'])

        mask = (df['week_day'] == operation_day) & \
               (df['Time'].dt.hour >= from_hour) & \
               (df['Time'].dt.hour < till_hour)

        df.loc[mask, 'working_time'] = 1            # set 'working_time' to 1 for rows matching the schedule

    try:
        print("Country symbol:", country_symbol)
        if country_symbol:
            unique_years = df['Time'].dt.year.unique()      # extract unique years from the 'Time' column
            print("Years in data:", unique_years)
            country_holidays = getattr(holidays, country_symbol)(years=unique_years)    # fetch holidays
            print("Holidays object for", country_name, ":", country_holidays)
            if country_holidays:
                print("Holidays in", country_name, ":")
                for date, name in sorted(country_holidays.items()):
                    print(date, "-", name)
            else:
                print("No holidays found for", country_name)
            # apply holiday logic to the df
            df['holiday'] = df['Time'].dt.date.apply(lambda date: 'holiday' if date in country_holidays else 'day')
        else:
            raise ValueError(f"Country abbreviation for '{country_name}' not found.")
    except Exception as e:
        print(f"Error in fetching holidays: {e}")
        df['holiday'] = 'day'

    df['Time'] = df['Time'].dt.strftime(known_format)       # format the 'Time' column back to the known_format
    df.to_csv(input_file_path, index=False)                 # save the extracted columns
    print(f"Extracted columns saved to {input_file_path}")

def enforce_temperature_limits(df, column, min_temp, max_temp):
    '''supporting function to enforce temperature limits'''
    if min_temp is not None:
        df[column] = df[column].apply(lambda x: max(x, min_temp))
    if max_temp is not None:
        df[column] = df[column].apply(lambda x: min(x, max_temp))
    return df

def enforce_humidity_limits(df, column, min_humi, max_humi):
    '''supporting function to enforce temperature limits'''
    if min_humi is not None:
        df[column] = df[column].apply(lambda x: max(x, min_humi))
    if max_humi is not None:
        df[column] = df[column].apply(lambda x: min(x, max_humi))
    return df

def verification_columns(df):
    '''generate additional verification columns for prediction inputs'''
    def conditional_binary_neg(value):
        if pd.isna(value):
            return value
        return -1 if value > 0 else 0

    def conditional_binary_pos(value):
        if pd.isna(value):
            return value
        return 1 if value > 0 else 0

    def conditional_complex(row):
        if pd.isna(row['D5']):
            return row['D5']
        elif row['D5'] > 0:
            if row['HDD'] > 0 and row['CDD'] == 0:
                return -1
            elif row['HDD'] == 0 and row['CDD'] > 0:
                return 1
            else:
                return 0
        else:
            return 0

    def conditional_rh_D4(row, rhi_avg):
        if pd.isna(row['D4']) or pd.isna(row['RHi']):
            return pd.NA
        if row['D4'] == 0:
            return 0
        if row['RHi'] < rhi_avg:
            return -1
        elif row['RHi'] > rhi_avg:
            return 1
        else:
            return 0

    def conditional_rh_D5(row, rhi_avg):
        if pd.isna(row['D5']) or pd.isna(row['RHi']):
            return pd.NA
        if row['D5'] == 0:
            return 0
        if row['RHi'] < rhi_avg:
            return -1
        elif row['RHi'] > rhi_avg:
            return 1
        else:
            return 0

    rhi_avg = df['RHi'].mean(skipna=True)
    print(rhi_avg)

    if 'D1' not in exclude_columns:
        df['neg_T-D1'] = df['D1'].apply(conditional_binary_neg)
    if 'D2' not in exclude_columns:
        df['pos_T-D2'] = df['D2'].apply(conditional_binary_pos)
    if 'D4' not in exclude_columns:
        df['mix_RH-D4'] = df.apply(lambda row: conditional_rh_D4(row, rhi_avg), axis=1)
    if 'D5' not in exclude_columns and 'D4' not in exclude_columns:
        df['mix_T-D5'] = df.apply(conditional_complex, axis=1)
        df['mix_RH-D4'] = df.apply(lambda row: conditional_rh_D4(row, rhi_avg), axis=1)
        df['mix_RH-D5'] = df.apply(lambda row: conditional_rh_D5(row, rhi_avg), axis=1)
    if 'D5' not in exclude_columns:
        df['mix_T-D5'] = df.apply(conditional_complex, axis=1)
        df['mix_RH-D5'] = df.apply(lambda row: conditional_rh_D5(row, rhi_avg), axis=1)
    return df

def modify_column_T(input_file_path, class_name, column, scenarios=optimization_schedule):
    '''function to modify the input file for predictions based on Ti for different schedules'''

    df = pd.read_csv(input_file_path)
    print("Initial DataFrame 'Time' column type in modify_column_T:", df['Time'].dtype)

    df['Time'] = pd.to_datetime(df['Time'], format=known_format, errors='coerce')
    na_rows = df[df['Time'].isna()]
    if not na_rows.empty:
        print("Rows with remaining conversion issues in modify_column_T:\n", na_rows)
        raise ValueError("There are remaining NaT values in the 'Time' column after conversion.")

    climate_class = next((cls for cls in climate_classes if cls['name'] == class_name), None)
    if not climate_class:
        raise ValueError(f"Invalid class name: {class_name}")

    Ti_short_offset = climate_class['Ti_short_offset']
    Ti_outer_min = climate_class['Ti_outer_min']
    Ti_outer_max = climate_class['Ti_outer_max']

    headers = []
    if 'D1' not in exclude_columns:
        headers.append('neg_T-D1')
    if 'D2' not in exclude_columns:
        headers.append('pos_T-D2')
    if 'D5' not in exclude_columns:
        headers.append('mix_T-D5')

    generated_columns = []

    def generate_column_name(header, offset, scenario):
        if header.startswith('neg_T-'):
            prefix = 'Heating'
        elif header.startswith('pos_T-'):
            prefix = 'Cooling'
        elif header == 'mix_T-D5':
            prefix = 'Total'
        else:
            prefix = 'Unknown'
        return f"{prefix}_{header}_{offset}_{scenario}"

    def apply_scenario(row, column, offset, header, scenario):
        if scenario == 's1':
            return row[column] + offset * row[header]
        elif scenario == 's2':
            return row[column] + offset * row[header] if row['week_day'] not in ['Saturday', 'Sunday'] else row[column]
        elif scenario == 's3':
            return row[column] + offset * row[header] if row['week_day'] in ['Saturday', 'Sunday'] else row[column]
        elif scenario == 's4':
            return row[column] + offset * row[header] if row['working_time'] == 0 else row[column]
        elif scenario == 's5':
            return row[column] + offset * row[header] if row['holiday'] == 'holiday' or row['working_time'] == 0 else row[column]
        return row[column]

    for header in headers:
        for scenario in scenarios:
            new_column = generate_column_name(header, Ti_short_offset, scenario)
            df[new_column] = df.apply(lambda row: apply_scenario(row, 'Ti', Ti_short_offset, header, scenario), axis=1)
            df = enforce_temperature_limits(df, new_column, Ti_outer_min, Ti_outer_max)
            if isinstance(Ti_short_offset, (int, float)):
                df[new_column] = df.apply(
                    lambda row: max(min(row[new_column], row['Ti_w'] + Ti_short_offset), row['Ti_w'] - Ti_short_offset),
                    axis=1
                )
            generated_columns.append(new_column)

    df['Time'] = df['Time'].dt.strftime(known_format)
    df.to_csv(input_file_path, index=False)
    print(f"Modified columns saved to {input_file_path}")
    return generated_columns

def modify_column_RH(input_file_path, class_name, column, scenarios=optimization_schedule):
    '''function to modify the input file for predictions based on Ti and RHi for different schedules'''

    df = pd.read_csv(input_file_path)
    print("Initial DataFrame 'Time' column type in modify_column_RH:", df['Time'].dtype)

    df['Time'] = pd.to_datetime(df['Time'], format=known_format, errors='coerce')
    na_rows = df[df['Time'].isna()]
    if not na_rows.empty:
        print("Rows with remaining conversion issues in modify_column_RH:\n", na_rows)
        raise ValueError("There are remaining NaT values in the 'Time' column after conversion.")

    climate_class = next((cls for cls in climate_classes if cls['name'] == class_name), None)
    if not climate_class:
        raise ValueError(f"Invalid class name: {class_name}")

    RHi_short_offset = climate_class['RHi_short_offset']
    RHi_outer_min = climate_class['RHi_outer_min']
    RHi_outer_max = climate_class['RHi_outer_max']

    headers = []
    if 'D4' not in exclude_columns:
        headers.append('mix_RH-D4')
    if 'D5' not in exclude_columns:
        headers.append('mix_RH-D5')

    generated_columns = []

    def generate_column_name(header, offset, scenario):
        if header == 'mix_RH-D4':
            prefix = 'Humidify'
        elif header == 'mix_RH-D5':
            prefix = 'Humidify'
        else:
            prefix = 'Unknown'
        return f"{prefix}_{header}_{offset}_{scenario}"

    def apply_scenario(row, column, offset, header, scenario):
        if scenario == 's1':
            return row[column] + offset * row[header]
        elif scenario == 's2':
            return row[column] + offset * row[header] if row['week_day'] not in ['Saturday', 'Sunday'] else row[column]
        elif scenario == 's3':
            return row[column] + offset * row[header] if row['week_day'] in ['Saturday', 'Sunday'] else row[column]
        elif scenario == 's4':
            return row[column] + offset * row[header] if row['working_time'] == 0 else row[column]
        elif scenario == 's5':
            return row[column] + offset * row[header] if row['holiday'] == 'holiday' or row['working_time'] == 0 else row[column]
        return row[column]

    for header in headers:
        for scenario in scenarios:
            if 'RH' in header:
                new_column = generate_column_name(header, RHi_short_offset, scenario)
                df[new_column] = df.apply(lambda row: apply_scenario(row, 'RHi', RHi_short_offset, header, scenario), axis=1)
                df = enforce_humidity_limits(df, new_column, RHi_outer_min, RHi_outer_max)
                # additional check for short offset range
                if isinstance(RHi_short_offset, (int, float)):
                    df[new_column] = df.apply(
                        lambda row: max(min(row[new_column], row['RHi_m_ave'] + RHi_short_offset), row['RHi_m_ave'] - RHi_short_offset),
                        axis=1
                    )
            generated_columns.append(new_column)

    df['Time'] = df['Time'].dt.strftime(known_format)
    df.to_csv(input_file_path, index=False)
    print(f"Modified columns saved to {input_file_path}")
    return generated_columns

def save_filtered_csv(input_file, output_file_name, column_filters, directory,
                      remove_columns=None, remove_exact_columns=None):
    ''''function to save the filtered files for predicitons'''
    df = pd.read_csv(input_file)
    filtered_columns = (['Time'] + demands_list +
                        [col for col in df.columns if any(f in col for f in column_filters)])
    filtered_df = df[filtered_columns]

    if remove_exact_columns:
        pattern = r'\b(?:' + '|'.join(re.escape(rc) for rc in remove_exact_columns) + r')\b'
        filtered_df = filtered_df[[col for col in filtered_df.columns if not re.search(pattern, col)]]

    if remove_columns:
        filtered_df = filtered_df[[col for col in filtered_df.columns if not any(substr in col for substr in remove_columns)]]

    duplicates = filtered_df.columns[filtered_df.columns.duplicated()]
    if not duplicates.empty:
        filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated(keep='first')]

    output_file = os.path.join(directory, output_file_name)
    filtered_df.to_csv(output_file, index=False)
    print(f"{output_file_name} saved successfully in {directory} with columns: {filtered_df.columns.tolist()}")


'''FUNCTIONALITIES'''
def main():
    from _00_CEAM_run import optimization_schedule
    input_file = 'h_comb_CC.csv'
    directory_input = os.path.join(directory, input_file)
    _06_2_factors = importlib.import_module('_06_2_factors')
    _06_1_Td_Twb_calc = importlib.import_module('_06_1_Td_Twb_calc')
    _06_3_abs_humi = importlib.import_module('_06_3_abs_humi')

    copy_file_path = os.path.join(directory, 'prediction_inputs_all.csv')
    shutil.copyfile(directory_input, copy_file_path)
    print(f"Copied {input_file} to prediction_inputs_all.csv")
    print(optimization_schedule)

    try:
        extract_columns(copy_file_path)
    except ValueError as e:
        print(e)

    df = pd.read_csv(copy_file_path)
    df = verification_columns(df)

    if pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = df['Time'].dt.strftime(known_format)

    df.to_csv(copy_file_path, index=False)
    print(f"Additional columns generated and saved to {copy_file_path}")

    Ti_columns_1 = modify_column_T(copy_file_path, 'A1', 'Ti')
    Ti_columns_2 = modify_column_T(copy_file_path, 'B', 'Ti')
    RHi_columns_1 = modify_column_RH(copy_file_path, 'A1','RHi')
    RHi_columns_2 = modify_column_RH(copy_file_path, 'B','RHi')

    # symbols explanation :
    #   neg - negative offsets added;
    #   pos - positive offsets added;
    #   mix - both negative and positive offsets added;
    #   complex - both negative and positive offsets added, simultaneously for T and RH
    Twb_columns_all = ['Cooling_pos_Twb-D2_2_s1', 'Cooling_pos_Twb-D2_2_s2', 'Cooling_pos_Twb-D2_2_s3', 'Cooling_pos_Twb-D2_2_s4', 'Cooling_pos_Twb-D2_2_s5',
                    'Cooling_pos_Twb-D2_5_s1', 'Cooling_pos_Twb-D2_5_s2', 'Cooling_pos_Twb-D2_5_s3', 'Cooling_pos_Twb-D2_5_s4', 'Cooling_pos_Twb-D2_5_s5',
                    'Total_mix_Twb-D5_2_s1', 'Total_mix_Twb-D5_2_s2', 'Total_mix_Twb-D5_2_s3', 'Total_mix_Twb-D5_2_s4', 'Total_mix_Twb-D5_2_s5',
                    'Total_mix_Twb-D5_5_s1', 'Total_mix_Twb-D5_5_s2', 'Total_mix_Twb-D5_5_s3', 'Total_mix_Twb-D5_5_s4', 'Total_mix_Twb-D5_5_s5']

    Td_columns_all = ['Total_mix_Td-D5_2_s1', 'Total_mix_Td-D5_2_s2', 'Total_mix_Td-D5_2_s3', 'Total_mix_Td-D5_2_s4', 'Total_mix_Td-D5_2_s5',
                    'Total_mix_Td-D5_5_s1', 'Total_mix_Td-D5_5_s2', 'Total_mix_Td-D5_5_s3', 'Total_mix_Td-D5_5_s4', 'Total_mix_Td-D5_5_s5',
                    'Total_complex_Td_T2RH5_s1', 'Total_complex_Td_T2RH5_s2', 'Total_complex_Td_T2RH5_s3', 'Total_complex_Td_T2RH5_s4', 'Total_complex_Td_T2RH5_s5',
                    'Total_complex_Td_T5RH10_s1', 'Total_complex_Td_T5RH10_s2', 'Total_complex_Td_T5RH10_s3', 'Total_complex_Td_T5RH10_s4', 'Total_complex_Td_T5RH10_s5']

    AbsH_columns_all = ['Humidify_mix_AbsH-D4_5_s1', 'Humidify_mix_AbsH-D4_5_s2', 'Humidify_mix_AbsH-D4_5_s3', 'Humidify_mix_AbsH-D4_5_s4', 'Humidify_mix_AbsH-D4_5_s5',
                        'Humidify_mix_AbsH-D4_10_s1', 'Humidify_mix_AbsH-D4_10_s2', 'Humidify_mix_AbsH-D4_10_s3', 'Humidify_mix_AbsH-D4_10_s4', 'Humidify_mix_AbsH-D4_10_s5']


    df = pd.read_csv(copy_file_path)

    # ---- Twb ----
    T_columns = [col for col in df.columns if '_T-' in col and not col.startswith(('TIST_', 'deltaT'))]

    for T_col in T_columns:
        if 'RHi' in df.columns:
            # compute Twb and store in a new column
            Twb_col = T_col.replace('_T-', '_Twb-')
            df[Twb_col] = df.apply(lambda row: _06_1_Td_Twb_calc.calculate_single_twb(row[T_col], row['RHi']), axis=1)
            Td_col = T_col.replace('_T-', '_Td-')
            df[Td_col] = df.apply(lambda row: _06_1_Td_Twb_calc.calculate_single_td(row[T_col], row['RHi']), axis=1)

    # ---- AbsH ----
    RH_columns = [col for col in df.columns if '_RH-D4' in col and not col.startswith('deltaRH')]

    for RH_col in RH_columns:
        if 'Ti' in df.columns:
            AbsH_col = RH_col.replace('_RH', '_AbsH')  # Rename RH to AbsH
            df[AbsH_col] = df.apply(lambda row: _06_3_abs_humi.calculate_absolute_humidity(row['Ti'], row[RH_col]),
                                    axis=1)

    # ---- Td for 'Total_n/p_T-D5' ----
    Td_column_pairs = [
        ('Total_mix_T-D5_2_s1', 'Humidify_mix_RH-D5_5_s1', 'Total_complex_Td_T2RH5_s1'),
        ('Total_mix_T-D5_2_s2', 'Humidify_mix_RH-D5_5_s2', 'Total_complex_Td_T2RH5_s2'),
        ('Total_mix_T-D5_2_s3', 'Humidify_mix_RH-D5_5_s3', 'Total_complex_Td_T2RH5_s3'),
        ('Total_mix_T-D5_2_s4', 'Humidify_mix_RH-D5_5_s4', 'Total_complex_Td_T2RH5_s4'),
        ('Total_mix_T-D5_2_s5', 'Humidify_mix_RH-D5_5_s5', 'Total_complex_Td_T2RH5_s5'),
        ('Total_mix_T-D5_5_s1', 'Humidify_mix_RH-D5_10_s1', 'Total_complex_Td_T5RH10_s1'),
        ('Total_mix_T-D5_5_s2', 'Humidify_mix_RH-D5_10_s2', 'Total_complex_Td_T5RH10_s2'),
        ('Total_mix_T-D5_5_s3', 'Humidify_mix_RH-D5_10_s3', 'Total_complex_Td_T5RH10_s3'),
        ('Total_mix_T-D5_5_s4', 'Humidify_mix_RH-D5_10_s4', 'Total_complex_Td_T5RH10_s4'),
        ('Total_mix_T-D5_5_s5', 'Humidify_mix_RH-D5_10_s5', 'Total_complex_Td_T5RH10_s5')
    ]

    for T_col, RH_col, Td_col in Td_column_pairs:
        if T_col in df.columns and RH_col in df.columns:
            df[Td_col] = df.apply(lambda row: _06_1_Td_Twb_calc.calculate_single_td(row[T_col], row[RH_col]),
                                  axis=1)
        else:
            print(f"Skipping {Td_col} calculation due to missing columns: {T_col} or {RH_col}")

    # exclude specific columns from the final saved dataset
    excluded_columns = ['neg_Twb-D1', 'pos_Twb-D2', 'mix_Twb-D5',
                        'neg_Td-D1', 'pos_Td-D2', 'mix_Td-D5',
                        'mix_AbsH-D4']
    df = df.drop(columns=[col for col in excluded_columns if col in df.columns], errors='ignore')
    df.to_csv(copy_file_path, index=False)
    print(f"Updated dataset columns and exclusions saved to {copy_file_path}")

    if solar_radiation == 'y':
        for Ti_1_param in Ti_columns_1:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[Ti_1_param, 'Te', None, None, None, None, None, None, 'Is'])
        for Ti_2_param in Ti_columns_2:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[Ti_2_param, 'Te', None, None, None, None, None, None, 'Is'])
        for RHi_1_param in RHi_columns_1:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, RHi_1_param, 'RHe', None, None, None, None, None])
        for RHi_2_param in RHi_columns_2:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, RHi_2_param, 'RHe', None, None, None, None, None])
        for Twb_param in Twb_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, None, None, None, None, Twb_param, 'Twbe', 'Is'])
        for Td_param in Td_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, None, None, Td_param, 'Tde', None, None, 'Is'])
        for AbsH_param in AbsH_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, AbsH_param, 'AbsHe', None, None, None, None, None])

        # do input files
        save_filtered_csv(copy_file_path, 'inputs_D1.csv', ['Te', 'Ti', 'D1', 'deltaT', 'TIST', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Twb', 'Td', 'Cooling', 'Total'],
                          remove_exact_columns=['D2', 'D3', 'D4', 'D5', 'neg_T-D1'])
        save_filtered_csv(copy_file_path, 'inputs_D2.csv', ['Te', 'Ti', 'Twbe', 'Twb_i', 'D2', 'deltaT', 'deltaTwb', 'TIST', 'TISTwb', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Td', 'Heating', 'Total'],
                          remove_exact_columns=['D1', 'D3', 'D4', 'D5', 'pos_T-D2'])
        save_filtered_csv(copy_file_path, 'inputs_D4.csv', ['RHe', 'RHi', 'AbsHe', 'AbsHi', 'D4', 'deltaRH', 'deltaAbsH', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['mix_RH-D5'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D5', 'mix_RH-D4', 'mix_RH-D5'])
        save_filtered_csv(copy_file_path, 'inputs_D5a.csv', ['Te', 'Ti', 'Twbe', 'Twbi', 'D5', 'deltaT', 'deltaTwb', 'TIST', 'TISTwb', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Humidify', 'Heating', 'Cooling', 'Td'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D4', 'mix_T-D5', 'mix_RH-D5'])
        save_filtered_csv(copy_file_path, 'inputs_D5b.csv', ['Tde', 'Tdi', 'D5', 'Total_complex', 'deltaTd', 'TISTd', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Humidify', 'Heating', 'Cooling', '_T-', 'Twb'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D4', 'mix_T-D5', 'mix_RH-D5'])
    else:
        for Ti_1_param in Ti_columns_1:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[Ti_1_param, 'Te', None, None, None, None, None, None, None])
        for Ti_2_param in Ti_columns_2:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[Ti_2_param, 'Te', None, None, None, None, None, None, None])
        for RHi_1_param in RHi_columns_1:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, RHi_1_param, 'RHe', None, None, None, None, None])
        for RHi_2_param in RHi_columns_2:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, RHi_2_param, 'RHe', None, None, None, None, None])
        for Twb_param in Twb_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, None, None, None, None, Twb_param, 'Twbe', None])
        for Td_param in Td_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, None, None, Td_param, 'Tde', None, None, None])
        for AbsH_param in AbsH_columns_all:
            _06_2_factors.process_factors_param(directory, filename='prediction_inputs_all.csv',
                                                parameters=[None, None, AbsH_param, 'AbsHe', None, None, None, None, None])

        # do input files
        save_filtered_csv(copy_file_path, 'inputs_D1.csv', ['Te', 'Ti', 'D1', 'deltaT', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Twb', 'Td', 'Cooling', 'Total', 'TIS'],
                          remove_exact_columns=['D2', 'D3', 'D4', 'D5', 'neg_T-D1'])
        save_filtered_csv(copy_file_path, 'inputs_D2.csv', ['Te', 'Ti', 'Twbe', 'Twb_i', 'D2', 'deltaT', 'deltaTwb', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Td', 'Heating', 'Total', 'TIS'],
                          remove_exact_columns=['D1', 'D3', 'D4', 'D5', 'pos_T-D2'])
        save_filtered_csv(copy_file_path, 'inputs_D4.csv', ['RHe', 'RHi', 'AbsHe', 'AbsHi', 'D4', 'deltaRH', 'deltaAbsH', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['mix_RH-D5'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D5', 'mix_RH-D4', 'mix_RH-D5'])
        save_filtered_csv(copy_file_path, 'inputs_D5a.csv', ['Te', 'Ti', 'Twbe', 'Twbi', 'D5', 'deltaT', 'deltaTwb', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Humidify', 'Heating', 'Cooling', 'Td', 'TIS'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D4', 'mix_T-D5', 'mix_RH-D5'])
        save_filtered_csv(copy_file_path, 'inputs_D5b.csv', ['Tde', 'Tdi', 'D5', 'Total_complex', 'deltaTd', 'Ti_w', 'RHi_m_ave'], directory,
                          remove_columns=['Humidify', 'Heating', 'Cooling', '_T-', 'Twb', 'TIS'],
                          remove_exact_columns=['D1', 'D2', 'D3', 'D4', 'mix_T-D5', 'mix_RH-D5'])


'''APPLICATION'''
if __name__ == "__main__":
    main()
