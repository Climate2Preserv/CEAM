import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import importlib         # import function from other codes
from datetime import datetime
from _00_CEAM_run import directory
from _11_initial_csv_check import directory_graphs_05, directory_others
from _18_1_best_predictions import prediction_selected_path


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
_11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
w_comb_path = os.path.join(directory, 'w_comb.csv')


'''FUNCTIONALITIES'''
def main():
    def extract_scenario_label(header):
        '''extracts scenario label from column header'''
        parts = header.split('_')
        scenario = None
        for part in parts:
            if part.startswith('s') and part[1:].isdigit():
                scenario = part
                break

        # check for additional scenario identifiers
        if scenario:
            for identifier in ['T2RH5', 'T5RH10']:
                if identifier in header:
                    return f'{scenario}_{identifier}_pred'
            for identifier in ['2', '5', '10']:
                if identifier in header:
                    return f'{scenario}_{identifier}_pred'

        return header

    def convert_datetime_column(df):
        '''method to convert datetime column'''
        # list of possible datetime formats to try
        datetime_formats = [
            "%d.%m.%Y %H:%M",       # OG format
            "%Y-%m-%d %H:%M:%S",    # ISO format
            "%d-%m-%Y %H:%M",       # alternative format
            "%m/%d/%Y %H:%M",       # US format
        ]

        time_column = df.columns[0]
        df[time_column] = df[time_column].str.strip()

        for fmt in datetime_formats:
            try:
                converted_dates = pd.to_datetime(df[time_column], format=fmt, dayfirst=True, errors='raise')
                df[time_column] = converted_dates
                break
            except (ValueError, TypeError):
                continue

        if df[time_column].dtype == 'object':
            try:
                df[time_column] = pd.to_datetime(df[time_column], dayfirst=True, infer_datetime_format=True)
            except Exception as e:
                print(f"Failed to convert datetime column: {e}")
                return None

        return df

    def extract_weekly_demand_distribution(df, start_time, end_time):
        '''filters the df to extract data for the week defined by the start and end times for plotting'''
        # filter the data based on the time range (start_time to end_time)
        weekly_data = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= end_time)]

        if weekly_data.empty:
            print("No data found for the specified week.")
            return None

        return weekly_data

    def highlight_missing_periods(ax, time_data, column_data, facecolor='#FAF3D3', alpha=0.75):
        '''highlight periods with missing values'''
        if column_data.isna().sum() == 0:
            return

        missing_periods = []
        start_time = None

        time_data = pd.Series(time_data)
        column_data = pd.Series(column_data)

        for i in range(len(column_data)):
            if pd.isna(column_data.iloc[i]):
                if start_time is None:
                    start_time = time_data.iloc[i - 1] if i > 0 else time_data.iloc[i]
            elif start_time is not None:
                end_time = time_data.iloc[i]
                missing_periods.append((start_time, end_time))
                start_time = None

        if start_time is not None:
            missing_periods.append((start_time, time_data.iloc[-1]))

        for start, end in missing_periods:
            try:
                ax.axvspan(start, end, facecolor=facecolor, alpha=alpha)
            except Exception as e:
                print(f"Error highlighting missing period from {start} to {end}: {e}")

    def create_energy_demand_plot(file_path, output_dir, w_comb_path):
        '''generates a scatter plot for energy demand data'''
        df = pd.read_csv(file_path)
        df = convert_datetime_column(df)

        if df is None:
            print(f"Skipping {file_path} due to datetime conversion failure")
            return

        w_comb_df = pd.read_csv(w_comb_path)

        # determine the correct 'D' column
        file_name = os.path.basename(file_path)
        d_column = None
        for col in w_comb_df.columns:
            if col.startswith('D') and col in file_name:
                d_column = col
                break

        if not d_column:
            print(f"Cannot find a matching 'D' column for {file_name} in {w_comb_path}. Using default D1.")
            d_column = 'D1'

        # find and retrieve the 'start' and 'end' records
        max_d_index = w_comb_df[d_column].idxmax()
        start_record = w_comb_df.iloc[max_d_index]['start']
        end_record = w_comb_df.iloc[max_d_index]['end']

        # convert start and end records to datetime
        start_time = pd.to_datetime(start_record, format="%d.%m.%Y %H:%M", dayfirst=True)
        end_time = pd.to_datetime(end_record, format="%d.%m.%Y %H:%M", dayfirst=True)

        # get the weekly demand distribution data
        weekly_data = extract_weekly_demand_distribution(df, start_time, end_time)

        # verification
        if weekly_data is None:
            return
        if df.iloc[:, 0].dtype == 'object':
            df.iloc[:, 0] = df.iloc[:, 0].str.strip()
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='raise')   # conversion to datetime64

        # verify if the column is properly converted
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            print("The 'Time' column is still not of datetime type.")
            return

        time_data = df.iloc[:, 0]
        column_names = df.columns[1:]

        # extract the day with the highest demand
        highest_demand_day = df.iloc[:, 1].idxmax()
        highest_demand_day_time = df.iloc[highest_demand_day, 0]

        day_data = df[df.iloc[:, 0].dt.date == highest_demand_day_time.date()]

        # create sub-directories for different scenarios
        for col in column_names:
            # extract scenario from column name
            scenario = None
            parts = col.split('_')
            for part in parts:
                if part.startswith('s') and part[1:].isdigit():
                    scenario = part
                    break
            if scenario:
                is_complex = any(tag in col for tag in ['T2RH5', 'T5RH10'])
                scenario_dir = os.path.join(output_dir, scenario)
                if not os.path.exists(scenario_dir):
                    os.makedirs(scenario_dir)
                file_suffix = '_complex' if is_complex else ''
                output_file = os.path.join(
                    scenario_dir,
                    os.path.basename(file_path).replace('.csv', f'_{scenario}{file_suffix}.png')
                )

                # include the base consumption, base prediction, and the current scenario's columns
                relevant_columns = [df.columns[1], df.columns[2]] + [
                    col for col in column_names if
                    scenario in col and (is_complex == any(tag in col for tag in ['T2RH5', 'T5RH10']))
                ]

                # create figure with 2 rows and 2 columns
                fig = plt.figure(figsize=(24, 12))
                ax_main = plt.subplot(2, 1, 1)      # main plot (upper row)
                ax_zoomed = plt.subplot(2, 2, 4)    # zoomed plot (lower right)
                ax_weekly = plt.subplot(2, 2, 3)    # weekly plot (lower left)

                ax_blank_2 = plt.subplot(2, 2, 2)
                ax_blank_2.axis('off')

                colors = ["#D3D3D3", "#77A5C9"] + ["#8FEBB4", "#218042"]

                # main plot
                for i, col in enumerate(relevant_columns):
                    label = extract_scenario_label(col)
                    ax_main.plot(time_data, df[col], linestyle='-', linewidth=1.5, color=colors[i],
                                 marker='o', markersize=1.5, markerfacecolor=colors[i], alpha=0.75, label=label)
                    baseline_columns = ['D1', 'D2', 'D4', 'D5']
                    if col in baseline_columns:
                        try:
                            highlight_missing_periods(ax_main, time_data, df[col])
                        except Exception as e:
                            print(f"Highlight error for {col}: {e}")

                ax_main.set_xlabel('Time', fontsize=14, fontweight='bold')
                ax_main.set_ylabel('Energy Demand [Wh]', fontsize=14, fontweight='bold')
                ax_main.set_title(f"Data-frame Distribution", fontsize=16, fontweight='bold')
                ax_main.set_xlim(time_data.min(), time_data.max())
                ax_main.set_ylim(bottom=0)
                ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
                ax_main.grid(True, linestyle='--', linewidth=0.5, color='grey')
                ax_main.legend(fontsize=16, loc='best')

                # zoomed plot (lower left)
                for i, col in enumerate(relevant_columns):
                    label = extract_scenario_label(col)
                    ax_zoomed.plot(day_data.iloc[:, 0], day_data[col], linestyle='-', linewidth=2.5,
                                   color=colors[i], marker='o', markersize=7.5,
                                   markeredgewidth=1.0, markerfacecolor=colors[i], markeredgecolor='black',
                                   alpha=0.75, label=label)
                    baseline_columns = ['D1', 'D2', 'D4', 'D5']
                    if col in baseline_columns:
                        try:
                            highlight_missing_periods(ax_zoomed, time_data, df[col])
                        except Exception as e:
                            print(f"Highlight error for {col}: {e}")

                ax_zoomed.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax_zoomed.set_ylabel('Energy Demand [Wh]', fontsize=12, fontweight='bold')
                ax_zoomed.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax_zoomed.set_xlim(day_data.iloc[:, 0].min(), day_data.iloc[:, 0].max())
                ax_zoomed.set_ylim(bottom=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                ax_zoomed.grid(True, linestyle='--', linewidth=0.5, color='grey')
                ax_zoomed.axvline(x=highest_demand_day_time,
                                  color='red', linestyle='--', linewidth=1.0, label="Peak demand")
                ax_zoomed.set_title(f"Peak Demand (recorded on: {highest_demand_day_time.strftime('%d.%m.%Y')})",
                                    fontsize=14, fontweight='bold')
                ax_zoomed.legend(fontsize=12, loc='best')

                # weekly demand distribution plot (lower right)
                for i, col in enumerate(relevant_columns):
                    label = extract_scenario_label(col)
                    ax_weekly.plot(weekly_data.iloc[:, 0], weekly_data[col], linestyle='-', linewidth=1.5,
                                   color=colors[i], marker='o', markersize=5.0,
                                   markeredgewidth=1.0, markerfacecolor=colors[i], markeredgecolor='black',
                                   alpha=0.75, label=label)
                    baseline_columns = ['D1', 'D2', 'D4', 'D5']
                    if col in baseline_columns:
                        try:
                            highlight_missing_periods(ax_weekly, time_data, df[col])
                        except Exception as e:
                            print(f"Highlight error for {col}: {e}")

                ax_weekly.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax_weekly.set_ylabel('Energy Demand [Wh]', fontsize=12, fontweight='bold')
                ax_weekly.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                ax_weekly.set_xlim(weekly_data.iloc[:, 0].min(), weekly_data.iloc[:, 0].max())
                ax_weekly.set_ylim(bottom=0)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                ax_weekly.grid(True, linestyle='--', linewidth=0.5, color='grey')
                ax_weekly.set_title("Week Distribution", fontsize=14, fontweight='bold')
                ax_weekly.legend(fontsize=12, loc='best')

                plt.tight_layout()
                plt.savefig(output_file, dpi=300)
                plt.close()

    def process_directory(input_dir, output_dir, w_comb_path):
        '''processes all CSV files in a directory and generates plots'''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in os.listdir(input_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(input_dir, file_name)
                create_energy_demand_plot(file_path, output_dir, w_comb_path)

    process_directory(prediction_selected_path, prediction_selected_path, w_comb_path)

    _11_initial_csv_check.move_files(prediction_selected_path, directory_graphs_05,
                                     move_folder=['s1', 's2', 's3', 's4', 's5'])
    _11_initial_csv_check.move_files(directory, directory_others, move_folder=['predictions_selected'])


'''APPLICATION'''
if __name__ == "__main__":
    main()
