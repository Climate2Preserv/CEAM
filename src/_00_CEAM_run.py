# CEAM (Climate and Energy Assessment of Museums)
# Copyright (C) 2025 Marcin Zygmunt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
import os
import shutil           # to handle file operations
import importlib        # import function from other codes
from datetime import datetime

''''GLOBAL ARGUMENTS AND FUNCTIONS'''
# initial variables
directory = ''
date_format_icd = ''
date_format_ecd = ''
date_format_ed = ''
frequency_icd = ''
frequency_ecd = ''
frequency_ed = ''
scripts = []
T_HDD = None
T_CDD = None
units = None
solar_radiation = ''
cc_reeval = ''
operation_schedule = []
optimization_schedule = []
optimizers_list = []
demands_list = []
country_name = ''
city_name = ''
known_format = '%d.%m.%Y %H:%M'
outlier_value = None
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')  # for greek symbols


'''FUNCTIONALITIES'''
def run_main_logic():
    global directory, T_HDD, T_CDD, units, solar_radiation, cc_reeval, operation_schedule, \
        date_format_icd, date_format_ecd, date_format_ed, \
        optimization_schedule, demands_list, optimizers_list, country_name, city_name, outlier_value, \
        frequency_icd, frequency_ecd, frequency_ed

    # clear the inputs defined as lists each time the run_main_logic (aka the software) runs
    scripts.clear()
    operation_schedule.clear()
    optimization_schedule.clear()
    demands_list.clear()
    optimizers_list.clear()

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")     # path to the desktop
    var_file_path = os.path.join(desktop_path, "var.txt")               # initial var.txt loacalization

    # verification and usage of the var.txt
    if not os.path.isfile(var_file_path):                               # if exist
        print(f"File not found: {var_file_path}")
        return

    with open(var_file_path, 'r') as file:                              # reading
        for line in file:
            if line.startswith("Working directory: "):
                directory = line.split("Working directory: ")[1].strip()    # getting the working directory
                break

    if directory:                                                       # move the var.txt file to working directory
        os.makedirs(directory, exist_ok=True)
        new_file_path = os.path.join(directory, "var.txt")
        shutil.move(var_file_path, new_file_path)
    else:
        print("No working directory found in var.txt")

    # list of scripts required for each module
    scripts_module_0 = ['_11_initial_csv_check', '_01_convert_units', '_02_pv_calc', '_03_degree_days',
                        '_12_data_overview_stats', '_04_extreme_weeks', '_05_1_combi_files',
                        '_05_2_adjust_combi_files', '_06_1_Td_Twb_calc', '_06_2_factors','_06_3_abs_humi']
    scripts_module_1 = ['_13_data_overview_plots']
    scripts_module_2 = ['_14_correlation']
    scripts_module_3 = ['_15_energy_signature']
    scripts_module_4 = ['_07_CC_file', '_16_CC_overview', '_08_psychrometric']
    scripts_module_5 = ['_09_prediction_inputs', '_17_predictions', '_18_1_best_predictions',
                        '_18_2_best_predictions_graphs', '_18_3_best_predictions_sorting', '_19_cc_reeval']

    module_scripts = {
        0: scripts_module_0,
        1: scripts_module_1,
        2: scripts_module_2,
        3: scripts_module_3,
        4: scripts_module_4,
        5: scripts_module_5
    }

    # load the user inputs from var.txt
    try:
        with open(os.path.join(directory, 'var.txt'), 'r') as file:
            for line in file:
                if line.startswith("ICD_format: "):
                    date_format_icd = line.split("ICD_format: ")[1].strip()
                elif line.startswith("ECD_format: "):
                    date_format_ecd = line.split("ECD_format: ")[1].strip()
                elif line.startswith("ED_format: "):
                    date_format_ed = line.split("ED_format: ")[1].strip()
                elif line.startswith("ICD_freq: "):
                    frequency_icd = line.split("ICD_freq: ")[1].strip()
                elif line.startswith("ECD_freq: "):
                    frequency_ecd = line.split("ECD_freq: ")[1].strip()
                elif line.startswith("ED_freq: "):
                    frequency_ed = line.split("ED_freq: ")[1].strip()
                elif line.startswith("Module "):
                    parts = line.split(":")
                    module_number = int(parts[0].replace("Module ", "").strip())
                    status = parts[1].strip().lower()
                    if status == 'on' and module_number in module_scripts:
                        for script in module_scripts[module_number]:
                            if script not in scripts:
                                scripts.append(script)
                elif line.startswith("Units: "):
                    units = line.split("Units: ")[1].strip()
                elif line.startswith("Solar radiation: "):
                    solar_radiation = line.split("Solar radiation: ")[1].strip()
                elif line.startswith("CC re-eval: "):
                    cc_reeval = line.split("CC re-eval: ")[1].strip()
                elif line.startswith("Country: "):
                    country_name = line.split("Country: ")[1].strip()
                elif line.startswith("City: "):
                    city_name = line.split("City: ")[1].strip()
                elif line.startswith("T_HDD: "):
                    value = line.split("T_HDD: ")[1].strip()
                    if value == 'N/A':
                        T_HDD = 16 if units == 'SI' else 61
                    else:
                        T_HDD = float(value) if units == 'SI' else 5 / 9 * (float(value) - 32)
                elif line.startswith("T_CDD: "):
                    value = line.split("T_CDD: ")[1].strip()
                    if value == 'N/A':
                        T_CDD = 21 if units == 'SI' else 70
                    else:
                        T_CDD = float(value) if units == 'SI' else 5 / 9 * (float(value) - 32)
                elif line.startswith("Operation day:"):
                    day_info = line.split("Operation day: ")[1].strip()
                    if ", from: " in day_info and ", till: " in day_info:
                        day_parts = day_info.split(", ")
                        day = day_parts[0].strip()
                        from_time = day_parts[1].split("from: ")[1].strip()
                        till_time = day_parts[2].split("till: ")[1].strip()
                    else:
                        day = day_info
                        from_time = "N/A"
                        till_time = "N/A"

                        from_line = next(file).strip()
                        if from_line.startswith("from:"):
                            from_time = from_line.split("from: ")[1].strip()
                        till_line = next(file).strip()
                        if till_line.startswith("till:"):
                            till_time = till_line.split("till: ")[1].strip()

                    schedule_entry = {
                        "Operation day": day,
                        "from": from_time,
                        "till": till_time
                    }
                    operation_schedule.append(schedule_entry)
                elif line.startswith("Optimization Schedule:"):
                    value = line.split("Optimization Schedule: ")[1].strip()
                    if value.lower() in ['n/a', 'not selected']:
                        optimization_schedule = ['s1', 's2', 's3']
                    else:
                        optimization_schedule = [s.strip() for s in value.split(',')]

                elif line.startswith("Optimizers:"):
                    value = line.split("Optimizers: ")[1].strip()
                    if value.lower() in ['n/a', 'not selected']:
                        optimizers_list = ['NN']
                    else:
                        optimizers_list = [s.strip() for s in value.split(',')]

                elif line.startswith("Examined demands:"):
                    value = line.split("Examined demands: ")[1].strip()
                    if value.lower() in ['n/a', 'not selected']:
                        demands_list = ['D1', 'D2', 'D3', 'D4', 'D5']
                    else:
                        demands_list = [s.strip() for s in value.split(',')]
                elif line.startswith("Normal distribution: "):
                    value = line.split("Normal distribution: ")[1].strip()
                    if value == '-1:+1':
                        outlier_value = 1
                    elif value == '-2:+2':
                        outlier_value = 2
                    elif value == '-3:+3':
                        outlier_value = 3
                    else:
                        outlier_value = None
    except FileNotFoundError:
        print(f"Error: {directory} not found. Please ensure the file exists on your Desktop.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {directory}: {e}")
        sys.exit(1)

    if not directory:       # logic if no working directory
        print("Error: Directory is not set. Please ensure it is defined in var.txt.")
        sys.exit(1)

    # logic to run each script with reloading
    terminal_outputs_filename = 'terminal_records.txt'
    terminal_records_path = os.path.join(directory, terminal_outputs_filename)
    with open(terminal_records_path, 'a', buffering=1, encoding="utf-8") as f:  # line-buffered for real-time logging
        sys.stdout = f                      # redirect stdout to the log file
        sys.stderr = f                      # redirect stderr to the log file
        try:
            print(f'\n\n  <<< Starting computing at {datetime.now().strftime('%d.%m.%Y %H:%M')} >>>\n\n')
            for script_name in scripts:
                try:
                    module = __import__(script_name)    # dynamic module import
                    importlib.reload(module)            # module reload
                    if hasattr(module, 'main'):
                        print(f"\nRunning {script_name}...")
                        print(directory, operation_schedule, demands_list, outlier_value)
                        print(date_format_icd, date_format_ecd, date_format_ed)
                        module.main()                   # execute the main function of each module
                        print(f"\n{script_name} executed successfully!\n")
                    else:
                        print(f"{script_name} has no main function.")
                except ImportError as e:
                    print(f"Error importing {script_name}: {e}")
                except Exception as e:
                    print(f"Error running {script_name}: {e}")
                    sys.exit(1)
        finally:
            sys.stdout = sys.__stdout__         # restore stdout
            sys.stderr = sys.__stderr__         # restore stderr


'''APPLICATION'''
if __name__ == "__main__":
    run_main_logic()
