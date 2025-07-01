import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import importlib
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from _00_CEAM_run import directory, demands_list
from _11_initial_csv_check import directory_reports, directory_graphs_03
from _13_data_overview_plots import units_dict


'''FUNCTIONALITIES'''
def main():
# functions to define energy signature (the piecewise linear functions) models
    def energy_signature_2p_C(x, x0, slope1):
        '''function to define energy signature for 2-parameters model - applicable for cooling'''
        return np.where(x <= x0, 0, slope1 * (x - x0))
    def energy_signature_3p_C(x, x0, slope1, intercept):
        '''function to define energy signature for 3-parameters model - applicable for cooling'''
        return np.where(x <= x0, intercept, slope1 * (x - x0) + intercept)
    def energy_signature_4p_C(x, x0, slope1, slope2, intercept):
        '''function to define energy signature for 4-parameters model - applicable for cooling'''
        return np.where(x <= x0, slope1 * (x - x0) + intercept, slope2 * (x0 - x) + intercept)
    def energy_signature_2p_H(x, x0, slope1):
        '''function to define energy signature for 2-parameters model - applicable for heating'''
        return np.where(x >= x0, 0, slope1 * (x0 - x))
    def energy_signature_3p_H(x, x0, slope1, intercept):
        '''function to define energy signature for 3-parameters model - applicable for heating'''
        return np.where(x >= x0, intercept, slope1 * (x0 - x) + intercept)
    def energy_signature_4p_H(x, x0, slope1, slope2, intercept):
        '''function to define energy signature for 4-parameters model - applicable for heating'''
        return np.where(x <= x0, slope1 * (x0 - x) + intercept, slope2 * (x - x0) + intercept)
    def energy_signature_5p(x, x1, x2, slope1, slope2, intercept):
        '''function to define energy signature for 5-parameters model - applicable for total demand'''
        return np.where(x <= x1, slope1 * (x1 - x) + intercept, np.where(x >= x2, slope2 * (x - x2) + intercept, intercept))

    def plot_data_and_model(directory, input_file, x_header, y_header, energy_signature_func, num_params, es_type='Total'):
        '''function to model and plot energy signature'''
        file_path = os.path.join(directory, input_file)
        file_prefix = input_file.split('_')[0]

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return f"Error: {input_file} not found."

        data = pd.read_csv(file_path)

        # clean the data: drop rows with NaN
        data = data.dropna(subset=[x_header, y_header])

        if data.empty:
            return f"Error: No valid data (records for variable {y_header}) found in {input_file} after cleaning."

        x_data = data[x_header].values
        y_data = data[y_header].values
        average_y = np.mean(y_data)

        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.25, random_state=42)

        # the initial guesses based on data characteristics
        if num_params == 2:
            initial_guess = [np.median(x_train), 1]
            es_id = '2p'        # for naming
        elif num_params == 3:
            initial_guess = [np.median(x_train), 1, np.mean(y_train)]
            es_id = '3p'        # for naming
        elif num_params == 4:
            initial_guess = [np.median(x_train), 1, 1, np.mean(y_train)]
            es_id = '4p'        # for naming
        else:  # num_params == 5
            initial_guess = [np.median(x_train), np.median(x_train), 1, 1, np.mean(y_train)]
            es_id = '5p'        # for naming

        try:
            params, covariance = curve_fit(energy_signature_func, x_train, y_train, p0=initial_guess, maxfev=10000)     # maxfev can be adjust
            y_pred_val = energy_signature_func(x_val, *params)
            r_squared_val = r2_score(y_val, y_pred_val)

            # assign regression equation
            if energy_signature_func == energy_signature_2p_C:
                regression_eq = f'y = {params[1]:.2f} * (x - {params[0]:.2f})'
            elif energy_signature_func == energy_signature_3p_C:
                regression_eq = f'y = {params[1]:.2f} * (x - {params[0]:.2f}) + {params[2]:.2f}'
            elif energy_signature_func == energy_signature_4p_C:
                regression_eq = f'y = {params[1]:.2f} * (x - {params[0]:.2f}) + {params[2]:.2f} * ({params[0]:.2f} - x) + {params[3]:.2f}'
            elif energy_signature_func == energy_signature_2p_H:
                regression_eq = f'y = {params[1]:.2f} * ({params[0]:.2f} - x)'
            elif energy_signature_func == energy_signature_3p_H:
                regression_eq = f'y = {params[1]:.2f} * ({params[0]:.2f} - x) + {params[2]:.2f}'
            elif energy_signature_func == energy_signature_4p_H:
                regression_eq = f'y = {params[1]:.2f} * ({params[0]:.2f} - x) + {params[2]:.2f} * (x - {params[0]:.2f}) + {params[3]:.2f}'
            elif energy_signature_func == energy_signature_5p:
                regression_eq = f'y = {params[2]:.2f} * ({params[0]:.2f} - x) + {params[3]:.2f} * (x - {params[1]:.2f}) + {params[4]:.2f}'

            # figure properties
            plt.figure(figsize=(16, 10), dpi=300)

            # plotting range
            x_min, x_max = min(x_data), max(x_data)
            if energy_signature_func == energy_signature_2p_C:
                x_range_model = np.linspace(params[0], x_max, 100)
            elif energy_signature_func == energy_signature_2p_H:
                x_range_model = np.linspace(x_min, params[0], 100)
            else:
                x_range_model = np.linspace(x_min, x_max, 100)

            y_model = energy_signature_func(x_range_model, *params)

            plt.scatter(x_data, y_data, label='Data points', color='skyblue', edgecolors='dodgerblue',
                        alpha=0.75, s=100)
            plt.plot(x_range_model, y_model, color='maroon', linewidth=3.0, label=f'Model: {regression_eq}')
            plt.axhline(y=average_y, color='steelblue', linestyle='--', linewidth=3.0,
                        label=f'{y_header}$_{{ave}}$={average_y:.2f} {units_dict[y_header]}')

            plt.xlim(x_min, x_max)  # ensure full data x-range is shown

            xlabel = f'{x_header} {units_dict[x_header]}'
            ylabel = f'{y_header} {units_dict[y_header]}'

            plt.xlabel(xlabel, fontsize=28, fontweight='bold')
            plt.ylabel(ylabel, fontsize=28, fontweight='bold')

            legend = plt.legend(loc='lower right', fontsize=16, edgecolor='black', facecolor='white', fancybox=False,
                                borderpad=1, borderaxespad=1)
            for text in legend.get_texts():
                text.set_color('black')
                text.set_weight('bold')
                text.set_fontsize(20)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.tight_layout()
            plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)

            output_file_name = f'{es_id}{es_type}_{file_prefix}_{x_header}_{y_header}.png'
            output_file = os.path.join(directory_graphs_03, output_file_name)
            plt.savefig(output_file)
            plt.close()

            # text summary
            result = f"\nExamined file: {input_file}\n"
            result += f'File type: {file_prefix}\n'
            result += f"ES assessment to correlate {x_header} with {y_header} demand\n"
            result += f'Graph name: {output_file_name}\n'
            result += f"Model: {regression_eq}\n"
            # specific parameters for different models
            if energy_signature_func in [energy_signature_2p_C, energy_signature_2p_H]:
                result += f"Breakpoint (x0): {params[0]:.3f}\n"
                result += f"Slope: {params[1]:.3f}\n"
            elif energy_signature_func in [energy_signature_3p_C, energy_signature_3p_H]:
                result += f"Breakpoint (x0): {params[0]:.3f}\n"
                result += f"Slope: {params[1]:.3f}\n"
                result += f"Intercept: {params[2]:.3f}\n"
            elif energy_signature_func in [energy_signature_4p_C, energy_signature_4p_H]:
                result += f"Breakpoint (x0): {params[0]:.3f}\n"
                result += f"Slope 1: {params[1]:.3f}\n"
                result += f"Slope 2: {params[2]:.3f}\n"
                result += f"Intercept: {params[3]:.3f}\n"
            elif energy_signature_func == energy_signature_5p:
                result += f"Breakpoint 1 (x1): {params[0]:.3f}\n"
                result += f"Breakpoint 2 (x2): {params[1]:.3f}\n"
                result += f"Slope 1: {params[2]:.3f}\n"
                result += f"Slope 2: {params[3]:.3f}\n"
                result += f"Intercept: {params[4]:.3f}\n"

            result += f"R^2 (coefficient of determination) = {r_squared_val:.3f}\n"
            return result

        except RuntimeError as e:
            return f"Error occurred while fitting data: {e}"

    report_filename = '05_ES_report.txt'

    output_log_file_C2 = os.path.join(directory, '05_ES_report_C2.txt')
    output_log_file_C3 = os.path.join(directory, '05_ES_report_C3.txt')
    output_log_file_C4 = os.path.join(directory, '05_ES_report_C4.txt')
    output_log_file_H2 = os.path.join(directory, '05_ES_report_H2.txt')
    output_log_file_H3 = os.path.join(directory, '05_ES_report_H3.txt')
    output_log_file_H4 = os.path.join(directory, '05_ES_report_H4.txt')
    output_log_file_5 = os.path.join(directory, '05_ES_report_5.txt')

    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')

    files = ['h_comb.csv', 'd_comb.csv']
    all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
    exclude_columns = all_demands.difference(demands_list)
    print(f"Exclude Columns: {exclude_columns}")

    # filtered variable pairs for cooling (C) assessment
    variable_pairs_C = [(var, demand, 'C') for var, demand, _ in [('Te', 'D2', 'C'), ('deltaT', 'D2', 'C')]
                        if demand not in exclude_columns]
    # filtered variable pairs for heating (H) assessment
    variable_pairs_H = [(var, demand, 'H') for var, demand, _ in [('Te', 'D1', 'H'), ('deltaT', 'D1', 'H')]
                        if demand not in exclude_columns]
    print(f"Variable pairs for heating: {variable_pairs_H}")
    # filtered variable pairs for total assessment
    variable_pairs = [(var, demand) for var, demand in [('Te', 'D5'), ('deltaT', 'D5')]
                      if demand not in exclude_columns]

    for input_file in files:
        for x_header, y_header, es_type in variable_pairs_C:
            print(f"Processing {input_file} with variables: X={x_header}, Y={y_header}")
            _11_initial_csv_check.save_output_to_file(output_log_file_C2, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_2p_C, 2, es_type)
            _11_initial_csv_check.save_output_to_file(output_log_file_C3, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_3p_C, 3, es_type)
            _11_initial_csv_check.save_output_to_file(output_log_file_C4, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_4p_C, 4, es_type)

    for input_file in files:
        for x_header, y_header, es_type in variable_pairs_H:
            print(f"Processing {input_file} with variables: X={x_header}, Y={y_header}")
            _11_initial_csv_check.save_output_to_file(output_log_file_H2, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_2p_H, 2, es_type)
            _11_initial_csv_check.save_output_to_file(output_log_file_H3, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_3p_H, 3, es_type)
            _11_initial_csv_check.save_output_to_file(output_log_file_H4, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_4p_H, 4, es_type)

    for input_file in files:
        for x_header, y_header in variable_pairs:
            print(f"Processing {input_file} with variables: X={x_header}, Y={y_header}")
            _11_initial_csv_check.save_output_to_file(output_log_file_5, plot_data_and_model, directory, input_file, x_header, y_header, energy_signature_5p, 5)

    _11_initial_csv_check.combine_txt_files(directory, report_filename, file_type='.txt', file_exception=['var.txt', 'terminal_records.txt'])
    _11_initial_csv_check.move_files(directory, directory_reports, file_name=report_filename)
    _11_initial_csv_check.delete_files(directory, file_type='.txt', exception_files=['var.txt', 'terminal_records.txt'])

    _11_initial_csv_check.create_folders(directory_graphs_03, folders=['03_selected', '02_daily', '01_hourly'])
    directory_graphs_ES_H = os.path.join(directory_graphs_03, '01_hourly')
    directory_graphs_ES_D = os.path.join(directory_graphs_03, '02_daily')
    directory_graphs_ES_selected = os.path.join(directory_graphs_03, '03_selected')

    def select_top_graphs(file_path):
        '''function to select the graphs for the most accurate ES'''
        selected_d_graphs = []
        selected_h_graphs = []

        with open(file_path, 'r') as file:
            content = file.read()

        content = content.replace("\r\n", "\n")
        blocks = content.strip().split("\n\n")

        file_type_pattern = re.compile(r"File type:\s*(\S+)")
        graph_name_pattern = re.compile(r"Graph name:\s*(\S+)")
        r2_pattern = re.compile(r"R\^2 \(coefficient of determination\)\s*=\s*([\d\.]+)")

        graphs = []

        for block in blocks:
            file_type_match = file_type_pattern.search(block)
            graph_name_match = graph_name_pattern.search(block)
            r2_match = r2_pattern.search(block)

            if file_type_match and graph_name_match and r2_match:
                file_type = file_type_match.group(1)
                graph_name = graph_name_match.group(1)
                r2_value = float(r2_match.group(1))

                graphs.append((file_type, graph_name, r2_value, block))

        # select top 3 daily graphs
        d_graphs_sorted = sorted([g for g in graphs if g[0] == 'd'], key=lambda x: x[2], reverse=True)[:3]
        selected_d_graphs = [g[1] for g in d_graphs_sorted]

        # find corresponding hourly graphs
        all_h_graphs = {g[1] for g in graphs if g[0] == 'h'}

        for d_graph in selected_d_graphs:
            h_graph = d_graph.replace("_d_", "_h_")  # replace 'd' with 'h' in the name
            if h_graph in all_h_graphs:
                selected_h_graphs.append(h_graph)

        # save results
        new_file_path = file_path.replace(".txt", "_selected.txt")
        with open(new_file_path, 'w') as new_file:
            for file_type, graph_name, r2_value, block in graphs:
                if (file_type == 'd' and graph_name in selected_d_graphs) or \
                        (file_type == 'h' and graph_name in selected_h_graphs):
                    new_file.write(block + "\n\n")

        return selected_h_graphs, selected_d_graphs

    base_report_path = os.path.join(directory_reports, '05_ES_report.txt')
    selected_h_graphs, selected_d_graphs = select_top_graphs(base_report_path)

    for selected_d_graph in selected_d_graphs:
        _11_initial_csv_check.copy_files(directory_graphs_03, directory_graphs_ES_selected,
                                         file_name=selected_d_graph)
    for selected_h_graph in selected_h_graphs:
        _11_initial_csv_check.copy_files(directory_graphs_03, directory_graphs_ES_selected,
                                         file_name=selected_h_graph)

    _11_initial_csv_check.move_files(directory_graphs_03, directory_graphs_ES_H, filename_include='_h_')
    _11_initial_csv_check.move_files(directory_graphs_03, directory_graphs_ES_D, filename_include='_d_')


'''APPLICATION'''
if __name__ == "__main__":
    main()
