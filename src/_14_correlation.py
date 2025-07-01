import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import importlib
import gc
import re
from scipy import stats
from _00_CEAM_run import directory, solar_radiation, demands_list
from _11_initial_csv_check import directory_reports, directory_graphs_02
from _13_data_overview_plots import units_dict


'''FUNCTIONALITIES'''
def main():
    def handle_exceptions(column_name, column_data):
        '''function to define axis range for different variables'''
        if column_name in ['RHi', 'RHe']:
            ax_min = 0
            ax_max = 100
        elif column_name in ['D1', 'D2', 'D3', 'D4', 'D5', 'Is']:
            ax_min = 0
            ax_max = max(column_data) * 1.10
        else:  # Other situations
            ax_min = 0 if min(column_data) == 0 else min(column_data) * 0.75
            ax_max = max(column_data) * 1.10
        return ax_min, ax_max

    def plot_regression(input_file, output_directory, column_x, column_y, marker_style='o', marker_size=25,
                        marker_color='skyblue', marker_edge_color='dodgerblue', figure_size=(12, 8), dpi=150):
        '''function to plot correlations'''
        if input_file.startswith('h_'):
            file_type = 'hourly'
        elif input_file.startswith('d_'):
            file_type = 'daily'
        elif input_file.startswith('m_'):
            file_type = 'monthly'
        # elif input_file.startswith('w_'):
        #     file_type = 'weekly'
        else:
            file_type = 'unknown'

        directory_input = os.path.join(output_directory, input_file)
        data = pd.read_csv(directory_input)
        data = data[[column_x, column_y]].dropna()

        if data.empty:
            return f"Error: No valid data available for columns {column_x} and {column_y} in {input_file}"

        # extract data after dropping missing values
        x = pd.to_numeric(data[column_x], errors='coerce')
        y = pd.to_numeric(data[column_y], errors='coerce')
        data = pd.DataFrame({column_x: x, column_y: y}).dropna()  # remove rows with NaN values in the columns

        if data.empty:
            return f"Error: No valid data after cleaning for columns {column_x} and {column_y} in {input_file}"

        x = data[column_x]
        y = data[column_y]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line = slope * x + intercept
        correlation_direction = '(P)ositive' if slope > 0 else '(N)egative' if slope < 0 else '(N)o(C)orrelation'
        r_squared = r_value ** 2

        x_min, x_max = handle_exceptions(column_x, x)
        y_min, y_max = handle_exceptions(column_y, y)

        plt.figure(figsize=figure_size, dpi=dpi)
        plt.scatter(x, y, label='Data points', marker=marker_style, s=marker_size, color=marker_color,
                    edgecolors=marker_edge_color, alpha=0.75)
        plt.plot(x, line, color='red', label=f'Regression line: y = {slope:.2f}x + {intercept:.2f}')
        plt.xlabel(f'{column_x} {units_dict.get(column_x, "")}', fontsize=20, fontweight='bold')
        plt.ylabel(f'{column_y} {units_dict.get(column_y, "")}', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        legend = plt.legend(loc='lower right', fontsize=20, edgecolor='black', facecolor='white', fancybox=False,
                            borderpad=1, borderaxespad=1)
        for text in legend.get_texts():
            text.set_color('black')
            text.set_weight('bold')
            text.set_fontsize(16)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=1.0)

        output_file_name = f'Cor_{input_file[:2]}{column_x}_{column_y}.png'
        output_file = os.path.join(directory_graphs_02, output_file_name)
        plt.savefig(output_file)
        plt.close('all')    # close all figures

        gc.collect()        # free up memory

        num_points = len(data)
        num_x_values = len(data[column_x].dropna())
        num_y_values = len(data[column_y].dropna())

        summary = [
            f'\nFile name: {input_file}',
            f'File type: {file_type}',
            f'Correlation between: {column_y} and {column_x}',
            f'Graph name: {output_file_name}',
            f'Number of data-points plotted: {num_points}',
            f'Number of {column_x} records in the input file: {num_x_values}',
            f'Number of {column_y} records in the input file: {num_y_values}',
            f'Regression line: y = {slope:.2f}x + {intercept:.2f}',
            f'Correlation is {correlation_direction}',
            f'R^2 (coefficient of determination) = {r_squared:.2f}',
            f'r (Pearson correlation coefficient) = {r_value:.2f}'
        ]
        return '\n'.join(summary)

    all_demands = {'D1', 'D2', 'D3', 'D4', 'D5'}
    exclude_columns = all_demands.difference(demands_list)

    def filter_combinations(combinations, exclude_columns):
        '''filters out column combinations containing any excluded columns.'''
        return [
            (col_x, col_y) for col_x, col_y in combinations
            if col_x not in exclude_columns and col_y not in exclude_columns
        ]

    # define the initial column_combinations based on the 'Is' condition
    if solar_radiation == 'y':
        column_combinations = [
            ('Te', 'RHe'), ('Te', 'Is'), ('Ti', 'RHi'), ('Ti', 'Is'),

            ('Ti', 'D1'), ('Ti', 'D2'), ('Ti', 'D5'),
            ('Te', 'D1'), ('Te', 'D2'), ('Te', 'D5'),

            ('RHi', 'D4'), ('RHi', 'D5'), ('AbsHi', 'D4'), ('AbsHi', 'D5'), ('pvi', 'D4'), ('pvi', 'D5'),
            ('RHe', 'D4'), ('RHe', 'D5'), ('AbsHe', 'D4'), ('AbsHe', 'D5'), ('pve', 'D4'), ('pve', 'D5'),

            ('HDD', 'D1'), ('CDD', 'D2'), ('HDD', 'D5'), ('CDD', 'D5'),

            ('Tdi', 'D1'), ('Tdi', 'D2'), ('Tdi', 'D4'), ('Tdi', 'D5'),
            ('Tde', 'D1'), ('Tde', 'D2'), ('Tde', 'D4'), ('Tde', 'D5'),

            ('Twbi', 'D1'), ('Twbi', 'D2'), ('Twbi', 'D4'), ('Twbi', 'D5'),
            ('Twbe', 'D1'), ('Twbe', 'D2'), ('Twbe', 'D4'), ('Twbe', 'D5'),

            ('deltaT', 'D1'), ('deltaT', 'D2'), ('deltaT', 'D5'),
            ('TIST', 'D1'), ('TIST', 'D2'), ('TIST', 'D5'),

            ('deltaRH', 'D4'), ('deltaRH', 'D5'),
            ('deltaAbsH', 'D4'), ('deltaAbsH', 'D5'),

            ('deltaTd', 'D1'), ('deltaTd', 'D2'), ('deltaTd', 'D4'), ('deltaTd', 'D5'),
            ('TISTd', 'D1'), ('TISTd', 'D2'), ('TISTd', 'D4'), ('TISTd', 'D5'),

            ('deltaTwb', 'D1'), ('deltaTwb', 'D2'), ('deltaTwb', 'D4'), ('deltaTwb', 'D5'),
            ('TISTwb', 'D1'), ('TISTwb', 'D2'), ('TISTwb', 'D4'), ('TISTwb', 'D5')
        ]
    else:
        # without 'Is' conditions
        column_combinations = [
            ('Te', 'RHe'), ('Ti', 'RHi'),

            ('Ti', 'D1'), ('Ti', 'D2'), ('Ti', 'D5'),
            ('Te', 'D1'), ('Te', 'D2'), ('Te', 'D5'),

            ('RHi', 'D4'), ('RHi', 'D5'), ('AbsHi', 'D4'), ('AbsHi', 'D5'), ('pvi', 'D4'), ('pvi', 'D5'),
            ('RHe', 'D4'), ('RHe', 'D5'), ('AbsHe', 'D4'), ('AbsHe', 'D5'), ('pve', 'D4'), ('pve', 'D5'),

            ('HDD', 'D1'), ('CDD', 'D2'), ('HDD', 'D5'), ('CDD', 'D5'),

            ('Tdi', 'D1'), ('Tdi', 'D2'), ('Tdi', 'D4'), ('Tdi', 'D5'),
            ('Tde', 'D1'), ('Tde', 'D2'), ('Tde', 'D4'), ('Tde', 'D5'),

            ('Twbi', 'D1'), ('Twbi', 'D2'), ('Twbi', 'D4'), ('Twbi', 'D5'),
            ('Twbe', 'D1'), ('Twbe', 'D2'), ('Twbe', 'D4'), ('Twbe', 'D5'),

            ('deltaT', 'D1'), ('deltaT', 'D2'), ('deltaT', 'D5'),

            ('deltaRH', 'D4'), ('deltaRH', 'D5'),
            ('deltaAbsH', 'D4'), ('deltaAbsH', 'D5'),

            ('deltaTd', 'D1'), ('deltaTd', 'D2'), ('deltaTd', 'D4'), ('deltaTd', 'D5'),

            ('deltaTwb', 'D1'), ('deltaTwb', 'D2'), ('deltaTwb', 'D4'), ('deltaTwb', 'D5')
        ]

    # filter the column_combinations
    column_combinations = filter_combinations(column_combinations, exclude_columns)

    input_files = ['h_comb.csv', 'd_comb.csv', 'm_comb.csv']  # all the input files
    output_log_file = os.path.join(directory, '04_correlation_report.txt')
    _11_initial_csv_check = importlib.import_module('_11_initial_csv_check')

    for input_file in input_files:
        for column_x, column_y in column_combinations:
            _11_initial_csv_check.save_output_to_file(output_log_file, plot_regression,
                                                      input_file, directory, column_x, column_y)

    _11_initial_csv_check.move_files(directory, directory_reports, file_extension='*.txt',
                                     exception_files=['var.txt', 'terminal_records.txt'])

    _11_initial_csv_check.create_folders(directory_graphs_02, folders=['04_selected', '03_monthly', '02_daily', '01_hourly'])
    directory_graphs_correlations_selected = os.path.join(directory_graphs_02, '04_selected')
    directory_graphs_correlations_M = os.path.join(directory_graphs_02, '03_monthly')
    directory_graphs_correlations_D = os.path.join(directory_graphs_02, '02_daily')
    directory_graphs_correlations_H = os.path.join(directory_graphs_02, '01_hourly')

    _11_initial_csv_check.move_files(directory_graphs_02, directory_graphs_correlations_M, file_extension='*.png',
                                     filename_include='_m_')
    _11_initial_csv_check.move_files(directory_graphs_02, directory_graphs_correlations_D, file_extension='*.png',
                                     filename_include='_d_')
    _11_initial_csv_check.move_files(directory_graphs_02, directory_graphs_correlations_H, file_extension='*.png',
                                     filename_include='_h_')

    def select_top_graphs(file_path):
        '''function to select the graphs for the best correlation'''
        selected_h_graphs = []
        selected_d_graphs = []
        selected_m_graphs = []

        with open(file_path, 'r') as file:
            content = file.read()

        blocks = content.split("\n\n")
        # regular expression patterns to extract needed data
        file_type_pattern = re.compile(r"File type:\s*(\S+)")
        graph_name_pattern = re.compile(r"Graph name:\s*(\S+)")
        r2_pattern = re.compile(r"R\^2\s*\(coefficient of determination\)\s*=\s*([\d\.]+)")

        # loop through each block to process the graph info
        for block in blocks:
            file_type_match = file_type_pattern.search(block)
            graph_name_match = graph_name_pattern.search(block)
            r2_match = r2_pattern.search(block)

            # extract the info: file type, graph name, and R^2 value
            if file_type_match and graph_name_match and r2_match:
                file_type = file_type_match.group(1)
                graph_name = graph_name_match.group(1)
                r2_value = float(r2_match.group(1))

                # sort and select the top 5 performing outputs (graphs)
                if file_type == 'hourly':
                    selected_h_graphs.append((graph_name, r2_value))
                elif file_type == 'daily':
                    selected_d_graphs.append((graph_name, r2_value))
                elif file_type == 'monthly':
                    selected_m_graphs.append((graph_name, r2_value))

        selected_h_graphs = [graph[0] for graph in sorted(selected_h_graphs, key=lambda x: x[1], reverse=True)[:5]]
        selected_d_graphs = [graph[0] for graph in sorted(selected_d_graphs, key=lambda x: x[1], reverse=True)[:5]]
        selected_m_graphs = [graph[0] for graph in sorted(selected_m_graphs, key=lambda x: x[1], reverse=True)[:5]]

        new_file_path = file_path.replace(".txt", "_selected.txt")

        with open(new_file_path, 'w') as new_file:
            for block in blocks:
                file_type_match = file_type_pattern.search(block)
                graph_name_match = graph_name_pattern.search(block)

                if file_type_match and graph_name_match:
                    file_type = file_type_match.group(1)
                    graph_name = graph_name_match.group(1)

                    if file_type == 'hourly' and graph_name in selected_h_graphs:
                        new_file.write(block + "\n\n")
                    elif file_type == 'daily' and graph_name in selected_d_graphs:
                        new_file.write(block + "\n\n")
                    elif file_type == 'monthly' and graph_name in selected_m_graphs:
                        new_file.write(block + "\n\n")

        return selected_h_graphs, selected_d_graphs, selected_m_graphs

    base_report_path = os.path.join(directory_reports, '04_correlation_report.txt')
    selected_h_graphs, selected_d_graphs, selected_m_graphs = select_top_graphs(base_report_path)

    print("Selected hourly graphs:", selected_h_graphs)
    print("Selected daily graphs:", selected_d_graphs)
    print("Selected monthly graphs:", selected_m_graphs)

    for selected_h_graph in selected_h_graphs:
        _11_initial_csv_check.copy_files(directory_graphs_correlations_H, directory_graphs_correlations_selected,
                                     file_name=selected_h_graph)
    for selected_d_graph in selected_d_graphs:
        _11_initial_csv_check.copy_files(directory_graphs_correlations_D, directory_graphs_correlations_selected,
                                     file_name=selected_d_graph)
    for selected_m_graph in selected_m_graphs:
        _11_initial_csv_check.copy_files(directory_graphs_correlations_M, directory_graphs_correlations_selected,
                                     file_name=selected_m_graph)


'''APPLICATION'''
if __name__ == "__main__":
    main()
