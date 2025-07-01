import os
import pandas as pd
import importlib
from _11_initial_csv_check import directory_graphs_05, directory_others, directory_reports


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
_11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
prediction_selected_path = os.path.join(directory_others, 'predictions_selected')


'''FUNCTIONALITIES'''
def main():
    def extract_selected_pairs(file_path):
        selected_pairs = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i in range(len(lines)):
            if 'File:' in lines[i]:
                filename = lines[i].split('File:')[1].strip()
                for j in range(i + 1, len(lines)):
                    if 'CCC:' in lines[j]:
                        ccc_value = float(lines[j].split('CCC:')[1].strip())
                        selected_pairs.append((filename, ccc_value))
                        break
        return selected_pairs

    def extract_best_outcome_from_folder(folder_path):
        best_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('_best models.txt'):
                file_path = os.path.join(folder_path, filename)
                selected_pairs = extract_selected_pairs(file_path)
                if selected_pairs:
                    best_pair = max(selected_pairs, key=lambda x: x[1])         # select file with the highest CCC
                    best_files.append(best_pair[0])                             # append only the best file
        return best_files
        print("Selected files:")
        print("\n".join(best_files))

    def process_selected_csv_files(folder_path, selected_files):
        output_file = os.path.join(folder_path, '07_prediction_summary.txt')
        with open(output_file, 'w') as output:
            for filename in selected_files:
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    demand, variable, model = filename.split('_')[:3]
                    if demand in ['D5a', 'D5b']:
                        original_demand = demand        # store OG demand
                        demand = 'D5'
                    else:
                        original_demand = None

                    variable_pred_col = f"{variable}_pred"
                    demand_sum = df[demand].sum()
                    variable_pred_sum = df[variable_pred_col].sum()
                    demand_diff = demand_sum - variable_pred_sum
                    demand_diff_pct = (demand_diff / variable_pred_sum) * 100 if variable_pred_sum != 0 else 0

                    output.write(f"\nProcessing: {filename}\n")
                    output.write(f"{demand}: {demand_sum:.2f}\n")
                    output.write(f"{variable_pred_col}: {variable_pred_sum:.2f}\n")
                    output.write(f"Difference: {demand_diff:.2f} ({demand_diff_pct:.2f}%)\n\n")

                    setpoints_data, total_data = [], []

                    for column in df.columns[1:]:           # skip Time (1st) column
                        if column != demand and column != variable_pred_col:
                            value_sum = df[column].sum()
                            savings = variable_pred_sum - value_sum
                            savings_pct = (savings / variable_pred_sum) * 100 if variable_pred_sum != 0 else 0

                            print(
                                f"Processing column: {column} | Sum: {value_sum:.2f} | Savings: {savings:.2f} ({savings_pct:.2f}%)")

                            if original_demand == "D5b":
                                if 'total' in column:
                                    total_data.append((column, value_sum, savings, savings_pct))
                                else:
                                    setpoint = column.split('_')[-2]
                                    setpoints_data.append((setpoint, column, value_sum, savings, savings_pct))
                            else:
                                setpoint = column.split('_')[-2]
                                setpoints_data.append((setpoint, column, value_sum, savings, savings_pct))

                    if not setpoints_data:
                        print(f"WARNING: No setpoints data found for {filename}")

                    setpoints_data.sort(key=lambda x: x[0])

                    if original_demand == 'D5b':
                        def sort_by_mix_complex(data):
                            # sort by 'mix' scenarios first, then 'complex'
                            return [item for item in data if 'mix' in item[1]] + [item for item in data if
                                                                                  'complex' in item[1]]

                        setpoints_data = sort_by_mix_complex(setpoints_data)

                    for setpoint, col_name, total, savings, savings_pct in setpoints_data:
                        output.write(f"{col_name}: {total:.2f} | Savings: {savings:.2f} ({savings_pct:.2f}%)\n")

                    if total_data:
                        output.write("\nTotal Columns Summary:\n")
                        for col_name, total, savings, savings_pct in total_data:
                            output.write(f"{col_name}: {total:.2f} | Savings: {savings:.2f} ({savings_pct:.2f}%)\n")

    best_files = extract_best_outcome_from_folder(prediction_selected_path)
    print(best_files)
    graphs_names = [file.rstrip('.csv') for file in best_files]
    print(graphs_names)

    process_selected_csv_files(prediction_selected_path, best_files)

    _11_initial_csv_check.create_folders(directory_others, folders=['predictions_best'])
    directory_predictions_best = os.path.join(directory_others, 'predictions_best')
    _11_initial_csv_check.create_folders(directory_graphs_05, folders=['bests'])
    directory_graphs_05_bests = os.path.join(directory_graphs_05, 'bests')

    for files in best_files:
        _11_initial_csv_check.copy_files(prediction_selected_path, directory_predictions_best, file_name=files)
    _11_initial_csv_check.move_files(prediction_selected_path, directory_predictions_best,
                                     file_name='07_prediction_summary.txt')
    _11_initial_csv_check.copy_files(directory_predictions_best, directory_reports,
                                     file_name='07_prediction_summary.txt')
    directory_graphs_05_s = [os.path.join(directory_graphs_05, 's1'), os.path.join(directory_graphs_05, 's2'),
                             os.path.join(directory_graphs_05, 's3'), os.path.join(directory_graphs_05, 's4'),
                             os.path.join(directory_graphs_05, 's5')]
    for paths in directory_graphs_05_s:
        for graphs in graphs_names:
            _11_initial_csv_check.copy_files(paths, directory_predictions_best, filename_including=graphs)
    _11_initial_csv_check.copy_files(directory_predictions_best, directory_graphs_05_bests, file_extension='png')


'''APPLICATION'''
if __name__ == "__main__":
    main()
