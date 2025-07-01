import os
import pandas as pd
import numpy as np
import importlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from _00_CEAM_run import directory
from _11_initial_csv_check import directory_others, directory_reports


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
prediction_outputs_path = os.path.join(directory, 'predictions_outputs')

_11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
_11_initial_csv_check.create_folders(directory, folders=['predictions_selected'])
prediction_selected_path = os.path.join(directory, 'predictions_selected')


def concordance_correlation_coefficient(y_true, y_pred):
    '''calculate the Concordance Correlation Coefficient (CCC)'''
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid_idx], y_pred[valid_idx]

    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan

    mean_true = np.nanmean(y_true)
    mean_pred = np.nanmean(y_pred)
    var_true = np.nanvar(y_true)
    var_pred = np.nanvar(y_pred)
    covariance = np.cov(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator if denominator != 0 else np.nan


'''FUNCTIONALITIES'''
def main():
    def evaluate_predictions(directory):
        selected_files = {}

        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if not os.path.isdir(folder_path):
                continue

            overview_path = os.path.join(folder_path, f"51_{folder}_predictions overview.txt")
            best_models_path = os.path.join(folder_path, f"52_{folder}_best models.txt")

            with open(overview_path, 'w', encoding='utf-8') as overview_file, open(best_models_path, 'w', encoding='utf-8') as best_models_file:
                overview_file.write(
                    f"{'=' * 52}\n=== Energy savings assessment for {folder} consumption ===\n{'=' * 52}\n\n")
                best_models_file.write(
                    f"{'=' * 65}\n=== Selection of the best prediction models for {folder} assessment ===\n{'=' * 65}\n\n")

                files = [f for f in os.listdir(folder_path) if f.endswith('_predictions.csv')]
                if not files:
                    overview_file.write(f"No prediction files found in folder: {folder}\n{'=' * 50}\n")
                    continue

                results = {}

                for file in files:
                    filename_parts = file.split('_')
                    demand_symbol = filename_parts[0]
                    if demand_symbol in ['D5a', 'D5b']:
                        demand_symbol = 'D5'
                    variable = filename_parts[1]
                    model = filename_parts[2]
                    predicted_column = f'{variable}_pred'

                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    if demand_symbol in df.columns and predicted_column in df.columns and 'Time' in df.columns:
                        actual = df[demand_symbol].values
                        predicted = df[predicted_column].values

                        valid_idx = ~np.isnan(actual) & ~np.isnan(predicted)
                        actual, predicted = actual[valid_idx], predicted[valid_idx]

                        if len(actual) == 0 or len(predicted) == 0:
                            continue

                        r2 = r2_score(actual, predicted)
                        rmse = np.sqrt(mean_squared_error(actual, predicted))
                        mae = mean_absolute_error(actual, predicted)
                        ccc = concordance_correlation_coefficient(actual, predicted)

                        total_base_consumption = np.nansum(actual)
                        total_predicted_consumption = np.nansum(predicted)
                        total_abs_diff = total_base_consumption - total_predicted_consumption
                        avg_rel_diff = (total_abs_diff / total_base_consumption) * 100 if total_base_consumption != 0 else 0

                        peak_base_value = np.nanmax(df[demand_symbol])
                        peak_index = df[demand_symbol].idxmax()
                        peak_predicted_value = np.nan_to_num(df.loc[peak_index, predicted_column], nan=0)
                        peak_time = df.loc[peak_index, 'Time']

                        peak_abs_diff = abs(peak_base_value - peak_predicted_value)
                        peak_rel_diff = (peak_abs_diff / peak_base_value) * 100 if peak_base_value != 0 else 0

                        key = (folder, variable)
                        if key not in results:
                            results[key] = []

                        results[key].append({
                            'model': model,
                            'r2': r2,
                            'ccc': ccc,
                            'peak_abs_diff': peak_abs_diff,
                            'peak_rel_diff': peak_rel_diff,
                            'file_path': file_path
                        })

                        overview_file.write(
                            f"Results for file: {file}\n"
                            f"\tExamined consumption: {folder}\n"
                            f"\tExamined variable: {variable}\n"
                            f"\tUsed prediction model: {model}\n"
                            f"\tTotal Base Consumption: {total_base_consumption:.2f}\n"
                            f"\tTotal Predicted Consumption: {total_predicted_consumption:.2f}\n"
                            f"\tDifference: {total_abs_diff:.2f} ({avg_rel_diff:.2f}%)\n"
                            f"\tModel performance assessment:\n"
                            f"\t\tR²: {r2:.4f}\n"
                            f"\t\tRMSE: {rmse:.4f}\n"
                            f"\t\tMAE: {mae:.4f}\n"
                            f"\t\tCCC: {ccc:.4f}\n"
                            f"\tPeak Consumption: {peak_base_value:.2f} (recorded at {peak_time})\n"
                            f"\tPredicted Peak: {peak_predicted_value:.2f}\n"
                            f"\tPeak Diff: {peak_abs_diff:.2f} ({peak_rel_diff:.2f}%)\n\n"
                            f"{'-' * 52}\n\n"
                        )

                for key, models in results.items():
                    folder, variable = key
                    models.sort(key=lambda x: (-x['r2'], -x['ccc'], x['peak_abs_diff']))
                    best_model = models[0]
                    best_models_file.write(
                        f"Best model for {variable} variable: {best_model['model']}\n"
                        f"\tFile: {folder}_{variable}_{best_model['model']}_predictions.csv\n"
                        f"\tR²: {best_model['r2']:.4f}\n"
                        f"\tCCC: {best_model['ccc']:.4f}\n"
                        f"\tPeak diff: {best_model['peak_abs_diff']:.2f} ({best_model['peak_rel_diff']:.2f}%)\n\n"
                        f"{'-' * 65}\n\n"
                    )

                    if folder not in selected_files:
                        selected_files[folder] = []
                    selected_files[folder].append(best_model['file_path'])

        return selected_files

    selected_files = evaluate_predictions(prediction_outputs_path)
    for folder, files in selected_files.items():
        for file_path in files:
            src_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            _11_initial_csv_check.copy_files(src_dir, prediction_selected_path, file_name=file_name)
            _11_initial_csv_check.copy_files(src_dir, prediction_selected_path, file_extension='txt')

    _11_initial_csv_check.combine_txt_files(prediction_selected_path, '07_predictions_overview.txt', file_type='.txt',
                                            prefix='51_')
    _11_initial_csv_check.combine_txt_files(prediction_selected_path, '07_best_predictions.txt', file_type='.txt',
                                            prefix='52_')
    _11_initial_csv_check.move_files(prediction_selected_path, directory_reports, filename_include='07_')
    _11_initial_csv_check.move_files(directory, directory_others, move_folder=['predictions_outputs'])


'''APPLICATION'''
if __name__ == "__main__":
    main()
