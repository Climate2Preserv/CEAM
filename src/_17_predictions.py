import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from _00_CEAM_run import directory, demands_list, optimizers_list
from _11_initial_csv_check import directory_others


''''GLOBAL ARGUMENTS AND FUNCTIONS'''
_11_initial_csv_check = importlib.import_module('_11_initial_csv_check')
_11_initial_csv_check.create_folders(directory, folders=['predictions_outputs'])
prediction_outputs_path = os.path.join(directory, 'predictions_outputs')
_11_initial_csv_check.create_folders(prediction_outputs_path, folders=['D1', 'D2', 'D4', 'D5a', 'D5b'])
prediction_outputs_D1_path = os.path.join(prediction_outputs_path, 'D1')
prediction_outputs_D2_path = os.path.join(prediction_outputs_path, 'D2')
prediction_outputs_D4_path = os.path.join(prediction_outputs_path, 'D4')
prediction_outputs_D5a_path = os.path.join(prediction_outputs_path, 'D5a')
prediction_outputs_D5b_path = os.path.join(prediction_outputs_path, 'D5b')


'''FUNCTIONALITIES'''
def main():
    def preprocess_data(X, y):
        '''remove rows where y or any feature is NaN or missing'''
        mask = ~pd.isna(X).any(axis=1) & ~pd.isna(y)
        return X[mask], y[mask]

    def train_neural_network_dense(X_train, y_train, X_val, y_val):
        '''train a Neural Network prediction model'''
        # check for NaN values in X_train and y_train
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("NaN values detected in training data!")
            print("X_train with NaN values:")
            print(X_train[X_train.isnull().any(axis=1)])
            print("y_train with NaN values:")
            print(y_train[y_train.isnull()])
            raise ValueError("NaN values detected in training data! Check preprocessing.")

        if X_val.isnull().values.any() or y_val.isnull().values.any():
            print("NaN values detected in validation data!")
            print("X_val with NaN values:")
            print(X_val[X_val.isnull().any(axis=1)])
            print("y_val with NaN values:")
            print(y_val[y_val.isnull()])
            raise ValueError("NaN values detected in validation data! Check preprocessing.")

        # convert to NumPy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_val = X_val.to_numpy()
        y_val = y_val.to_numpy()

        # scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = Sequential()
        model.add(Dense(128, input_shape=(X_train_scaled.shape[1],)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1, activation='softplus'))

        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])     # temporarily switch to 'mae' for debugging NaN issues

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model.fit(X_train_scaled, y_train, epochs=200, batch_size=32,
                  validation_data=(X_val_scaled, y_val), verbose=1,
                  callbacks=[early_stopping, reduce_lr])

        return model, scaler

    def train_lstm_model(X_train, y_train, X_val, y_val):
        '''train an LSTM model'''
        # check for NaN values
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("NaN values detected in training data!")
            raise ValueError("Check preprocessing.")

        if X_val.isnull().values.any() or y_val.isnull().values.any():
            print("NaN values detected in validation data!")
            raise ValueError("Check preprocessing.")

        # convert to NumPy
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        X_val, y_val = X_val.to_numpy(), y_val.to_numpy()

        # reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        # scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.3),

            Dense(1, activation='softplus')  # Prevents negative values
        ])

        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model.fit(X_train_scaled, y_train, epochs=200, batch_size=32,
                  validation_data=(X_val_scaled, y_val), verbose=1,
                  callbacks=[early_stopping, reduce_lr])

        return model, scaler

    def train_gru_model(X_train, y_train, X_val, y_val):
        '''train a GRU model'''
        # check for NaN values
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("NaN values detected in training data!")
            raise ValueError("Check preprocessing.")

        if X_val.isnull().values.any() or y_val.isnull().values.any():
            print("NaN values detected in validation data!")
            raise ValueError("Check preprocessing.")

        # convert to NumPy
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        X_val, y_val = X_val.to_numpy(), y_val.to_numpy()

        # reshape for GRU
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        # scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            BatchNormalization(),
            Dropout(0.2),

            GRU(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.3),

            Dense(1, activation='softplus')
        ])

        optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        model.fit(X_train_scaled, y_train, epochs=200, batch_size=32,
                  validation_data=(X_val_scaled, y_val), verbose=1,
                  callbacks=[early_stopping, reduce_lr])

        return model, scaler

    def train_decision_tree(X_train, y_train, X_val, y_val):
        '''train a Decision Tree Regressor model'''
        # check for NaN values
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            print("NaN values detected in training data!")
            raise ValueError("Check preprocessing.")

        if X_val.isnull().values.any() or y_val.isnull().values.any():
            print("NaN values detected in validation data!")
            raise ValueError("Check preprocessing.")

        # convert to NumPy
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        X_val, y_val = X_val.to_numpy(), y_val.to_numpy()

        # reshape for DT
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # train DT model
        model = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5)
        model.fit(X_train_scaled, y_train)

        return model, scaler

    def predict_with_model(model, X_test_scaled):
        '''prediction using the trained model'''
        predictions = model.predict(X_test_scaled).flatten()
        return np.clip(predictions, 0, None) if hasattr(model, 'predict') else predictions

    def apply_trained_model(model_function, localization, x_train, y_train, x_prediction, r2_evaluation_columns):
        '''apply a trained prediction model to new data and evaluate'''
        df = pd.read_csv(localization)

        model_symbols = {
            'train_decision_tree': 'DT',
            'train_gru_model': 'GRU',
            'train_lstm_model': 'LSTM',
            'train_neural_network_dense': 'NN'
        }
        model_symbol = model_symbols.get(model_function.__name__, 'UNK')

        if x_train not in df.columns:
            print(f"Warning: Training feature '{x_train}' is missing in the input CSV!")
            x_prediction = []

        required_columns = [x_train, y_train, 'Time'] + x_prediction
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in CSV: {missing_columns}")
            x_prediction = [col for col in x_prediction if col not in missing_columns]

        if not x_prediction:
            print(f"No valid prediction columns found in the input CSV. Skipping predictions for {x_train}.")
            return

        df_filtered = df[df[y_train] != 0].dropna()
        X_train, y_train_values = preprocess_data(df_filtered[[x_train]], df_filtered[y_train])

        trained_model, scaler = model_function(X_train, y_train_values, X_train, y_train_values)

        predictions = {}
        for col in x_prediction:
            if col in df.columns:
                X_test = df[[col]].values
                X_test_scaled = scaler.transform(X_test)
                predictions[col + '_pred'] = predict_with_model(trained_model, X_test_scaled)

        output_df = df[['Time', y_train]].copy()
        for key, values in predictions.items():
            output_df[key] = values

        output_df.loc[output_df[y_train] == 0, predictions.keys()] = 0

        output_file = localization.replace('inputs_', '', 1).replace('.csv', f'_{x_train}_{model_symbol}_predictions.csv')
        output_df.to_csv(output_file, index=False)

        print(f"Predictions saved to: {output_file}")

        for col in r2_evaluation_columns:
            if col + '_pred' in predictions:
                y_pred = predictions[col + '_pred']
                y_true = df[y_train].values

                valid_idx = ~np.isnan(y_pred) & ~np.isnan(y_true)
                y_pred_clean = y_pred[valid_idx]
                y_true_clean = y_true[valid_idx]

                if len(y_pred_clean) > 1:
                    r2 = r2_score(y_true_clean, y_pred_clean)
                    print(f"Model {model_symbol}: R² score for {col}: {r2:.4f}")
                else:
                    print(f"Not enough valid data for R² evaluation for {col}.")

    training_models = []        # initialize an empty list
    # adjust training models based on the optimizers_list
    if 'DT' in optimizers_list:
        training_models.append(train_decision_tree)
    if 'NN' in optimizers_list:
        training_models.append(train_neural_network_dense)
    if 'LSTM' in optimizers_list:
        training_models.append(train_lstm_model)
    if 'GRU' in optimizers_list:
        training_models.append(train_gru_model)

    for model_function in training_models:
        if 'D1' in demands_list:
            file_D1 = os.path.join(directory, 'inputs_D1.csv')
            y_train = "D1"
            apply_trained_model(
                model_function=model_function,
                localization=file_D1,
                x_train="deltaT",
                y_train=y_train,
                x_prediction=["deltaT",
                              "deltaT_Heating_neg_T-D1_2_s1", "deltaT_Heating_neg_T-D1_2_s2",
                              "deltaT_Heating_neg_T-D1_2_s3", "deltaT_Heating_neg_T-D1_2_s4",
                              "deltaT_Heating_neg_T-D1_2_s5",
                              "deltaT_Heating_neg_T-D1_5_s1", "deltaT_Heating_neg_T-D1_5_s2",
                              "deltaT_Heating_neg_T-D1_5_s3", "deltaT_Heating_neg_T-D1_5_s4",
                              "deltaT_Heating_neg_T-D1_5_s5"],
                r2_evaluation_columns=["deltaT"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D1,
                x_train="Ti",
                y_train=y_train,
                x_prediction=["Ti",
                              "Heating_neg_T-D1_2_s1", "Heating_neg_T-D1_2_s2", "Heating_neg_T-D1_2_s3",
                              "Heating_neg_T-D1_2_s4", "Heating_neg_T-D1_2_s5",
                              "Heating_neg_T-D1_5_s1", "Heating_neg_T-D1_5_s2", "Heating_neg_T-D1_5_s3",
                              "Heating_neg_T-D1_5_s4","Heating_neg_T-D1_5_s5"],
                r2_evaluation_columns=["Ti"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D1,
                x_train="TIST",
                y_train=y_train,
                x_prediction=["TIST",
                              "TIST_Heating_neg_T-D1_2_s1", "TIST_Heating_neg_T-D1_2_s2", "TIST_Heating_neg_T-D1_2_s3",
                              "TIST_Heating_neg_T-D1_2_s4", "TIST_Heating_neg_T-D1_2_s5",
                              "TIST_Heating_neg_T-D1_5_s1", "TIST_Heating_neg_T-D1_5_s2", "TIST_Heating_neg_T-D1_5_s3",
                              "TIST_Heating_neg_T-D1_5_s4", "TIST_Heating_neg_T-D1_5_s5"],
                r2_evaluation_columns=["TIST"]
            )

        if 'D4' in demands_list:
            file_D4 = os.path.join(directory, 'inputs_D4.csv')
            y_train = "D4"
            apply_trained_model(
                model_function=model_function,
                localization=file_D4,
                x_train="RHi",
                y_train=y_train,
                x_prediction=["RHi",
                              "Humidify_mix_RH-D4_5_s1", "Humidify_mix_RH-D4_5_s2", "Humidify_mix_RH-D4_5_s3",
                              "Humidify_mix_RH-D4_5_s4", "Humidify_mix_RH-D4_5_s5",
                              "Humidify_mix_RH-D4_10_s1", "Humidify_mix_RH-D4_10_s2", "Humidify_mix_RH-D4_10_s3",
                              "Humidify_mix_RH-D4_10_s4", "Humidify_mix_RH-D4_10_s5"],
                r2_evaluation_columns=["RHi"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D4,
                x_train="AbsHi",
                y_train=y_train,
                x_prediction=["AbsHi",
                              "Humidify_mix_AbsH-D4_5_s1", "Humidify_mix_AbsH-D4_5_s2", "Humidify_mix_AbsH-D4_5_s3",
                              "Humidify_mix_AbsH-D4_5_s4", "Humidify_mix_AbsH-D4_5_s5",
                              "Humidify_mix_AbsH-D4_10_s1", "Humidify_mix_AbsH-D4_10_s2", "Humidify_mix_AbsH-D4_10_s3",
                              "Humidify_mix_AbsH-D4_10_s4", "Humidify_mix_AbsH-D4_10_s5"],
                r2_evaluation_columns=["AbsHi"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D4,
                x_train="deltaRH",
                y_train=y_train,
                x_prediction=["deltaRH",
                              "deltaRH_Humidify_mix_RH-D4_5_s1", "deltaRH_Humidify_mix_RH-D4_5_s2",
                              "deltaRH_Humidify_mix_RH-D4_5_s3", "deltaRH_Humidify_mix_RH-D4_5_s4",
                              "deltaRH_Humidify_mix_RH-D4_5_s5",
                              "deltaRH_Humidify_mix_RH-D4_10_s1", "deltaRH_Humidify_mix_RH-D4_10_s2",
                              "deltaRH_Humidify_mix_RH-D4_10_s3", "deltaRH_Humidify_mix_RH-D4_10_s4",
                              "deltaRH_Humidify_mix_RH-D4_10_s5"],
                r2_evaluation_columns=["deltaRH"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D4,
                x_train="deltaAbsH",
                y_train=y_train,
                x_prediction=["deltaAbsH",
                              "deltaAbsH_Humidify_mix_AbsH-D4_5_s1", "deltaAbsH_Humidify_mix_AbsH-D4_5_s2",
                              "deltaAbsH_Humidify_mix_AbsH-D4_5_s3", "deltaAbsH_Humidify_mix_AbsH-D4_5_s4",
                              "deltaAbsH_Humidify_mix_AbsH-D4_5_s5",
                              "deltaAbsH_Humidify_mix_AbsH-D4_10_s1", "deltaAbsH_Humidify_mix_AbsH-D4_10_s2",
                              "deltaAbsH_Humidify_mix_AbsH-D4_10_s3", "deltaAbsH_Humidify_mix_AbsH-D4_10_s4",
                              "deltaAbsH_Humidify_mix_AbsH-D4_10_s5"],
                r2_evaluation_columns=["deltaAbsH"]
            )

        if 'D2' in demands_list:
            file_D2 = os.path.join(directory, 'inputs_D2.csv')
            y_train = "D2"
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="Ti",
                y_train=y_train,
                x_prediction=["Ti",
                              "Cooling_pos_T-D2_2_s1", "Cooling_pos_T-D2_2_s2", "Cooling_pos_T-D2_2_s3",
                              "Cooling_pos_T-D2_2_s4", "Cooling_pos_T-D2_2_s5",
                              "Cooling_pos_T-D2_5_s1", "Cooling_pos_T-D2_5_s2", "Cooling_pos_T-D2_5_s3",
                              "Cooling_pos_T-D2_5_s4", "Cooling_pos_T-D2_5_s5"],
                r2_evaluation_columns=["Ti"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="deltaT",
                y_train=y_train,
                x_prediction=["deltaT",
                              "deltaT_Cooling_pos_T-D2_2_s1", "deltaT_Cooling_pos_T-D2_2_s2",
                              "deltaT_Cooling_pos_T-D2_2_s3", "deltaT_Cooling_pos_T-D2_2_s4",
                              "deltaT_Cooling_pos_T-D2_2_s5",
                              "deltaT_Cooling_pos_T-D2_5_s1", "deltaT_Cooling_pos_T-D2_5_s2",
                              "deltaT_Cooling_pos_T-D2_5_s3", "deltaT_Cooling_pos_T-D2_5_s4",
                              "deltaT_Cooling_pos_T-D2_5_s5"],
                r2_evaluation_columns=["deltaT"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="TIST",
                y_train=y_train,
                x_prediction=["TIST",
                              "TIST_Cooling_pos_T-D2_2_s1", "TIST_Cooling_pos_T-D2_2_s2", "TIST_Cooling_pos_T-D2_2_s3",
                              "TIST_Cooling_pos_T-D2_2_s4", "TIST_Cooling_pos_T-D2_2_s5",
                              "TIST_Cooling_pos_T-D2_5_s1", "TIST_Cooling_pos_T-D2_5_s2", "TIST_Cooling_pos_T-D2_5_s3",
                              "TIST_Cooling_pos_T-D2_5_s4", "TIST_Cooling_pos_T-D2_5_s5"],
                r2_evaluation_columns=["TIST"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="Twbi",
                y_train=y_train,
                x_prediction=["Twbi",
                              "Cooling_pos_Twb-D2_2_s1", "Cooling_pos_Twb-D2_2_s2", "Cooling_pos_Twb-D2_2_s3",
                              "Cooling_pos_Twb-D2_2_s4", "Cooling_pos_Twb-D2_2_s5",
                              "Cooling_pos_Twb-D2_5_s1", "Cooling_pos_Twb-D2_5_s2", "Cooling_pos_Twb-D2_5_s3",
                              "Cooling_pos_Twb-D2_5_s4", "Cooling_pos_Twb-D2_5_s5"],
                r2_evaluation_columns=["Twbi"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="deltaTwb",
                y_train=y_train,
                x_prediction=["deltaTwb",
                              "deltaTwb_Cooling_pos_Twb-D2_2_s1", "deltaTwb_Cooling_pos_Twb-D2_2_s2",
                              "deltaTwb_Cooling_pos_Twb-D2_2_s3", "deltaTwb_Cooling_pos_Twb-D2_2_s4",
                              "deltaTwb_Cooling_pos_Twb-D2_2_s5",
                              "deltaTwb_Cooling_pos_Twb-D2_5_s1", "deltaTwb_Cooling_pos_Twb-D2_5_s2",
                              "deltaTwb_Cooling_pos_Twb-D2_5_s3", "deltaTwb_Cooling_pos_Twb-D2_5_s4",
                              "deltaTwb_Cooling_pos_Twb-D2_5_s5"],
                r2_evaluation_columns=["delta_Twb"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D2,
                x_train="TISTwb",
                y_train=y_train,
                x_prediction=["TISTwb",
                              "TISTwb_Cooling_pos_Twb-D2_2_s1", "TISTwb_Cooling_pos_Twb-D2_2_s2",
                              "TISTwb_Cooling_pos_Twb-D2_2_s3",
                              "TISTwb_Cooling_pos_Twb-D2_2_s4", "TISTwb_Cooling_pos_Twb-D2_2_s5",
                              "TISTwb_Cooling_pos_Twb-D2_5_s1", "TISTwb_Cooling_pos_Twb-D2_5_s2",
                              "TISTwb_Cooling_pos_Twb-D2_5_s3",
                              "TISTwb_Cooling_pos_Twb-D2_5_s4", "TISTwb_Cooling_pos_Twb-D2_5_s5"],
                r2_evaluation_columns=["TISTwb"]
            )

        if 'D5' in demands_list:
            file_D5a = os.path.join(directory, 'inputs_D5a.csv')
            file_D5b = os.path.join(directory, 'inputs_D5b.csv')
            y_train = "D5"
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="Ti",
                y_train=y_train,
                x_prediction=["Ti",
                              "Total_mix_T-D5_2_s1", "Total_mix_T-D5_2_s2", "Total_mix_T-D5_2_s3",
                              "Total_mix_T-D5_2_s4", "Total_mix_T-D5_2_s5",
                              "Total_mix_T-D5_5_s1", "Total_mix_T-D5_5_s2", "Total_mix_T-D5_5_s3",
                              "Total_mix_T-D5_5_s4", "Total_mix_T-D5_5_s5"],
                r2_evaluation_columns=["Ti"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="deltaT",
                y_train=y_train,
                x_prediction=["deltaT",
                              "deltaT_Total_mix_T-D5_2_s1", "deltaT_Total_mix_T-D5_2_s2", "deltaT_Total_mix_T-D5_2_s3",
                              "deltaT_Total_mix_T-D5_2_s5", "deltaT_Total_mix_T-D5_2_s5",
                              "deltaT_Total_mix_T-D5_5_s1", "deltaT_Total_mix_T-D5_5_s2", "deltaT_Total_mix_T-D5_5_s3",
                              "deltaT_Total_mix_T-D5_5_s5", "deltaT_Total_mix_T-D5_5_s5"],
                r2_evaluation_columns=["deltaT"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="TIST",
                y_train=y_train,
                x_prediction=["TIST",
                              "TIST_Total_mix_T-D5_2_s1", "TIST_Total_mix_T-D5_2_s2", "TIST_Total_mix_T-D5_2_s3",
                              "TIST_Total_mix_T-D5_2_s5", "TIST_Total_mix_T-D5_2_s5",
                              "TIST_Total_mix_T-D5_5_s1", "TIST_Total_mix_T-D5_5_s2", "TIST_Total_mix_T-D5_5_s3",
                              "TIST_Total_mix_T-D5_5_s5", "TIST_Total_mix_T-D5_5_s5"],
                r2_evaluation_columns=["TIST"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="Twbi",
                y_train=y_train,
                x_prediction=["Twbi",
                              "Total_mix_Twb-D5_2_s1", "Total_mix_Twb-D5_2_s2", "Total_mix_Twb-D5_2_s3",
                              "Total_mix_Twb-D5_2_s4", "Total_mix_Twb-D5_2_s5",
                              "Total_mix_Twb-D5_5_s1", "Total_mix_Twb-D5_5_s2", "Total_mix_Twb-D5_5_s3",
                              "Total_mix_Twb-D5_5_s4", "Total_mix_Twb-D5_5_s5"],
                r2_evaluation_columns=["Twbi"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="deltaTwb",
                y_train=y_train,
                x_prediction=["deltaTwb",
                              "deltaTwb_Total_mix_Twb-D5_2_s1", "deltaTwb_Total_mix_Twb-D5_2_s2",
                              "deltaTwb_Total_mix_Twb-D5_2_s3", "deltaTwb_Total_mix_Twb-D5_2_s5",
                              "deltaTwb_Total_mix_Twb-D5_2_s5",
                              "deltaTwb_Total_mix_Twb-D5_5_s1", "deltaTwb_Total_mix_Twb-D5_5_s2",
                              "deltaTwb_Total_mix_Twb-D5_5_s3", "deltaTwb_Total_mix_Twb-D5_5_s5",
                              "deltaTwb_Total_mix_Twb-D5_5_s5"],
                r2_evaluation_columns=["deltaTwb"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5a,
                x_train="TISTwb",
                y_train=y_train,
                x_prediction=["TISTwb",
                              "TISTwb_Total_mix_Twb-D5_2_s1", "TISTwb_Total_mix_Twb-D5_2_s2",
                              "TISTwb_Total_mix_Twb-D5_2_s3", "TISTwb_Total_mix_Twb-D5_2_s4",
                              "TISTwb_Total_mix_Twb-D5_2_s5",
                              "TISTwb_Total_mix_Twb-D5_5_s1", "TISTwb_Total_mix_Twb-D5_5_s2",
                              "TISTwb_Total_mix_Twb-D5_5_s3", "TISTwb_Total_mix_Twb-D5_5_s4",
                              "TISTwb_Total_mix_Twb-D5_5_s5"],
                r2_evaluation_columns=["TISTwb"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5b,
                x_train="Tdi",
                y_train=y_train,
                x_prediction=["Tdi",
                              "Total_mix_Td-D5_2_s1", "Total_mix_Td-D5_2_s2", "Total_mix_Td-D5_2_s3",
                              "Total_mix_Td-D5_2_s4", "Total_mix_Td-D5_2_s5",
                              "Total_mix_Td-D5_5_s1", "Total_mix_Td-D5_5_s2", "Total_mix_Td-D5_5_s3",
                              "Total_mix_Td-D5_5_s4", "Total_mix_Td-D5_5_s5",
                              "Total_complex_Td_T2RH5_s1", "Total_complex_Td_T2RH5_s2", "Total_complex_Td_T2RH5_s3",
                              "Total_complex_Td_T2RH5_s4", "Total_complex_Td_T2RH5_s5",
                              "Total_complex_Td_T5RH10_s1", "Total_complex_Td_T5RH10_s2", "Total_complex_Td_T5RH10_s3",
                              "Total_complex_Td_T5RH10_s4", "Total_complex_Td_T5RH10_s5"],
                r2_evaluation_columns=["Tdi"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5b,
                x_train="deltaTd",
                y_train=y_train,
                x_prediction=["deltaTd",
                              "deltaTd_Total_mix_Td-D5_2_s1", "deltaTd_Total_mix_Td-D5_2_s2",
                              "deltaTd_Total_mix_Td-D5_2_s3", "deltaTd_Total_mix_Td-D5_2_s4",
                              "deltaTd_Total_mix_Td-D5_2_s5",
                              "deltaTd_Total_mix_Td-D5_5_s1", "deltaTd_Total_mix_Td-D5_5_s2",
                              "deltaTd_Total_mix_Td-D5_5_s3", "deltaTd_Total_mix_Td-D5_5_s4",
                              "deltaTd_Total_mix_Td-D5_5_s5",
                              "deltaTd_Total_complex_Td_T2RH5_s1", "deltaTd_Total_complex_Td_T2RH5_s2",
                              "deltaTd_Total_complex_Td_T2RH5_s3",
                              "deltaTd_Total_complex_Td_T2RH5_s4", "deltaTd_Total_complex_Td_T2RH5_s5",
                              "deltaTd_Total_complex_Td_T5RH10_s1", "deltaTd_Total_complex_Td_T5RH10_s2",
                              "deltaTd_Total_complex_Td_T5RH10_s3",
                              "deltaTd_Total_complex_Td_T5RH10_s4", "deltaTd_Total_complex_Td_T5RH10_s5"],
                r2_evaluation_columns=["deltaTd"]
            )
            apply_trained_model(
                model_function=model_function,
                localization=file_D5b,
                x_train="TISTd",
                y_train=y_train,
                x_prediction=["TISTd",
                              "TISTd_Total_mix_Td-D5_2_s1", "TISTd_Total_mix_Td-D5_2_s2", "TISTd_Total_mix_Td-D5_2_s3",
                              "TISTd_Total_mix_Td-D5_2_s4", "TISTd_Total_mix_Td-D5_2_s5",
                              "TISTd_Total_mix_Td-D5_5_s1", "TISTd_Total_mix_Td-D5_5_s2", "TISTd_Total_mix_Td-D5_5_s3",
                              "TISTd_Total_mix_Td-D5_5_s4", "TISTd_Total_mix_Td-D5_5_s5",
                              "TISTd_Total_complex_Td_T2RH5_s1", "TISTd_Total_complex_Td_T2RH5_s2",
                              "TISTd_Total_complex_Td_T2RH5_s3",
                              "TISTd_Total_complex_Td_T2RH5_s4", "TISTd_Total_complex_Td_T2RH5_s5",
                              "TISTd_Total_complex_Td_T5RH10_s1", "TISTd_Total_complex_Td_T5RH10_s2",
                              "TISTd_Total_complex_Td_T5RH10_s3",
                              "TISTd_Total_complex_Td_T5RH10_s4", "TISTd_Total_complex_Td_T5RH10_s5"],
                r2_evaluation_columns=["TISTd"]
            )

    _11_initial_csv_check.move_files(directory, prediction_outputs_D1_path, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='D1_')
    _11_initial_csv_check.move_files(directory, prediction_outputs_D2_path, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='D2_')
    _11_initial_csv_check.move_files(directory, prediction_outputs_D4_path, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='D4_')
    _11_initial_csv_check.move_files(directory, prediction_outputs_D5a_path, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='D5a_')
    _11_initial_csv_check.move_files(directory, prediction_outputs_D5b_path, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='D5b_')
    _11_initial_csv_check.move_files(directory, directory_others, file_extension=None, file_name=None,
                                     exception_files=None, filename_include='inputs')


'''APPLICATION'''
if __name__ == "__main__":
    main()
