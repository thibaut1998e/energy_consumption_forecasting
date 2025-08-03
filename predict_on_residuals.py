import matplotlib.pyplot as plt
import pandas as pd
from trasnformers_2 import train_time_series_seq_to_seq_transformer, to_tensor, to_numpy
import torch
import numpy as np



if __name__ == '__main__':
    df = pd.read_csv('prohet_modelisation.csv')
    exogeneous_variables = ['hour_sin', 'dayofweek_sin', 'dayofyear_sin']
    nb_test_vals = 600
    df_train, df_test = df.iloc[:-nb_test_vals], df.iloc[-nb_test_vals:]
    df_train_y = df_train['resiudals']
    df_train_exogenous = df_train[exogeneous_variables]
    df_test_exogenous = df_test[exogeneous_variables]

    train_y = to_tensor(df_train_y.values)
    train_exogenous = to_tensor(df_train_exogenous.to_numpy())
    test_exogenous = to_tensor(df_test_exogenous.to_numpy())

    in_size = 200

    model, _, _, _, _ = train_time_series_seq_to_seq_transformer(train_y, train_exogenous, in_size, 100)

    predicted_residuals = model.long_term_forecasting(train_y[-in_size:], train_exogenous[-in_size:], test_exogenous, 450)
    predicted_residuals = to_numpy(predicted_residuals)

    predicted_signal_plus_residuals = predicted_residuals + df_test['prophet_pred'].to_numpy()[:len(predicted_residuals)]

    mse_prophet = np.mean(df_test['resiudals'].to_numpy()**2)
    mse_prophet_transformer = np.mean((predicted_signal_plus_residuals - df_test['y'].to_numpy())**2)
    print('mse prophet', np.mean(df_test['resiudals'].to_numpy()**2))
    print('mse prophet + transformer on residuals', mse_prophet_transformer)


    plt.plot(df_test['y'].to_numpy(), label='true signal')
    plt.plot(df_test['prophet_pred'].to_numpy(), label='prediction prophet')
    plt.plot(predicted_signal_plus_residuals, label='prediction prophet + transformers')
    plt.legend()
    plt.show()






