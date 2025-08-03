import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
from agence_ore_data import manage_date_in_df, select_part_of_data_frame
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv("data/consommation_inf_36Kva.csv", delimiter=";")
    df = df.fillna(method='bfill')
    manage_date_in_df(df, utc=True)
    region = 'Bretagne'
    df_bretagne = select_part_of_data_frame(df, select_all=True, region=region)
    df_bretagne["ENERGIE_SOUTIREE"] /= 10 ** 9
    sequence = df_bretagne["ENERGIE_SOUTIREE"].to_numpy()
    exog_variables = df_bretagne[['hour_sin', 'dayofweek_sin', 'dayofyear_sin']].to_numpy()
    nb_test_vals = 10000
    #train_sequence, test_sequence = sequence[:-nb_test_vals], sequence[-nb_test_vals:]
    #train_exog, test_exog = exog_variables[:-nb_test_vals], exog_variables[-nb_test_vals:]


    df_prohet = pd.DataFrame(columns=['ds', 'y'])
    df_prohet['ds'] = df_bretagne['HORODATE'].dt.tz_localize(None)
    print('horodate', df_prohet['ds'])
    df_prohet['y'] = df_bretagne["ENERGIE_SOUTIREE"]
    print('head', df_prohet.head())

    df_prohet_train = df_prohet.iloc[:-nb_test_vals, :]
    df_prohet_test = df_prohet.iloc[-nb_test_vals:, :]



    prohet_mod = Prophet()
    prohet_mod.fit(df_prohet_train)

    prediction_time_stamps = df_prohet_test[['ds']]
    forecast = prohet_mod.predict(prediction_time_stamps)

    plt.plot(prediction_time_stamps, df_prohet_test['y'])
    plt.plot(prediction_time_stamps, forecast['yhat'])
    plt.show()


    prediction_time_stamps_all = df_prohet[['ds']]
    forecast_all = prohet_mod.predict(prediction_time_stamps_all)
    plt.plot(prediction_time_stamps_all, df_prohet['y'])
    plt.plot(prediction_time_stamps_all, forecast_all['yhat'])
    plt.show()

    print('ff', df_prohet['y'].index)
    print('gg', forecast_all['yhat'].index)
    print((df_prohet['y'] - forecast_all['yhat']).shape)
    residual = df_prohet['y'].values - forecast_all['yhat'].values
    plt.plot(prediction_time_stamps_all, residual)
    plt.show()


    time_stamp_chars = ['hour_of_day', 'day_of_week', 'day_of_year', 'hour_sin', 'dayofweek_sin', 'dayofyear_sin']
    prophet_results_df = pd.DataFrame(columns=['y', 'prophet_pred', 'resiudals', 'ds']+ time_stamp_chars)

    prophet_results_df['prophet_pred'] = forecast_all['yhat'].values
    prophet_results_df['y'] = df_prohet['y'].values
    prophet_results_df['resiudals'] = residual
    prophet_results_df['ds'] = df_prohet['ds'].values
    for g in time_stamp_chars:
        prophet_results_df[g] = df_bretagne[g].values
    print('shape df', prophet_results_df.shape)
    prophet_results_df.to_csv('prohet_modelisation.csv')






