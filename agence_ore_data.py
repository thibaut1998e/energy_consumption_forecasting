import pandas as pd
import numpy as np
from dateutil.parser import parse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from manage_files import make_dir
from LSTM_model import LSTM, split_train_test, training_loop
import torch
from torch import nn
from torch import optim



def select_part_of_data_frame(df, jour_debut='2024-05-13T16:00:00+02:00', jour_fin='2024-05-14T16:00:00+02:00', region='Bretagne', select_all=False):
    jour_debut_obj = parse(jour_debut)
    jour_fin_obj = parse(jour_fin)
    if not select_all:
        sub_df = df[(df["REGION"]==region) & (jour_debut_obj<=df["HORODATE"]) & (df["HORODATE"]<=jour_fin_obj)]
    else:
        sub_df = df[df["REGION"]==region]
    df_sorted = sub_df.sort_values(by="HORODATE")
    return df_sorted


def plot_temporal_series(dates, values, day_interval=5, label=''):
    # Incline les dates automatiquement
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, marker='o', label=label)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Valeurs")
    plt.legend(loc='best')
    return ax

def manage_date_in_df(df, utc=True):
    df['HORODATE'] = pd.to_datetime(df['HORODATE'], infer_datetime_format=True, utc=utc)
    df['hour_of_day'] = df['HORODATE'].dt.hour  # 0 to 23
    df['day_of_week'] = df['HORODATE'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['day_of_year'] = df['HORODATE'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)


if __name__ == '__main__':
    print('start to load data set')
    df = pd.read_csv("data/consommation_inf_36Kva.csv", delimiter=";")
    print('end of loading data set')
    df = df.fillna(method='bfill')
    manage_date_in_df(df)
    #df['HORODATE'] = pd.to_datetime(df['HORODATE'], infer_datetime_format=True, utc=True)
    region = 'Bretagne'
    df_bretagne = select_part_of_data_frame(df, select_all=True, region=region)
    conso_bretagne = np.array(df_bretagne["ENERGIE_SOUTIREE"])
    conso_bretagne_norm = conso_bretagne / 10 ** 9
    print('size temporal serie', len(conso_bretagne_norm))
    alpha = 0.5
    save_fold = f'results_proba_input_outout_alpha_{alpha}'
    make_dir(save_fold)
    plot_temporal_series(df_bretagne["HORODATE"], conso_bretagne_norm, day_interval=100)
    plt.savefig(f'{save_fold}/energy_consumption_inf_36Kva_bretagne.png')
    plt.close()

    # plot part of the temporal serie
    deb = -1000
    fin = -1
    plot_temporal_series(df_bretagne["HORODATE"][deb:fin], conso_bretagne_norm[deb:fin], day_interval=5)
    plt.savefig(f'{save_fold}/energy_consumption_inf_36Kva_bretagne_end.png')
    plt.close()

    temp_series = np.array([[conso_bretagne_norm, df_bretagne['hour_sin'], df_bretagne['dayofweek_sin'], df_bretagne['dayofyear_sin']]])
    train_input, train_target, values_to_predict, test_input, test_target = split_train_test(temp_series, nb_test_sequences=0, prop_seq_test=0.05)
    #print(df_bretagne["HORODATE"].dt.hour[:24])
    model = LSTM(hidden_layers=32,input_size=4, alpha=alpha)  # input features : conso_bretagne_norm, hour_of_day, day_of_week, day_of_year
    criterion = nn.MSELoss()
    n_epochs = 101
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    training_loop(n_epochs, model, optimiser, criterion, train_input, train_target, values_to_predict,
                  test_input, test_target, np.array(df_bretagne["HORODATE"]), save_fold=save_fold, title=f'Half-hourly energy consumption in {region} by delivery point with power under 36 kVA')




