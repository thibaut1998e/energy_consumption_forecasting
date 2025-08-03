import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import pandas as pd
import numpy as np

from LSTM_model import draw_plot
from agence_ore_data import manage_date_in_df, select_part_of_data_frame

# Chargement et préparation des données (votre code existant)
df = pd.read_csv("data/consommation_inf_36Kva.csv", delimiter=";")
df = df.fillna(method='bfill')
manage_date_in_df(df)
region = 'Bretagne'
df_bretagne = select_part_of_data_frame(df, select_all=True, region=region)
df_bretagne["ENERGIE_SOUTIREE"] /= 10**9
dates = df_bretagne["HORODATE"]
exog_list = ['hour_sin', 'dayofweek_sin', 'dayofyear_sin']
cols_to_keep = ['ENERGIE_SOUTIREE'] + exog_list + ['available_mask']
df_bretagne["available_mask"] = 1
df_bretagne = df_bretagne[cols_to_keep]

h = 1000

# Convertir au format NeuralForecast (long format avec unique_id, ds, y)
df_nf = pd.DataFrame({
    'unique_id': [region] * len(df_bretagne),
    'ds': range(len(df_bretagne)),
    'y': df_bretagne['ENERGIE_SOUTIREE'].values
})

# Ajouter les variables exogènes
for col in exog_list:
    df_nf[col] = df_bretagne[col].values

print("Format du DataFrame NeuralForecast:")
print(df_nf.head())
print("Colonnes:", df_nf.columns.tolist())

# Séparer train/test
train_df = df_nf.iloc[:-h].copy()
test_df = df_nf.iloc[-h:].copy()

print(f"Taille train: {len(train_df)}, Taille test: {len(test_df)}")

# Créer les données futures (avec exogènes mais sans y)
futr_df = test_df[['unique_id', 'ds'] + exog_list].copy()

print("Format des données futures:")
print(futr_df.head())

# Initialiser le modèle NHITS
nhits = NHITS(
    h=h,
    input_size=1000,
    futr_exog_list=exog_list,
    stack_types=['identity', 'identity', 'identity'],
    n_pool_kernel_size=[8, 4, 1],
    n_freq_downsample=[8, 4, 1],
    max_steps=1000  # Ajoutez ceci pour limiter l'entraînement si nécessaire
)

# Créer l'instance NeuralForecast
nf = NeuralForecast(models=[nhits], freq=1)

print("Début de l'entraînement...")
# Entraîner le modèle
nf.fit(df=train_df)

print("Début de la prédiction...")
# Faire les prédictions
forecasts = nf.predict(futr_df=futr_df)

print("Prédictions terminées!")
print("Shape des forecasts:", forecasts.shape)
print("Colonnes des forecasts:", forecasts.columns.tolist())
print("Premières prédictions:")
print(forecasts.head())

# Optionnel : Comparer avec les vraies valeurs
if len(test_df) > 0:
    # Fusionner les prédictions avec les vraies valeurs
    comparison = forecasts.merge(test_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
    comparison['error'] = comparison['y'] - comparison['NHITS']
    print("\nComparaison prédictions vs réalité:")
    print(comparison[['ds', 'y', 'NHITS', 'error']].head(10))

print('forecast shape', forecasts.shape)
print('train df shape', train_df.shape)
print('test_df shape', test_df.shape)
print('colonnes train df', train_df.columns.tolist())
print('colonnes test df', test_df.columns.to_list())

#plt.plot(train_df['ds'], train_df['y'], c='b')
plt.plot(test_df['ds'], test_df['y'], c='b')
plt.plot(forecasts['ds'], forecasts['NHITS'], c='r', linestyle='--')
plt.show()

#draw_plot(forecasts, train_df.to_numpy(), test_df.to_numpy(), len(train_df), )