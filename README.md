# 🔮 Time Series Forecasting — ORE Electricity Consumption ⚡

## 📌 Context

This Python project presents several methods to **forecast future values** of a time series.  
The use case relies on data provided by **ORE (Opérateurs de Réseaux d’Énergie)**, specifically the **half-hourly aggregated consumption** of delivery points with a subscribed power below 36 kVA ([download the data here](https://www.agenceore.fr)).

**ORE** unites all French electricity and gas distribution operators to offer a comprehensive view of distribution in France through a **single, free data portal**. It provides open data sets for electricity and gas distribution across multiple grid operators, along with visualizations, to **support the energy transition** in French regions.

---

## ⚙️ Methodology

### 1️⃣ Prophet Model

The initial modeling uses the **[Prophet](https://facebook.github.io/prophet/)** model developed by Facebook.  
**Prophet** models the signal as the sum of:
- 🔁 a **seasonal component** (daily and yearly periodicity),
- 📈 a **trend component**.

This model is particularly suitable for forecasting energy consumption data, which typically shows strong periodic patterns.

---

### 2️⃣ Models on the Residuals

To further improve the forecast, two **deep learning** models are trained on the **residuals** from the Prophet model:  

#### 🔄 Transformer Sequence-to-Sequence

A **Transformer Sequence-to-Sequence** is a model based on the **Transformer** architecture, originally developed for natural language processing.  
The models take as input an input sequence of specified lenght. It emdbends all the elements of the input sequence into a latent space, where the embedings of a given element of the context window is adjusted by using the embedding of the other elements. Then the embedings are used to predict the following elements of the time series. 


#### 🔁 LSTM (Long Short-Term Memory)

An **LSTM** is a type of **Recurrent Neural Network (RNN)** designed to handle sequential data and solve the **vanishing gradient problem**.  
It consists of **memory cells** that allow it to retain information over long periods.

---

## 📊 Results & Visualizations

➡️ For more details on the methodology, results, and interactive visualizations, please refer to the **notebook [`predictions_visualizations.ipynb`](./predictions_visualizations.ipynb)**.


