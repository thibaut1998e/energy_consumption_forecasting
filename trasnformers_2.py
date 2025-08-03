import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy.physics.quantum.circuitutils import random_reduce

from LSTM_model import draw_plot, to_numpy
from agence_ore_data import manage_date_in_df, select_part_of_data_frame
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch
from torch import optim




class Seq2SeqTransformer(nn.Module):
    """class that takes as input a sequence of lenght window_size and that returns a sequence of same lenght.
    The mapping between inputs and outputs is modeled by a transformer encoder followed by a linear block. This class can
    be used for time series forecasting by mapping the values of the sequences at times (t-window_size, .... t-1) to
    the values at time (t, ...., t+window_size). The forward methods also includes the possibility to give to the model
    exogeneous varaibles associated both to the input sequence and the output sequence"""
    def __init__(self, n_features,  window_size, n_head=4, num_layer=5, latent_space_size=64):
        """INPUTS
            - n_features : nombre de features = number of features of the input signal + 2*number of features of exogenous
                variables. Indeed, both present and future exogenous variables must be passed to the model (see forward function)
            - window_size : the architecture maps a sequence of lenght window_size to an other sequence of lenght window_size
            - n_head : number of head of attention blocks
            - num_layer : number of transformer layers
            - latent_space_size : size of the latent space that is used to encodesignal values
            """
        super().__init__()
        self.linear_in = nn.Linear(n_features, latent_space_size).cuda(0)
        self.transformer_layer = TransformerEncoderLayer(latent_space_size, n_head).cuda(0)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layer).cuda(0)
        self.linear_out = nn.Linear(window_size*latent_space_size, window_size).cuda(0)
        self.window_size = window_size

    def forward(self, seq, exog_variables, future_exog_variables):
        """forward pass through the model
        INPUTS :
        - seq : tensor of shape (window_size) with signal values
        - exog_variables : exogeneous variables associated to the input sequence. Tensor of shape (window_size, N) with
            N the number of exogenous variables
        - future_exog_variables : exogeneous variables asoociated to the output sequence. Tensor of shape (window_size, N)
        OUTPUT :
        - out : Tensor of shape (window_size). output sequences predicted by teh model
        """
        inp = concat_seq_exogeneous(seq, exog_variables, future_exog_variables)
        fetaures_1 = self.linear_in(inp)
        features = self.transformer(fetaures_1)
        out = self.linear_out(torch.flatten(features))
        return out

    def long_term_forecasting(self, input_seq, exog_input_seq, future_exog_variables, prediction_horizon):
        """Function used to predict at long term, beyond the window_size horizon, by iteratively putting the outputs of the prediction
        as the input of the next prediction. This is subject to accumulation of errors
        INPUTS :
            - input_seq : tensor of shape (window_size) with signal values
            - exog_input_seq : exogeneous variables associated to the input sequence. Tensor of shape (window_size, N) with
            N the number of exogenous variables
            - future_exog_variables : exogeneous variables associated to future values to predict. Shape (X, N) with X greater than prediction horizon
            - prediction horizon : number of steps in the future we want to predict
        OUTPUT
            - allforecast : tensor avec les predictions futures"""
        prediction = self.forward(input_seq, exog_input_seq, future_exog_variables[:self.window_size])
        all_forecast = [prediction.clone()]
        for i in range(prediction_horizon//self.window_size):
            prediction_i = self.forward(all_forecast[-1], future_exog_variables[i*self.window_size:(i+1)*self.window_size],
                                      future_exog_variables[(i+1)*self.window_size:(i+2)*self.window_size])
            all_forecast.append(prediction_i.clone())
        all_forecast = torch.cat(all_forecast)
        return all_forecast

def concat_seq_exogeneous(seq, exog_variables, future_exog_variables):
    seq=seq.unsqueeze(1)
    inp = torch.cat((seq, exog_variables, future_exog_variables), dim=1)
    return inp

def split_input_output_exogenous(sequence, exogenous_variables, window_size, step=20):
    """
    Split the data set in (inputs/outputs) couples. (Input/output) couples are in the form (sequence[i:i+window_size], sequence[i+window_size:i+2*window_size])
    INPUTS:
        sequence : np array or tensor that represents a temporal series of size L
        exogenous varianbles : np array ou tensor that represents the exogeneous varaiebles. Size (L, nb_variables)
        step : number of element between each window
    """
    inputs = [sequence[i:i+window_size] for i in range(0, len(sequence)-2*window_size, step)]
    ouputs = [sequence[i+window_size:i+2*window_size] for i in range(0,len(sequence)-2*window_size, step)]
    exog_inputs = [exogenous_variables[i:i + window_size] for i in range(0, len(sequence) - 2*window_size, step)]
    exog_ouputs = [exogenous_variables[i + window_size:i + 2*window_size] for i in range(0,len(sequence) - 2*window_size,step)]
    return inputs, ouputs, exog_inputs, exog_ouputs


def train_time_series_seq_to_seq_transformer(sequence, exogenous_variables, window_size, nb_epochs):
    """train a Seq to Seq transformer on the time series sequence, with specified exogeneous variales
    ENTREES :
        sequence  : tensor de shape (L)
        exogenous_variables : variables exogènes, de taille (L, N) with N the number of exogeneous variables
        window_size : size of the window used to map previous values of sequence to the next values
        nb_epochs : number of epochs of the training"""
    inputs, ouputs, exog_inputs, exog_ouputs = split_input_output_exogenous(sequence, exogenous_variables, window_size)
    n_features = concat_seq_exogeneous(inputs[0], exog_inputs[0], exog_ouputs[0]).shape[1]
    model = Seq2SeqTransformer(n_features, window_size)
    optimiser = optim.Adam(model.parameters(), lr=3*10**-5)
    losses = []
    for ep in range(nb_epochs):
        print('ep', ep)
        loss_epoch = 0
        for i in range(len(inputs)):
            input_seq, output_seq, exog_inp, exog_out = inputs[i], ouputs[i], exog_inputs[i], exog_ouputs[i]
            estimated_out = model.forward(input_seq, exog_inp, exog_out)
            loss = torch.mean((estimated_out-output_seq)**2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            loss_float = to_numpy(loss)
            loss_epoch += loss_float
        loss_epoch/=len(inputs)
        losses.append(loss_epoch)
        print('loss epoch', loss_epoch)
    return model, inputs, ouputs, exog_inputs, exog_ouputs

def to_tensor(arr):
    return torch.FloatTensor(arr).cuda(0)

if __name__ == '__main__':
    df = pd.read_csv("data/consommation_inf_36Kva.csv", delimiter=";")
    df = df.fillna(method='bfill')
    manage_date_in_df(df)
    region = 'Bretagne'
    df_bretagne = select_part_of_data_frame(df, select_all=True, region=region)
    df_bretagne["ENERGIE_SOUTIREE"] /= 10 ** 9
    sequence = df_bretagne["ENERGIE_SOUTIREE"].to_numpy()
    exog_variables = df_bretagne[['hour_sin', 'dayofweek_sin', 'dayofyear_sin']].to_numpy()
    nb_test_vals = 2000
    train_sequence, test_sequence = sequence[:-nb_test_vals], sequence[-nb_test_vals:]
    train_exog, test_exog = exog_variables[:-nb_test_vals], exog_variables[-nb_test_vals:]
    window_size = 100

    train_sequence, train_exog, test_exog = to_tensor(train_sequence), to_tensor(train_exog), to_tensor(test_exog)
    model, inputs, ouputs, exog_inputs, exog_ouputs = train_time_series_seq_to_seq_transformer(train_sequence, train_exog, window_size,100)

    forecasted = model.long_term_forecasting(train_sequence[-window_size:], train_exog[-window_size:], test_exog, 1800)
    forecasted = to_numpy(forecasted)


    plt.plot(forecasted)
    plt.plot(test_sequence)
    plt.show()
