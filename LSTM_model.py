import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.pyplot import legend
from torch import nn
from torch import optim
from manage_files import make_dir
from generate_simulated_signals import sin_waves_with_different_periods, sum_two_sin_waves, add_linear_tendencies
import matplotlib.dates as mdates

colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c']

class LSTM(nn.Module):
    """a LSTM model used to model a time series. It is composed of consecutive LSTM blocks (2 blocks for each unit of the model)"""
    def __init__(self, hidden_layers=64, input_size=1, alpha=0.1):
        """hidden layer : number of layers in one LSTM block
        input size : dimensionality of the input sequence
        alpha : during the training, the input a LSTM block is set to the actual value at the corresponding time step with probability 1-alpha
                and is set to the output of previous block with probability alpha"""
        super(LSTM, self).__init__()
        self.alpha = alpha
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(input_size, self.hidden_layers).cuda(0)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers).cuda(0)
        self.linear = nn.Linear(self.hidden_layers, 1).cuda(0)

    def forward(self, y, secondary_variables, future_preds=0, inference=False):
        """forward pass through the LSTM model. For each time step t, the values of y at time t is put inside an LSTM block
        and the output of the block is stored in a list that is finally returned. With probability p, the input of an LSTM block
         is instead the output of that last block. Future predictions are then made at horizon future_preds by iteratively
         putting the output of a block as the input of the following block.
         INPUT :
         - y (torch.tensor) shape (N_time_series, N_features, N_pt) with N_time_series the number of time
            series used to train the model, N_features the number of features, the first one is considered as the signal values and the others
            the exogeneous variables, N_pt the number of points in the train part of the time series
         - future_preds (int) : number of steps ahead for the prediction (0 if we do not want to make a prediction)
         - secondary_variables (N_time_series, N_features-1, N_pt-1) if inference is False, else  (N_time_series, N_features-1, N_pt_test-1)
                exogenous variables used for predction
        OUTPUT :
         - outputs (torch.tensor) Shape (N_time_series, N_pt + future_preds). Successive outputs of the LSTM blocks
        """
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).cuda(0)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).cuda(0)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).cuda(0)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).cuda(0)
        x = 0
        for it in range(y.shape[2]):
            input_t = y[:,:,it] # n_sample, input_size
            x+=1
            rd = np.random.random()
            #print('it', it)
            inp = torch.cat((output, secondary_variables[:,:,it]), axis=1) if it > 1 and rd < self.alpha and not inference else input_t
            h_t, c_t = self.lstm1(inp, (h_t, c_t))  # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))  # new hidden and cell states
            output = self.linear(h_t2)  # output from the last FC layer
            outputs.append(output)

        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            output = output if secondary_variables is None else torch.cat((output, secondary_variables[:,:,i]), axis=1)
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor
        outputs = torch.cat(outputs, dim=1)
        return outputs

def draw_plot(predicted_values, train_target, values_to_predict, n, iter, X_axis, title='', save_fold=''):
    def draw(x_min, x_max, save_name=f'prediction_{iter}'):
        plt.figure(figsize=(12, 6))
        plt.title(title, fontsize=18)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.plot(X_axis[np.arange(n)], train_target.squeeze(), 'g', linewidth=0.5)
        plt.plot(X_axis[np.arange(n, predicted_values.shape[1])], values_to_predict.squeeze()[0], 'g', linewidth=0.5, label='True values')
        plt.plot(X_axis[np.arange(n)], predicted_values[0, :n], 'r', linewidth=2.0,
                 label='One step ahead prediction on train data')
        plt.plot(X_axis[np.arange(n - 1, predicted_values.shape[1])], predicted_values[0, n - 1:], 'r:', linewidth=2.0,
                 label='Forecasting')
        ax = plt.gca()
        days_interval = len(X_axis)//500
        plt.grid()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=days_interval))
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.xlabel('Date', fontsize=20)
        plt.ylabel(r'Energy (Wh/$10^{9}$)', fontsize=20)
        plt.xlim(x_min, x_max)
        plt.savefig(f"{save_fold}/{save_name}.png", dpi=200)
        plt.close()

    draw(X_axis[0], X_axis[-1])
    draw(X_axis[int(0.95*n)], X_axis[-1], save_name=f'prediction_{iter}_sub_part')
    draw(X_axis[n-10], X_axis[n+100], save_name=f'prediction_{iter}_small_sub_part')
    draw(X_axis[n + 100], X_axis[n + 400], save_name=f'prediction_{iter}_small_later_sub_part')

def draw_losses_graph(losses, save_fold, title):
    plt.plot(losses)
    plt.xlabel('iterations')
    plt.ylabel('Losses')
    plt.title(title)
    plt.grid()
    plt.savefig(f'{save_fold}/{title}.png')
    plt.close()

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def training_loop(n_epochs, model, optimiser, loss_fn,train_input, train_target, values_to_predict, test_input,
                  test_target, X_axis, save_fold='', draw_plot_iter=10, title=''):
    """train an LSTM model and make prediction for future time steps, plot the predictions and saves the best model.
    INPUTS :
        - n_epochs (int) : number of epochs of training
        - model : instance of the class LSTM
        - optimiser : optimize used (for example Adam)
        - loss_fn : loss finction used for training (for example nn.MseLoss)
        - train_input (torch.tensor). Shape (N_time_series, N_features, N_pt) with N_time_series the number of time
            series used to train the model, N_features the number of features, the first one is considered as the signal values and the others
            the exogeneous variables, N_pt the number of points in trhe train part of the time series
        - train_target (torch.tensor). Shape (N_time_series, N_pt) : the expected outputs at each block of the LSTM (the values
            of the time series shifted from one time stamp)
        - values_to_predict (torch.tensor). Shape (N_time_series, N_features, N_pt_test). A prediction is made at the horizon
            N_pt_test and the predicted values are compared to values_to_predict.
        - test_inputs (torch.tensor). Shape (N_time_series_test, N_features, N_pt). Time series not present in the
        training data set on which predictions are made. The result of the prediction is compared to test_target (torch.tensor
        with shape (N_time_series_test, N_features, N_pt_test)
        - X_axis (np.array) Shape N_pt + N_pt_test. Values of the X_axis w.r.t the predicted values are plot:
        - save_fold (str): location where the results are saved.
        - draw_plot_iter (int) : the plot of the results are saved every draw_plot_iter iterations
        - title (str) : title of the graph as well as the saved model """

    make_dir(save_fold)
    train_losses = []
    pred_losses_train = []
    pred_losses_test = []

    best_pred_loss_train = float('inf')
    best_model_path = f'{save_fold}/best_model.pth'
    best_pred_seq = []
    print("Train loss |  Prediction loss test sequences | Prediction loss train sequences ")
    for i in range(n_epochs):
        print('ep', i)
        def closure():
            optimiser.zero_grad()
            out = model(train_input, secondary_variables=train_input[:, 1:, :])
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)

        with torch.no_grad():
            future = values_to_predict.shape[2]
            pred_train_seq = model(train_input, secondary_variables=values_to_predict[:, 1:, :], future_preds=future, inference=True)
            pred_test_seq = model(test_input, secondary_variables=test_target[:, 1:, :], future_preds=future, inference=True)
            pred_loss_test_sequences = to_numpy(loss_fn(pred_test_seq[:, -future:], test_target[:, 0, :]))
            pred_loss_train_sequences = to_numpy(loss_fn(pred_train_seq[:, -future:], values_to_predict[:, 0, :]))
            y_train_seq = to_numpy(pred_train_seq)
            y_test_seq = to_numpy(pred_test_seq)

        if i % draw_plot_iter == 0:
            draw_plot(y_train_seq, to_numpy(train_target), to_numpy(values_to_predict), train_input.shape[2], i, X_axis, save_fold=save_fold, title=title)

        if pred_loss_train_sequences < best_pred_loss_train:
            best_pred_loss_train = pred_loss_train_sequences
            best_pred_seq = y_train_seq
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {i} with train pred loss {best_pred_loss_train:.3e}")



        out = model.forward(train_input,secondary_variables=train_input[:, 1:, :], inference=True)
        loss_print = to_numpy(loss_fn(out, train_target))
        train_losses.append(loss_print)
        pred_losses_train.append(pred_loss_train_sequences)
        pred_losses_test.append(pred_loss_test_sequences)

        print(
            f"Step: {i} {loss_print:.3e} | "
            f"{pred_loss_test_sequences:.3e} | "
            f"{pred_loss_train_sequences:.3e}"
        )

    draw_losses_graph(train_losses, save_fold, "Train losses")
    draw_losses_graph(pred_losses_train, save_fold, "Prediction losses on train sequences")
    if test_input.shape[0] > 0:
        draw_losses_graph(pred_losses_test, save_fold, "Prediction losses on test sequences")

    draw_plot(best_pred_seq, to_numpy(train_target), to_numpy(values_to_predict), train_input.shape[2], 'end', X_axis, save_fold=save_fold,
              title=title)
    torch.save(model.state_dict(), f'{save_fold}/end_model.pth')
    return y_train_seq

def split_train_test(temp_series, prop_seq_test=0.2, nb_test_sequences=0, nb_test=None):
    """temp_series is an np or pandas array of shape (N, n_input, M) with N the number of time series and M the number of points
    in one time serie, n_input the number of features. This function split the data in the following way : nb_test_sequences are used for validation, and the rest for training.
    In the training sequences, not all the data is used but only the first propotion 1 - prop_seq_test . The end of the
    training sequences is also used for validation"""
    N, _, M = temp_series.shape
    if prop_seq_test is not None:
        x = int(prop_seq_test*M)
    else:
        x=nb_test
    temp_series = torch.FloatTensor(temp_series).cuda(0)
    train_input = temp_series[nb_test_sequences:,:, :-x]
    train_target = temp_series[nb_test_sequences:,0, 1:-x+1]
    values_to_predict = temp_series[nb_test_sequences:,:, -x:] # target values to predict at the end of the train sequences
    test_input = temp_series[:nb_test_sequences,:, :-x] # start of test sequences, used to predict the following values of the test sequence
    test_target = temp_series[:nb_test_sequences,:, -x:] # values to predict in the test sequences
    return train_input, train_target, values_to_predict, test_input, test_target


if __name__ == '__main__':
    Te = 0.08 # sampling period : quite important parameter to set. If to big, the long term prediction will diverge quiclky due too much error propagation (chaos),
                #if to small, the signal is undersampled and we miss information.
    lenght = 100
    N_pt = int(lenght/Te)
    print('pt', N_pt)
    n_sample = 20
    n_test_sequences = 0
    noise_variance = 0.1
    coeff_tendency = 0
    #_, temp_series = sin_waves_with_different_periods(N_pt, lenght, n_sample, T_min=1, T_max=1)
    X = np.linspace(0, lenght, N_pt)
    T1 = 5
    T2 = 1
    input_size = 2
    temp_series = add_linear_tendencies(sum_two_sin_waves(1,0.5,5,1,X, noise_variance=noise_variance), X, coeff_tendency)
    secondary_variable = np.sin(2*np.pi*X/T1)
    temp_series = np.array([temp_series, secondary_variable])
    #temp_series = np.expand_dims(temp_series, axis=0)
    temp_series = np.array([temp_series]*n_sample) #shape n_sample, nb_input, sample_lenght
    print('shp', temp_series.shape)
    train_input, train_target, values_to_predict, test_input, test_target = split_train_test(temp_series,
                                                                                             nb_test_sequences=n_test_sequences, prop_seq_test=0.2)

    model = LSTM(hidden_layers=64, input_size=input_size)
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 100
    X_axis = np.arange(temp_series.shape[2])
    print('train input shape', train_input.shape)
    training_loop(n_epochs, model, optimiser, criterion, train_input, train_target, values_to_predict,
                  test_input, test_target, X_axis, save_fold='predictions_with_secondary_variable')