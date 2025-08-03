import matplotlib.pyplot as plt
import numpy as np

def sin_waves_with_different_periods(N_pt = 1000, lenght = 20, n_sample = 50, T_min=0.5, T_max=10):
    X = np.linspace(0, lenght, N_pt)
    signals = [np.sin(2*np.pi*X/T) for T in np.linspace(T_min, T_max, n_sample)]
    return X, np.array(signals)

def sum_two_sin_waves(A1, A2, T1, T2, X, phi1=0, phi2=0, noise_variance=0):
    return A1*np.sin(2*np.pi*X/T1 + phi1) + A2*np.sin(2*np.pi*X/T2 + phi2) + noise_variance * np.random.randn(len(X))

def add_linear_tendencies(signal, X, coeff_dirr, origin=0):
    return signal + coeff_dirr * X + origin


if __name__ == '__main__':
    lenght = 40
    N_pt = 2000
    X = np.linspace(0, lenght, N_pt)
    signal = sum_two_sin_waves(1,0.5,5,1,X, noise_variance=0.1)
    signal = add_linear_tendencies(signal, X, 0.1)
    plt.plot(signal)
    plt.show()