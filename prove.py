import matplotlib.pyplot as plt
import numpy as np

from sum_time_series import parallel_generate_series
from properties_time_series import Misure, plot_return_dist, plot_log_returns, plot_autocorrelation

def res(var, hyper_param):
    exponent, freq_range, C = var
    n_series_range, num_repetitions, series_length, sampling_freq, = hyper_param

    serie_generate, freqs_generate, amps_generate = parallel_generate_series(n_series_range, num_repetitions, series_length, sampling_freq, exponent, freq_range, C)

    m = Misure(serie_generate, 1001, 1, grid_variables=var)

    lags = [1, 10, 100, 1000, 10000]
    m.calc_misure(lags)

    return m, lags


def plot_all(m, lags):
    fig, ax = plt.subplots(2, 4, figsize=(20, 6))
    ax[0, 0].plot(m.s[:35])
    ax[0, 1].plot(m.s[:350])
    ax[0, 2].plot(m.s[:3500])
    ax[0, 3].plot(m.s[:35000])

    # fig.delaxes(ax[0, 3])
    #ax[0, 0].plot(m.s_normalized);

    plot_return_dist(m.return_distributions, lags, ax[0, 1])
    plot_log_returns(m.log_return_distributions, lags, m.kurtosis, m.skew, ax[0, 2])
    plot_autocorrelation(m.mean_acf, m.max_lag, m.Hurst_exponent, ax[0, 3])



    plt.suptitle(f"exponent, freq_range, C = {m.grid_variables}")
    plt.tight_layout()
    plt.show()



series_length = 36000  # Length of each series
sampling_freq = 1  # Sampling frequency in Hz

n_series_range = range(1001, 1, -790)

# Numero di ripetizioni per ogni valore intero nel range
num_repetitions = 15

C = 1
# freq_range = [(0.00001, 50), (0.000001, 500), (0.0001, 1), (0.0001, 0.1), (0.001, 1)]
freq_range = (1e-6, 500)
exp_range = [-0.7, -0.85, -0.999, -1.001, -1.15, -1.3]


def p1():
    for e in exp_range:
        m, lags = res((e, freq_range, C), (n_series_range, num_repetitions, series_length, sampling_freq))
        plot_all(m, lags)



if __name__ == '__main__':
    p1()