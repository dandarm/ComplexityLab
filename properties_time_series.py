import numpy as np
import pandas as pd
from scipy.signal import resample
import matplotlib.pyplot as plt

import concurrent.futures

#import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

import sys

sys.path.append("../LorenzCaosTest")
from Test_Chaos_noparallel import caostest10_noparallel

from sum_time_series import (generate_sinusoidal_series, rndm, generate_power_law_series_negative, generate_inversely_related_series,
                             calculate_spectral_entropy, perform_fft, identify_peaks,
                             calculate_f1_score, calculate_f1_score_corrected, analyze_spectrum_detail, plot_exp_distrib,
                             parallel_generate_series
                            )

from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis


def check_shape(numpy_array, for_pandas=False):
    assert isinstance(numpy_array, np.ndarray), "Non è un numpy array"
    assert numpy_array.ndim == 2, "Non è una matrice a 2 dimensioni"

    if for_pandas:
        timesteps, num_series = numpy_array.shape
    else:
        num_series, timesteps = numpy_array.shape

    # se non è come mi aspetto, traspongo l'array
    if timesteps < num_series:
        return numpy_array.T
    else:
        return numpy_array


def calc_volatility(array, window=10):
    array = check_shape(array, for_pandas=True)
    df = pd.DataFrame(array)
    vol = df.rolling(window=window).std() * np.sqrt(window)
    return vol.fillna(0).values


# region autocorrelazione

def calc_autocorrelation(time_series_array, max_lag=1000):
    """
    Calcola l'autocorrelazione media su tutte le serie temporali per lag fino a max_lag.
    """

    time_series_array = check_shape(time_series_array)
    num_series, timesteps = time_series_array.shape

    max_lag = min(max_lag, timesteps)

    # Inizializziamo un array per salvare le autocorrelazioni medie
    mean_acf = np.zeros(max_lag)
    H = 0

    # Calcoliamo l'autocorrelazione per ciascuna serie e facciamo la media
    for series in time_series_array:
        # Calcoliamo l'autocorrelazione della serie per lag fino a max_lag
        acf_values = acf(series, nlags=max_lag, fft=True)  # Otteniamo un array di autocorrelazioni

        # Aggiungiamo l'autocorrelazione ai valori totali
        mean_acf += acf_values[:max_lag]

        # Calcola l'esponente di Hurst usando il metodo R/S
        H += nolds.hurst_rs(series)

        # Facciamo la media dividendo per il numero di serie
    mean_acf /= num_series
    H /= num_series

    return mean_acf, H


def plot_autocorrelation(mean_acf, max_lag, Hurst_exponent, ax):
    """
    Plotta l'autocorrelazione media fino a max_lag.
    """
    ax.plot(range(1, max_lag + 1), mean_acf, marker='o')
    ax.set_title(f'Autocorrelazione - Esponente Hurst = {round(Hurst_exponent, 4)}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelazione Media')
    #ax.set_ylim(0.85, 1.0)
    ax.grid(True)

# endregion



# region Ritorni

def calc_ret_dist(time_series_array, lags):
    # Calcoliamo i ritorni percentuali per ciascun lag
    # e media su tutte le serie
    return_distributions = {lag: [] for lag in lags}
    log_return_distributions = {lag: [] for lag in lags}

    for lag in lags:  # sorted(lags, reverse=True):
        # Calcoliamo i ritorni per ciascuna serie
        all_returns = calc_ret(time_series_array, lag)
        all_log_returns = calc_log_ret(time_series_array.T, lag)

        # Calcoliamo la media lungo l'asse 0 (ossia, la media tra tutte le serie)
        mean_returns = np.mean(all_returns, axis=0)
        return_distributions[lag] = mean_returns

        mean_log_returns = np.mean(all_log_returns, axis=0)
        log_return_distributions[lag] = mean_log_returns

    return return_distributions, log_return_distributions


def calc_ret(time_series_array, lag=1):
    all_returns = [(serie[lag:] - serie[:-lag]) / serie[:-lag] for serie in time_series_array]
    all_returns = np.array(all_returns / np.std(all_returns))
    # Shape (num_series, n_values_per_lag)
    return all_returns


def calc_log_ret(time_series_array, lag=1):
    # Calcoliamo i rendimenti logaritmici
    time_series_array = check_shape(time_series_array, for_pandas=True)

    prices = pd.DataFrame(time_series_array)
    returns = np.log(prices / prices.shift(lag)).dropna()
    return returns.values.T


def pos_neg_rets(returns, n_bins=50):
    # Separiamo i ritorni positivi e negativi
    positive_returns = returns[returns > 0]
    negative_returns = -returns[returns < 0]  # Prendiamo i valori assoluti dei negativi

    # Verifica se ci sono valori positivi e negativi per evitare errore di array vuoto
    # maxim, minim = positive_returns.min(), positive_returns.max()
    maxim, minim = 1e+2, 1e-6
    #if len(positive_returns) > 0:
    bins_positive = np.logspace(np.log10(minim), np.log10(maxim), n_bins)
    #else:
    #    bins_positive = np.array([])

    #if len(negative_returns) > 0:
    bins_negative = np.logspace(np.log10(minim), np.log10(maxim), n_bins)
    #else:
    #    bins_negative = np.array([])

    return positive_returns, negative_returns, bins_positive, bins_negative


def plot_distribution(array, ax, **kwargs):
    assert isinstance(array, np.ndarray), "Errore, non è un numyp array"
    logx = kwargs.get('logx')
    logy = kwargs.get('logy')
    nome_variabile = kwargs.get('nome_variabile')

    val_neg = np.any([np.any(np.array(v).ravel() < 0) for v in array])
    colormap = plt.cm.rainbow
    if isinstance(array, dict):  # abbiamo tanti valori
        for i, (parametro, valori) in enumerate(array.items()):
            color = colormap(i / len(array))
            if logx and val_neg:
                positive_returns, negative_returns, bins_positive, bins_negative = pos_neg_rets(valori, n_bins=50)
                ax.hist(positive_returns, bins=bins_positive, color=color, density=True, histtype="step", label=f'Lag {parametro}')
                ax.hist(-negative_returns, bins=-bins_negative[::-1], color=color, density=True, histtype="step")

    if logx:
        if val_neg:
            ax.set_xscale('symlog', linthresh=1e-5)
        else:
            ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(nome_variabile)
    ax.set_ylabel('Densità')

    ax.legend(prop={'size': 6})

def plot_return_dist(return_distributions, lags, ax):
    # Variazione di alpha in base al numero di lag
    # min_alpha = 0.1  # Alpha per il lag più piccolo
    # max_alpha = 0.99  # Alpha per il lag più grande
    # alphas = np.linspace(max_alpha, min_alpha, len(lags))
    colormap = plt.cm.rainbow
    # Tracciamo le distribuzioni dei ritorni
    #chiavi = sorted(return_distributions.keys(), reverse=True)
    #da_ciclare = {k: return_distributions[k] for k in chiavi}
    for i, (lag, mean_returns) in enumerate(return_distributions.items()):
        color = colormap(i / len(lags))
        positive_returns, negative_returns, bins_positive, bins_negative = pos_neg_rets(mean_returns, n_bins=50)
        ax.hist(positive_returns, bins=bins_positive, label=f'Lag {lag}', color=color, density=True, histtype="step")  # alpha=alphas[i],
        ax.hist(-negative_returns, bins=-bins_negative[::-1], color=color, density=True, histtype="step")

    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_yscale('log')

    ax.set_xlabel('Ritorno')
    ax.set_ylabel('Densità')
    ax.set_title('Ritorni standardizzati per diversi Lag')
    ax.legend(prop={'size': 6})
    ax.grid(True)


def plot_log_returns(log_return_distributions, lags, kurt, skew, ax):
    colormap = plt.cm.rainbow
    ypos = 0.98

    for i, (lag, returns) in enumerate(log_return_distributions.items()):
        color = colormap(i / len(lags))
        positive_returns, negative_returns, bins_positive, bins_negative = pos_neg_rets(returns, n_bins=50)
        ax.hist(positive_returns, bins=bins_positive, density=True, color=color, histtype="step", label=f'Lag {lag}')
        ax.hist(-negative_returns, bins=-bins_negative[::-1], density=True, histtype="step", color=color)
        # Aggiungiamo il testo con i valori di kurtosi e skewness
        textstr = f'Kurt: {kurt[lag]:.2f}\nSkew: {skew[lag]:.2f}'

        # Posizioniamo il testo nell'angolo in alto a sinistra
        # Posiziona il testo al 90% del limite superiore per evitare sovrapposizioni
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.02, ypos, textstr, transform=ax.transAxes, fontsize=7, color=color,
                verticalalignment='top', horizontalalignment='left')#, bbox=props)
        ypos -= 0.11


    ax.set_xscale('symlog', linthresh=1e-5)
    ax.set_yscale('log')
    ax.legend(loc='upper right', prop={'size': 6})
    ax.grid(True)

# endregion


def show_stazionariety(returns):
    """
    Se il valore p è inferiore a 0.05, possiamo rifiutare l'ipotesi nulla di non stazionarietà (la serie è stazionaria).
    Se il valore p è maggiore di 0.05, non possiamo rifiutare l'ipotesi nulla (la serie potrebbe essere non stazionaria).
    """
    adf_result = adfuller(returns)
    print('Test ADF:')
    print(f'Statistica di test: {adf_result[0]:.4f}')
    print(f'Valore p: {adf_result[1]:.4f}')
    for key, value in adf_result[4].items():
        print(f'Valore critico {key}: {value:.4f}')


def asimmetria(returns):
    """ Calcoliamo la skewness (asimmetria) e la kurtosis (curtosi) dei rendimenti.
    returns è un dizionario per ogni lag
    """
    skewness = {}
    kurt = {}
    for lag, ret in returns.items():
        skewness[lag] = skew(ret)
        kurt[lag] = kurtosis(ret, fisher=True)  # fisher=True per avere la kurtosi in eccesso

    return skewness, kurt


class Misure():
    def __init__(self, time_series_array, num_serie_sommate, i_example, grid_variables):
        self.time_series_array = time_series_array
        self.i_example = i_example

        ss = np.array(time_series_array[num_serie_sommate]).T
        self.s_normalized = (ss - np.mean(ss, axis=0)) / (np.std(ss, axis=0))
        self.ss = ss
        self.s = ss[:, i_example]

        self.return_distributions, self.log_return_distributions = None, None
        self.mean_acf, self.Hurst_exponent = None, None
        self.mean_acf_logret, self.Hurst_exponent_logret = None, None
        self.max_lag = 0

        self.grid_variables = grid_variables

    def calc_misure(self, lags):
        self.return_distributions, self.log_return_distributions = calc_ret_dist(self.ss.T, lags)

        self.max_lag = max(lags)
        # Autocorrelazione delle serie
        self.mean_acf, self.Hurst_exponent = calc_autocorrelation(self.ss.T, self.max_lag)
        # Autocorrelazione dei ritorni
        self.mean_acf_logret, self.Hurst_exponent_logret = calc_autocorrelation(self.log_return_distributions, self.max_lag)

        self.kurtosis, self.skew = asimmetria(self.log_return_distributions)


