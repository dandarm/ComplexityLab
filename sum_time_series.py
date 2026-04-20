import numpy as np
import matplotlib.pyplot as plt

import concurrent.futures

from scipy.signal import welch, find_peaks
from scipy.stats import entropy


def generate_sinusoidal_series(n, length, fs):
    """
    Generate n sinusoidal time series of a given length and sampling frequency (fs).
    Each sinusoidal wave will have a random frequency and amplitude.
    """
    t = np.arange(length) / fs
    series = np.zeros((n, length))
    freqs = []

    for i in range(n):
        freq = np.random.uniform(0.1, 5)  # Random frequency between 0.1 and 10 Hz
        amp = np.random.uniform(0.1, 40)    # Random amplitude between 0.5 and 2
        series[i] = amp * np.sin(2 * np.pi * freq * t)
        freqs.append(freq)

    return series, t, freqs

def rndm(v_min, v_max, exponent, size=1, rng=None):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b
    Secondo questa formula il vero esponente è g-1 quindi se voglio una power law con -3 devo passare g=-2
    ---> aggiungo quì 1."""
    g = exponent + 1.0
    #print(f"a: {a}, b: {b}, g: {g}")
    if rng is None:
        rng = np.random.default_rng()
    #r = np.random.random(size=size)
    r = rng.random(size=size)
    ag, bg = v_min**g, v_max**g
    return (ag + (bg - ag)*r)**(1./g)

def generate_power_law_series_negative(n, length, fs, exponent=-2.0, freq_range=(0.1, 10), amp_range=(0.5, 2)):
    """
    Generate n sinusoidal time series of a given length and sampling frequency (fs),
    where frequencies and amplitudes follow a power law distribution with a negative exponent.
    
    Parameters:
    - n: Number of sinusoidal series to generate.
    - length: Length of each time series.
    - fs: Sampling frequency.
    - alpha: Negative exponent of the power law distribution.
    - freq_range: Tuple indicating the range of frequencies.
    - amp_range: Tuple indicating the range of amplitudes.
    
    Returns:
    - series: Generated sinusoidal series.
    - t: Time vector.
    - freqs: Frequencies used.
    - amps: Amplitudes used.
    """
    t = np.arange(length) / fs
    series = np.zeros((n, length))
    
    # Genera frequenze e ampiezze con legge di potenza inversa
    #pos_alpha = abs(alpha)
    # sbagliato!!!!
    #freqs = freq_range[1] - (freq_range[1] - freq_range[0]) * np.random.power(pos_alpha, size=n)
    #amps = amp_range[1] - (amp_range[1] - amp_range[0]) * np.random.power(pos_alpha, size=n)
    
    #uso la distribuzione che gia' conosco
    freqs = rndm(freq_range[0], freq_range[1], exponent, size=n)
    amps = rndm(amp_range[0], amp_range[1], exponent, size=n)
    
    
    for i in range(n):
        series[i] = amps[i] * np.sin(2 * np.pi * freqs[i] * t)
    
    return series, t, freqs, amps

def generate_inversely_related_series(n, length, fs, exponent=-1.3, freq_range=(0.1, 10), r_min=0.1, r_max=10.0, C=1.0):
    """
    Generate n sinusoidal time series of a given length and sampling frequency (fs),
    where frequencies and amplitudes are inversely related with some random variation.
    
    Parameters:
    - n: Number of sinusoidal component series to generate.
    - length: Length of each time series.
    - fs: Sampling frequency.
    - exponent: Exponent for the power law distribution.
    - freq_range: Tuple indicating the range of frequencies.
    - r_min, r_max: Range for the random factor applied to the inverse relation.
    - C: Constant to scale the amplitude inversely related to frequency.
    
    Returns:
    - series: Generated sinusoidal series.
    - t: Time vector.
    - freqs: Frequencies used.
    - amps: Amplitudes used.
    """
    t = np.arange(length) / fs
    series = np.zeros((n, length))
    
    rng = np.random.default_rng()
    # Generare le frequenze usando la funzione rndm per la distribuzione a esponente negativo
    freqs = rndm(freq_range[0], freq_range[1], exponent, size=n, rng=rng)
    
    r = np.logspace(np.log10(r_min), np.log10(r_max), num=n)  # Random factor to maintain variability    
    # Calcola le ampiezze con un esponente per cui risulti Lo spettro di potenza proporzionale a f^-1
    gamma = 1  # Esponente desiderato della PSD \prop A(f)^2 * P(f)
    amp_exponent = (- gamma - exponent) / 2
    amp_exponent = - 0.5 #exponent
    amps = (C * freqs ** amp_exponent) * r
    #amps = C * r * freqs  ** (-0.2) # exponent 
    
    
    # fase random
    phi = rng.uniform(0, 2 * np.pi, size=n)
    
    series = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * t + phi[:, None])        
    composite_serie = np.sum(series, axis=0)
    
    return composite_serie, t, freqs, amps


def generate_single_series(n, series_length, sampling_freq, exponent, freq_range, C):
    """Genera una singola serie usando i parametri forniti."""
    composite_serie, _, freqs, amps = generate_inversely_related_series(n, series_length, sampling_freq, exponent, freq_range, C=C)
    # return composite_serie, freqs, amps
    return np.copy(composite_serie), np.copy(freqs), np.copy(amps)


# Funzione per parallelizzare la generazione delle serie
def parallel_generate_series(n_series_range, num_repetitions, series_length, sampling_freq, exponent, freq_range, C):
    serie_generate = {}
    freqs_generate = {}
    amps_generate = {}
    # Lista che ripete ogni valore di n per num_repetitions volte
    # extended_n_series_range = [n for n in n_series_range for _ in range(num_repetitions)]

    # Funzione parallela
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for n in n_series_range:
            serie_generate[n] = []
            freqs_generate[n] = []
            amps_generate[n] = []
            for _ in range(num_repetitions):
                # Sottomettiamo ogni task al pool di processi e raccogliamo i futures
                futures[(n, _)] = executor.submit(generate_single_series, n, series_length, sampling_freq, exponent, freq_range, C)

        # Recuperiamo i risultati
        for (n, _), future in futures.items():
            try:
                composite_serie, freqs, amps = future.result()
                serie_generate[n].append(composite_serie)
                freqs_generate[n].append(freqs)
                amps_generate[n].append(amps)
            except Exception as e:
                print(f"Errore durante il recupero del risultato per n={n}: {e}")

    return serie_generate, freqs_generate, amps_generate



def calculate_spectral_entropy(series, fs):
    """
    Calculate the spectral entropy of a time series.
    """
    f, Pxx = welch(series, fs=fs)
    Pxx_norm = Pxx / np.sum(Pxx)  # Normalize the power spectrum
    return entropy(Pxx_norm)

def perform_fft(series, fs):
    """
    Perform Fast Fourier Transform (FFT) on a given time series.
    """
    n = len(series)
    f = np.fft.fftfreq(n, 1/fs)
    fft_values = np.fft.fft(series)
    return f[:n // 2], np.abs(fft_values)[:n // 2]  # Only return positive frequencies

def identify_peaks(fft_magnitude, fft_frequencies, height=0.1):
    # Identificare i picchi significativi nel segnale FFT
    peaks, _ = find_peaks(fft_magnitude, height=height * np.max(fft_magnitude))
    identified_frequencies = fft_frequencies[peaks]
    return peaks, identified_frequencies

def calculate_f1_score(original_frequencies, fft_frequencies, fft_magnitude, tolerance=0.08):
    """
    Calculate the F1-score by comparing the original frequencies 
    with those identified in the FFT magnitude spectrum.
    
    Parameters:
    - original_frequencies: The true frequencies used to generate the sinusoidal series.
    - fft_frequencies: The frequencies obtained from the FFT.
    - fft_magnitude: The magnitude of the FFT at each frequency.
    - tolerance: The tolerance within which a frequency match is considered correct.
    
    Returns:
    - f1_score: The F1-score combining precision and recall.
    - precision: The precision of the identified frequencies.
    - recall: The recall of the identified frequencies.
    """
    # Identificare i picchi significativi nel segnale FFT
    peaks, identified_frequencies = identify_peaks(fft_magnitude, fft_frequencies)
    
    correct_matches = 0
    
    # Verificare se le frequenze identificate corrispondono alle frequenze originali
    for original_freq in original_frequencies:
        if any(np.abs(identified_frequencies - original_freq) <= tolerance):
            correct_matches += 1
    
    # Precisione: quante delle frequenze identificate sono corrette
    precision = correct_matches / len(identified_frequencies) if len(identified_frequencies) > 0 else 0
    if precision > 1:
        print(f"La precisione e'stranamente maggiore di 1: {precision}")
    
    # Richiamo: quante delle frequenze originali sono state identificate
    recall = correct_matches / len(original_frequencies)
    
    # Calcolare l'F1-score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score, precision, recall, identified_frequencies

def calculate_f1_score_corrected(original_frequencies, fft_frequencies, fft_magnitude, tolerance=0.2, height=0.1):
    """
    Calcola l'F1-score evitando la duplicazione di associazione dei picchi a frequenze originali vicine.
    
    Parameters:
    - original_frequencies: Le frequenze originali utilizzate per generare le serie sinusoidali.
    - fft_frequencies: Le frequenze ottenute dalla FFT.
    - fft_magnitude: L'ampiezza della FFT a ciascuna frequenza.
    - tolerance: La tolleranza entro cui una corrispondenza di frequenze è considerata corretta.
    
    Returns:
    - f1_score: L'F1-score che combina precisione e richiamo.
    - precision: La precisione delle frequenze identificate.
    - recall: Il richiamo delle frequenze identificate.
    - identified_frequencies: Le frequenze identificate nei picchi dello spettro.
    """
    # Identificare i picchi significativi nel segnale FFT
    peaks, identified_frequencies = identify_peaks(fft_magnitude, fft_frequencies, height)
    
    correct_matches = 0
    matched_original_freqs = []
    matched_peaks = []

    # Verificare se le frequenze identificate corrispondono alle frequenze originali
    for original_freq in original_frequencies:
        # Trova il picco identificato più vicino alla frequenza originale
        closest_peak = min(identified_frequencies, key=lambda x: abs(x - original_freq))
        
        # Verifica se la distanza tra il picco e la frequenza originale è entro la tolleranza
        if abs(closest_peak - original_freq) <= tolerance and closest_peak not in matched_peaks:
            if original_freq not in matched_original_freqs:
                correct_matches += 1
                matched_original_freqs.append(original_freq)
                matched_peaks.append(closest_peak)
    
    # Precisione: quante delle frequenze identificate sono corrette
    precision = correct_matches / len(identified_frequencies) if len(identified_frequencies) > 0 else 0
    
    # Richiamo: quante delle frequenze originali sono state identificate
    recall = correct_matches / len(original_frequencies)
    
    # Calcolare l'F1-score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return f1_score, precision, recall, identified_frequencies


def analyze_spectrum_detail(original_frequencies, fft_frequencies, fft_magnitude, tolerance=0.1):
    """
    Analizza lo spettro delle frequenze e confronta i picchi identificati con le frequenze originali.
    """
    # Identificare i picchi significativi nel segnale FFT
    peaks, identified_frequencies = identify_peaks(fft_magnitude, fft_frequencies, height=tolerance)
    
    # Visualizzazione dello spettro con i picchi identificati
    plt.figure(figsize=(12, 6))
    plt.plot(fft_frequencies, fft_magnitude, label='Spectral Magnitude')
    plt.plot(identified_frequencies, fft_magnitude[peaks], 'x', color='red', label='Identified Peaks')
    
    # Segnalazione delle frequenze originali
    for original_freq in original_frequencies:
        plt.axvline(x=original_freq, color='green', linestyle='--', label='Original Frequency' if original_freq == original_frequencies[0] else "")
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Spectral Analysis with Identified Peaks')
    plt.legend()
    plt.grid(True)
    plt.xlim(0.001,30)
    plt.xscale('log')
    plt.show()
    

def plot_exp_distrib(vrs):
    bin_min, bin_max = np.min(vrs), np.max(vrs)
    bins = 10**(np.linspace(np.log10(bin_min), np.log10(bin_max), 50))
    counts, edges = np.histogram(vrs, bins, density=True)
    centers = (edges[1:] + edges[:-1])/2.
    return centers, counts
        
    
    
    
def reconstruct_time_series(identified_frequencies, fft_magnitude, length, fs):
    """
    Ricostruisce la serie temporale a partire dalle frequenze identificate e le loro ampiezze.
    
    Parameters:
    - identified_frequencies: Le frequenze identificate nella FFT.
    - fft_magnitude: L'ampiezza delle frequenze identificate.
    - length: La lunghezza della serie temporale originale.
    - fs: La frequenza di campionamento.
    
    Returns:
    - reconstructed_series: La serie temporale ricostruita.
    """
    t = np.arange(length) / fs
    reconstructed_series = np.zeros(length)
    
    for i, freq in enumerate(identified_frequencies):
        # Estrai ampiezza e fase dalla FFT
        amp = np.abs(fft_magnitude[i])# / (length/fs)  # Normalizzazione rispetto alla lunghezza del segnale
        phase = np.angle(fft_magnitude[i])
        reconstructed_series += amp * np.sin(2 * np.pi * freq * t + phase)
        
    num_identified_freq = len(identified_frequencies)
    
    return reconstructed_series / num_identified_freq