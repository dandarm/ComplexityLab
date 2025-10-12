# region Lyapunov
import numpy as np
from scipy.spatial import cKDTree


def time_delay_embedding(series, emb_dim, delay):
    series = np.asarray(series, dtype=float)
    n_vectors = len(series) - (emb_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError("Serie troppo corta per l'embedding richiesto")
    return np.array([
        series[i : i + emb_dim * delay : delay] for i in range(n_vectors)
    ])


def largest_lyapunov_rosenstein(series, dt, emb_dim=6, delay=8, theiler=50, fit_range=(5, 25)):
    embedded = time_delay_embedding(series, emb_dim, delay)
    tree = cKDTree(embedded)
    distances, indices = tree.query(embedded, k=emb_dim + 2)
    nn_indices = np.full(len(embedded), -1, dtype=int)

    for i in range(len(embedded)):
        for neighbor in indices[i, 1:]:
            if neighbor == i:
                continue
            if abs(neighbor - i) > theiler:
                nn_indices[i] = neighbor
                break
        if nn_indices[i] == -1:
            nn_indices[i] = indices[i, 1]

    max_t = min(200, len(embedded))
    log_divergence = []
    valid_steps = []
    for t in range(max_t):
        separations = []
        for idx, neighbor in enumerate(nn_indices):
            if neighbor < 0:
                continue
            if idx + t >= len(embedded) or neighbor + t >= len(embedded):
                continue
            dist = np.linalg.norm(embedded[idx + t] - embedded[neighbor + t])
            if dist > 0:
                separations.append(np.log(dist))
        if separations:
            log_divergence.append(np.mean(separations))
            valid_steps.append(t * dt)
        elif log_divergence:
            break

    if len(valid_steps) < 6:
        return np.nan

    start, end = fit_range
    end = min(end, len(valid_steps))
    start = min(start, end - 2)
    coeffs = np.polyfit(valid_steps[start:end], log_divergence[start:end], 1)
    return coeffs[0]

# endregion


# region IAAFT

def iaaft_surrogate(series, n_iterations=100):
    series = np.asarray(series, dtype=float)
    sorted_series = np.sort(series)
    target_amplitude = np.abs(np.fft.rfft(series))
    surrogate = np.random.permutation(series)
    for _ in range(n_iterations):
        surrogate_fft = np.fft.rfft(surrogate)
        surrogate = np.fft.irfft(
            target_amplitude * np.exp(1j * np.angle(surrogate_fft)), n=len(series)
        )
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_series[ranks]
    return surrogate


def iaaft_nonlinearity_test(series, n_surrogates=20):
    reduced = series[::2]
    statistics = []
    for _ in range(n_surrogates):
        surrogate = iaaft_surrogate(reduced, n_iterations=50)
        statistics.append(largest_lyapunov_rosenstein(surrogate, dt))
    statistics = np.array([s for s in statistics if np.isfinite(s)])
    observed = largest_lyapunov_rosenstein(reduced, dt)
    if statistics.size == 0 or not np.isfinite(observed):
        return np.nan, np.nan, np.nan, np.nan
    mean = statistics.mean()
    std = statistics.std(ddof=1) if statistics.size > 1 else np.nan
    z_score = (observed - mean) / std if np.isfinite(std) and std > 0 else np.nan
    p_value = (np.sum(statistics >= observed) + 1) / (statistics.size + 1)
    return observed, mean, std, z_score, p_value

# endregion


# region HUSRT exponent

def hurst_rs(series, min_window=16, max_window=None, num_windows=20):
    series = np.asarray(series, dtype=float)
    series = series - np.mean(series)
    n = len(series)
    if max_window is None:
        max_window = n // 6
    windows = np.unique(
        np.logspace(
            np.log10(min_window), np.log10(max_window), num=num_windows, dtype=int
        )
    )
    rs_values = []
    valid_windows = []
    for w in windows:
        if w < min_window or w >= n // 2:
            continue
        n_segments = n // w
        if n_segments < 2:
            continue
        data = series[: n_segments * w].reshape((n_segments, w))
        data = data - data.mean(axis=1, keepdims=True)
        cumulative = np.cumsum(data, axis=1)
        ranges = cumulative.max(axis=1) - cumulative.min(axis=1)
        stds = data.std(axis=1, ddof=1)
        valid = stds > 0
        if not np.any(valid):
            continue
        rs = np.mean(ranges[valid] / stds[valid])
        rs_values.append(rs)
        valid_windows.append(w)
    if len(rs_values) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_windows), np.log(rs_values), 1)
    return slope


def dfa_alpha(series, min_window=16, max_window=None, num_windows=20, order=1):
    series = np.asarray(series, dtype=float)
    series = series - np.mean(series)
    n = len(series)
    if max_window is None:
        max_window = n // 6
    windows = np.unique(
        np.logspace(
            np.log10(min_window), np.log10(max_window), num=num_windows, dtype=int
        )
    )
    profile = np.cumsum(series)
    flucts = []
    valid_windows = []
    for w in windows:
        if w < min_window or w >= n // 2:
            continue
        n_segments = n // w
        if n_segments < 2:
            continue
        segments = profile[: n_segments * w].reshape((n_segments, w))
        x = np.arange(w)
        rms = []
        for segment in segments:
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))
        rms = np.array(rms)
        if np.all(rms == 0):
            continue
        flucts.append(np.mean(rms))
        valid_windows.append(w)
    if len(flucts) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_windows), np.log(flucts), 1)
    return slope

# endregion