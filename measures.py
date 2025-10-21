# region Lyapunov
from __future__ import annotations

from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


def _norm_parameter(metric: float | str) -> float:
    if metric in {"euclidean", 2}:
        return 2.0
    if metric in {"manhattan", 1}:
        return 1.0
    if metric in {"chebyshev", "infinity", np.inf}:  # type: ignore[unreachable]
        return np.inf
    if isinstance(metric, (int, float)) and metric > 0:
        return float(metric)
    raise ValueError("Metrica non supportata")


def autocorrelation(
    series: ArrayLike,
    max_lag: Optional[int] = None,
    *,
    demean: bool = True,
    normalize: bool = True,
    method: str = "fft",
) -> NDArray[np.float64]:
    """Compute the (auto-)correlation function of a real-valued series.

    Parameters
    ----------
    series:
        Input time series.
    max_lag:
        Largest lag for which the autocorrelation is computed.  If ``None``
        the default is ``len(series) // 2``.
    demean:
        Whether to remove the mean before computing the correlations.
    normalize:
        If ``True`` the returned values are normalised by the variance so that
        ``acf[0] == 1``.
    method:
        ``"fft"`` (default) uses the convolution theorem and is ``O(n log n)``.
        ``"direct"`` performs an explicit summation and is ``O(n * max_lag)``.

    Returns
    -------
    numpy.ndarray
        Array of length ``max_lag + 1`` containing the autocorrelation values.
    """

    x = np.asarray(series, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("Serie vuota")

    if max_lag is None:
        max_lag = max(1, n // 2)
    if max_lag >= n:
        raise ValueError("'max_lag' deve essere minore della lunghezza della serie")

    if demean:
        x = x - x.mean()

    if method not in {"fft", "direct"}:
        raise ValueError(f"Metodo non supportato: {method}")

    if method == "fft":
        fft_size = 1 << (2 * n - 1).bit_length()
        fft = np.fft.rfft(x, n=fft_size)
        acf = np.fft.irfft(fft * np.conjugate(fft), n=fft_size)[: max_lag + 1]
        acf /= np.arange(n, n - max_lag - 1, -1)
    else:
        acf = np.array(
            [np.dot(x[: n - lag], x[lag:]) / (n - lag) for lag in range(max_lag + 1)]
        )

    if normalize:
        variance = acf[0]
        if variance == 0:
            return np.zeros_like(acf)
        acf = acf / variance
    return acf


def integrated_autocorrelation_time(
    series: ArrayLike,
    max_lag: Optional[int] = None,
    *,
    window: Optional[int] = None,
    method: str = "fft",
) -> float:
    """Estimate the integrated autocorrelation time.

    The sum is truncated at ``window`` if provided, otherwise at the first
    non-positive correlation or at ``max_lag``.
    """

    acf = autocorrelation(series, max_lag=max_lag, method=method)
    if window is None:
        # heuristic: stop at first negative value of ACF
        positive = np.where(acf <= 0)[0]
        cutoff = positive[0] if positive.size > 0 else len(acf)
    else:
        cutoff = min(window + 1, len(acf))
    tau = 1 + 2 * np.sum(acf[1:cutoff])
    return float(tau)


def time_delay_embedding(series: ArrayLike, emb_dim: int, delay: int) -> NDArray[np.float64]:
    series = np.asarray(series, dtype=float)
    if emb_dim <= 0 or delay <= 0:
        raise ValueError("'emb_dim' e 'delay' devono essere positivi")
    n_vectors = len(series) - (emb_dim - 1) * delay
    if n_vectors <= 0:
        raise ValueError("Serie troppo corta per l'embedding richiesto")
    return np.array([
        series[i : i + emb_dim * delay : delay] for i in range(n_vectors)
    ])


def correlation_integral(
    embedded: NDArray[np.float64],
    radii: Sequence[float],
    *,
    theiler: int = 0,
    metric: float | str = "euclidean",
) -> NDArray[np.float64]:
    """Compute the Grassberger–Procaccia correlation integral.

    Parameters
    ----------
    embedded:
        Delay vectors in the reconstructed phase space.
    radii:
        Iterable of radii at which the correlation integral is evaluated.
    theiler:
        Theiler window: pairs of points closer than this delay are ignored.
    metric:
        Norm used for distances (passed to :meth:`cKDTree.query_pairs`).  Use
        ``np.inf`` for Chebyshev metric.
    """

    points = np.asarray(embedded, dtype=float)
    if points.ndim != 2:
        raise ValueError("'embedded' deve essere una matrice 2D di vettori ritardati")
    if points.shape[0] < 2:
        raise ValueError("Servono almeno due vettori per calcolare l'integrale di correlazione")

    tree = cKDTree(points)
    n = len(points)
    radii = np.asarray(radii, dtype=float)
    if np.any(radii <= 0):
        raise ValueError("I raggi devono essere positivi")

    counts = np.empty_like(radii)
    p = _norm_parameter(metric)
    for idx, r in enumerate(radii):
        pairs = tree.query_pairs(r, p=p, output_type="ndarray")
        if theiler > 0 and pairs.size:
            mask = np.abs(pairs[:, 0] - pairs[:, 1]) > theiler
            pairs = pairs[mask]
        counts[idx] = pairs.shape[0]
    norm = n * (n - 1) / 2
    return counts / norm


def correlation_dimension(
    series: ArrayLike,
    *,
    emb_dim: int,
    delay: int,
    radii: Sequence[float],
    theiler: int = 0,
    metric: float | str = "euclidean",
    max_points: Optional[int] = 5000,
    fit_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate the correlation dimension via the Grassberger–Procaccia method.

    Returns the radii, the correlation integral values and the slope of the
    scaling region (log-log fit).  ``fit_range`` specifies the slice over which
    the linear regression in log-space is computed; if ``None`` the full range
    is used.
    """

    data = np.asarray(series, dtype=float)
    embedded = time_delay_embedding(data, emb_dim, delay)
    if max_points is not None and len(embedded) > max_points:
        rng = np.random.default_rng(12345)
        indices = rng.choice(len(embedded), size=max_points, replace=False)
        embedded = embedded[np.sort(indices)]

    corr = correlation_integral(embedded, radii, theiler=theiler, metric=metric)
    positive = corr > 0
    if not np.any(positive):
        return np.asarray(radii), corr, np.nan

    log_r = np.log(radii[positive])
    log_c = np.log(corr[positive])

    if fit_range is None:
        start, end = 0, len(log_r)
    else:
        start = max(0, fit_range[0])
        end = min(len(log_r), fit_range[1])
    if end - start < 2:
        return np.asarray(radii), corr, np.nan

    slope, _ = np.polyfit(log_r[start:end], log_c[start:end], 1)
    return np.asarray(radii), corr, float(slope)


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


# region Recurrence plots and entropies


def recurrence_matrix(
    embedded: NDArray[np.float64],
    *,
    threshold: Optional[float] = None,
    metric: float | str = "euclidean",
    percentage: Optional[float] = None,
    theiler: int = 0,
) -> Tuple[NDArray[np.bool_], float]:
    """Compute the recurrence matrix for a set of embedded vectors.

    Parameters
    ----------
    embedded:
        Delay vectors (``n × d`` array).
    threshold:
        Distance threshold ``ε``.  If ``None`` it is estimated so that the
        recurrence rate matches ``percentage`` if provided, otherwise the 10th
        percentile of the distance distribution is used.
    percentage:
        Desired recurrence rate (between 0 and 1).  Ignored if ``threshold`` is
        given.  The Theiler window is not considered when computing this rate.
    metric:
        Norm used for distances (``np.inf`` corresponds to the Chebyshev norm).
    theiler:
        Sets to zero the band of width ``2 * theiler + 1`` around the main
        diagonal.
    """

    points = np.asarray(embedded, dtype=float)
    if points.ndim != 2:
        raise ValueError("'embedded' deve essere un array 2D")
    n = len(points)
    if n == 0:
        raise ValueError("L'array embedding è vuoto")

    tree = cKDTree(points)
    p = _norm_parameter(metric)
    if threshold is None:
        rng = np.random.default_rng(12345)
        sample_size = min(n, 2000)
        if sample_size < 2:
            eps = 0.0
        else:
            indices = rng.choice(n, size=sample_size, replace=False)
            subset = points[indices]
            diffs = subset[:, None, :] - subset[None, :, :]
            if np.isinf(p):
                distances = np.max(np.abs(diffs), axis=-1)
            else:
                distances = np.linalg.norm(diffs, ord=p, axis=-1)
            tri = distances[np.triu_indices(sample_size, k=1)]
            if tri.size == 0:
                eps = 0.0
            else:
                if percentage is not None:
                    if not (0 < percentage < 1):
                        raise ValueError("'percentage' deve essere nell'intervallo (0, 1)")
                    k = int(np.floor(percentage * tri.size))
                    k = np.clip(k, 0, tri.size - 1)
                    eps = float(np.partition(tri, k)[k])
                else:
                    eps = float(np.percentile(tri, 10))
    else:
        eps = float(threshold)

    if eps < 0:
        raise ValueError("La soglia deve essere non negativa")

    # Build recurrence matrix
    pairs = tree.query_pairs(eps, p=p, output_type="ndarray")
    R = np.zeros((n, n), dtype=bool)
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        R[i, j] = True
        R[j, i] = True
    np.fill_diagonal(R, True)

    if theiler > 0:
        for offset in range(-theiler, theiler + 1):
            diag = np.diagonal(R, offset=offset)
            diag[:] = False

    return R, eps


def _run_lengths(binary: NDArray[np.bool_]) -> List[int]:
    if binary.ndim != 1:
        raise ValueError("L'array deve essere monodimensionale")
    lengths: List[int] = []
    count = 0
    for value in binary:
        if value:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _diagonal_lengths(R: NDArray[np.bool_]) -> List[int]:
    n = R.shape[0]
    lengths: List[int] = []
    for offset in range(-n + 1, n):
        diag = np.diagonal(R, offset=offset)
        if diag.size:
            lengths.extend(_run_lengths(diag))
    return lengths


def _vertical_lengths(R: NDArray[np.bool_]) -> List[int]:
    lengths: List[int] = []
    for col in range(R.shape[1]):
        lengths.extend(_run_lengths(R[:, col]))
    return lengths


def recurrence_quantification(
    R: NDArray[np.bool_],
    *,
    l_min: int = 2,
    v_min: int = 2,
) -> Dict[str, float]:
    """Compute standard RQA (Recurrence Quantification Analysis) measures."""

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("La matrice di ricorrenza deve essere quadrata")
    n = R.shape[0]
    R = R.astype(bool, copy=True)
    np.fill_diagonal(R, False)
    total = n * n - n  # exclude main diagonal
    off_diag = R.sum()
    rr = off_diag / total if total > 0 else np.nan

    diag_lengths = [L for L in _diagonal_lengths(R) if L >= l_min]
    if off_diag == 0:
        det = np.nan
        l_mean = np.nan
        l_max = 0
        entr = np.nan
    else:
        diag_points = sum(L for L in diag_lengths)
        det = diag_points / off_diag if off_diag > 0 else np.nan
        l_mean = diag_points / len(diag_lengths) if diag_lengths else np.nan
        l_max = max(diag_lengths) if diag_lengths else 0
        if diag_lengths:
            counts = np.bincount(diag_lengths)
            probs = counts[counts > 0] / counts.sum()
            entr = -np.sum(probs * np.log(probs))
        else:
            entr = np.nan

    vert_lengths = [L for L in _vertical_lengths(R) if L >= v_min]
    vert_points = sum(vert_lengths)
    lam = vert_points / off_diag if off_diag > 0 else np.nan
    tt = vert_points / len(vert_lengths) if vert_lengths else np.nan

    return {
        "RR": float(rr),
        "DET": float(det),
        "L_mean": float(l_mean),
        "L_max": float(l_max),
        "LAM": float(lam),
        "TT": float(tt),
        "ENTR": float(entr),
    }


def permutation_entropy(
    series: ArrayLike,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Permutation entropy as defined by Bandt & Pompe."""

    x = np.asarray(series, dtype=float)
    if order < 2:
        raise ValueError("L'ordine deve essere almeno 2")
    if delay <= 0:
        raise ValueError("Il ritardo deve essere positivo")

    n = len(x) - (order - 1) * delay
    if n <= 0:
        raise ValueError("Serie troppo corta per l'ordine richiesto")

    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(n):
        window = x[i : i + order * delay : delay]
        ranks = tuple(np.argsort(np.argsort(window, kind="mergesort")))
        patterns[ranks] = patterns.get(ranks, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    if normalize:
        entropy /= np.log(factorial(order))
    return float(entropy)


def sample_entropy(
    series: ArrayLike,
    *,
    m: int = 2,
    r: float = 0.2,
    theiler: int = 0,
) -> float:
    """Sample entropy (SampEn) following Richman & Moorman (2000)."""

    x = np.asarray(series, dtype=float)
    if m < 1:
        raise ValueError("'m' deve essere almeno 1")
    if r < 0:
        raise ValueError("'r' deve essere non negativo")
    if len(x) <= m + 1:
        raise ValueError("Serie troppo corta per i parametri richiesti")

    std = np.std(x)
    if std == 0:
        return 0.0
    tol = r * std

    emb_m = time_delay_embedding(x, m, 1)
    emb_m1 = time_delay_embedding(x, m + 1, 1)

    def _count_pairs(embedded: NDArray[np.float64]) -> float:
        tree = cKDTree(embedded)
        pairs = tree.query_pairs(tol, p=np.inf, output_type="ndarray")
        if theiler > 0 and pairs.size:
            mask = np.abs(pairs[:, 0] - pairs[:, 1]) > theiler
            pairs = pairs[mask]
        return float(pairs.shape[0])

    count_m = _count_pairs(emb_m)
    count_m1 = _count_pairs(emb_m1)

    nm = len(emb_m)
    nm1 = len(emb_m1)
    total_m = nm * (nm - 1) / 2
    total_m1 = nm1 * (nm1 - 1) / 2
    if total_m == 0 or total_m1 == 0:
        return np.nan

    B = count_m / total_m
    A = count_m1 / total_m1
    if A == 0 or B == 0:
        return np.inf
    return float(-np.log(A / B))


__all__ = [
    "autocorrelation",
    "integrated_autocorrelation_time",
    "time_delay_embedding",
    "correlation_integral",
    "correlation_dimension",
    "largest_lyapunov_rosenstein",
    "iaaft_surrogate",
    "iaaft_nonlinearity_test",
    "hurst_rs",
    "dfa_alpha",
    "recurrence_matrix",
    "recurrence_quantification",
    "permutation_entropy",
    "sample_entropy",
]

# endregion