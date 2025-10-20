"""Generatori di serie temporali per sistemi dinamici classici."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _lorenz_derivatives(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


def _rk4_step(state: np.ndarray, sigma: float, rho: float, beta: float, dt: float) -> np.ndarray:
    k1 = _lorenz_derivatives(state, sigma, rho, beta)
    k2 = _lorenz_derivatives(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = _lorenz_derivatives(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = _lorenz_derivatives(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz63(
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    steps: int = 10000,
    transient: int = 0,
    initial_state: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Integra il sistema di Lorenz 63 e restituisce la traiettoria."""

    if steps <= 0:
        raise ValueError("'steps' deve essere positivo")
    if transient < 0:
        raise ValueError("'transient' non può essere negativo")

    total_steps = steps + transient
    trajectory = np.zeros((total_steps, 3), dtype=float)
    trajectory[0] = np.asarray(initial_state, dtype=float)

    for i in range(total_steps - 1):
        trajectory[i + 1] = _rk4_step(trajectory[i], sigma, rho, beta, dt)

    return trajectory[transient:]


def logistic_map(
    *,
    r: float = 3.9,
    x0: float = 0.5,
    steps: int = 5000,
    transient: int = 0,
) -> np.ndarray:
    """Genera la successione della mappa logistica x_{n+1} = r x_n (1 - x_n)."""

    if steps <= 0:
        raise ValueError("'steps' deve essere positivo")
    if not 0 < x0 < 1:
        raise ValueError("'x0' deve appartenere all'intervallo (0, 1)")
    if transient < 0:
        raise ValueError("'transient' non può essere negativo")

    total_steps = steps + transient
    values = np.empty(total_steps, dtype=float)
    values[0] = x0

    for i in range(1, total_steps):
        values[i] = r * values[i - 1] * (1.0 - values[i - 1])

    return values[transient:]


def henon_map(
    *,
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.1,
    y0: float = 0.3,
    steps: int = 5000,
    transient: int = 0,
) -> np.ndarray:
    """Genera la traiettoria della mappa di Hénon."""

    if steps <= 0:
        raise ValueError("'steps' deve essere positivo")
    if transient < 0:
        raise ValueError("'transient' non può essere negativo")

    total_steps = steps + transient
    x = np.empty(total_steps, dtype=float)
    y = np.empty(total_steps, dtype=float)
    x[0], y[0] = x0, y0

    for i in range(1, total_steps):
        x_prev = x[i - 1]
        y_prev = y[i - 1]
        x[i] = 1.0 - a * x_prev**2 + y_prev
        y[i] = b * x_prev

    return np.column_stack((x[transient:], y[transient:]))


__all__ = ["lorenz63", "logistic_map", "henon_map"]
