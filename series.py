"""Utility per la gestione di serie temporali e generatori deterministici classici."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np


@dataclass
class Series:
    """Contenitore leggero per serie temporali."""

    data: Union[np.ndarray, Iterable[float]]
    name: str = ""
    dt: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data, dtype=float)
        if array.ndim == 0:
            raise ValueError("Una serie temporale deve contenere almeno un campione")
        self.data = array

    def __len__(self) -> int:  # pragma: no cover - semplice wrapper
        return len(self.data)

    def copy(self) -> "Series":
        """Restituisce una copia indipendente della serie."""

        return Series(
            data=np.copy(self.data),
            name=self.name,
            dt=self.dt,
            parameters=dict(self.parameters),
        )

    def standardized(self, axis: int = 0, inplace: bool = False) -> "Series":
        """Normalizza la serie sottraendo la media e dividendo per la deviazione standard."""

        data = self.data
        if data.ndim == 1:
            mean = data.mean()
            std = data.std()
            std = 1.0 if std == 0 else std
            normalized = (data - mean) / std
        else:
            mean = data.mean(axis=axis, keepdims=True)
            std = data.std(axis=axis, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            normalized = (data - mean) / std

        if inplace:
            self.data = normalized
            self.parameters["standardized"] = True
            return self

        params = dict(self.parameters)
        params["standardized"] = True
        return Series(data=normalized, name=self.name, dt=self.dt, parameters=params)

    def as_array(self) -> np.ndarray:
        """Ritorna i dati come ``numpy.ndarray`` (copia)."""

        return np.array(self.data, copy=True)


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


_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2}


def lorenz63(
    rho: float = 28.0,
    *,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    steps: int = 10000,
    discard: int = 0,
    initial_state: Optional[Iterable[float]] = None,
    component: Optional[Union[str, int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Genera una traiettoria del sistema di Lorenz 63 usando integrazione RK4."""

    if steps <= 0:
        raise ValueError("'steps' deve essere un intero positivo")
    if discard < 0:
        raise ValueError("'discard' non può essere negativo")

    total_steps = steps + discard
    trajectory = np.empty((total_steps, 3), dtype=float)
    if initial_state is None:
        trajectory[0] = np.array([1.0, 1.0, 1.0], dtype=float)
    else:
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (3,):
            raise ValueError("Lo stato iniziale deve contenere tre componenti")
        trajectory[0] = state

    for i in range(total_steps - 1):
        trajectory[i + 1] = _rk4_step(trajectory[i], sigma, rho, beta, dt)

    result = trajectory[discard:]
    if component is not None:
        if isinstance(component, str):
            try:
                idx = _COMPONENT_INDEX[component.lower()]
            except KeyError as exc:
                raise ValueError("Componente non valida: usare 'x', 'y' o 'z'") from exc
        else:
            idx = int(component)
            if idx not in (0, 1, 2):
                raise ValueError("L'indice di componente deve essere 0, 1 o 2")
        result = result[:, idx]

    if normalize:
        if result.ndim == 1:
            std = result.std()
            std = 1.0 if std == 0 else std
            result = (result - result.mean()) / std
        else:
            mean = result.mean(axis=0, keepdims=True)
            std = result.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            result = (result - mean) / std

    return result


__all__ = ["Series", "lorenz63"]
