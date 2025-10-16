"""Utility per rappresentare serie temporali generate da modelli dinamici."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, MutableMapping, Optional

import numpy as np


@dataclass
class Series:
    """Rappresentazione di una serie temporale generata esternamente.

    Parameters
    ----------
    name:
        Nome descrittivo della serie.
    data:
        Sequenza numerica (anche multidimensionale) contenente i campioni.
    parameters:
        Parametri utilizzati per generare la serie.
    metadata:
        Informazioni aggiuntive opzionali.
    """

    name: str
    data: np.ndarray
    parameters: MutableMapping[str, Any] = field(default_factory=dict)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim == 1:
            # normalizza forma a (n_samples, 1) per un trattamento uniforme
            self.data = self.data[:, np.newaxis]

    @property
    def n_samples(self) -> int:
        """Numero di campioni disponibili."""

        return self.data.shape[0]

    @property
    def n_components(self) -> int:
        """Numero di componenti della serie (es. x,y,z per Lorenz)."""

        return self.data.shape[1]

    def component(self, index: int) -> np.ndarray:
        """Restituisce una singola componente come vettore 1D."""

        if not 0 <= index < self.n_components:
            raise IndexError("Indice di componente fuori intervallo")
        return self.data[:, index]

    def to_array(self) -> np.ndarray:
        """Restituisce una copia dei dati come array NumPy."""

        return np.array(self.data, copy=True)

    def copy(self, *, name: Optional[str] = None, **updates: Any) -> "Series":
        """Crea una copia profonda della serie con eventuali aggiornamenti."""

        new_params: Dict[str, Any] = dict(self.parameters)
        new_params.update(updates.pop("parameters", {}))
        new_metadata: Dict[str, Any] = dict(self.metadata)
        new_metadata.update(updates.pop("metadata", {}))
        return Series(
            name=name or self.name,
            data=self.to_array(),
            parameters=new_params,
            metadata=new_metadata,
        )


__all__ = ["Series"]
