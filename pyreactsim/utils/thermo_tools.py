# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast

# SECTION: total heat capacity calculation


def total_heat_capacity(self, n: np.ndarray, cp: np.ndarray) -> float:
    """
    Calculate the total heat capacity of the system.

    Parameters
    ----------
    n : np.ndarray
        An array of molar amounts of each component.
    cp : np.ndarray
        An array of heat capacities (Cp) for each component.

    Returns
    -------
    float
        The total heat capacity of the system.
    """
    return np.sum(n * cp)
