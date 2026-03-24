# import libs
import logging
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, Pressure, Temperature, CustomProperty
from pythermodb_settings.utils import set_component_id
from pyreactlab_core.models.reaction import Reaction
# locals
from ..models.rate_exp import ReactionRateExpression
from ..models.rate_exp_refs import rArgs, rParams, rRet, rXs
from .unit_tools import to_m3, to_K

# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION:


def calc_heat_exchange(
    temperature: float,
    jacket_temperature: float,
    heat_transfer_area: float,
    heat_transfer_coefficient: float,
) -> float:
    """
    Calculate the heat exchange with the surroundings based on the current temperature of the system.

    Parameters
    ----------
    temperature : float
        Current temperature of the system [K].
    jacket_temperature : float
        Temperature of the jacket or surroundings [K].
    heat_transfer_area : float
        Area available for heat transfer [m2].
    heat_transfer_coefficient : float
        Heat transfer coefficient [W/m2.K].

    Returns
    -------
    float
        Heat exchange with the surroundings [W] or [J/s].
    """
    # NOTE: Convert units if necessary
    T = temperature
    T_s = jacket_temperature
    A = heat_transfer_area
    U = heat_transfer_coefficient

    # ! calculate heat exchange using the formula: Q = U * A * (T_s - T)
    # unit check: U [W/m^2.K], A [m^2], T_s [K], temp [K] => Q [W] or [J/s]
    return U * A * (T_s - T)
