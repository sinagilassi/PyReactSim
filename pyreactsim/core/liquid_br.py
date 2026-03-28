# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
# locals
from .br import BatchReactor
from .thermo_source import ThermoSource
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.reaction_tools import stoichiometry_mat_key, stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..utils.opt_tools import calc_heat_exchange, set_component_X

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidBatchReactor(BatchReactor):
    """
    Liquid Batch Reactor (LBR) class for simulating batch reactions in liquid phase.

    """
