# import libs
import numpy as np
from typing import Any, Dict, Optional, List
from pythermodb_settings.models import CustomProp
from pyThermoLinkDB.models.component_models import ComponentEquationSource


class ThermoPropertyFields:
    """
    Typed attribute declarations for dynamically assigned thermodynamic properties.
    """
    # ! heat capacity at constant pressure (Cp_IG) for ideal gas components
    Cp_IG: np.ndarray
    Cp_IG_comp: Dict[str, float]
    Cp_IG_src: Dict[str, ComponentEquationSource]

    # ! heat capacity at constant pressure (Cp_LIQ) for liquid components
    Cp_LIQ: np.ndarray
    Cp_LIQ_comp: Dict[str, float]
    Cp_LIQ_src: Dict[str, ComponentEquationSource]

    # ! liquid density (rho_LIQ) for liquid components
    rho_LIQ: np.ndarray
    rho_LIQ_comp: Dict[str, float]
    rho_LIQ_src: Dict[str, ComponentEquationSource]

    # ! ideal gas formation enthalpy at 298 K (EnFo_IG_298) for ideal gas components
    EnFo_IG_298: np.ndarray
    EnFo_IG_298_comp: Dict[str, float]
    EnFo_IG_298_src: Dict[str, Any]

    # ! liquid formation enthalpy at 298 K (EnFo_LIQ_298) for liquid components
    EnFo_LIQ_298: np.ndarray
    EnFo_LIQ_298_comp: Dict[str, float]
    EnFo_LIQ_298_src: Dict[str, Any]

    # ! molecular weight (MW) for components
    MW: np.ndarray
    MW_comp: Dict[str, float]
    MW_src: Dict[str, Any]

    # ! reaction enthalpy (dH_rxns) for reactions in the reactor, calculated based on the current temperature and reaction enthalpy mode
    dH_rxns: Optional[Dict[str, Dict[str, Any]]]

    # ! mixture density for liquid phase (rho_LIQ_MIX)
    rho_LIQ_MIX: Optional[CustomProp]

    # ! total heat capacities for gas mixture
    Cp_IG_MIX_TOTAL: Optional[CustomProp]
    # ! total heat capacities for liquid mixture
    Cp_LIQ_MIX_TOTAL: Optional[CustomProp]
    # ! volumetric heat capacity for liquid mixture
    Cp_LIQ_MIX_VOLUMETRIC: Optional[CustomProp]

    # ! average heat capacity change for reactions (dCp_rxns)
    dCp_rxns: np.ndarray

    # ! enthalpy of reactions at 298 K (dH_rxns_298)
    dH_rxns_298: np.ndarray

    def __init__(self) -> None:
        # NOTE: default-initialize all optional runtime attributes so missing
        # source selections do not raise AttributeError on access.
        self.dH_rxns = None
        self.rho_LIQ_MIX = None
        self.Cp_IG_MIX_TOTAL = None
        self.Cp_LIQ_MIX_TOTAL = None
        self.Cp_LIQ_MIX_VOLUMETRIC = None
