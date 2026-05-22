# import libs
import numpy as np
from typing import Any, Dict, Optional, List
from pythermodb_settings.models import CustomProp
from pyThermoLinkDB.models.component_models import ComponentEquationSource


class ThermoPropertyFields:
    """
    Typed attribute declarations for dynamically assigned thermodynamic properties.
    """
    Cp_IG: np.ndarray
    Cp_IG_comp: Dict[str, float]
    Cp_IG_src: Dict[str, ComponentEquationSource]

    Cp_LIQ: np.ndarray
    Cp_LIQ_comp: Dict[str, float]
    Cp_LIQ_src: Dict[str, ComponentEquationSource]

    rho_LIQ: np.ndarray
    rho_LIQ_comp: Dict[str, float]
    rho_LIQ_src: Dict[str, ComponentEquationSource]

    EnFo_IG_298: np.ndarray
    EnFo_IG_298_comp: Dict[str, float]
    EnFo_IG_298_src: Dict[str, Any]

    MW: np.ndarray
    MW_comp: Dict[str, float]
    MW_src: Dict[str, Any]

    dH_rxns: Optional[Dict[str, Dict[str, Any]]]
    rho_LIQ_MIX: Optional[CustomProp]
    Cp_IG_MIX_TOTAL: Optional[CustomProp]
    Cp_LIQ_MIX_TOTAL: Optional[CustomProp]
    Cp_LIQ_MIX_VOLUMETRIC: Optional[CustomProp]

    dCp_rxns: np.ndarray
    dH_rxns_298: np.ndarray

    thermo_properties: List[str] = [
        "Cp_IG",
        "Cp_LIQ",
        "rho_LIQ",
        "EnFo_IG_298",
        "MW",
        "dH_rxns",
        "rho_LIQ_MIX",
        "Cp_IG_MIX_TOTAL",
        "Cp_LIQ_MIX_TOTAL",
        "Cp_LIQ_MIX_VOLUMETRIC",
    ]

    def __init__(self) -> None:
        # NOTE: default-initialize all optional runtime attributes so missing
        # source selections do not raise AttributeError on access.
        self.dH_rxns = None
        self.rho_LIQ_MIX = None
        self.Cp_IG_MIX_TOTAL = None
        self.Cp_LIQ_MIX_TOTAL = None
        self.Cp_LIQ_MIX_VOLUMETRIC = None
