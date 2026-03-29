from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from pythermodb_settings.models import (
    Component,
    ComponentKey,
    Pressure,
    Temperature,
)
from pyreactlab_core.models.reaction import Reaction
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyThermoLinkDB.thermo import Source

from ..models.br import BatchReactorOptions, GasModel
from ..models.rate_exp import ReactionRateExpression


class TS(ABC):
    """Abstract thermodynamic source contract."""

    T_ref = Temperature(value=298.15, unit="K")
    T_ref_K = 298.15
    P_ref = Pressure(value=101325, unit="Pa")

    @abstractmethod
    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
        component_key: ComponentKey,
    ) -> None:
        ...

    @abstractmethod
    def prop_eq_src(self, prop_name: str) -> Dict[str, ComponentEquationSource]:
        ...

    @abstractmethod
    def prop_dt_src(
        self,
        component_ids: List[str],
        prop_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def _config_constant_gas_heat_capacity(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ...

    @abstractmethod
    def _config_constant_liquid_heat_capacity(
        self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        ...

    @abstractmethod
    def _config_constant_liquid_density(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ...

    @abstractmethod
    def build_reactions(self) -> List[Reaction]:
        ...

    @abstractmethod
    def build_stoichiometry(self) -> np.ndarray:
        ...

    @abstractmethod
    def calc_Cp_IG(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def calc_Cp_IG_real(self, inputs: Dict[str, Any]) -> np.ndarray:
        ...

    @abstractmethod
    def calc_Cp_LIQ(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def calc_Cp_LIQ_real(self, inputs: Dict[str, Any]) -> np.ndarray:
        ...

    @abstractmethod
    def calc_dCp_IG(self) -> np.ndarray:
        ...

    @abstractmethod
    def calc_dH_rxns_298(self) -> np.ndarray:
        ...

    @abstractmethod
    def calc_dH_rxns(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def calc_dH_rxns_real(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def calc_dH_rxns_linear(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def _config_EnFo_IG_unit(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def _config_MW_unit(self) -> Tuple[np.ndarray, Dict[str, float]]:
        ...

    @abstractmethod
    def calc_tot_pressure(
        self,
        n_total: float,
        temperature: float,
        reactor_volume_value: float,
        R: float,
        gas_model: GasModel,
    ) -> float:
        ...

    @abstractmethod
    def calc_gas_volume(
        self,
        n_total: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: GasModel,
    ) -> float:
        ...

    @abstractmethod
    def calc_liquid_volume(
        self,
        n: np.ndarray,
        molecular_weights: np.ndarray,
        density: np.ndarray,
    ) -> float:
        ...

    @abstractmethod
    def calc_rho_LIQ(self, temperature: Temperature) -> np.ndarray:
        ...

    @abstractmethod
    def calc_rho_LIQ_real(self, inputs: Dict[str, Any]) -> np.ndarray:
        ...
