import logging
import numpy as np
from typing import Dict, List
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Temperature
# locals
from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .pbrc import PBRReactorCore

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidPBRReactor:
    """
    Liquid-phase packed-bed reactor model.

    PBR difference from PFR:
    rates are interpreted on catalyst-mass basis r' [mol/kg.s] and converted to
    reactor-volume basis r_V [mol/m3.s] via r_V = rho_B * r'.
    """

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        pbr_reactor_core: PBRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        self.components = components
        self.component_key = component_key
        self.thermo_source = thermo_source
        self.pbr_reactor_core = pbr_reactor_core

        # SECTION: options and heat-transfer configuration
        self.heat_transfer_mode = pbr_reactor_core.heat_transfer_mode
        self.operation_mode = pbr_reactor_core.operation_mode
        self.liquid_density_mode = pbr_reactor_core.liquid_density_mode

        self.heat_exchange = pbr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = pbr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = pbr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = pbr_reactor_core.jacket_temperature_value
        self.heat_rate_value = pbr_reactor_core.heat_rate_value

        # NOTE: packed-bed catalyst bulk density [kg/m3]
        self._rho_B = pbr_reactor_core._rho_B

        # SECTION: reaction and stoichiometry mapping
        self.reaction_rates = reaction_rates
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        self.reaction_stoichiometry = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: component references
        self.component_num = self.thermo_source.component_refs["component_num"]
        self.component_formula_state = self.thermo_source.component_refs["component_formula_state"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: inlet and reactor geometry
        self._F_in = pbr_reactor_core._F_in
        self.temperature_in = pbr_reactor_core.temperature_inlet
        self._T_in = pbr_reactor_core._T_in
        self._Vr = pbr_reactor_core.reactor_volume_value

        self._rho_LIQ_in = self.thermo_source.calc_rho_LIQ(
            temperature=self.temperature_in
        )
        self._q_in = self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=self._F_in,
            molecular_weights=self.thermo_source.MW,
            density=self._rho_LIQ_in
        )

        # SECTION: final configuration checks
        self.pbr_reactor_core.config_model()

    @property
    def F_in(self) -> np.ndarray:
        return self._F_in

    def build_y0(self) -> np.ndarray:
        f0 = self.F_in.astype(float)
        if self.heat_transfer_mode == "isothermal":
            return f0
        return np.concatenate([f0, np.array([float(self._T_in)], dtype=float)])

    def rhs(
            self,
            V: float,
            y: np.ndarray
    ) -> np.ndarray:
        ns = self.component_num

        F = np.clip(y[:ns], 0.0, None)

        if self.heat_transfer_mode == "non-isothermal":
            temp = float(y[ns])
        else:
            temp = float(self._T_in)
        temperature = Temperature(value=temp, unit="K")

        rho_LIQ = self.thermo_source.calc_rho_LIQ(
            temperature=temperature
        )
        q_vol = self._calc_q_vol(
            F=F,
            rho_LIQ=rho_LIQ
        )
        q_vol = max(q_vol, 1e-30)
        concentration = F / q_vol

        concentration_std = {
            sp: CustomProperty(
                value=concentration[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # NOTE: raw rates are catalyst-mass basis r' [mol/kg.s]
        rates_prime = self._calc_rates(
            concentration=concentration_std,
            temperature=temperature
        )

        # NOTE: packed-bed conversion to reactor-volume basis r_V [mol/m3.s]
        rates_v = self._rho_B * rates_prime

        dF_dV = self._build_dF_dV(rates=rates_v)

        if self.heat_transfer_mode == "isothermal":
            return dF_dV

        dT_dV = self._build_dT_dV(
            F=F,
            rates_v=rates_v,
            temp=temp
        )
        return np.concatenate([dF_dV, np.array([dT_dV], dtype=float)])

    def _calc_q_liquid(
            self,
            flow: np.ndarray,
            rho_LIQ: np.ndarray
    ) -> float:
        return self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=flow,
            molecular_weights=self.thermo_source.MW,
            density=rho_LIQ
        )

    def _calc_q_vol(
            self,
            F: np.ndarray,
            rho_LIQ: np.ndarray
    ) -> float:
        if self.operation_mode == "constant_volume":
            return float(self._q_in)
        if self.operation_mode == "constant_pressure":
            return float(self._calc_q_liquid(flow=F, rho_LIQ=rho_LIQ))
        raise ValueError(
            f"Invalid operation_mode '{self.operation_mode}' for liquid PBR."
        )

    def _calc_rates(
        self,
        concentration: Dict[str, CustomProperty],
        temperature: Temperature
    ) -> np.ndarray:
        rates = []

        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis
            if basis != "concentration":
                raise ValueError(
                    f"Invalid basis '{basis}' for liquid PBR reaction rate expression '{rate_exp.name}'. "
                    "Liquid PBR supports basis='concentration' only."
                )

            r_k = rate_exp.calc(
                xi=concentration,
                temperature=temperature,
                pressure=None
            )
            rates.append(float(r_k.value))

        return np.array(rates, dtype=float)

    def _build_dF_dV(self, rates: np.ndarray) -> np.ndarray:
        ns = self.component_num
        dF_dV = np.zeros(ns, dtype=float)

        for k, _ in enumerate(self.reactions):
            r_k = rates[k]
            for sp_name, nu_ik in self.reaction_stoichiometry[k].items():
                i = self.component_id_to_index[sp_name]
                dF_dV[i] += nu_ik * r_k

        return dF_dV

    def _build_dT_dV(
        self,
        F: np.ndarray,
        rates_v: np.ndarray,
        temp: float
    ) -> float:
        temperature = Temperature(value=temp, unit="K")

        cp_liq_values = self.thermo_source.calc_Cp_LIQ(
            temperature=temperature
        )
        cp_flow = calc_total_heat_capacity(x=F, cp=cp_liq_values)
        if cp_flow <= 1e-16:
            raise ValueError(
                "Total flowing liquid heat capacity is too small or zero."
            )

        delta_h = self.thermo_source.calc_dH_rxns_IG_ref(
            temperature=temperature
        )
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates_v,
            reactor_volume=1.0
        )

        q_exchange = 0.0
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=self._Vr
            )

        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value / self._Vr

        return (q_rxn + q_exchange + q_constant) / cp_flow
