import logging
import numpy as np
from typing import Dict, List
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Temperature
from pyreactsim_core.models import ReactionRateExpression
# locals
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .pfrc import PFRReactorCore
# auxiliary
from .react_aux import ReactorAuxiliary
# log
from .react_log import ReactLog

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidPFRReactor(ReactorAuxiliary, ReactLog):
    """
    Liquid-phase plug-flow reactor model.

    Modeling basis
    --------------
    - Steady-state PFR in reactor-volume coordinate V.
    - Primary states are component molar flows F_i(V) [mol/s].
    - Optional thermal state T(V) [K] for non-isothermal mode.
    - Kinetics are concentration-basis only for this liquid implementation.

    State vector by mode
    --------------------
    - isothermal: [F1, ..., FNc]
    - non-isothermal: [F1, ..., FNc, T]
    """

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        pfr_reactor_core: PFRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        # LINK: ReactorAuxiliary initialization
        super().__init__(
            components=components,
            reaction_rates=reaction_rates,
            thermo_source=thermo_source,
            reactor_core=pfr_reactor_core,
            component_key=component_key,
        )
        ReactLog.__init__(self)

        # SECTION: PFR reactor core configuration
        self.pfr_reactor_core = pfr_reactor_core

        # SECTION: Options and heat-transfer configuration
        self.heat_transfer_mode = pfr_reactor_core.heat_transfer_mode
        self.operation_mode = pfr_reactor_core.operation_mode
        self.liquid_density_mode = pfr_reactor_core.liquid_density_mode

        self.heat_exchange = pfr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = pfr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = pfr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = pfr_reactor_core.jacket_temperature_value
        self.heat_rate_value = pfr_reactor_core.heat_rate_value

        # SECTION: Inlet and reactor geometry
        # ! F_in: inlet component molar flow rates [mol/s]
        self._F_in = pfr_reactor_core._F_in
        self._F_in_total = pfr_reactor_core._F_in_total
        # ! T_in: inlet temperature [K]
        self.temperature_in = pfr_reactor_core.temperature_inlet
        self._T_in = pfr_reactor_core._T_in
        # ! V_R: total reactor volume [m3]
        self._Vr = pfr_reactor_core.reactor_volume_value

        # ! rho_LIQ: liquid density [g/m3]
        self._rho_LIQ_in = self.thermo_source.calc_rho_LIQ(
            temperature=self.temperature_in
        )

        # ! q_in: inlet volumetric flow rate [m3/s]
        # constant-volume liquid closure uses this fixed value.
        self._q_in = self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=self._F_in,
            molecular_weights=self.thermo_source.MW,
            density=self._rho_LIQ_in
        )

    @property
    def F_in(self) -> np.ndarray:
        """Inlet component molar-flow vector [mol/s]."""
        return self._F_in

    def build_y0(self) -> np.ndarray:
        """
        Build inlet state vector for solve_ivp.

        Notes
        -----
        - F_i(0) is always included.
        - T(0) is added only for non-isothermal mode.
        """
        # NOTE: inlet component molar flows [mol/s]
        f0 = self.F_in.astype(float)
        if self.heat_transfer_mode == "isothermal":
            return f0
        return np.concatenate([f0, np.array([float(self._T_in)], dtype=float)])

    def rhs(
            self,
            V: float,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Right-hand side for liquid PFR ODE system in reactor-volume coordinate.

        Equations
        ---------
        - Species: dF_i/dV = Σ_j(ν_i,j r_j)
        - Energy (optional): dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^L)
        """
        ns = self.component_num

        # SECTION: unpack state vector
        # ! species states: component molar flows [mol/s]
        F = np.clip(y[:ns], 0.0, None)

        # ! thermal state [K]
        if self.heat_transfer_mode == "non-isothermal":
            temp = float(y[ns])
        else:
            temp = float(self._T_in)

        # NOTE: temperature
        temperature = Temperature(value=temp, unit="K")

        # NOTE: density
        rho_LIQ = self.thermo_source.calc_rho_LIQ(
            temperature=temperature
        )

        # NOTE: volumetric flow rate from closure
        # ! volumetric flow from selected liquid closure [m3/s]
        q_vol = self._calc_q_vol(
            F=F,
            rho_LIQ=rho_LIQ
        )
        # >> avoid zero or negative volumetric flow for concentration calculation
        q_vol = max(q_vol, 1e-30)

        # NOTE: concentration from flow
        # ! C_i = F_i / Q [mol/m3]
        concentration = F / q_vol

        # NOTE: standardized concentration dict for rate interface
        concentration_std = {
            sp: CustomProperty(
                value=concentration[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # SECTION: kinetics evaluation
        rates = self._calc_rates_concentration_basis(
            concentration=concentration_std,
            temperature=temperature
        )

        # SECTION: species balance
        dF_dV = self._build_dF_dV(rates=rates)

        if self.heat_transfer_mode == "isothermal":
            return dF_dV

        # SECTION: energy balance (optional)
        dT_dV = self._build_dT_dV(
            F=F,
            rates=rates,
            temp=temp
        )

        # NOTE: concatenate species and thermal derivatives for non-isothermal mode
        return np.concatenate([dF_dV, np.array([dT_dV], dtype=float)])

    # SECTION: helper methods for balances and closures
    # ! Calculate liquid volumetric flow rate
    def _calc_q_liquid(
            self,
            flow: np.ndarray,
            rho_LIQ: np.ndarray
    ) -> float:
        """
        Estimate liquid volumetric flow rate from molar-flow composition.

        Formula
        -------
        Q = Σ_i(F_i MW_i / rho_i)
        """
        return self.thermo_source.calc_liquid_volumetric_flow_rate(
            molar_flow_rates=flow,
            molecular_weights=self.thermo_source.MW,
            density=rho_LIQ
        )

    # ! Calculate volumetric flow rate closure for concentration calculation
    def _calc_q_vol(
            self,
            F: np.ndarray,
            rho_LIQ: np.ndarray
    ) -> float:
        """
        Compute volumetric-flow closure used to recover concentrations.

        Closures
        --------
        - constant_volume: Q = Q_in (fixed)
        - constant_pressure: Q = Q(F, T) from density/molecular-weight mixing
        """
        if self.operation_mode == "constant_volume":
            return float(self._q_in)
        if self.operation_mode == "constant_pressure":
            return float(self._calc_q_liquid(flow=F, rho_LIQ=rho_LIQ))
        raise ValueError(
            f"Invalid operation_mode '{self.operation_mode}' for liquid PFR."
        )

    # NOTE: species derivative builder for all modes
    def _build_dF_dV(self, rates: np.ndarray) -> np.ndarray:
        """
        Build species derivatives dF_i/dV.

        Formula
        -------
        dF_i/dV = Σ_j(ν_i,j r_j)
        """
        ns = self.component_num
        dF_dV = np.zeros(ns, dtype=float)

        for k, _ in enumerate(self.reactions):
            r_k = rates[k]
            for sp_name, nu_ik in self.reaction_stoichiometry[k].items():
                i = self.component_id_to_index[sp_name]
                dF_dV[i] += nu_ik * r_k

        return dF_dV

    # NOTE: thermal derivative builder for non-isothermal mode
    def _build_dT_dV(
        self,
        F: np.ndarray,
        rates: np.ndarray,
        temp: float
    ) -> float:
        """
        Build thermal derivative dT/dV for non-isothermal liquid PFR.

        Formula
        -------
        dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^L)
        """
        # NOTE: temperature wrapper for thermo API
        temperature = Temperature(value=temp, unit="K")

        # NOTE: flowing heat-capacity rate denominator
        # ! [J/s.K]
        Cp_LIQ_values_out = self.thermo_source.calc_Cp_LIQ(
            temperature=temperature
        )

        # ! total flowing liquid heat capacity [J/s.K]
        Cp_LIQ_total = calc_total_heat_capacity(x=F, cp=Cp_LIQ_values_out)

        # >> check
        if Cp_LIQ_total <= 1e-16:
            raise ValueError(
                "Total flowing liquid heat capacity is too small or zero."
            )

        # NOTE: reaction heat source term [W/m3]
        # ! Q_rxn = sum_k((-dH_k) * r_k)
        # ??? ΔH_k [J/mol]
        delta_h = self.thermo_source.calc_dH_rxns_LIQ(
            temperature=temperature
        )

        # ??? Q_rxn [W/m3] or [J/s.m3] = sum_k((-ΔH_k) * r_k) [J/mol * mol/m3.s]
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=1.0
        )

        # NOTE: jacket/surrounding heat exchange
        # ! [W/m3] or [J/s.m3]
        # ! Q_exchange = U A (T - T_jacket) / V
        q_exchange = 0.0

        # >> check if heat exchange is enabled for this reactor configuration
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=self._Vr
            )

        # NOTE: user-defined constant heat source
        # ! [W/m3]
        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value / self._Vr

        # ! dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^L) [K/m3]
        return (q_rxn + q_exchange + q_constant) / Cp_LIQ_total
