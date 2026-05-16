import logging
import numpy as np
from typing import Dict, List, cast, Optional, cast, Literal
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature
from pyreactsim_core.models import ReactionRateExpression
# locals
from ..configs.constants import R_J_per_mol_K
from ..models.ref import GasModel
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity, calc_pressure_using_PFT
from .pbrc import PBRReactorCore
# auxiliary
from .react_aux import ReactorAuxiliary
# log
from .react_log import ReactLog

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasPBRReactor(ReactorAuxiliary, ReactLog):
    """
    Gas-phase packed-bed reactor model.

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
        # LINK: ReactorAuxiliary initialization
        super().__init__(
            components=components,
            reaction_rates=reaction_rates,
            thermo_source=thermo_source,
            reactor_core=pbr_reactor_core,
            component_key=component_key,
        )
        ReactLog.__init__(self)

        # SECTION: core model and configuration
        self.pbr_reactor_core = pbr_reactor_core

        # SECTION: options and heat-transfer configuration
        self.heat_transfer_mode = pbr_reactor_core.heat_transfer_mode
        self.operation_mode = pbr_reactor_core.operation_mode
        self.pressure_mode = pbr_reactor_core.pressure_mode
        self.gas_model = pbr_reactor_core.gas_model

        self.heat_exchange = pbr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = pbr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = pbr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = pbr_reactor_core.jacket_temperature_value
        self.heat_rate_value = pbr_reactor_core.heat_rate_value

        # NOTE: packed-bed catalyst bulk density [kg/m3]
        self._rho_B_value = pbr_reactor_core._rho_B_value
        self.rho_B = pbr_reactor_core.rho_B
        self.rho_B_arg = {
            "rho_B": self.rho_B
        }

        # SECTION: inlet and reactor geometry
        self._F_in = pbr_reactor_core._F_in
        self._F_in_total = pbr_reactor_core._F_in_total
        self._T_in = pbr_reactor_core._T_in
        self._P0 = pbr_reactor_core._P0
        self._V_R = pbr_reactor_core.reactor_volume_value

        # SECTION: final configuration checks
        self.pbr_reactor_core.config_model()

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
        - T(0) is added for non-isothermal mode.
        - Pressure state is not included in current PBR scope.
        """
        # NOTE: inlet component molar flows [mol/s]
        f0 = self.F_in.astype(float)
        # NOTE: build state vector parts based on selected case
        y0_parts: List[np.ndarray] = [f0]

        if self.heat_transfer_mode == "non-isothermal":
            y0_parts.append(np.array([float(self._T_in)], dtype=float))

        # NOTE: state-variable pressure is currently not implemented by design.
        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def rhs(
            self,
            V: float,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Right-hand side for gas PBR ODE system in reactor-volume coordinate.

        Equations
        ---------
        - Species: dF_i/dV = Σ_j(ν_i,j r_V,j)
        - Energy (optional): dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^g)

        Packed-bed conversion
        ---------------------
        - Raw kinetic rate from expression: r' [mol/kg.s]
        - Converted reactor-volume rate: r_V = rho_B * r' [mol/m3.s]
        """
        ns = self.component_num

        # SECTION: unpack state vector
        # ! species states: component molar flows [mol/s]
        F = np.clip(y[:ns], 0.0, None)

        # NOTE: temperature state configuration
        # ! thermal state [K]
        if self.pbr_reactor_core.is_non_isothermal:
            temp = float(y[ns])
        else:
            temp = float(self._T_in)
        temp = max(temp, 1.0)
        # >>> set
        temperature = Temperature(value=temp, unit="K")

        # SECTION: closure relations
        # ! total molar flow [mol/s]
        F_total = max(float(np.sum(F)), 1e-30)
        # ! gas mole fractions [-]
        y_mole = F / F_total

        # NOTE: pressure closure
        if self.pressure_mode == "shortcut":
            p_total = calc_pressure_using_PFT(
                P_in=self._P0,
                F_in_total=self._F_in_total,
                T_in=self._T_in,
                F_out_total=F_total,
                T_out=temp,
                heat_transfer_mode=cast(
                    Literal['isothermal', 'non-isothermal'],
                    self.heat_transfer_mode
                )
            )
        elif self.pressure_mode == "constant":
            p_total = float(self._P0)
        elif self.pressure_mode == "state_variable":
            raise NotImplementedError(
                "PBR pressure_mode='state_variable' is not implemented yet."
            )
        else:
            raise ValueError(
                f"Invalid pressure_mode '{self.pressure_mode}' for gas PBR."
            )

        # >>> set
        pressure = Pressure(value=p_total, unit="Pa")

        # NOTE: gas volumetric flow
        # ! Q = F_total * R * T / P [m3/s]
        q_vol = self.thermo_source.calc_gas_volumetric_flow_rate(
            molar_flow_rate=F_total,
            temperature=temp,
            pressure=p_total,
            R=self.R,
            gas_model=cast(GasModel, self.gas_model)
        )
        q_vol = max(q_vol, 1e-30)

        # ! concentration from flow form: C_i = F_i / Q [mol/m3]
        concentration = F / q_vol

        # NOTE: standardized properties for rate expression interface
        # ! partial pressures: P_i = y_i * P_total [Pa]
        partial_pressures_std = {
            sp: CustomProperty(
                value=y_mole[i] * p_total, unit="Pa", symbol="P")
            for i, sp in enumerate(self.component_formula_state)
        }

        # ! concentrations: C_i = F_i / Q [mol/m3]
        concentration_std = {
            sp: CustomProperty(
                value=concentration[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # SECTION: kinetics evaluation
        # NOTE: raw rates are catalyst-mass basis r' [mol/kg.s]
        # packed-bed conversion to reactor-volume basis r_V [mol/m3.s]
        rates_v = self._calc_rates(
            partial_pressures=partial_pressures_std,
            concentration=concentration_std,
            temperature=temperature,
            pressure=pressure,
            args=self.rho_B_arg
        )

        # NOTE: species balance
        dF_dV = self._build_dF_dV(rates=rates_v)

        # isothermal case: only species balances
        if self.heat_transfer_mode == "isothermal":
            return dF_dV

        # NOTE: energy balance (optional)
        dT_dV = self._build_dT_dV(F=F, rates_v=rates_v, temp=temp)
        return np.concatenate([dF_dV, np.array([dT_dV], dtype=float)])

    def _build_dF_dV(self, rates: np.ndarray) -> np.ndarray:
        """
        Build species derivatives dF_i/dV.

        Formula
        -------
        dF_i/dV = Σ_j(ν_i,j r_V,j)
        """
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
        """
        Build thermal derivative dT/dV for non-isothermal gas PBR.

        Formula
        -------
        dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^g)
        """
        # NOTE: temperature wrapper for thermo API
        temperature = Temperature(value=temp, unit="K")

        # NOTE: flowing heat-capacity rate denominator [J/s.K]
        cp_g_values = self.thermo_source.calc_Cp_IG(temperature=temperature)

        # ! total flowing gas heat capacity [J/s.K] = Σ_i(F_i Cp_i^g)
        cp_flow = calc_total_heat_capacity(x=F, cp=cp_g_values)

        # check
        if cp_flow <= 1e-16:
            raise ValueError(
                "Total flowing gas heat capacity is too small or zero.")

        # NOTE: reaction heat source term [w/m3] uses converted r_V rates
        # ! calculate heat generated by reactions: Q_rxn = Σ_k [(-ΔH_k) r_k]
        # ? enthalpy change of reactions [J/mol]
        delta_h = self._calc_dH_rxns(
            temperature=temperature,
            phase=cast(Literal['gas', 'liquid'], 'gas')
        )

        # calculate generation heat
        # ? reaction heat generation [W/m3] = Σ_k [(-ΔH_k) r_V,k]
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates_v,
            reactor_volume=1.0
        )

        # NOTE: jacket/surrounding heat exchange [W/m3]
        q_exchange = 0.0
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=self._V_R
            )

        # NOTE: user-defined constant heat source [W/m3]
        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value / self._V_R

        # ! dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^g)
        return (q_rxn + q_exchange + q_constant) / cp_flow
