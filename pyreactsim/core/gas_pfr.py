import logging
import numpy as np
from typing import Dict, List
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature
# locals
from ..configs.constants import R_J_per_mol_K
from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .pfrc import PFRReactorCore

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasPFRReactor:
    """
    Gas-phase plug-flow reactor model.

    Modeling basis
    --------------
    - Steady-state PFR in reactor-volume coordinate V.
    - Primary states are component molar flows F_i(V) [mol/s].
    - Optional thermal state T(V) [K] for non-isothermal mode.
    - Optional pressure state P(V) [Pa] for calculated-pressure mode.

    State vector by mode
    --------------------
    - isothermal + constant pressure: [F1, ..., FNc]
    - non-isothermal + constant pressure: [F1, ..., FNc, T]
    - isothermal + calculated pressure: [F1, ..., FNc, P]
    - non-isothermal + calculated pressure: [F1, ..., FNc, T, P]
    """
    R = R_J_per_mol_K

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        pfr_reactor_core: PFRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        self.components = components
        self.component_key = component_key
        self.thermo_source = thermo_source
        self.pfr_reactor_core = pfr_reactor_core

        # SECTION: Options and heat-transfer configuration
        self.heat_transfer_mode = pfr_reactor_core.heat_transfer_mode
        self.operation_mode = pfr_reactor_core.operation_mode
        self.pressure_mode = pfr_reactor_core.pressure_mode
        self.gas_model = pfr_reactor_core.gas_model

        self.heat_exchange = pfr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = pfr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = pfr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = pfr_reactor_core.jacket_temperature_value
        self.heat_rate_value = pfr_reactor_core.heat_rate_value

        # SECTION: Reaction and stoichiometry mapping
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

        # SECTION: Component references
        self.component_num = self.thermo_source.component_refs["component_num"]
        self.component_formula_state = self.thermo_source.component_refs[
            "component_formula_state"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: Inlet and reactor geometry
        # ! F_in: inlet component molar flow rates [mol/s]
        self._F_in = pfr_reactor_core._F_in
        self._F_in_total = pfr_reactor_core._F_in_total
        # ! T_in: inlet temperature [K]
        self._T_in = pfr_reactor_core._T_in
        # ! P0: inlet/reference pressure [Pa]
        self._P0 = pfr_reactor_core._P0
        # ! V_R: total reactor volume [m3]
        self._V_R = pfr_reactor_core.reactor_volume_value

        # SECTION: Mode validation
        if self.pressure_mode == "constant" and self.operation_mode == "constant_volume":
            raise ValueError(
                "Invalid gas PFR setup: operation_mode='constant_volume' is incompatible with pressure_mode='constant'."
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
        - T(0) is added for non-isothermal mode.
        - P(0) is added for calculated-pressure mode.
        """
        # NOTE: inlet component molar flows [mol/s]
        f0 = self.F_in.astype(float)
        # NOTE: mode checks
        is_nonisothermal = self.heat_transfer_mode == "non-isothermal"
        is_pressure_ode = self.pressure_mode == "calculated"

        # NOTE: build state vector parts based on selected case
        y0_parts: List[np.ndarray] = [f0]
        if is_nonisothermal:
            y0_parts.append(np.array([float(self._T_in)], dtype=float))
        if is_pressure_ode:
            y0_parts.append(np.array([float(self._P0)], dtype=float))

        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def rhs(self, V: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side for gas PFR ODE system in reactor-volume coordinate.

        Equations
        ---------
        - Species: dF_i/dV = Σ_j(ν_i,j r_j)
        - Energy (optional): dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^g)
        - Pressure (optional): dP/dV = placeholder (NotImplementedError)
        """
        ns = self.component_num

        # NOTE: mode flags for current simulation case
        is_nonisothermal = self.heat_transfer_mode == "non-isothermal"
        is_pressure_ode = self.pressure_mode == "calculated"

        # SECTION: unpack state vector
        # ! species states: component molar flows [mol/s]
        F = np.clip(y[:ns], 0.0, None)
        idx = ns

        # ! thermal state [K]
        if is_nonisothermal:
            temp = float(y[idx])
            idx += 1
        else:
            temp = float(self._T_in)

        # ! pressure state [Pa] or fixed pressure closure
        if is_pressure_ode:
            p_total = float(max(y[idx], 1e-9))
        else:
            p_total = float(self._P0)

        # SECTION: closure relations
        # ! total molar flow [mol/s]
        F_total = float(np.sum(F))
        F_total = max(F_total, 1e-30)
        # ! gas mole fractions [-]
        y_mole = F / F_total

        # ! ideal-gas volumetric flow: Q = F_total * R * T / P [m3/s]
        q_vol = F_total * self.R * temp / p_total
        q_vol = max(q_vol, 1e-30)

        # ! concentration from flow form: C_i = F_i / Q [mol/m3]
        concentration = F / q_vol

        # NOTE: standardized properties for rate expression interface
        partial_pressures_std = {
            sp: CustomProperty(
                value=y_mole[i] * p_total, unit="Pa", symbol="P")
            for i, sp in enumerate(self.component_formula_state)
        }
        concentration_std = {
            sp: CustomProperty(
                value=concentration[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # SECTION: kinetics evaluation
        rates = self._calc_rates(
            partial_pressures=partial_pressures_std,
            concentration=concentration_std,
            temperature=Temperature(value=temp, unit="K"),
            pressure=Pressure(value=p_total, unit="Pa")
        )

        # SECTION: species balance
        dF_dV = self._build_dF_dV(rates=rates)

        out: List[np.ndarray] = [dF_dV]

        # SECTION: energy balance (optional)
        if is_nonisothermal:
            dT_dV = self._build_dT_dV(F=F, rates=rates, temp=temp)
            out.append(np.array([dT_dV], dtype=float))

        # SECTION: pressure balance (optional)
        if is_pressure_ode:
            dP_dV = self._calc_dP_dV(F=F, temp=temp, p_total=p_total)
            out.append(np.array([dP_dV], dtype=float))

        return np.concatenate(out)

    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ) -> np.ndarray:
        """
        Evaluate reaction rates for all reactions [mol/m3.s].

        Basis support
        -------------
        - basis='pressure': rate uses partial pressures
        - basis='concentration': rate uses concentrations
        """
        rates = []

        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis
            if basis == "pressure":
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for gas PFR reaction rate expression '{rate_exp.name}'."
                )
            rates.append(float(r_k.value))

        return np.array(rates, dtype=float)

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

    def _build_dT_dV(
        self,
        F: np.ndarray,
        rates: np.ndarray,
        temp: float
    ) -> float:
        """
        Build thermal derivative dT/dV for non-isothermal gas PFR.

        Formula
        -------
        dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^g)
        """
        # NOTE: temperature wrapper for thermo API
        temperature = Temperature(value=temp, unit="K")

        # NOTE: flowing heat-capacity rate denominator [J/s.K]
        cp_g_values = self.thermo_source.calc_Cp_IG(temperature=temperature)
        cp_flow = calc_total_heat_capacity(x=F, cp=cp_g_values)
        if cp_flow <= 1e-16:
            raise ValueError(
                "Total flowing gas heat capacity is too small or zero.")

        # NOTE: reaction heat source term [W/m3]
        delta_h = self.thermo_source.calc_dH_rxns(temperature=temperature)
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
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

        return (q_rxn + q_exchange + q_constant) / cp_flow

    def _calc_dP_dV(
        self,
        F: np.ndarray,
        temp: float,
        p_total: float
    ) -> float:
        """
        Placeholder for gas PFR pressure-drop model.

        Expected future form
        --------------------
        - Ergun-type or friction-factor pressure-drop relation:
          dP/dV = f(F, T, P, geometry, fluid properties)
        """
        raise NotImplementedError(
            "Gas PFR pressure ODE placeholder: dP/dV model is not implemented yet. "
            "Add a pressure-drop relation (e.g., Ergun/friction model) here."
        )
