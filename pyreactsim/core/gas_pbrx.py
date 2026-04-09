import logging
import numpy as np
from typing import Dict, List, cast, Optional, Tuple
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature
# locals
from .pbrc import PBRReactorCore
from ..configs.constants import R_J_per_mol_K
from ..models.rate_exp import ReactionRateExpression
from ..models.ref import GasModel
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity, calc_pressure_using_PFT
from ..utils.tools import smooth_floor

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasPBRReactorX:
    """
    Gas-phase packed-bed reactor model.

    PBR difference from PFR:
    rates are interpreted on catalyst-mass basis r' [mol/kg.s] and converted to
    reactor-volume basis r_V [mol/m3.s] via r_V = rho_B * r'.
    """
    R = R_J_per_mol_K

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
        self.component_formula_state = self.thermo_source.component_refs[
            "component_formula_state"
        ]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: inlet and reactor geometry
        self._F_in = pbr_reactor_core._F_in
        self._F_in_total = pbr_reactor_core._F_in_total
        self._T_in = pbr_reactor_core._T_in
        self._P0 = pbr_reactor_core._P0
        self._V_R = pbr_reactor_core.reactor_volume_value

        # SECTION: final configuration checks
        self.pbr_reactor_core.config_model()

        # SECTION: solver scaling
        # Species scaling uses inlet molar flows with a protective floor so
        # product species with zero inlet flow still get a meaningful scale.
        self.F_scale = np.maximum(self._F_in.astype(float), 1e-8)

        # Temperature scaling is written as a deviation scale rather than a
        # ratio scale because reactor temperatures may increase or decrease
        # around the inlet temperature.
        self.T_ref = float(self._T_in)
        self.T_scale = 100.0  # K, practical default for non-isothermal gas PBRs

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

    def build_y0_scaled(self) -> np.ndarray:
        """
        Build scaled inlet state vector for solve_ivp.

        Scaling
        -------
        - Species: f_i = F_i / F_scale_i
        - Temperature: theta = (T - T_ref) / T_scale
        """
        f0_scaled = self.F_in.astype(float) / self.F_scale
        y0_parts: List[np.ndarray] = [f0_scaled]

        if self.heat_transfer_mode == "non-isothermal":
            theta0 = (float(self._T_in) - self.T_ref) / self.T_scale
            y0_parts.append(np.array([theta0], dtype=float))

        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert a scaled state vector to physical units.
        """
        ns = self.component_num

        # NOTE: species molar flows [mol/s]
        F = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.F_scale

        # NOTE: thermal state [K]
        if self.pbr_reactor_core.is_non_isothermal:
            theta = float(y_scaled[ns])
            temp = self.T_ref + self.T_scale * theta
            temp = float(smooth_floor(temp, xmin=1.0, s=1e-3))
        else:
            temp = float(self._T_in)

        return F, temp

    def _scale_rhs(
        self,
        dF_dV: np.ndarray,
        dT_dV: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert physical derivatives to scaled derivatives.
        """
        dF_scaled_dV = dF_dV / self.F_scale

        if dT_dV is None:
            return dF_scaled_dV

        dtheta_dV = dT_dV / self.T_scale
        return np.concatenate([dF_scaled_dV, np.array([dtheta_dV], dtype=float)])

    def rhs_physical(
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
        F = np.asarray(
            smooth_floor(y[:ns], xmin=0.0, s=1e-12),
            dtype=float
        )

        # NOTE: temperature state configuration
        # ! thermal state [K]
        if self.pbr_reactor_core.is_non_isothermal:
            temp = float(y[ns])
        else:
            temp = float(self._T_in)
        # >>> set
        temperature = Temperature(value=temp, unit="K")

        # SECTION: closure relations
        # ! total molar flow [mol/s]
        F_total = float(smooth_floor(float(np.sum(F)), xmin=1e-30, s=1e-31))
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
        q_vol = float(smooth_floor(q_vol, xmin=1e-30, s=1e-31))

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

    def rhs_scaled(
            self,
            V: float,
            y_scaled: np.ndarray
    ) -> np.ndarray:
        """
        Scaled right-hand side for solve_ivp.

        Notes
        -----
        The reactor model itself is still evaluated in physical units.
        Only the state vector seen by the ODE solver is scaled.
        """
        ns = self.component_num

        # NOTE: convert scaled state to physical units
        F, temp = self._unscale_state(y_scaled)

        if self.pbr_reactor_core.is_non_isothermal:
            y_physical = np.concatenate([F, np.array([temp], dtype=float)])
        else:
            y_physical = F

        dy_physical_dV = self.rhs_physical(V, y_physical)

        dF_dV = dy_physical_dV[:ns]

        if self.heat_transfer_mode == "isothermal":
            return self._scale_rhs(dF_dV=dF_dV)

        dT_dV = float(dy_physical_dV[ns])
        return self._scale_rhs(dF_dV=dF_dV, dT_dV=dT_dV)

    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure,
        args: Optional[Dict[str, CustomProperty]] = None,
    ) -> np.ndarray:
        """
        Evaluate raw reaction rates for all reactions based on the provided state and closure properties. The raw rates are on catalyst-mass basis r' [mol/kg.s] and will be converted to reactor-volume basis r_V [mol/m3.s] via r_V = rho_B * r'.

        Notes
        -----
        - The args parameter contains the bulk density rho_B for packed-bed conversion, which is passed to the rate expression calculations.
        - The final rate unit is mol/m3.s after conversion, which is suitable for the species balance equations in the PBR model.
        """
        rates = []

        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis

            # >> check
            if basis == "pressure":
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    args=args,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                r_k = rate_exp.calc(
                    xi=concentration,
                    args=args,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for gas PBR reaction rate expression '{rate_exp.name}'."
                )

            # check unit (already done)
            # store
            rates.append(float(r_k.value))

        return np.array(rates, dtype=float)

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

        cp_flow = calc_total_heat_capacity(x=F, cp=cp_g_values)

        # check
        if cp_flow <= 1e-16:
            raise ValueError(
                "Total flowing gas heat capacity is too small or zero.")

        # NOTE: reaction heat source term [w/m3] uses converted r_V rates
        delta_h = self.thermo_source.calc_dH_rxns(temperature=temperature)

        # calculate generation heat
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

        return (q_rxn + q_exchange + q_constant) / cp_flow
