import logging
import time
import numpy as np
from typing import Dict, List, Optional, cast, Literal
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Temperature
from pyreactsim_core.models import ReactionRateExpression
# locals
# from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import calc_residence_time
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity
from .pbrc import PBRReactorCore
# auxiliary
from .react_aux import ReactorAuxiliary
# log
from .react_log import ReactLog

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidPBRReactor(ReactorAuxiliary, ReactLog):
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
        # LINK: ReactorAuxiliary initialization
        super().__init__(
            components=components,
            reaction_rates=reaction_rates,
            thermo_source=thermo_source,
            reactor_core=pbr_reactor_core,
            component_key=component_key,
        )
        ReactLog.__init__(self)

        # section: core model and configuration
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

        # volumetric inlet flow rate mode for liquid PFR closure
        self.volumetric_inlet_flow_rate_mode = pbr_reactor_core.volumetric_inlet_flow_rate_mode

        # NOTE: packed-bed catalyst bulk density (without unit conversion)
        self._rho_B_value = pbr_reactor_core._rho_B_value
        self._rho_B = pbr_reactor_core.rho_B
        self.rho_B_arg = {
            "rho_B": self._rho_B
        }

        # SECTION: inlet and reactor geometry
        # ! feed flow rate [mol/s]
        self._F_in = pbr_reactor_core._F_in

        # ! inlet temperature [K]
        self.temperature_in = pbr_reactor_core.temperature_inlet
        self._T_in = pbr_reactor_core._T_in

        # ! reactor volume [m3]
        self._Vr = pbr_reactor_core.reactor_volume_value

        # ! inlet liquid density [g/m3]
        self._rho_LIQ_in = self.thermo_source.calc_rho_LIQ(
            temperature=self.temperature_in
        )

        # ! q_in: inlet volumetric flow rate [m3/s]
        # constant-volume liquid closure uses this fixed value.
        self._q_in = self.config_volumetric_inlet_flow()

        # ! inlet concentration [mol/m3] from closure
        self._C_in = self._F_in / self._q_in

        # NOTE: residence time
        self.residence_time = calc_residence_time(
            volume=self._Vr,
            volumetric_flow_rate=self._q_in
        )

        # SECTION: final configuration checks
        self.pbr_reactor_core.config_model()

        # SECTION: rhs logging configuration
        # Use `rhs_log_interval` (or `log_v_interval`) to log once every X in V.
        log_interval = kwargs.get(
            "rhs_log_interval", kwargs.get("log_v_interval"))
        if log_interval is None:
            self.rhs_log_interval: Optional[float] = None
        else:
            self.rhs_log_interval = float(log_interval)
            if self.rhs_log_interval < 0.0:
                raise ValueError("rhs_log_interval must be >= 0.")

        enabled_default = self.rhs_log_interval is not None
        self.rhs_log_enabled = bool(
            kwargs.get("rhs_log_enabled", kwargs.get(
                "enable_rhs_logging", enabled_default))
        )
        self.rhs_log_level = int(kwargs.get("rhs_log_level", logging.INFO))
        self.rhs_log_timing_enabled = bool(
            kwargs.get("rhs_log_timing_enabled", True)
        )
        self._rhs_next_log_v: Optional[float] = None
        self._rhs_last_v: Optional[float] = None
        self._rhs_log_active = False
        self._rhs_log_v: Optional[float] = None
        self._rhs_wall_t0: Optional[float] = None
        self._rhs_last_log_wall: Optional[float] = None

    # SECTION: model configuration methods
    # NOTE: inlet volumetric flow rate configuration for liquid PBR based on selected mode
    def config_volumetric_inlet_flow(self) -> float:
        """
        Configure inlet volumetric flow rate for liquid PBR based on selected mode.

        Modes
        -----
        - constant: fixed inlet volumetric flow rate from model inputs.
        - density-dependant: calculate inlet volumetric flow rate based on inlet mass flow rate and density at inlet conditions.
        """
        if self.volumetric_inlet_flow_rate_mode == "constant":
            # ! q_in is fixed from model inputs [m3/s]
            q_in = self.pbr_reactor_core._q0
            # >> check
            if q_in is None:
                raise ValueError(
                    "Inlet volumetric flow rate (q_in) must be provided in model inputs for constant mode."
                )

            return q_in
        elif self.volumetric_inlet_flow_rate_mode == "density-dependant":
            # ! q_in is calculated from inlet molar flow rates and density at inlet conditions [m3/s]
            q_in = self.thermo_source.calc_liquid_volumetric_flow_rate(
                molar_flow_rates=self._F_in,
                molecular_weights=self.thermo_source.MW,
                density=self._rho_LIQ_in
            )
            return q_in
        else:
            raise ValueError(
                f"Invalid volumetric_inlet_flow_rate_mode '{self.volumetric_inlet_flow_rate_mode}' for liquid PFR."
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
        Right-hand side for liquid PBR ODE system in reactor-volume coordinate.

        Equations
        ---------
        - Species: dF_i/dV = Σ_j(ν_i,j r_V,j)
        - Energy (optional): dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^L)

        Packed-bed conversion
        ---------------------
        - Raw kinetic rate from expression: r' [mol/kg.s]
        - Converted reactor-volume rate: r_V = rho_B * r' [mol/m3.s]
        """
        ns = self.component_num
        self._rhs_log_active = self._should_log_rhs(V)
        self._rhs_log_v = float(V) if self._rhs_log_active else None
        if self._rhs_log_active:
            self._rhs_wall_t0 = time.perf_counter()
            self._rhs_last_log_wall = self._rhs_wall_t0
        else:
            self._rhs_wall_t0 = None
            self._rhs_last_log_wall = None

        # SECTION: unpack state vector
        # ! species states: component molar flows [mol/s]
        F = np.clip(y[:ns], 0.0, None)

        # ! thermal state [K]
        if self.heat_transfer_mode == "non-isothermal":
            temp = float(y[ns])
        else:
            temp = float(self._T_in)
        # >>> set
        temperature = Temperature(value=temp, unit="K")

        # NOTE: density
        # ! rho_LIQ = ρ(T) from selected liquid density model, used for concentration and volumetric flow calculations [g/m3]
        # ? constant volume -> constant density,
        # ? variable volume --> variable density
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

        # ! concentration from flow form: C_i = F_i / Q [mol/m3]
        # ! C_i [mol/m3] = F_i [mol/s] / q_vol [m3/s]
        concentration = F / q_vol

        # NOTE: standardized concentration dict for rate interface
        concentration_std = {
            sp: CustomProperty(
                value=concentration[i], unit="mol/m3", symbol="C")
            for i, sp in enumerate(self.component_formula_state)
        }

        # SECTION: kinetics evaluation
        # NOTE: raw rates are catalyst-mass basis r' [mol/kg.s]
        # packed-bed conversion to reactor-volume basis r_V [mol/m3.s]
        rates_v = self._calc_rates_concentration_basis(
            concentration=concentration_std,
            temperature=temperature,
            args=self.rho_B_arg
        )

        # SECTION: species balance
        dF_dV = self._build_dF_dV(rates=rates_v)

        if self.heat_transfer_mode == "isothermal":
            return dF_dV

        # SECTION: energy balance (optional)
        dT_dV = self._build_dT_dV(
            F=F,
            rates_v=rates_v,
            temp=temp,
            volumetric_flow_rate=q_vol
        )

        # NOTE: concatenate species and thermal derivatives for non-isothermal mode
        out = np.concatenate([dF_dV, np.array([dT_dV], dtype=float)])
        return out

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

        Parameters
        ----------
        flow: np.ndarray
            Component molar flow rates [mol/s].
        rho_LIQ: np.ndarray
            Component liquid densities [g/m3].
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
            f"Invalid operation_mode '{self.operation_mode}' for liquid PBR."
        )

    # NOTE: species derivative builder for all modes
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

        self._log_rhs("_build_dF_dV")
        return dF_dV

    # NOTE: thermal derivative builder for non-isothermal mode
    def _build_dT_dV(
        self,
        F: np.ndarray,
        rates_v: np.ndarray,
        temp: float,
        volumetric_flow_rate: float
    ) -> float:
        """
        Build thermal derivative dT/dV for non-isothermal liquid PBR.

        Formula
        -------
        dT/dV = (q_rxn + q_exchange + q_constant) / Σ_i(F_i Cp_i^L)
        """
        # NOTE: temperature wrapper for thermo API
        temperature = Temperature(value=temp, unit="K")

        # NOTE: calculate total flowing heat capacity of liquid mixture based on selected mode
        # ? if use_liquid_mixture_volumetric_heat_capacity is True, Cp_LIQ_MIX_TOTAL = Cp_LIQ_MIX_VOLUMETRIC * Q where Cp_LIQ_MIX_VOLUMETRIC is in J/m3.K and Q is in m3/s, giving J/s.K
        Cp_LIQ_MIX_TOTAL_FLOWING = self._calc_total_flowing_heat_capacity_liquid(
            F=F,
            temperature=temperature,
            volumetric_flow_rate=volumetric_flow_rate,
            mode=cast(
                Literal['calculate', 'constant'],
                self.Cp_LIQ_MIX_VOLUMETRIC_MODE
            )
        )

        # NOTE: reaction heat source term [W/m3] uses converted r_V rates
        # ! Q_rxn = sum_k((-dH_k) * r_k)
        # ??? ΔH_k [J/mol]
        # delta_h is reaction enthalpy change [J/mol] for each reaction, positive for endothermic
        delta_h = self._calc_dH_rxns(
            temperature=temperature,
            phase=cast(
                Literal['gas', 'liquid'],
                'liquid'
            )
        )

        # ??? Q_rxn [W/m3] or [J/s.m3] = sum_k((-ΔH_k) * r_k) [J/mol * mol/m3.s]
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates_v,
            reactor_volume=1.0
        )

        # NOTE: jacket/surrounding heat exchange [W/m3]
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

        # ! final dT/dV calculation
        dT_dV = (q_rxn + q_exchange + q_constant) / Cp_LIQ_MIX_TOTAL_FLOWING

        # results in K/m3, since q_rxn, q_exchange, and q_constant are in W/m3 or J/s.m3, and Cp_LIQ_MIX_TOTAL_FLOWING is in J/s.K
        return dT_dV
