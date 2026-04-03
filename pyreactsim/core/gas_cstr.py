import logging
import numpy as np
from typing import Dict, List, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature
# locals
from ..configs.constants import R_J_per_mol_K
from .cstrc import CSTRReactorCore
from ..models.br import GasModel
from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasCSTRReactor:
    """
    Gas-phase CSTR dynamic reactor model.

    State vector:
    - isothermal cases: [n1, n2, ..., nNc]
    - non-isothermal cases: [n1, n2, ..., nNc, T]

    Physical interpretation
    ------------------------
    - Constant volume -> pressure varies
    - Constant pressure -> volume varies or fixed (holdup) volume

    Holdup volume
    --------------
    - Fixed holdup -> the gas always fills a constant volume (rigid vessel + level/pressure control keeps the effective reacting volume fixed).
    - Dynamic holdup -> the gas volume can expand/contract (piston, variable-level tank, balloon-like reactor, or the reactor simply fills/empties).

    Cases
    -----
    - `Rigid vessel with pressure control` (most common industrial/lab gas CSTR)
        operation_mode = "constant_pressure"
        holdup_volume_mode = "fixed"
        outlet_flow_mode = "calculated" (default) to maintain constant P and V with the formula as:
            F_out_total = F_in_total + V * Σ_j(ν_i,j r_j) (total mole balance)

    What happens:
        P stays constant (back-pressure regulator or controller).
        V stays constant (rigid vessel, gas always fills it).
        F_out_total is automatically adjusted by the model to approximately keep n_total consistent with PV = nRT (accounting for reaction mole change and temperature effects).
        T can vary (non-isothermal) or be fixed (isothermal).
        This is the case where outlet flow "fights" to maintain constant P and V.

    - `Constant pressure, variable volume` (e.g. piston reactor, balloon, or level-changing gas holder)
        operation_mode = "constant_pressure"
        holdup_volume_mode = "dynamic"
        outlet_flow_mode = "calculated" (often simplest).

    What happens:
        P fixed.
        V varies freely: V(t) = n_total(t) × R × T(t) / P.
        F_out_total usually set ≈ F_in_total (simple mass balance; volume absorbs the difference).
        Good when the reactor can expand/contract.

    - `Fixed volume, pressure drifts` (classic constant-volume gas reactor)
        operation_mode = "constant_volume"
        holdup_volume_mode is ignored.
        outlet_flow_mode = "calculated" (recommended)

    What happens:
        V fixed.
        P varies: P(t) = n_total(t) × R × T(t) / V.
        F_out_total usually ≈ F_in_total.
        Pressure will rise or fall depending on whether the reaction increases/decreases total moles and on temperature changes.
    """
    # NOTE: Constants
    # ! universal gas constant [J/mol.K]
    R = R_J_per_mol_K

    # NOTE: Primary state/configuration values
    # ! initial component moles in reactor [mol]
    _N0: np.ndarray = np.array([])
    # ! inlet component molar flow rates [mol/s]
    _F_in: np.ndarray = np.array([])
    # ! outlet total molar flow rate [mol/s]
    _F_out_total: float = 0.0
    # ! initial reactor temperature [K]
    _T0: float = 0.0
    # ! inlet stream temperature [K]
    _T_in: float = 0.0
    # ! reactor pressure for constant-pressure modes [Pa]
    _P0: float = 0.0
    # ! reactor holdup volume for fixed-volume modes [m3]
    _V0: float = 0.0

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        cstr_reactor_core: CSTRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        """
        Initialize the gas-phase CSTR model from configured options and thermo source.

        Parameters
        ----------
        components : List[Component]
            Chemical components participating in reactor chemistry.
        reaction_rates : List[ReactionRateExpression]
            Kinetic expressions for reactions, either pressure- or concentration-based.
        thermo_source : ThermoSource
            Thermodynamic source wrapper with Cp, dH_rxn, EOS, and reaction definitions.
        cstr_reactor_core : CSTRReactorCore
            Validated CSTR inputs/options provider.
        component_key : ComponentKey
            Component ID key used to map stoichiometry and states.
        """
        self.components = components
        self.component_key = component_key
        self.thermo_source = thermo_source
        self.cstr_reactor_core = cstr_reactor_core

        # SECTION: Reactor mode configuration
        self.cstr_reactor_core = cstr_reactor_core
        # >>>
        self.gas_model = cstr_reactor_core.gas_model
        self.heat_transfer_mode = cstr_reactor_core.heat_transfer_mode
        self.operation_mode = cstr_reactor_core.operation_mode

        # ! Heat transfer configuration
        self.heat_exchange = cstr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = cstr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = cstr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = cstr_reactor_core.jacket_temperature_value
        # ! heat rate [W]
        self.heat_rate = cstr_reactor_core.heat_rate
        self.heat_rate_value = cstr_reactor_core.heat_rate_value

        # SECTION: Reaction and stoichiometry mapping
        self.reaction_rates = reaction_rates
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
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
        self.component_ids = self.thermo_source.component_refs["component_ids"]
        self.component_formula_state = self.thermo_source.component_refs[
            "component_formula_state"]
        self.component_mapper = self.thermo_source.component_refs["component_mapper"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: CSTR input streams and initial holdup
        # ! initial moles [mole]
        _, self._N0 = self.cstr_reactor_core.config_initial_mole()

        # ! inlet molar flow rates [mol/s]
        _, self._F_in = self.cstr_reactor_core.config_inlet_mole_flows()

        # ! outlet total molar flow rate [mol/s]
        self._F_out_total = self.cstr_reactor_core.config_outlet_mole_flow_total()

        # ! T_in: initial temperature [K]
        self.temperature_initial = self.cstr_reactor_core.config_initial_temperature()
        self._T_in = self.temperature_initial.value

        # ! T: inlet temperature [K]
        self.temperature_inlet = self.cstr_reactor_core.config_inlet_temperature()
        self._T_in = self.temperature_inlet.value

        # ! Cp: initial heat capacity configuration
        self.Cp_IG_values_in = self.thermo_source.calc_Cp_IG(
            temperature=self.temperature_initial
        )

        # ! En_IG: ideal gas enthalpy for inlet stream components [J/mol]
        _, self.En_IG_values_in = self.thermo_source.calc_En_IG(
            temperature=self.temperature_inlet
        )

        # SECTION: Case-based pressure/volume initialization
        self._configure_pressure_volume_initial()

    def _configure_pressure_volume_initial(self):
        """
        Configure initial pressure and/or volume based on selected CSTR case closure.

        Notes
        -----
        - constant-pressure cases: pressure is required
        - constant-volume cases: volume is required
        - fixed-volume constant-pressure cases (1, 3, 5): volume is also required
        """
        # ! P: initial pressure [Pa]
        if self.operation_mode == "constant_volume":
            # retrieve
            # ! volume [m3]
            self.volume = self.cstr_reactor_core.config_reactor_volume()
            self._V0 = self.volume.value

            # calc
            # ? P = n_total * R * T / V
            self._P0 = self.thermo_source.calc_tot_pressure(
                n_total=np.sum(self._N0),
                temperature=self._T0,
                reactor_volume_value=self._V0,
                R=self.R,
                gas_model=cast(GasModel, self.gas_model)
            )
        # ! V: initial volume [m3]
        elif self.operation_mode == "constant_pressure":
            # retrieve
            self.pressure = self.cstr_reactor_core.config_pressure()
            self._P0 = self.pressure.value

            # calc
            # ? V = N_total * R * T / P
            self._V0 = self.thermo_source.calc_gas_volume(
                n_total=np.sum(self._N0),
                temperature=self._T0,
                pressure=self._P0,
                R=self.R,
                gas_model=cast(GasModel, self.gas_model)
            )

        else:
            raise ValueError(
                f"Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

    @property
    def N0(self) -> np.ndarray:
        """Initial holdup moles vector [mol]."""
        if self._N0 is None:
            raise ValueError("N0 has not been set.")
        return self._N0

    @property
    def F_in(self) -> np.ndarray:
        """Inlet component molar flow vector [mol/s]."""
        if self._F_in is None:
            raise ValueError("F_in has not been set.")
        return self._F_in

    # SECTION: Build initial value for n and T
    def build_y0(self) -> np.ndarray:
        # NOTE: initial moles
        n0 = self.N0

        # NOTE: initial temperature
        T0 = self._T0

        # NOTE: build initial value vector
        if self.heat_transfer_mode == "isothermal":
            # state vector: [n1, n2, ..., nNc]
            y0 = n0
        else:
            # state vector: [n1, n2, ..., nNc, T]
            y0 = np.concatenate([n0, np.array([T0], dtype=float)])

        return y0

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the dynamic CSTR ODE system.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        y : np.ndarray
            Current state vector of moles and (if applicable) temperature.

        Returns
        -------
        np.ndarray
            Time derivatives of model states.
        """
        ns = self.component_num

        # NOTE: unpack state vector
        if self.heat_transfer_mode == "isothermal":
            if self._T0 is None:
                raise ValueError(
                    "initial temperature must be provided for isothermal simulation."
                )
            n = y[:ns]
            temp = float(self._T0)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # NOTE: enforce physically non-negative moles
        n = np.clip(n, 0.0, None)
        # ! total moles [mol]
        n_total = float(np.sum(n))
        n_total = max(n_total, 1e-30)
        # ! mole fractions [-]
        y_mole = n / n_total

        # NOTE: closure calculation
        # ! total pressure and reactor volume
        p_total, reactor_volume = self._calc_pressure_volume(
            n_total=n_total,
            temperature=temp
        )

        # ! partial pressures [Pa]
        (
            _,
            partial_pressures_std,
            _
        ) = self._calc_partial_pressure(
            y_mole=y_mole,
            p_total=p_total
        )

        # ! concentrations [mol/m3]
        (
            _,
            concentration_std,
            _
        ) = self._calc_concentration(
            n=n,
            reactor_volume=reactor_volume
        )

        # NOTE: evaluate reaction rates for each reaction [mol/m3.s]
        rates = self._calc_rates(
            partial_pressures=partial_pressures_std,
            concentration=concentration_std,
            temperature=Temperature(value=temp, unit="K"),
            pressure=Pressure(value=p_total, unit="Pa")
        )

        # NOTE: calculate total outlet molar flow rate based on operating policy and case closure
        F_out_total = self._calc_F_out_total(
            n=n,
            temp=temp,
            reactor_volume=reactor_volume,
            p_total=p_total,
            rates=rates
        )

        # NOTE: species balance
        # ! dn_i/dt = F_in,i - F_out,i + V * Σ_j(ν_i,j r_j)
        dn_dt = self._build_dn_dt(
            ns=ns,
            rates=rates,
            reactor_volume=reactor_volume,
            y_mole=y_mole,
            F_out_total=F_out_total
        )

        # isothermal cases: only moles are states
        if self.cstr_reactor_core.is_isothermal:
            return dn_dt

        # NOTE: non-isothermal energy balance
        dT_dt = self._build_dT_dt(
            n=n,
            rates=rates,
            temp=temp,
            reactor_volume=reactor_volume,
            y_mole=y_mole,
            F_out_total=F_out_total
        )

        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    # SECTION: Closure calculations
    def _calc_pressure_volume(
            self,
            n_total: float,
            temperature: float
    ) -> Tuple[float, float]:
        """
        Compute closure variables (P_total, V_reactor) for the active case.

        Parameters
        ----------
        n_total : float
            Total reactor moles [mol].
        temperature : float
            Reactor temperature [K].

        Returns
        -------
        Tuple[float, float]
            Total pressure [Pa], reactor volume [m3].
        """
        if self.cstr_reactor_core.is_constant_pressure:
            # NOTE: Case group: P = constant
            p_total = self._P0

            if self.cstr_reactor_core.holdup_volume_mode == "dynamic":
                # ! V = n_total * R * T / P
                reactor_volume = self.thermo_source.calc_gas_volume(
                    n_total=n_total,
                    temperature=temperature,
                    pressure=p_total,
                    R=self.R,
                    gas_model=cast(GasModel, self.gas_model)
                )
            else:
                # fixed holdup volume
                reactor_volume = self._V0
        else:
            # NOTE: Case group: V = constant, pressure varies
            reactor_volume = self._V0

            # ! P = n_total * R * T / V
            p_total = self.thermo_source.calc_tot_pressure(
                n_total=n_total,
                temperature=temperature,
                reactor_volume_value=reactor_volume,
                R=self.R,
                gas_model=cast(GasModel, self.gas_model)
            )

        if reactor_volume <= 1e-30:
            raise ValueError(
                "Calculated reactor volume is too small or non-positive. "
                "Check case closure, model inputs, and gas_model support."
            )

        if p_total <= 0.0:
            raise ValueError(
                "Calculated total pressure is non-positive. "
                "Check case closure, model inputs, and gas_model support."
            )

        return p_total, reactor_volume

    def _calc_partial_pressure(
        self,
        y_mole: np.ndarray,
        p_total: float
    ) -> Tuple[Dict[str, float], Dict[str, CustomProperty], float]:
        """
        Calculate component partial pressures from mole fractions.

        Formula
        -------
        P_i = y_i * P_total
        """
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }
        partial_pressures_std = {
            k: CustomProperty(value=v, unit="Pa", symbol="P")
            for k, v in partial_pressures.items()
        }
        return partial_pressures, partial_pressures_std, p_total

    def _calc_concentration(
        self,
        n: np.ndarray,
        reactor_volume: float
    ) -> Tuple[np.ndarray, Dict[str, CustomProperty], float]:
        """
        Calculate component concentrations in reactor gas holdup.

        Formula
        -------
        C_i = n_i / V
        """
        concentration = n / reactor_volume
        concentration_total = float(np.sum(n)) / reactor_volume
        concentration_std = {
            sp: CustomProperty(value=conc, unit="mol/m3", symbol="C")
            for sp, conc in zip(self.component_formula_state, concentration)
        }
        return concentration, concentration_std, concentration_total

    # SECTION: Calculate rates
    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ) -> np.ndarray:
        """
        Evaluate reaction rates for all defined reactions.

        Parameters
        ----------
        partial_pressures : Dict[str, CustomProperty]
            Partial pressure of the components in the reactor (in Pa).
        concentration : Dict[str, CustomProperty]
            Concentration of the components in the reactor (in mol/m3).
        temperature : Temperature
            Current temperature of the system (in K).
        pressure : Pressure
            Total pressure of the system (in Pa).

        Returns
        -------
        np.ndarray
            Array of reaction rates evaluated at the current state (in mol/m3.s or mol/s depending on rate basis).

        Notes
        -----
        - basis='pressure' -> xi = partial pressures
        - basis='concentration' -> xi = concentrations
        """
        # ! r_k = k(T, P_i) for each reaction k
        rates = []

        # iterate over reactions and evaluate rates based on their defined basis
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
                    f"Invalid basis '{basis}' for reaction rate expression '{rate_exp.name}'."
                )
            rates.append(float(r_k.value))
        return np.array(rates, dtype=float)

    # SECTION: Building dn/dt
    def _build_dn_dt(
        self,
        ns: int,
        rates: np.ndarray,
        reactor_volume: float,
        y_mole: np.ndarray,
        F_out_total: float
    ) -> np.ndarray:
        """
        Build species-time derivatives for CSTR mole balances.

        Parameters
        ----------
        ns : int
            Number of components.
        rates : np.ndarray
            Array of reaction rates evaluated at the current state (in mol/m3.s or mol/s depending on rate basis).
        reactor_volume : float
            Current reactor volume (in m3).
        y_mole : np.ndarray
            Mole fractions of the components in the reactor.
        F_out_total : float
            Total outlet molar flow rate (in mol/s).

        Balance
        -------
        dn_i/dt = F_in,i - F_out,i + V * Σ_j(ν_i,j r_j)
        """
        # CSTR mole balance term initialization
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        # ! outlet component molar flows: F_out,i = y_i * F_out,total
        F_out_i = y_mole * F_out_total
        # ??? flow terms: F_in,i - F_out,i
        f_term = self.F_in - F_out_i

        # add flow terms to dn/dt
        dn_dt += f_term

        # ! reaction source terms: V * Σ_j(ν_i,j r_j)
        # volume [m3], stoichiometry [mol/mol], rates [mol/m3.s] => source term [mol/s]
        for k, _ in enumerate(self.reactions):
            # > calculate reaction rate for reaction k
            r_k = rates[k]

            # > extract stoichiometry for reaction k
            stoich_k = self.reaction_stoichiometry[k].items()

            # >> generation term for reaction k
            # ??? g[k] = V * Σ_k ν[i][k] * r[k]

            # >> calculate dn/dt for each component i based on reaction k
            for sp_name, nu_ik in stoich_k:
                i = name_to_idx[sp_name]
                dn_dt[i] += reactor_volume * nu_ik * r_k

        return dn_dt

    # SECTION: Building dT/dt
    def _build_dT_dt(
        self,
        n: np.ndarray,
        rates: np.ndarray,
        temp: float,
        reactor_volume: float,
        y_mole: np.ndarray,
        F_out_total: float
    ) -> float:
        """
        Build temperature derivative from gas-phase CSTR energy balance.

        Parameters
        ----------
        n : np.ndarray
            Current moles of components in the reactor.
        rates : np.ndarray
            Array of reaction rates evaluated at the current state (in mol/m3.s or mol/s depending on rate basis).
        temp : float
            Current reactor temperature (in K).
        reactor_volume : float
            Current reactor volume (in m3).
        y_mole : np.ndarray
            Mole fractions of the components in the reactor.

        Returns
        -------
        float
            Time derivative of temperature (in K/s).

        Energy balance
        --------------
        (Σ_i n_i Cp_i) dT/dt =
            Σ_i F_in,i Cp_i (T_in - T)
            + V Σ_j [(-ΔH_j) r_j]
            + UA (T_j - T)
            + Q_constant
        """
        # NOTE: current temperature wrapper
        temperature = Temperature(value=temp, unit="K")
        temperature_value = temperature.value

        # NOTE: Cp_i(T) in gas phase [J/mol.K]
        Cp_IG_values_out = self.thermo_source.calc_Cp_IG(
            temperature=temperature
        )

        # ! thermal inventory coefficient: Σ_i n_i Cp_i [J/K]
        Cp_IG_total = calc_total_heat_capacity(n, Cp_IG_values_out)

        if Cp_IG_total <= 1e-16:
            raise ValueError("Total gas heat capacity is too small or zero.")

        # NOTE: reaction heat term [W]
        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        # ??? ΔH_k
        delta_h = self.thermo_source.calc_dH_rxns(temperature=temperature)

        # ??? Q_rxn
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=reactor_volume
        )

        # NOTE: inlet sensible heat term [W]
        # ! calculate enthalpy flow
        # ??? inlet enthalpy flow Fi[mol/s] * En_IG(Ti)[J/mol] => [J/s] or [W]
        En_in = np.sum(self._F_in * self.En_IG_values_in)

        # ??? outlet enthalpy flow
        En_out = np.sum(F_out_total * Cp_IG_values_out)

        # ??? enthalpy flow term
        En = En_in - En_out

        # NOTE: jacket/surrounding heat transfer term [W]
        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T)
        # U[W/m2.K], A[m2], (T_s - T)[K] => Q_exchange [W] or [J/s]
        # ??? Q_exchange
        q_exchange = 0.0

        # >> check
        if self.heat_exchange:
            # use reactor_volume=1 to obtain total UA(Tj-T) [W]
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=1.0
            )

        # NOTE: user-defined constant heat term [W]
        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value

        # ! dT/dt [K/s]
        return (En + q_rxn + q_exchange + q_constant) / Cp_IG_total

    # SECTION: Outlet Flow Configuration methods
    def _calc_F_out_total(
        self,
        n: np.ndarray,
        temp: float,
        reactor_volume: float,
        p_total: float,
        rates: np.ndarray
    ) -> float:
        """
        Compute total outlet molar flow rate [mol/s] based on outlet mode and reactor closure.
        """
        # NOTE: fixed mode
        if self.cstr_reactor_core.outlet_flow_mode == "fixed":
            if self._F_out_total is None:
                raise ValueError(
                    "Fixed outlet flow mode selected but no outlet flow was provided.")
            return float(self._F_out_total)

        # NOTE: calculated mode
        if (
            self.operation_mode == "constant_pressure" and
            self.cstr_reactor_core.holdup_volume_mode == "fixed"
        ):
            # ! fixed P and fixed V -> outlet must enforce total mole constraint
            # ! rigid vessel with inlet/outlet flow
            dTdt_est = 0.0
            dn_total_dt = -(
                p_total * reactor_volume / (self.R * temp**2)
            ) * dTdt_est

            # sum_i nu_i,j for each reaction j
            nu_sum = np.sum(
                self.reaction_stoichiometry_matrix, axis=0
            )

            generation_total = reactor_volume * float(np.dot(nu_sum, rates))
            F_in_total = float(np.sum(self.F_in))

            F_out_total = F_in_total + generation_total - dn_total_dt
            return max(F_out_total, 0.0)

        elif (
            self.operation_mode == "constant_pressure" and
            self.cstr_reactor_core.holdup_volume_mode == "dynamic"
        ):
            # ! pressure fixed by variable V, so outlet need not enforce pressure
            # ! expandable reactor
            return float(np.sum(self.F_in))

        elif self.operation_mode == "constant_volume":
            # ! volume fixed, pressure may vary, so use operating policy
            return float(np.sum(self.F_in))

        else:
            raise ValueError("Unsupported outlet-flow calculation mode.")
