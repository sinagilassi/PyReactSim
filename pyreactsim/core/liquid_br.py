# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
# locals
from .brc import BatchReactorCore
from ..sources.thermo_source import ThermoSource
from ..models.rate_exp import ReactionRateExpression
from ..utils.reaction_tools import stoichiometry_mat_key, stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..utils.opt_tools import calc_heat_exchange, set_component_X

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidBatchReactor:
    """
    Liquid Batch Reactor (LBR) class for simulating batch reactions in liquid phase.

    Modeling assumptions
    --------------------
    - The reactor is perfectly mixed.
    - The liquid phase is spatially homogeneous.
    - Reactions occur in a single liquid phase.
    - There is no inlet and no outlet.
    - Primary state variables are moles, not concentrations.
    - Concentrations are derived quantities.
    - Internal energy balance is written on a molar volumetric heat-capacity basis.
    - SI units are used throughout.
    """
    # NOTE: Attributes

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        batch_reactor_core: BatchReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        """
        Initializes the GasBatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the gas-phase batch reactor.
        source : Source
            A Source object containing information about the source of the data or equations used in the gas-phase batch reactor simulations.
        model_inputs : Dict[str, Any]
            A dictionary containing the model inputs for the gas-phase batch reactor simulations, such as temperature, pressure, and initial mole numbers.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the options and parameters specific to the gas-phase batch reactor setup, such as heat transfer mode, volume mode, and gas model.
        reaction_rates : List[ReactionRateExpression]
            A list containing the reaction rate expressions for the reactions occurring in the gas-phase batch reactor,
            where the keys are the names of the reactions and the values are ReactionRateExpression objects.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the model source.
        **kwargs
            Additional keyword arguments that can be passed to the initialization of the GasBatchReactor instance.
        """
        # NOTE: set
        self.components = components
        self.component_key = component_key

        # SECTION: thermo source
        self.thermo_source = thermo_source

        # SECTION: batch reactor core
        self.batch_reactor_core = batch_reactor_core
        # >>>
        self.heat_transfer_mode = batch_reactor_core.heat_transfer_mode
        self.gas_model = batch_reactor_core.gas_model
        self.operation_mode = batch_reactor_core.operation_mode
        # ! heat exchange
        self.heat_exchange = batch_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = batch_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = batch_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = batch_reactor_core.jacket_temperature_value
        # ! heat rate [W]
        self.heat_rate = batch_reactor_core.heat_rate
        self.heat_rate_value = batch_reactor_core.heat_rate_value

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        # >>> build stoichiometry matrix
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        # >> matrix
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: component references
        self.component_num = self.thermo_source.component_refs['component_num']
        self.component_ids = self.thermo_source.component_refs['component_ids']
        self.component_formula_state = self.thermo_source.component_refs[
            'component_formula_state'
        ]
        self.component_mapper = self.thermo_source.component_refs['component_mapper']
        self.component_id_to_index = self.thermo_source.component_refs['component_id_to_index']

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        # >>> build stoichiometry matrix
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        # >> matrix
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: Configuration Input Stream
        # ! N: initial mole [-]
        (
            _,
            self._N0
        ) = self.batch_reactor_core.config_mole()

        # ! T: initial temperature [K]
        self.temperature = self.batch_reactor_core.config_temperature()
        self._T0 = self.temperature.value

        # ! rho: density of liquid phase [g/m3]
        # FIXME:
        self._rho_LIQ0 = self.thermo_source.calc_rho_LIQ(
            temperature=self.temperature
        )

        # ! V: initial volume [m3]
        if self.operation_mode == "constant_volume":
            # retrieve
            self.volume = self.batch_reactor_core.config_reactor_volume()
            self._V0 = self.volume.value

        elif self.operation_mode == "variable_volume":
            # calc
            self._V0 = self.thermo_source.calc_liquid_volume(
                n=self._N0,
                molecular_weights=self.thermo_source.MW,
                density=self._rho_LIQ0
            )
        else:
            raise ValueError(
                f"Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

    # SECTION: Properties
    @property
    def N0(self) -> np.ndarray:
        if self._N0 is None:
            raise ValueError("N0 has not been set.")
        return self._N0

    @N0.setter
    def N0(self, value: np.ndarray):
        self._N0 = value

    @property
    def T0(self) -> float:
        if self._T0 is None:
            raise ValueError("T0 has not been set.")
        return self._T0

    @T0.setter
    def T0(self, value: float):
        self._T0 = value

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

    # SECTION: ODE system for solve_ivp
    def rhs(
            self,
            t: float,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Right-hand side for solve_ivp.

        State vector:
        - isothermal: [n1, n2, ..., nNc]
        - non-isothermal: [n1, n2, ..., nNc, T]

        Parameters
        ----------
        t : float
            Current time in the simulation (in seconds).
        y : np.ndarray
            Current state vector of the system, which includes the moles of each component and, if applicable, the temperature.
        """
        ns = self.component_num

        if self.heat_transfer_mode == "isothermal":
            if self._T0 is None:
                raise ValueError(
                    "initial temperature fixed must be provided for isothermal simulation."
                )
            n = y[:ns]
            temp = float(self._T0)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # NOTE: Calculate total moles
        n_total = np.sum(n)
        n_total = max(n_total, 1e-30)

        # mole fraction
        y_mole = n / n_total

        # ! temperature [K]
        temperature = Temperature(value=temp, unit="K")

        # ! calculate density of liquid phase [g/m3]
        rho_LIQ = self.thermo_source.calc_rho_LIQ(
            temperature=temperature
        )

        # ! calculate system volume [m3]
        reactor_volume = self._calc_system_volume(
            n=n,
            rho_LIQ=rho_LIQ,
            temperature=temp,
        )

        # ! calculate concentration: C_i = n_i / V [mol/m3]
        (
            c,
            concentration_std,
            _
        ) = self._calc_concentration(
            n=n,
            reactor_volume=reactor_volume
        )

        # NOTE: Calculate Reaction rates for each component (partial pressures and temperature)
        # ! r_k = k(T, P_i) for each reaction k [mol/m3.s]
        rates = self._calc_rates(
            concentration=concentration_std,
            temperature=temperature,
        )

        # NOTE: Species balances:
        # ! dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = self._build_dn_dt(
            ns=ns,
            rates=rates,
            reactor_volume=reactor_volume
        )

        # >>> calculate dn/dt for isothermal case
        if self.heat_transfer_mode == "isothermal":
            return dn_dt

        # NOTE: Energy balance:
        # ! (Σ_i c_i Cp_i) dT/dt = Σ_k [(-ΔH_k) r_k] + UA (T_s - T)/V
        # >>> calculate dT/dt
        dT_dt = self._build_dT_dt(
            c=c,
            rates=rates,
            temp=temp,
            reactor_volume=reactor_volume
        )

        # >>> calculate both dn/dt and dT/dt
        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    # SECTION: Calculate rates
    def _calc_rates(
        self,
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
    ):
        """
        Calculate reaction rates for each reaction based on the current partial pressures and temperature.

        Parameters
        ----------
        concentration : Dict[str, CustomProperty]
            Concentration of the components in the reactor (in mol/m3).
        temperature : Temperature
            Current temperature of the system (in K).

        Returns
        -------
        np.ndarray
            An array of reaction rates for each reaction in the reactor, calculated based on the current partial pressures and temperature.
        """
        # ! r_k = k(T, P_i) for each reaction k
        rates = []

        # iterate over reaction rate expressions
        for rate_exp in self.reaction_rates:
            # >> check basis
            basis = rate_exp.basis

            # >> calculate rate for reaction
            if basis == "concentration":
                # >> calculate rate based on concentrations
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=None
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for reaction rate expression '{rate_exp.name}'. Must be 'concentration'."
                )

            # extract rate value
            r_k_value = r_k.value
            # append to rates list
            rates.append(r_k_value)

        # >> to array
        rates = np.array(rates, dtype=float)

        return rates

    # SECTION: Building dn/dt
    def _build_dn_dt(
            self,
            ns: int,
            rates: np.ndarray,
            reactor_volume: float
    ):
        """
        Build the rate of change of moles (dn/dt) for each component based on the reaction rates and stoichiometry.

        Parameters
        ----------
        ns : int
            Number of components in the reactor.
        rates : np.ndarray
            Array of reaction rates for each reaction in the reactor.
        reactor_volume : float
            Volume of the reactor (in m3).

        Returns
        -------
        np.ndarray
            An array of the rate of change of moles (dn/dt) for each component in the reactor, calculated based on the reaction rates and stoichiometry.
        """
        # ! dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        for k, rxn in enumerate(self.reactions):
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

            # >> Alternatively, using matrix multiplication:
            # > stoichiometry matrix
            # stoich_k_matrix = self.reaction_stoichiometry_matrix[k]

            # g_k = reactor_volume * np.dot(stoich_k_matrix, r_k)

            # for i in range(ns):
            #     dn_dt[i] += g_k[i]

        return dn_dt

    # SECTION: Building dT/dt
    def _build_dT_dt(
            self,
            c: np.ndarray,
            rates: np.ndarray,
            temp: float,
            reactor_volume: float
    ):
        """
        Calculate the rate of change of temperature (dT/dt) based on the energy balance for the non-isothermal gas-phase batch reactor.

        Parameters
        ----------
        c : np.ndarray
            Array of concentration of each component in the reactor.
        rates : np.ndarray
            Array of reaction rates for each reaction in the reactor.
        temp : float
            Current temperature of the system (in K).
        reactor_volume : float
            Volume of the reactor (in m3).

        Returns
        -------
        float
            The rate of change of temperature (dT/dt) for the non-isothermal gas-phase batch reactor.
        """
        # ! temperature
        temperature = Temperature(value=temp, unit="K")
        temperature_value = temperature.value

        # ! (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        # ??? Cp_i(T)
        Cp_LIQ_values = self.thermo_source.calc_Cp_LIQ(
            temperature=temperature
        )

        # ??? Σ_i n_i Cp_i
        Cp_LIQ_total = calc_total_heat_capacity(c, Cp_LIQ_values)

        if Cp_LIQ_total <= 1e-16:
            raise ValueError("Total heat capacity is too small or zero.")

        # ! calculate heat generated by reactions: Q_rxn = Σ_k [(-ΔH_k) r_k]
        # ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s.m^3] or [W/m^3]
        # ??? ΔH_k
        delta_h = self.thermo_source.calc_dH_rxns_IG_ref(
            temperature=temperature
        )

        # ??? Q_rxn
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=1
        )

        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T) / Vr
        # U [W/m^2.K], A [m^2], T_s [K], temp [K], Vr [m^3] => Q_exchange [W/m^3] or [J/s.m^3]
        # ??? Q_exchange
        q_exchange = 0.0

        # >>> check if heat exchange is enabled
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temperature_value,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=reactor_volume
            )

        # ??? Q_constant: constant heat rate (in W or J/s)
        # W/m^3 => J/s.m^3
        q_constant = 0.0

        # >>> check if constant heat rate is provided
        if self.heat_rate_value:
            q_constant = self.heat_rate_value/reactor_volume

        # ! >>> calculate dT/dt
        # (K/s) = (J/s.m^3) / (J/K.m^3) => K/s
        dT_dt = (q_rxn + q_exchange + q_constant) / Cp_LIQ_total

        return dT_dt

    # SECTION: Building xi (partial pressure)
    def _calc_concentration(
            self,
            n: np.ndarray,
            reactor_volume: float
    ) -> Tuple[np.ndarray, Dict[str, CustomProperty], float]:
        """
        Calculate the concentration of each component in the reactor based on the moles and reactor volume.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        reactor_volume : float
            Volume of the reactor (in m3).

        Returns
        -------
        Tuple[np.ndarray, Dict[str, CustomProperty], float]
            A tuple containing:
            - An array of concentrations for each component (in mol/m3).
            - A dictionary of concentrations for each component as CustomProperty objects (in mol/m3).
            - The total concentration of the system (in mol/m3).
        """
        # ! C_i = n_i / V
        # unit check: n_i [mol], V [m3] => C_i [mol/m3]
        concentration = n / reactor_volume

        # total concentration
        # ! C_total = N_total / V
        n_total = np.sum(n)
        concentration_total = n_total / reactor_volume

        # NOTE: create ids for concentration array
        conc_ids = [
            sp for sp in self.component_formula_state
        ]

        # std concentration as dict
        concentration_std = {
            sp: CustomProperty(
                value=conc,
                unit="mol/m3",
                symbol="C"
            ) for sp, conc in zip(conc_ids, concentration)
        }

        return concentration, concentration_std, concentration_total

    def _calc_system_volume(
            self,
            n: np.ndarray,
            rho_LIQ: np.ndarray,
            temperature: float,
    ):
        """
        Calculate the system volume based on the current moles, temperature, and pressure.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        rho_LIQ : np.ndarray
            Density of the liquid phase (in g/m3).
        temperature : float
            Current temperature of the system (in K).

        Returns
        -------
        float
            The calculated system volume (in m3) based on the current moles, temperature, and pressure, taking into account the operation mode of the reactor (constant volume or variable volume).
        """
        if self.operation_mode == "constant_volume":
            # ??? Constant volume assumption: V = V0
            reactor_volume = self._V0
        elif self.operation_mode == "variable_volume":
            # ??? Variable volume assumption: V = f(n_total(t), T(t), P(t))
            reactor_volume = self.thermo_source.calc_liquid_volume(
                n=n,
                molecular_weights=self.thermo_source.MW,
                density=rho_LIQ
            )
        else:
            raise ValueError(
                "Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

        return reactor_volume
