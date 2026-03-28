# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
# locals
from .br import BatchReactor
from .thermo_source import ThermoSource
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.reaction_tools import stoichiometry_mat_key, stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..utils.opt_tools import calc_heat_exchange, set_component_X

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasBatchReactor(BatchReactor, ThermoSource):
    """
    GasBatchReactor class for simulating chemical reactions in a gas-phase batch reactor setup. This class inherits from the BatchReactor class and is specifically designed to handle gas-phase reactions, incorporating properties and methods relevant to gas-phase systems.

    Assumptions
    -----------
    - Constant heat capacity (Cp) for energy balance calculations.
    """
    # NOTE: Attributes

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
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
        reaction_rates : Dict[str, ReactionRateExpression]
            A dictionary containing the reaction rate expressions for the reactions occurring in the gas-phase batch reactor,
            where the keys are the names of the reactions and the values are ReactionRateExpression objects.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the model source.
        **kwargs
            Additional keyword arguments that can be passed to the initialization of the GasBatchReactor instance.
        """
        # LINK: Initialize the parent BatchReactor class
        BatchReactor.__init__(
            self,
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            component_key=component_key
        )
        # LINK: Initialize the parent ThermoSource class
        ThermoSource.__init__(
            self,
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key
        )

        # ! N: initial mole [-]
        _, self._N0 = set_component_X(
            components=components,
            prop_name="mole",
            component_key=component_key
        )

        # ! P: initial pressure [Pa]
        if self.operation_mode == "constant_volume":
            # retrieve
            self.volume = self._config_reactor_volume()
            self._V0 = self.volume.value

            # calc
            self._P0 = self.calc_tot_pressure(
                n_total=np.sum(self._N0),
                temperature=self._T0,
                reactor_volume_value=self._V0,
                R=self.R,
                gas_model=self.gas_model
            )

        # ! V: initial volume [m3]
        elif self.operation_mode == "constant_pressure":
            # retrieve
            self.pressure = self._config_pressure()
            self._P0 = self.pressure.value

            # calc
            self._V0 = self.calc_volume(
                n_total=np.sum(self._N0),
                temperature=self._T0,
                pressure=self._P0,
                R=self.R,
                gas_model=self.gas_model
            )

        else:
            raise ValueError(
                f"Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

        # SECTION: Model inputs
        self.model_inputs = model_inputs

        # SECTION: GasBatchReactor-specific properties
        self.reactor_inputs = reactor_inputs

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.build_reactions()
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

        # SECTION: Thermodynamic properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        # ! Ideal Gas Enthalpy of formation at 298 K

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
            if self.temperature_fixed is None:
                raise ValueError(
                    "temperature_fixed must be provided for isothermal simulation.")
            n = y[:ns]
            temp = float(self.temperature_fixed)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # NOTE: Calculate total moles
        n_total = np.sum(n)
        n_total = max(n_total, 1e-30)

        # Calculate partial pressures
        y_mole = n / n_total

        # ! calculate total pressure using ideal gas law: P = N_total * R * T / V
        # ! unit check: N_total [mol], R [J/mol.K], T [K], V [m3] => P [Pa]
        (
            _,
            partial_pressures_std,
            p_total,
            reactor_volume
        ) = self._calc_partial_pressure(
            n_total=n_total,
            y_mole=y_mole,
            T=temp
        )

        # ! calculate concentration: C_i = n_i / V
        (
            _,
            concentration_std,
            C_total
        ) = self._calc_concentration(
            n=n,
            reactor_volume=reactor_volume
        )

        # NOTE: Calculate Reaction rates for each component (partial pressures and temperature)
        # ! r_k = k(T, P_i) for each reaction k
        rates = self._calc_rates(
            partial_pressures=partial_pressures_std,
            concentration=concentration_std,
            temperature=Temperature(value=temp, unit="K"),
            pressure=Pressure(value=p_total, unit="Pa")
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
        # ! (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        # >>> calculate dT/dt
        dT_dt = self._build_dT_dt(
            n=n,
            rates=rates,
            temp=temp,
            reactor_volume=reactor_volume
        )

        # >>> calculate both dn/dt and dT/dt
        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    # SECTION: Calculate rates
    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ):
        """
        Calculate reaction rates for each reaction based on the current partial pressures and temperature.

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
            An array of reaction rates for each reaction in the reactor, calculated based on the current partial pressures and temperature.
        """
        # ! r_k = k(T, P_i) for each reaction k
        rates = []

        # iterate over reaction rate expressions
        for rxn_name, rate_exp in self.reaction_rates.items():
            # >> check basis
            basis = rate_exp.basis

            # >> calculate rate for reaction
            if basis == "pressure":
                # >> calculate rate based on partial pressures
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                # >> calculate rate based on concentrations
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for reaction rate expression '{rxn_name}'. Must be 'pressure' or 'concentration'."
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
            n: np.ndarray,
            rates: np.ndarray,
            temp: float,
            reactor_volume: float
    ):
        """
        Calculate the rate of change of temperature (dT/dt) based on the energy balance for the non-isothermal gas-phase batch reactor.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
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
        Cp_IG_values = self.calc_Cp_IG(
            temperature=temperature
        )

        # ??? Σ_i n_i Cp_i
        c_total = calc_total_heat_capacity(n, Cp_IG_values)

        if c_total <= 1e-16:
            raise ValueError("Total heat capacity is too small or zero.")

        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        # ??? ΔH_k
        delta_h = self.calc_dH_rxns(
            temperature=temperature
        )

        # ??? Q_rxn
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=reactor_volume
        )

        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T)
        # ??? Q_exchange
        q_exchange = 0.0

        # >>> check if heat exchange is enabled
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temperature_value,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value
            )

        # >>> calculate dT/dt
        dT_dt = (q_rxn + q_exchange) / c_total

        return dT_dt

    # SECTION: Building xi (partial pressure)
    def _calc_partial_pressure(
        self,
        n_total: float,
        y_mole: np.ndarray,
        T: float,
    ):
        """
        Calculate the partial pressures of the components based on the total moles, mole fractions, and temperature.

        Parameters
        ----------
        n_total : float
            Total moles of all components in the reactor.
        y_mole : np.ndarray
            Mole fractions of the components in the reactor.
        T : float
            Current temperature of the system (in K).
        component_key : ComponentKey
            The key to be used for the components in the model source, which is necessary for mapping the component names to their corresponding properties in the model source.

        Returns
        -------
        Tuple[Dict[str, CustomProperty], Dict[str, CustomProperty], float]
            A tuple containing:
            - A dictionary of partial pressures for each component (in Pa).
            - A dictionary of partial pressures for each component as CustomProperty objects (in Pa).
            - The total pressure of the system (in Pa).
        """
        # ! calculate total pressure using ideal gas law: P = N_total * R * T / V
        # ! unit check: N_total [mol], R [J/mol.K], T [K], V [m3] => P [Pa]
        if self.operation_mode == "constant_volume":
            # ??? Constant volume assumption: V = V0
            reactor_volume = self._V0

            # NOTE: calculate total pressure
            # ! P_total = f(n_total(t), P(t))
            p_total = self.calc_tot_pressure(
                n_total=n_total,
                temperature=T,
                reactor_volume_value=reactor_volume,
                R=self.R,
                gas_model=self.gas_model
            )
        elif self.operation_mode == "constant_pressure":
            # ??? Constant pressure assumption: P = P0
            p_total = self._P0

            # NOTE: calculate volume
            # ! V(t) = f(n_total(t), T(t))
            reactor_volume = self.calc_volume(
                n_total=n_total,
                temperature=T,
                pressure=p_total,
                R=self.R,
                gas_model=self.gas_model
            )
        else:
            raise ValueError(
                f"Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

        # NOTE: partial pressures:
        # ! P_i = y_i * P_total
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }

        # NOTE: standardize partial pressures to be used in rate calculations:
        # ??? r[k] = k(T, P_i) for each reaction k
        # >> std partial pressures
        partial_pressures_std = {}

        for k, v in partial_pressures.items():
            partial_pressures_std[k] = CustomProperty(
                value=v,
                unit="Pa",
                symbol="P"
            )

        return partial_pressures, partial_pressures_std, p_total, reactor_volume

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
